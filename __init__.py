bl_info = {
    "name": "Glow Creatures",
    "author": "Claude",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar (N) > Glow Creatures",
    "description": "Animals (manta, whale, dolphin, fish or any mesh) made of glowing particles "
                   "that react to audio and to key presses",
    "category": "Animation",
}

import math
import time

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)
from bpy.types import Operator, Panel, PropertyGroup

from . import core

# ---------------------------------------------------------------------------
# Runtime caches (never saved in the .blend, rebuilt on demand)
# ---------------------------------------------------------------------------
_RT = {}      # object name -> Runtime
_AUDIO = {}   # (abs path, fps) -> analysis dict


class Runtime:
    def __init__(self):
        self.sim = None
        self.sig = None          # settings signature the sim was built with
        self.last_frame = None
        self.live = []           # (frame, strength) pulses from live key mode (not saved)
        self.ev_sig = None
        self.ev = (np.zeros(0), np.zeros(0))


def _clear_caches():
    _RT.clear()
    _AUDIO.clear()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def _owner_object(settings):
    idb = settings.id_data
    return idb if isinstance(idb, bpy.types.Object) else None


def _live_object(settings):
    """The creature object that owns these settings, or None (scene template / still being built)."""
    obj = _owner_object(settings)
    return obj if (obj is not None and settings.is_creature) else None


def _upd_look(self, context):
    obj = _live_object(self)
    if obj is not None:
        apply_look(obj)


def _upd_audio(self, context):
    obj = _live_object(self)
    if obj is None:
        return
    _AUDIO.clear()
    rt = _RT.get(obj.name)
    if rt:
        rt.ev_sig = None


def _refresh(self, context, mode):
    obj = _live_object(self)
    if obj is not None:
        try:
            update_creature(obj, context.scene, None, mode=mode)
        except Exception as exc:  # never break the UI because of a slider
            print("[Glow Creatures]", exc)


def _upd_refresh(self, context):
    _refresh(self, context, "refresh")


def _upd_pose(self, context):
    _refresh(self, context, "reset")  # size / motion changes: snap particles to the new pose


def _upd_kind(self, context):
    """Picking another animal loads that animal's natural size and flap speed."""
    idb = self.id_data
    if isinstance(idb, bpy.types.Object) and not self.is_creature:
        return  # object is still being set up by the Create operator
    self.size = core.DEFAULT_SIZE.get(self.kind, 3.0)
    self.flap_speed = core.DEFAULT_SPEED.get(self.kind, 0.5)


class GC_Pulse(PropertyGroup):
    frame: IntProperty(name="Frame")
    strength: FloatProperty(name="Strength", default=1.0)


KIND_ITEMS = [
    ("MANTA", "Manta Ray", "Flapping manta ray"),
    ("WHALE", "Whale", "Big slow whale with flukes"),
    ("DOLPHIN", "Dolphin", "Dolphin with a beak"),
    ("FISH", "Fish", "Little fish"),
    ("CUSTOM", "Custom Mesh", "Use the surface of any mesh object (head toward +Y)"),
]
MOTION_ITEMS = [
    ("AUTO", "Automatic", "Pick the motion that fits the creature"),
    ("WING", "Wing flap", "Wings flap along the X axis"),
    ("BODY_V", "Swim up/down", "Wave travelling along the body, moving Z (whales, dolphins)"),
    ("BODY_H", "Swim sideways", "Wave travelling along the body, moving X (fish, sharks)"),
    ("NONE", "Still", "No body motion"),
]


class GC_Settings(PropertyGroup):
    # --- shape -------------------------------------------------------------
    kind: EnumProperty(name="Animal", items=KIND_ITEMS, default="MANTA", update=_upd_kind)
    source: PointerProperty(name="Source Mesh", type=bpy.types.Object,
                            poll=lambda self, o: o.type == 'MESH')
    count: IntProperty(name="Particles", default=60000, min=1000, max=1000000)
    dust_count: IntProperty(name="Dust", default=15000, min=0, max=500000)
    seed: IntProperty(name="Seed", default=1, min=0)
    size: FloatProperty(name="Size", default=3.0, min=0.01, soft_max=30.0, update=_upd_pose)
    controller: PointerProperty(name="Controller", type=bpy.types.Object)
    is_creature: BoolProperty(default=False)
    swim_loop: IntProperty(name="Loop Frames", default=240, min=24)
    demo_swim: BoolProperty(name="Demo swim path", default=True,
                            description="Animate the controller on a looping path")

    # --- look --------------------------------------------------------------
    color_base: FloatVectorProperty(name="Color", subtype='COLOR', size=4, min=0.0, max=1.0,
                                    default=(0.05, 0.45, 1.0, 1.0), update=_upd_look)
    color_hot: FloatVectorProperty(name="Hot Color", subtype='COLOR', size=4, min=0.0, max=1.0,
                                   default=(0.75, 0.95, 1.0, 1.0), update=_upd_look)
    intensity: FloatProperty(name="Emission", default=6.0, min=0.0, soft_max=50.0,
                             update=_upd_look)
    point_size: FloatProperty(name="Point Size", default=0.010, min=0.0005, soft_max=0.05,
                              precision=4, update=_upd_refresh)
    size_jitter: FloatProperty(name="Size Variation", default=0.6, min=0.0, max=1.0,
                               update=_upd_refresh)
    base_glow: FloatProperty(name="Base Glow", default=1.0, min=0.0, soft_max=4.0,
                             update=_upd_refresh)
    trail_fade: FloatProperty(name="Straggler Fade", default=1.0, min=0.0, soft_max=5.0,
                              description="How much particles dim when they lag behind",
                              update=_upd_refresh)
    dust_glow: FloatProperty(name="Dust Glow", default=0.5, min=0.0, soft_max=3.0,
                             update=_upd_refresh)
    dust_size: FloatProperty(name="Dust Size", default=0.012, min=0.0005, soft_max=0.05,
                             precision=4, update=_upd_refresh)
    bokeh_frac: FloatProperty(name="Orb Fraction", default=0.04, min=0.0, max=1.0,
                              description="Fraction of the dust that becomes big soft orbs",
                              update=_upd_refresh)
    bokeh_size: FloatProperty(name="Orb Size", default=6.0, min=1.0, soft_max=20.0,
                              update=_upd_refresh)
    dust_speed: FloatProperty(name="Dust Drift", default=0.5, min=0.0, soft_max=5.0,
                              update=_upd_refresh)
    dust_box: FloatProperty(name="Dust Area", default=6.0, min=0.5, soft_max=40.0,
                            description="Half size of the cube of dust that follows the creature",
                            update=_upd_refresh)

    # --- motion ------------------------------------------------------------
    motion: EnumProperty(name="Motion", items=MOTION_ITEMS, default="AUTO", update=_upd_pose)
    flap_speed: FloatProperty(name="Flap Speed (Hz)", default=0.45, min=0.0, soft_max=4.0,
                              update=_upd_pose)
    flap_amp: FloatProperty(name="Flap Amount", default=1.0, min=0.0, soft_max=3.0,
                            update=_upd_pose)
    stiffness: FloatProperty(name="Stiffness", default=40.0, min=1.0, soft_max=200.0,
                             description="How tightly particles hold their place", update=_upd_refresh)
    damping: FloatProperty(name="Damping", default=0.7, min=0.05, soft_max=2.0,
                           description="Low = springy and wobbly", update=_upd_refresh)
    trail: FloatProperty(name="Tail Lag", default=0.7, min=0.0, max=1.0,
                         description="How much more the tail lags behind the head", update=_upd_refresh)
    loose: FloatProperty(name="Looseness", default=0.5, min=0.0, max=1.0,
                         description="Random variation in how tightly each particle holds on",
                         update=_upd_refresh)
    noise: FloatProperty(name="Swirl", default=0.5, min=0.0, soft_max=3.0, update=_upd_refresh)
    noise_scale: FloatProperty(name="Swirl Scale", default=1.2, min=0.05, soft_max=5.0,
                               update=_upd_refresh)

    # --- audio -------------------------------------------------------------
    use_audio: BoolProperty(name="Use Audio File", default=False, update=_upd_audio)
    audio_path: StringProperty(name="Audio", subtype='FILE_PATH', update=_upd_audio)
    audio_offset: IntProperty(name="Offset (frames)", default=0, update=_upd_audio,
                              description="Shift the audio relative to the start of the timeline")
    bass_gain: FloatProperty(name="Bass Glow", default=1.5, min=0.0, soft_max=6.0,
                             update=_upd_refresh)
    mid_gain: FloatProperty(name="Mid Glow", default=1.0, min=0.0, soft_max=6.0,
                            update=_upd_refresh)
    high_gain: FloatProperty(name="High Sparkle", default=1.5, min=0.0, soft_max=6.0,
                             update=_upd_refresh)
    onset_gain: FloatProperty(name="Beat Strength", default=1.0, min=0.0, soft_max=3.0,
                              description="How strong the pulses triggered by detected beats are",
                              update=_upd_audio)
    bass_push: FloatProperty(name="Bass Push", default=0.5, min=0.0, soft_max=3.0,
                             description="Bass makes the body swell outward", update=_upd_refresh)
    audio_decay: FloatProperty(name="Audio Release (s)", default=0.15, min=0.01, soft_max=1.0,
                               update=_upd_refresh)
    use_bpm: BoolProperty(name="Metronome Pulses", default=False, update=_upd_audio,
                          description="Pulse on a fixed BPM grid (useful without audio)")
    bpm: FloatProperty(name="BPM", default=120.0, min=20.0, max=300.0, update=_upd_audio)
    bpm_offset: IntProperty(name="Beat Offset", default=0, update=_upd_audio)

    # --- pulses (keys) -------------------------------------------------------
    pulse_gain: FloatProperty(name="Flash", default=2.0, min=0.0, soft_max=8.0,
                              update=_upd_refresh)
    pulse_push: FloatProperty(name="Burst", default=0.6, min=0.0, soft_max=3.0,
                              update=_upd_refresh)
    dust_push: FloatProperty(name="Dust Kick", default=1.0, min=0.0, soft_max=5.0,
                             description="How strongly a pulse shoves and lights up the dust around it",
                             update=_upd_refresh)
    ripple_gain: FloatProperty(name="Ripple Glow", default=2.5, min=0.0, soft_max=8.0,
                               update=_upd_refresh)
    ripple_speed: FloatProperty(name="Ripple Speed", default=1.2, min=0.1, soft_max=6.0,
                                description="Body lengths per second the wave travels",
                                update=_upd_refresh)
    ripple_width: FloatProperty(name="Ripple Width", default=0.15, min=0.02, max=1.0,
                                update=_upd_refresh)
    pulse_decay: FloatProperty(name="Pulse Release (s)", default=0.35, min=0.02, soft_max=3.0,
                               update=_upd_refresh)
    record_keys: BoolProperty(name="Record Key Presses", default=True,
                              description="Store pulses so renders reproduce them")
    events: CollectionProperty(type=GC_Pulse)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scene_fps(scene):
    return scene.render.fps / max(scene.render.fps_base, 1e-6)


def _active_creature(context):
    obj = context.active_object
    if obj is not None and obj.type == 'MESH' and getattr(obj, "glow_creature", None) \
            and obj.glow_creature.is_creature:
        return obj
    return None


def _settings_for_ui(context):
    obj = _active_creature(context)
    return (obj.glow_creature if obj else context.scene.glow_creature_new), obj


def _copy_settings(src, dst):
    for p in src.bl_rna.properties:
        pid = p.identifier
        if pid in {"rna_type", "name", "events", "source", "controller", "is_creature"}:
            continue
        try:
            val = getattr(src, pid)
            if hasattr(val, "__len__") and not isinstance(val, str):
                val = tuple(val)
            setattr(dst, pid, val)
        except Exception:
            pass
    dst.source = src.source


def _params(st):
    motion = st.motion
    if motion == "AUTO":
        motion = core.DEFAULT_MOTION.get(st.kind, "BODY_V")
    vals = {n: getattr(st, n) for n in core.PARAM_NAMES if n != "motion"}
    return core.SimParams(motion=motion, **vals)


def _extract_mesh(obj, depsgraph):
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        n = len(me.vertices)
        v = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3).astype(np.float64)
        M = np.array(obj.matrix_world, dtype=np.float64)
        v = v @ M[:3, :3].T + M[:3, 3]
        tris = None
        try:
            me.calc_loop_triangles()
        except Exception:
            pass
        nt = len(me.loop_triangles)
        if nt:
            t = np.empty(nt * 3, np.int32)
            me.loop_triangles.foreach_get("vertices", t)
            tris = t.reshape(-1, 3)
    finally:
        ev.to_mesh_clear()
    return v, tris


# ---------------------------------------------------------------------------
# Material + geometry nodes
# ---------------------------------------------------------------------------
def _find_output(node, name):
    for o in node.outputs:
        if o.name == name and getattr(o, "enabled", True):
            return o
    return node.outputs[0]


def apply_look(obj):
    st = obj.glow_creature
    mat = bpy.data.materials.get(obj.get("gc_mat", ""))
    if mat is None or mat.node_tree is None:
        return
    nodes = mat.node_tree.nodes
    ramp = nodes.get("GC_Ramp")
    if ramp is not None:
        base = tuple(st.color_base)
        hot = tuple(st.color_hot)
        els = ramp.color_ramp.elements
        els[0].color = (base[0] * 0.2, base[1] * 0.2, base[2] * 0.2, 1.0)
        els[1].color = base
        els[2].color = hot
    mul = nodes.get("GC_Intensity")
    if mul is not None:
        mul.inputs[1].default_value = st.intensity


def make_material(obj):
    name = "GC_Mat_" + obj.name
    mat = bpy.data.materials.get(name)
    if mat is not None:
        bpy.data.materials.remove(mat)
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emis = nt.nodes.new("ShaderNodeEmission")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = 'GEOMETRY'
    attr.attribute_name = "glow"
    mapr = nt.nodes.new("ShaderNodeMapRange")
    mapr.inputs[1].default_value = 0.0   # From Min
    mapr.inputs[2].default_value = 2.0   # From Max
    mapr.inputs[3].default_value = 0.0
    mapr.inputs[4].default_value = 1.0
    mapr.clamp = True
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.name = "GC_Ramp"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].position = 0.5
    ramp.color_ramp.elements.new(1.0)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.name = "GC_Intensity"
    mul.operation = 'MULTIPLY'
    nt.links.new(attr.outputs["Fac"], mapr.inputs[0])
    nt.links.new(mapr.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], emis.inputs["Color"])
    nt.links.new(attr.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs[0], emis.inputs["Strength"])
    nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (0.1, 0.5, 1.0, 1.0)
    return mat


def make_gn_group(obj, mat):
    name = "GC_Points_" + obj.name
    old = bpy.data.node_groups.get(name)
    if old is not None:
        bpy.data.node_groups.remove(old)
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    if hasattr(ng, "interface"):
        ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    else:  # Blender 3.x
        ng.inputs.new('NodeSocketGeometry', "Geometry")
        ng.outputs.new('NodeSocketGeometry', "Geometry")
    n = ng.nodes
    gin = n.new("NodeGroupInput")
    gout = n.new("NodeGroupOutput")
    m2p = n.new("GeometryNodeMeshToPoints")
    named = n.new("GeometryNodeInputNamedAttribute")
    named.data_type = 'FLOAT'
    named.inputs["Name"].default_value = "size"
    setmat = n.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    gin.location = (-500, 0)
    named.location = (-500, -200)
    m2p.location = (-250, 0)
    setmat.location = (0, 0)
    gout.location = (250, 0)
    ng.links.new(gin.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(_find_output(named, "Attribute"), m2p.inputs["Radius"])
    ng.links.new(m2p.outputs["Points"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], gout.inputs[0])
    return ng


def _ensure_attributes(me):
    for nm in ("glow", "size"):
        if nm not in me.attributes:
            me.attributes.new(nm, 'FLOAT', 'POINT')


# ---------------------------------------------------------------------------
# Building / updating creatures
# ---------------------------------------------------------------------------
def _build_sim(obj, context_or_depsgraph, st):
    verts = tris = None
    if st.kind == "CUSTOM":
        if st.source is None:
            raise RuntimeError("Pick a Source Mesh for the Custom animal")
        dg = context_or_depsgraph
        verts, tris = _extract_mesh(st.source, dg)
    cr = core.build_creature(st.kind, st.count, st.seed, verts=verts, tris=tris)
    P = _params(st)
    return core.CreatureSim(cr, st.dust_count, st.seed, P.motion)


def _sig(st):
    return (st.kind, st.count, st.dust_count, st.seed, st.source.name if st.source else "")


def _fill_mesh(obj, sim):
    me = obj.data
    me.clear_geometry()
    me.vertices.add(sim.N)
    _ensure_attributes(me)
    me.update()


def _get_runtime(obj, context):
    st = obj.glow_creature
    rt = _RT.get(obj.name)
    sig = _sig(st)
    if rt is None or rt.sim is None or rt.sig != sig:
        if rt is None:
            rt = Runtime()
            _RT[obj.name] = rt
        dg = context.evaluated_depsgraph_get() if hasattr(context, "evaluated_depsgraph_get") \
            else bpy.context.evaluated_depsgraph_get()
        rt.sim = _build_sim(obj, dg, st)
        rt.sig = sig
        rt.last_frame = None
        if len(obj.data.vertices) != rt.sim.N:
            _fill_mesh(obj, rt.sim)
    return rt


def _controller_matrix(st, depsgraph):
    ctrl = st.controller
    if ctrl is None:
        return np.eye(4, dtype=np.float32)
    try:
        m = ctrl.evaluated_get(depsgraph).matrix_world if depsgraph else ctrl.matrix_world
    except Exception:
        m = ctrl.matrix_world
    return np.array(m, dtype=np.float32)


def _audio_analysis(st, fps):
    if not (st.use_audio and st.audio_path):
        return None
    path = bpy.path.abspath(st.audio_path)
    key = (path, round(fps, 3))
    if key in _AUDIO:
        return _AUDIO[key]
    try:
        data, rate = core.load_audio(path)
        res = core.analyze_audio(data, rate, fps)
    except Exception as exc:
        print("[Glow Creatures] could not read audio '%s': %s" % (path, exc))
        res = None
    _AUDIO[key] = res
    return res


def _events(obj, rt, st, scene, fps):
    sig = (st.use_audio, st.audio_path, st.audio_offset, st.onset_gain, st.use_bpm, st.bpm,
           st.bpm_offset, len(st.events), len(rt.live), scene.frame_start, scene.frame_end,
           round(fps, 3))
    if rt.ev_sig == sig:
        return rt.ev
    frames = []
    strengths = []
    an = _audio_analysis(st, fps)
    if an is not None and len(an["events"]):
        base = scene.frame_start + st.audio_offset
        frames.append(an["events"][:, 0] + base)
        strengths.append(an["events"][:, 1] * st.onset_gain)
    if st.use_bpm:
        bf = core.beat_events(st.bpm, scene.frame_start + st.bpm_offset, fps,
                              scene.frame_start, scene.frame_end + 240)
        frames.append(bf)
        strengths.append(np.ones(len(bf)))
    if len(st.events):
        frames.append(np.array([e.frame for e in st.events], np.float64))
        strengths.append(np.array([e.strength for e in st.events], np.float64))
    if rt.live:
        frames.append(np.array([f for f, _ in rt.live], np.float64))
        strengths.append(np.array([s for _, s in rt.live], np.float64))
    if frames:
        f = np.concatenate(frames)
        s = np.concatenate(strengths)
        o = np.argsort(f, kind="stable")
        rt.ev = (f[o], s[o])
    else:
        rt.ev = (np.zeros(0), np.zeros(0))
    rt.ev_sig = sig
    return rt.ev


def _signals(obj, rt, st, scene, frame_f, fps):
    sig = {"bass": 0.0, "mid": 0.0, "high": 0.0}
    an = _audio_analysis(st, fps)
    if an is not None:
        idx = frame_f - (scene.frame_start + st.audio_offset)
        for k in ("bass", "mid", "high"):
            arr = an[k]
            if 0.0 <= idx <= len(arr) - 1:
                sig[k] = float(np.interp(idx, np.arange(len(arr)), arr))
    evf, evs = _events(obj, rt, st, scene, fps)
    env, age, strength = core.eval_events(evf, evs, frame_f, st.pulse_decay * fps, fps)
    sig.update(pulse=env, age=age, strength=strength)
    return sig


def update_creature(obj, scene, depsgraph, mode="step"):
    """mode: 'step' (frame changed), 'refresh' (re-evaluate current frame), 'reset'."""
    st = obj.glow_creature
    ctx = bpy.context
    rt = _get_runtime(obj, ctx)
    sim = rt.sim
    fps = _scene_fps(scene)
    frame_f = scene.frame_current + scene.frame_subframe
    t = frame_f / fps
    P = _params(st)
    M = _controller_matrix(st, depsgraph)
    sig = _signals(obj, rt, st, scene, frame_f, fps)

    if mode == "reset" or rt.last_frame is None:
        sim.reset(t, M, P)
        sim.step(t, 1e-4, M, P, sig, _no_move=True)
        rt.last_frame = frame_f
    elif mode == "refresh" or frame_f == rt.last_frame:
        sim.step(t, 1e-4, M, P, sig, _no_move=True)
    else:
        diff = frame_f - rt.last_frame
        if 0.0 < diff <= 4.0:
            sim.step(t, diff / fps, M, P, sig)
        else:  # jumped (scrub / loop): no history, snap to pose
            sim.reset(t, M, P)
            sim.step(t, 1e-4, M, P, sig, _no_move=True)
        rt.last_frame = frame_f

    # write into mesh, in the object's local space
    me = obj.data
    if len(me.vertices) != sim.N:
        _fill_mesh(obj, sim)
    _ensure_attributes(me)
    pos = sim.pos
    mw = np.array(obj.matrix_world, dtype=np.float64)
    if not np.allclose(mw, np.eye(4)):
        inv = np.linalg.inv(mw)
        pos = (pos.astype(np.float64) @ inv[:3, :3].T + inv[:3, 3]).astype(np.float32)
    me.vertices.foreach_set("co", np.ascontiguousarray(pos).ravel())
    me.attributes["glow"].data.foreach_set("value", sim.glow)
    me.attributes["size"].data.foreach_set("value", sim.size)
    me.update()
    obj.update_tag()


@persistent
def _on_frame(scene, depsgraph=None):
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        st = getattr(obj, "glow_creature", None)
        if st is None or not st.is_creature:
            continue
        try:
            update_creature(obj, scene, depsgraph, mode="step")
        except Exception as exc:
            if getattr(_on_frame, "_last_err", None) != str(exc):
                print("[Glow Creatures]", obj.name, "->", exc)
                _on_frame._last_err = str(exc)


@persistent
def _on_load(_):
    _clear_caches()


def build_into(context, obj, st):
    """(Re)generate the particle data of an existing creature object."""
    _RT.pop(obj.name, None)
    rt = _get_runtime(obj, context)
    _fill_mesh(obj, rt.sim)
    update_creature(obj, context.scene, None, mode="reset")


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class GC_OT_create(Operator):
    bl_idname = "glow_creature.create"
    bl_label = "Create Glow Creature"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        tmpl = scene.glow_creature_new
        if tmpl.kind == "CUSTOM" and tmpl.source is None:
            self.report({'ERROR'}, "Pick a Source Mesh first (or choose another animal)")
            return {'CANCELLED'}
        coll = context.collection
        loc = scene.cursor.location.copy()
        base = {"MANTA": "Manta", "WHALE": "Whale", "DOLPHIN": "Dolphin",
                "FISH": "Fish", "CUSTOM": "Creature"}[tmpl.kind]

        me = bpy.data.meshes.new(base + "_Particles")
        obj = bpy.data.objects.new(base + "_Particles", me)
        coll.objects.link(obj)
        _copy_settings(tmpl, obj.glow_creature)
        st = obj.glow_creature
        st.is_creature = True
        st.source = tmpl.source
        obj.lock_location = obj.lock_rotation = obj.lock_scale = (True, True, True)

        ctrl = bpy.data.objects.new(base + "_Controller", None)
        ctrl.empty_display_type = 'ARROWS'
        ctrl.empty_display_size = 1.0
        ctrl.location = loc
        coll.objects.link(ctrl)
        st.controller = ctrl

        mat = make_material(obj)
        obj["gc_mat"] = mat.name
        ng = make_gn_group(obj, mat)
        mod = obj.modifiers.new("GlowPoints", 'NODES')
        mod.node_group = ng
        apply_look(obj)

        try:
            build_into(context, obj, st)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        if st.demo_swim:
            _demo_swim(scene, ctrl, loc, st.swim_loop)

        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        self.report({'INFO'}, "Created %s. Move or animate '%s' to make it swim; set the viewport "
                              "to Material Preview / Rendered to see the glow." % (obj.name, ctrl.name))
        return {'FINISHED'}


def _demo_swim(scene, ctrl, center, loop):
    f0 = scene.frame_start
    radius = 9.0
    step = max(loop // 40, 2)
    frames = list(range(0, loop + 1, step))
    if frames[-1] != loop:
        frames.append(loop)
    for k in frames:
        th = 2.0 * math.pi * k / loop
        ctrl.location = (center[0] + radius * math.cos(th),
                         center[1] + radius * math.sin(th),
                         center[2] + 1.2 * math.sin(2.0 * th))
        ctrl.rotation_euler = (0.0, 0.25 * math.sin(th), th)
        ctrl.keyframe_insert("location", frame=f0 + k)
        ctrl.keyframe_insert("rotation_euler", frame=f0 + k)
    ad = ctrl.animation_data
    if ad and ad.action:
        fcs = getattr(ad.action, "fcurves", None)
        if fcs is None:  # Blender 5.0 layered actions
            try:
                fcs = [fc for layer in ad.action.layers for strip in layer.strips
                       for bag in strip.channelbags for fc in bag.fcurves]
            except Exception:
                fcs = []
        for fc in fcs:
            fc.modifiers.new('CYCLES')
    ctrl.location = (center[0] + radius, center[1], center[2])


class GC_OT_rebuild(Operator):
    bl_idname = "glow_creature.rebuild"
    bl_label = "Rebuild Particles"
    bl_description = "Regenerate the particles with the current animal / count / seed"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_creature(context) is not None

    def execute(self, context):
        obj = _active_creature(context)
        try:
            build_into(context, obj, obj.glow_creature)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class GC_OT_setup_scene(Operator):
    bl_idname = "glow_creature.setup_scene"
    bl_label = "Set Up Glow Scene"
    bl_description = "Black world, standard color, EEVEE bloom/glare so the particles glow"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        done, failed = [], []

        def attempt(label, fn):
            try:
                fn()
                done.append(label)
            except Exception as exc:
                failed.append("%s (%s)" % (label, exc))

        def world():
            w = scene.world or bpy.data.worlds.new("World")
            scene.world = w
            try:
                w.use_nodes = True
            except Exception:
                pass
            bg = w.node_tree.nodes.get("Background")
            if bg is not None:
                bg.inputs["Color"].default_value = (0.0, 0.0, 0.003, 1.0)
                bg.inputs["Strength"].default_value = 1.0

        def color():
            scene.view_settings.view_transform = 'Standard'
            try:
                scene.view_settings.look = 'None'
            except Exception:
                pass

        def engine():
            items = [e.identifier for e in
                     bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
            for cand in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
                if cand in items:
                    scene.render.engine = cand
                    return
            raise RuntimeError("no EEVEE engine found")

        def bloom():  # Blender <= 4.1 only
            ee = scene.eevee
            if hasattr(ee, "use_bloom"):
                ee.use_bloom = True
                ee.bloom_threshold = 0.6
                ee.bloom_intensity = 0.25
                ee.bloom_radius = 6.5

        def glare():
            _setup_glare(scene)

        def viewport():
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    sh = area.spaces.active.shading
                    sh.type = 'MATERIAL'
                    try:
                        sh.use_compositor = 'ALWAYS'
                    except Exception:
                        pass

        attempt("black world", world)
        attempt("standard color", color)
        attempt("EEVEE", engine)
        attempt("bloom", bloom)
        attempt("compositor glare", glare)
        attempt("viewport shading", viewport)
        msg = "Scene set up: " + ", ".join(done)
        if failed:
            msg += ". Could not set: " + "; ".join(failed) + \
                   " -> add a Glare (Fog Glow) node in the Compositor yourself."
        self.report({'WARNING' if failed else 'INFO'}, msg)
        return {'FINISHED'}


def _set_any(node, name, value):
    if hasattr(node, name):
        setattr(node, name, value)
        return True
    for key in (name, name.title(), name.replace("_", " ").title()):
        if key in node.inputs:
            node.inputs[key].default_value = value
            return True
    return False


def _setup_glare(scene):
    tree = None
    if hasattr(scene, "use_nodes"):
        try:
            scene.use_nodes = True
        except Exception:
            pass
    tree = getattr(scene, "node_tree", None)
    group_mode = False
    if tree is None and hasattr(scene, "compositing_node_group"):
        tree = bpy.data.node_groups.get("GC_Compositor") or \
            bpy.data.node_groups.new("GC_Compositor", 'CompositorNodeTree')
        scene.compositing_node_group = tree
        group_mode = True
    if tree is None:
        raise RuntimeError("no compositor node tree available")
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    gl = tree.nodes.new("CompositorNodeGlare")
    gl.name = "GC_Glare"
    try:
        gl.glare_type = 'FOG_GLOW'
    except Exception:
        pass
    for k, v in (("quality", 'HIGH'), ("threshold", 0.4), ("size", 7), ("mix", 0.0),
                 ("strength", 0.6)):
        try:
            _set_any(gl, k, v)
        except Exception:
            pass
    if group_mode:
        if hasattr(tree, "interface") and not any(
                it.name == "Image" for it in tree.interface.items_tree):
            tree.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
        out = tree.nodes.new("NodeGroupOutput")
    else:
        out = tree.nodes.new("CompositorNodeComposite")
    rl.location = (-300, 0)
    gl.location = (0, 0)
    out.location = (300, 0)
    tree.links.new(rl.outputs["Image"], gl.inputs["Image"])
    tree.links.new(gl.outputs["Image"], out.inputs[0])


class GC_OT_add_audio(Operator):
    bl_idname = "glow_creature.add_audio"
    bl_label = "Add Audio To Timeline"
    bl_description = "Put the audio file in the sequencer so you hear it during playback and render"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        st, _ = _settings_for_ui(context)
        return bool(st.audio_path)

    def execute(self, context):
        st, _ = _settings_for_ui(context)
        scene = context.scene
        path = bpy.path.abspath(st.audio_path)
        se = scene.sequence_editor or scene.sequence_editor_create()
        seqs = se.strips if hasattr(se, "strips") else se.sequences
        for s in list(seqs):
            if s.type == 'SOUND' and bpy.path.abspath(getattr(s.sound, "filepath", "")) == path:
                self.report({'INFO'}, "Audio is already in the timeline")
                return {'FINISHED'}
        try:
            seqs.new_sound("GC_Audio", path, channel=1, frame_start=scene.frame_start)
        except Exception as exc:
            self.report({'ERROR'}, "Could not add audio: %s" % exc)
            return {'CANCELLED'}
        return {'FINISHED'}


class GC_OT_analyze(Operator):
    bl_idname = "glow_creature.analyze"
    bl_label = "Analyze Audio"
    bl_description = "Read the audio file and find the bass / mid / high energy and the beats"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        st, _ = _settings_for_ui(context)
        return bool(st.audio_path)

    def execute(self, context):
        st, obj = _settings_for_ui(context)
        _AUDIO.clear()
        fps = _scene_fps(context.scene)
        t0 = time.time()
        try:
            path = bpy.path.abspath(st.audio_path)
            data, rate = core.load_audio(path)
            res = core.analyze_audio(data, rate, fps)
            _AUDIO[(path, round(fps, 3))] = res
        except Exception as exc:
            self.report({'ERROR'}, "Audio analysis failed: %s" % exc)
            return {'CANCELLED'}
        if obj is not None and obj.name in _RT:
            _RT[obj.name].ev_sig = None
        self.report({'INFO'}, "%.1f s of audio, %d beats found (%.1fs). Turn on 'Use Audio File'."
                    % (res["duration"], len(res["events"]), time.time() - t0))
        return {'FINISHED'}


def add_pulse(obj, frame, strength=1.0):
    st = obj.glow_creature
    rt = _RT.get(obj.name)
    if rt is None:
        rt = _get_runtime(obj, bpy.context)
    rt.live.append((float(frame), float(strength)))
    rt.ev_sig = None
    if st.record_keys:
        e = st.events.add()
        e.frame = int(frame)
        e.strength = float(strength)


class GC_OT_pulse(Operator):
    bl_idname = "glow_creature.pulse"
    bl_label = "Pulse At Current Frame"
    bl_description = "Fire a flash + ripple at the current frame (saved, so renders reproduce it)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_creature(context) is not None

    def execute(self, context):
        obj = _active_creature(context)
        st = obj.glow_creature
        e = st.events.add()
        e.frame = context.scene.frame_current
        e.strength = 1.0
        rt = _RT.get(obj.name)
        if rt:
            rt.ev_sig = None
        update_creature(obj, context.scene, None, mode="refresh")
        return {'FINISHED'}


class GC_OT_clear_pulses(Operator):
    bl_idname = "glow_creature.clear_pulses"
    bl_label = "Clear Recorded Pulses"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_creature(context) is not None

    def execute(self, context):
        obj = _active_creature(context)
        obj.glow_creature.events.clear()
        rt = _RT.get(obj.name)
        if rt:
            rt.live.clear()
            rt.ev_sig = None
        return {'FINISHED'}


_PULSE_KEYS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {
    "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
    "COMMA", "PERIOD", "SEMI_COLON", "QUOTE", "SLASH", "MINUS", "EQUAL",
    "LEFT_BRACKET", "RIGHT_BRACKET", "BACK_SLASH", "ACCENT_GRAVE",
    "NUMPAD_0", "NUMPAD_1", "NUMPAD_2", "NUMPAD_3", "NUMPAD_4", "NUMPAD_5", "NUMPAD_6",
    "NUMPAD_7", "NUMPAD_8", "NUMPAD_9", "RET", "BACK_SPACE", "TAB"}
_PULSE_KEYS -= {"TAB", "RET", "BACK_SPACE"}


class GC_OT_live_keys(Operator):
    bl_idname = "glow_creature.live_keys"
    bl_label = "Play + Key Pulses"
    bl_description = ("Start playback; every key you press fires a flash and ripple through the "
                      "creature. ESC stops. Pulses are recorded so you can render them")

    obj_name: StringProperty()

    @classmethod
    def poll(cls, context):
        return _active_creature(context) is not None

    def invoke(self, context, event):
        obj = _active_creature(context)
        self.obj_name = obj.name
        context.window_manager.modal_handler_add(self)
        if not context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()
        context.workspace.status_text_set(
            "Glow Creatures: press any key to pulse  |  SPACE pause  |  ESC stop")
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        context.workspace.status_text_set(None)
        if context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            self._finish(context)
            return {'FINISHED'}
        obj = bpy.data.objects.get(self.obj_name)
        if obj is None:
            self._finish(context)
            return {'CANCELLED'}
        if (event.type in _PULSE_KEYS and event.value == 'PRESS' and not event.is_repeat
                and not (event.ctrl or event.alt or event.oskey)):
            add_pulse(obj, context.scene.frame_current, 1.0)
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}

    def cancel(self, context):
        context.workspace.status_text_set(None)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
class _Base:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Glow Creatures"


class GC_PT_main(_Base, Panel):
    bl_label = "Glow Creatures"
    bl_idname = "GC_PT_main"

    def draw(self, context):
        lay = self.layout
        st, obj = _settings_for_ui(context)
        if obj is None:
            lay.label(text="New creature", icon='ADD')
        else:
            lay.label(text=obj.name, icon='OUTLINER_OB_POINTCLOUD')
        lay.prop(st, "kind")
        if st.kind == "CUSTOM":
            lay.prop(st, "source")
            lay.label(text="Head should point toward +Y", icon='INFO')
        col = lay.column(align=True)
        col.prop(st, "count")
        col.prop(st, "dust_count")
        col.prop(st, "seed")
        lay.prop(st, "size")
        if obj is None:
            lay.prop(st, "demo_swim")
            if st.demo_swim:
                lay.prop(st, "swim_loop")
            lay.operator("glow_creature.create", icon='PARTICLES')
        else:
            lay.prop(st, "controller")
            lay.operator("glow_creature.rebuild", icon='FILE_REFRESH')
            lay.label(text="Move / animate the Controller to make it swim", icon='INFO')
        lay.operator("glow_creature.setup_scene", icon='WORLD')


class GC_PT_look(_Base, Panel):
    bl_label = "Look"
    bl_parent_id = "GC_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        st, _ = _settings_for_ui(context)
        lay = self.layout
        col = lay.column(align=True)
        col.prop(st, "color_base")
        col.prop(st, "color_hot")
        col.prop(st, "intensity")
        col = lay.column(align=True)
        col.prop(st, "point_size")
        col.prop(st, "size_jitter")
        col.prop(st, "base_glow")
        col.prop(st, "trail_fade")
        lay.label(text="Dust & orbs")
        col = lay.column(align=True)
        col.prop(st, "dust_glow")
        col.prop(st, "dust_size")
        col.prop(st, "bokeh_frac")
        col.prop(st, "bokeh_size")
        col.prop(st, "dust_speed")
        col.prop(st, "dust_box")


class GC_PT_motion(_Base, Panel):
    bl_label = "Motion"
    bl_parent_id = "GC_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        st, _ = _settings_for_ui(context)
        col = self.layout.column(align=True)
        col.prop(st, "motion")
        col.prop(st, "flap_speed")
        col.prop(st, "flap_amp")
        col = self.layout.column(align=True)
        col.prop(st, "stiffness")
        col.prop(st, "damping")
        col.prop(st, "trail")
        col.prop(st, "loose")
        col = self.layout.column(align=True)
        col.prop(st, "noise")
        col.prop(st, "noise_scale")


class GC_PT_audio(_Base, Panel):
    bl_label = "Audio Reaction"
    bl_parent_id = "GC_PT_main"

    def draw(self, context):
        st, obj = _settings_for_ui(context)
        lay = self.layout
        lay.prop(st, "use_audio")
        col = lay.column()
        col.active = st.use_audio
        col.prop(st, "audio_path", text="")
        row = col.row(align=True)
        row.operator("glow_creature.analyze", icon='SOUND')
        row.operator("glow_creature.add_audio", icon='SEQUENCE')
        col.prop(st, "audio_offset")
        col = lay.column(align=True)
        col.prop(st, "bass_gain")
        col.prop(st, "mid_gain")
        col.prop(st, "high_gain")
        col.prop(st, "bass_push")
        col.prop(st, "audio_decay")
        col.prop(st, "onset_gain")
        lay.prop(st, "use_bpm")
        if st.use_bpm:
            row = lay.row(align=True)
            row.prop(st, "bpm")
            row.prop(st, "bpm_offset")


class GC_PT_keys(_Base, Panel):
    bl_label = "Key Pulses"
    bl_parent_id = "GC_PT_main"

    def draw(self, context):
        st, obj = _settings_for_ui(context)
        lay = self.layout
        lay.operator("glow_creature.live_keys", icon='PLAY')
        lay.prop(st, "record_keys")
        row = lay.row(align=True)
        row.operator("glow_creature.pulse", icon='KEYFRAME_HLT')
        row.operator("glow_creature.clear_pulses", text="", icon='X')
        if obj is not None:
            lay.label(text="%d pulses recorded" % len(st.events))
        col = lay.column(align=True)
        col.prop(st, "pulse_gain")
        col.prop(st, "pulse_push")
        col.prop(st, "dust_push")
        col.prop(st, "ripple_gain")
        col.prop(st, "ripple_speed")
        col.prop(st, "ripple_width")
        col.prop(st, "pulse_decay")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
classes = (
    GC_Pulse, GC_Settings,
    GC_OT_create, GC_OT_rebuild, GC_OT_setup_scene, GC_OT_add_audio, GC_OT_analyze,
    GC_OT_pulse, GC_OT_clear_pulses, GC_OT_live_keys,
    GC_PT_main, GC_PT_look, GC_PT_motion, GC_PT_audio, GC_PT_keys,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Object.glow_creature = PointerProperty(type=GC_Settings)
    bpy.types.Scene.glow_creature_new = PointerProperty(type=GC_Settings)
    bpy.app.handlers.frame_change_post.append(_on_frame)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    for h, fn in ((bpy.app.handlers.frame_change_post, _on_frame),
                  (bpy.app.handlers.load_post, _on_load)):
        if fn in h:
            h.remove(fn)
    del bpy.types.Scene.glow_creature_new
    del bpy.types.Object.glow_creature
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    _clear_caches()


if __name__ == "__main__":
    register()
