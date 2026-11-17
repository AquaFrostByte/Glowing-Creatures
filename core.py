"""Glow Creatures - pure numpy core (no bpy import, so it can be unit tested).

Coordinate convention for every creature: rest pose is centred near the origin,
the head points to +Y, the "up" axis is +Z, wings / flukes spread along X.
Roughly 2 units long, so the Blender side only has to multiply by `size`.
"""
import math
import wave
from dataclasses import dataclass, fields

import numpy as np

KINDS = ("MANTA", "WHALE", "DOLPHIN", "FISH", "CUSTOM")
DEFAULT_MOTION = {
    "MANTA": "WING",
    "WHALE": "BODY_V",
    "DOLPHIN": "BODY_V",
    "FISH": "BODY_H",
    "CUSTOM": "BODY_V",
}
# sensible per-kind flap / swim frequency (Hz)
DEFAULT_SPEED = {"MANTA": 0.45, "WHALE": 0.25, "DOLPHIN": 0.75, "FISH": 1.4, "CUSTOM": 0.5}
DEFAULT_SIZE = {"MANTA": 3.0, "WHALE": 5.0, "DOLPHIN": 2.5, "FISH": 1.5, "CUSTOM": 3.0}


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------
def sample_tris(rng, tris, n):
    """Area-weighted random points on triangles. tris: (m, 3, 3)."""
    tris = np.asarray(tris, dtype=np.float64)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tot = area.sum()
    if tot <= 1e-12:
        return a[rng.integers(0, len(a), n)]
    idx = rng.choice(len(tris), size=n, p=area / tot)
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    w0, w1, w2 = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
    return w0[:, None] * a[idx] + w1[:, None] * b[idx] + w2[:, None] * c[idx]


def strip_tris(p, q):
    """Two polylines (k,3) -> triangles that fill the strip between them."""
    t1 = np.stack([p[:-1], p[1:], q[1:]], axis=1)
    t2 = np.stack([p[:-1], q[1:], q[:-1]], axis=1)
    return np.concatenate([t1, t2], axis=0)


# --------------------------------------------------------------------------
# Procedural creatures
# --------------------------------------------------------------------------
def gen_manta(rng, n):
    n_wing = int(n * 0.86)
    n_tail = int(n * 0.05)
    n_lobe = n - n_wing - n_tail

    m = n_wing * 4
    u = rng.uniform(-1.0, 1.0, m)
    au = np.abs(u)
    y_lead = 0.45 - 0.85 * au ** 1.25
    y_trail = -0.55 + 0.10 * au + 0.10 * np.sin(np.pi * au)
    chord = y_lead - y_trail
    ok = np.flatnonzero(rng.random(m) < chord / chord.max())
    if len(ok) < n_wing:  # extremely unlikely; pad
        ok = np.concatenate([ok, rng.choice(ok, n_wing - len(ok))])
    ok = ok[:n_wing]
    u, au, y_lead, y_trail, chord = u[ok], au[ok], y_lead[ok], y_trail[ok], chord[ok]
    v = rng.random(n_wing)
    y = y_trail + v * chord
    x = u * 1.0
    thick = 0.11 * (1.0 - au) ** 1.6 * np.sin(np.pi * (0.12 + 0.76 * v)) + 0.008
    side = np.where(rng.random(n_wing) < 0.5, 1.0, -0.6)
    z = side * thick - 0.05 * au ** 2
    wings = np.stack([x, y, z], axis=1)

    t = rng.random(n_tail)
    ang = rng.random(n_tail) * 2 * np.pi
    r = 0.018 * (1.0 - t) ** 1.2 + 0.003
    tail = np.stack([r * np.cos(ang), -0.52 - 1.1 * t, r * np.sin(ang)], axis=1)

    lobes = []
    for sx in (-1.0, 1.0):
        tri = np.array([[[sx * 0.06, 0.40, 0.0], [sx * 0.17, 0.35, 0.0], [sx * 0.12, 0.60, 0.0]]])
        lobes.append(tri)
    lobe = sample_tris(rng, np.concatenate(lobes, axis=0), n_lobe)
    return np.concatenate([wings, tail, lobe], axis=0)


_BODY_CFG = {
    "WHALE": dict(L=2.4, R=0.30, head=0.28, beak=0.0, wx=1.0, wz=0.85, fluke=0.55,
                  fluke_vertical=False, flipper=0.22, dorsal=0.05, dorsal_s=0.55, dorsal_w=0.10),
    "DOLPHIN": dict(L=1.8, R=0.17, head=0.22, beak=1.0, wx=1.0, wz=0.92, fluke=0.30,
                    fluke_vertical=False, flipper=0.17, dorsal=0.17, dorsal_s=0.42, dorsal_w=0.13),
    "FISH": dict(L=1.3, R=0.20, head=0.24, beak=0.0, wx=0.42, wz=1.0, fluke=0.30,
                 fluke_vertical=True, flipper=0.12, dorsal=0.17, dorsal_s=0.32, dorsal_w=0.30),
}


def _body_radius(s, R, head, beak):
    s = np.asarray(s, dtype=np.float64)
    r_head = R * np.sqrt(np.clip(1.0 - ((head - s) / head) ** 2, 0.0, 1.0))
    t = np.clip((s - head) / (0.93 - head), 0.0, 1.0)
    r_tail = R * (1.0 - t) ** 1.15
    r = np.where(s < head, r_head, r_tail)
    r = np.maximum(r, 0.06 * R)
    if beak > 0:
        nose = beak * 0.30 * R * (0.7 + 0.3 * np.minimum(s / 0.04, 1.0))
        r = np.where(s < 0.16, np.maximum(r, nose), r)
    return r


def gen_body(rng, n, kind):
    c = _BODY_CFG[kind]
    L, R, head, beak = c["L"], c["R"], c["head"], c["beak"]
    wx, wz = c["wx"], c["wz"]

    def ypos(s):
        return L * (0.5 - np.asarray(s))

    def rad(s):
        return _body_radius(s, R, head, beak)

    n_body = int(n * 0.78)
    n_fluke = int(n * 0.08)
    n_flip = int(n * 0.06)
    n_dors = n - n_body - n_fluke - n_flip

    # body: s sampled proportional to circumference
    grid = np.linspace(0.0, 0.94, 1024)
    w = rad(grid)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    s = np.interp(rng.random(n_body), cdf, grid)
    r = rad(s)
    th = rng.random(n_body) * 2 * np.pi
    inside = np.where(rng.random(n_body) < 0.12, np.sqrt(rng.random(n_body)), 1.0)
    r = r * inside * (0.94 + 0.06 * rng.random(n_body))
    body = np.stack([r * np.cos(th) * wx, ypos(s), r * np.sin(th) * wz], axis=1)

    # tail flukes (crescent) / caudal fin
    F = c["fluke"]
    a = np.linspace(-1.0, 1.0, 17)
    aa = np.abs(a)
    y0 = ypos(0.90)
    lead = y0 - 0.55 * F * aa ** 1.2
    chord = F * (0.30 * np.sin(np.pi * aa) ** 0.8 + 0.10 * (1.0 - aa))
    trail = lead - chord
    span = a * F
    zeros = np.zeros_like(a)
    if c["fluke_vertical"]:
        P = np.stack([zeros, lead, span], axis=1)
        Q = np.stack([zeros, trail, span], axis=1)
    else:
        P = np.stack([span, lead, zeros], axis=1)
        Q = np.stack([span, trail, zeros], axis=1)
    fluke = sample_tris(rng, strip_tris(P, Q), n_fluke)

    # pectoral fins
    fl = c["flipper"] * L
    sf = head + 0.05
    yf = ypos(sf)
    rb = rad(sf)
    tris = []
    for sx in (-1.0, 1.0):
        bx = sx * rb * wx * 0.85
        bz = -rb * wz * 0.35
        A = [bx, yf + 0.02 * L, bz]
        B = [bx, yf - 0.08 * L, bz]
        C = [bx + sx * 0.75 * fl, yf - 0.17 * L, bz - 0.45 * fl]
        tris.append([A, B, C])
    flip = sample_tris(rng, np.array(tris), n_flip)

    # dorsal fin
    s0 = c["dorsal_s"]
    s1 = s0 + c["dorsal_w"]
    top0 = rad(s0) * wz
    top1 = rad(s1) * wz
    h = c["dorsal"] * L
    tri = np.array([[[0.0, ypos(s0), top0], [0.0, ypos(s1), top1],
                     [0.0, ypos(s0 + 0.75 * c["dorsal_w"]), top0 + h]]])
    dors = sample_tris(rng, tri, n_dors)

    return np.concatenate([body, fluke, flip, dors], axis=0)


def normalize_custom(points):
    pts = np.asarray(points, dtype=np.float64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pts = pts - (lo + hi) * 0.5
    ext = max((hi - lo).max(), 1e-9)
    return pts * (2.0 / ext)


@dataclass
class Creature:
    rest: np.ndarray   # (n,3) float32 rest pose
    s: np.ndarray      # (n,) 0 at head -> 1 at tail (along Y)
    rnd: np.ndarray    # (n,4) per-particle random numbers
    bright: np.ndarray  # (n,) base brightness


def build_creature(kind, n, seed=1, verts=None, tris=None, fuzz=0.004):
    """Create the rest-pose particle cloud.

    For kind == 'CUSTOM' pass `verts` (v,3) and optionally `tris` (t,3) vertex indices.
    """
    rng = np.random.default_rng(int(seed))
    n = max(int(n), 8)
    if kind == "MANTA":
        pts = gen_manta(rng, n)
    elif kind in _BODY_CFG:
        pts = gen_body(rng, n, kind)
    elif kind == "CUSTOM":
        if verts is None or len(verts) == 0:
            raise ValueError("Custom creature needs a source mesh")
        verts = np.asarray(verts, dtype=np.float64)
        if tris is not None and len(tris):
            pts = sample_tris(rng, verts[np.asarray(tris)], n)
        else:
            pts = verts[rng.integers(0, len(verts), n)]
            pts = pts + rng.normal(0, 0.01 * max(np.ptp(verts, axis=0).max(), 1e-6), pts.shape)
        pts = normalize_custom(pts)
    else:
        raise ValueError("unknown kind %r" % (kind,))

    pts = pts + rng.normal(0.0, fuzz, pts.shape)
    if kind == "CUSTOM":
        pts = pts  # already normalised before fuzz
    pts = pts.astype(np.float32)
    y = pts[:, 1]
    s = ((y.max() - y) / max(float(y.max() - y.min()), 1e-6)).astype(np.float32)
    rnd = rng.random((len(pts), 4)).astype(np.float32)
    bright = 0.35 + 0.65 * rnd[:, 0] ** 1.5
    bright = np.where(rnd[:, 1] > 0.97, bright * 1.8, bright).astype(np.float32)
    return Creature(pts, s, rnd, bright)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
@dataclass
class SimParams:
    size: float = 3.0
    motion: str = "WING"
    flap_speed: float = 0.45
    flap_amp: float = 1.0
    stiffness: float = 40.0
    damping: float = 0.7
    trail: float = 0.7
    loose: float = 0.5
    noise: float = 0.5
    noise_scale: float = 1.2
    trail_fade: float = 1.0
    point_size: float = 0.010
    size_jitter: float = 0.6
    base_glow: float = 1.0
    bass_gain: float = 1.5
    mid_gain: float = 1.0
    high_gain: float = 1.5
    onset_gain: float = 1.0
    bass_push: float = 0.5
    pulse_gain: float = 2.0
    pulse_push: float = 0.6
    ripple_gain: float = 2.5
    ripple_speed: float = 1.2
    ripple_width: float = 0.15
    pulse_decay: float = 0.35
    audio_decay: float = 0.15
    dust_glow: float = 0.5
    dust_size: float = 0.012
    bokeh_frac: float = 0.04
    bokeh_size: float = 6.0
    dust_speed: float = 0.5
    dust_box: float = 6.0
    dust_push: float = 1.0


PARAM_NAMES = tuple(f.name for f in fields(SimParams))


def flow(p, t, sc):
    """Cheap divergence-ish pseudo noise vector field, (n,3) -> (n,3)."""
    x = p[:, 0] * sc
    y = p[:, 1] * sc
    z = p[:, 2] * sc
    fx = np.sin(y * 1.7 + t * 0.9) + np.cos(z * 2.3 - t * 0.7)
    fy = np.sin(z * 1.9 + t * 0.8) + np.cos(x * 2.1 + t * 0.6)
    fz = np.sin(x * 1.5 - t * 0.7) + np.cos(y * 2.7 + t * 0.5)
    return np.stack([fx, fy, fz], axis=1) * 0.5


class CreatureSim:
    SUBSTEPS = 3

    def __init__(self, creature, n_dust, seed=1, motion="WING"):
        self.c = creature
        self.rest = creature.rest
        self.s = creature.s
        self.rnd = creature.rnd
        self.bright = creature.bright
        self.n = len(self.rest)
        self.nd = max(int(n_dust), 0)
        self.N = self.n + self.nd
        self.motion = motion
        self.half_span = max(float(np.abs(self.rest[:, 0]).max()), 1e-6)
        rng = np.random.default_rng(int(seed) + 101)
        self.d_rand = rng.random((self.nd, 4)).astype(np.float32)
        self.d_unit = rng.uniform(-1.0, 1.0, (self.nd, 3)).astype(np.float32)
        self.push_vec = (0.5 + self.rnd[:, 2]).astype(np.float32)
        self.pos = np.zeros((self.N, 3), np.float32)
        self.vel = np.zeros((self.N, 3), np.float32)
        self.glow = np.zeros(self.N, np.float32)
        self.size = np.zeros(self.N, np.float32)
        self.sm = {"bass": 0.0, "mid": 0.0, "high": 0.0}

    # -- pose ---------------------------------------------------------------
    def local_pose(self, t, P, bass=0.0):
        q = self.rest.copy()
        amp = P.flap_amp * (1.0 + 0.5 * min(bass, 1.5))
        ph = 2.0 * math.pi * P.flap_speed * t
        s = self.s
        mode = P.motion
        if mode == "WING":
            xn = np.abs(self.rest[:, 0]) / self.half_span
            wave_ = np.sin(ph - 2.2 * xn)
            q[:, 2] += 0.42 * amp * xn ** 1.3 * wave_
            q[:, 0] *= 1.0 - 0.10 * amp * xn * wave_
            tailw = np.clip((s - 0.5) / 0.5, 0.0, 1.0)
            q[:, 2] += 0.12 * amp * tailw ** 1.5 * np.sin(ph - 2.5 - 3.0 * tailw)
            q[:, 2] += 0.05 * amp * math.sin(ph - 1.0)
        elif mode == "BODY_V":
            q[:, 2] += 0.22 * amp * (0.03 + s ** 2) * np.sin(ph - 5.5 * s)
        elif mode == "BODY_H":
            q[:, 0] += 0.22 * amp * (0.03 + s ** 2) * np.sin(ph - 5.5 * s)
        return q

    def home_world(self, t, M, P, bass=0.0):
        q = self.local_pose(t, P, bass) * np.float32(P.size)
        R = np.asarray(M, np.float32)[:3, :3]
        T = np.asarray(M, np.float32)[:3, 3]
        return q @ R.T + T

    # -- control ------------------------------------------------------------
    def reset(self, t, M, P):
        n = self.n
        self.pos[:n] = self.home_world(t, M, P)
        self.vel[:] = 0.0
        center = np.asarray(M, np.float32)[:3, 3]
        self.pos[n:] = center + self.d_unit * np.float32(P.dust_box)
        self.sm = {"bass": 0.0, "mid": 0.0, "high": 0.0}
        self.step(t, 1.0 / 24.0, M, P, {}, _no_move=True)

    def step(self, t, dt, M, P, sig, _no_move=False):
        """Advance the simulation by dt seconds and update glow/size arrays.

        sig keys (all optional): bass, mid, high (0..1.5), pulse (envelope 0..1.5),
        age (seconds since the last pulse event) and strength (of that event).
        """
        n = self.n
        dt = float(min(max(dt, 1e-4), 0.25))
        dec = math.exp(-dt / max(P.audio_decay, 1e-3))
        for k in ("bass", "mid", "high"):
            self.sm[k] = max(float(sig.get(k, 0.0)), self.sm[k] * dec)
        bass, mid, high = self.sm["bass"], self.sm["mid"], self.sm["high"]
        pulse = float(sig.get("pulse", 0.0))
        age = float(sig.get("age", 1e9))
        strength = float(sig.get("strength", 1.0))
        Mn = np.asarray(M, np.float32)
        center = Mn[:3, 3]
        size = np.float32(P.size)

        home = self.home_world(t, Mn, P, bass)
        pos = self.pos[:n]
        vel = self.vel[:n]
        s = self.s
        rnd = self.rnd

        rad = home - center
        rl = np.linalg.norm(rad, axis=1, keepdims=True)
        rad_u = rad / np.maximum(rl, 1e-4)

        push = P.bass_push * bass + P.pulse_push * pulse
        target = home + rad_u * (push * 0.25 * size) * self.push_vec[:, None]

        front = age * P.ripple_speed
        rip_amp = strength * math.exp(-min(age, 50.0) / max(P.pulse_decay * 2.5, 1e-3))
        ripple = rip_amp * np.exp(-((s - front) / max(P.ripple_width, 1e-3)) ** 2)
        ripple = ripple.astype(np.float32)
        target = target + rad_u * (ripple[:, None] * (0.12 * P.pulse_push * size))

        if not _no_move:
            k_scale = (1.0 - P.trail * 0.85 * s ** 1.3) * (1.0 - P.loose * 0.7 * rnd[:, 3])
            k = np.float32(min(P.stiffness, 500.0)) * k_scale.astype(np.float32)
            c = 2.0 * np.sqrt(k) * np.float32(P.damping)
            fl = flow(pos, t, P.noise_scale) * (
                (P.noise * P.stiffness * 0.08) * size * (0.3 + rnd[:, 3:4]))
            h = np.float32(dt / self.SUBSTEPS)
            kk = k[:, None]
            cc = c[:, None]
            for _ in range(self.SUBSTEPS):
                acc = kk * (target - pos) - cc * vel + fl
                vel += acc * h
                pos += vel * h

        disp = np.linalg.norm(target - pos, axis=1) / size
        fade = 1.0 / (1.0 + P.trail_fade * disp * 4.0)
        tw = 0.5 + 0.5 * np.sin(t * (5.0 + 9.0 * rnd[:, 0]) + 6.2832 * rnd[:, 1])
        spark = np.where(rnd[:, 2] > 0.5, tw ** 6, 0.0)
        g = self.bright * (P.base_glow + P.bass_gain * bass + P.mid_gain * 0.6 * mid
                           + P.pulse_gain * pulse)
        g = g + P.ripple_gain * ripple * (0.5 + self.bright)
        g = g + P.high_gain * 3.0 * high * spark
        g = g * fade
        self.glow[:n] = g
        jit = P.size_jitter
        sz = P.point_size * (1.0 - 0.5 * jit + 1.5 * jit * rnd[:, 1] ** 2)
        self.size[:n] = sz * (1.0 + 0.25 * np.minimum(g, 3.0))

        # ---- ambient dust / bokeh -------------------------------------------
        if self.nd:
            dp = self.pos[n:]
            dv = self.vel[n:]
            dr = self.d_rand
            box = np.float32(max(P.dust_box, 0.1))
            fdust = flow(dp, t, 0.35) * (P.dust_speed * (0.3 + dr[:, 0:1]))
            a = 1.0 - math.exp(-dt * 1.5)
            dv += (fdust - dv) * np.float32(a)
            drad = dp - center
            ddist = np.maximum(np.linalg.norm(drad, axis=1, keepdims=True), 1e-4)
            dru = drad / ddist
            shock = np.exp(-ddist / (0.5 * box))  # shockwave only affects dust near the creature
            dv += dru * shock * np.float32(
                (bass * P.bass_push + pulse * P.pulse_push) * 2.0 * dt) * size
            # pulse shockwave: a ring expands from the creature and shoves the dust it passes
            front_d = age * P.ripple_speed * size * 2.5
            width = max(0.5 * float(size), 0.15 * float(box))
            ring = strength * math.exp(-min(age, 50.0) / max(P.pulse_decay * 2.5, 1e-3)) * np.exp(
                -((ddist[:, 0] - front_d) / width) ** 2)
            ring = ring.astype(np.float32)
            kick = (P.dust_push * 3.0 * float(size) * dt) * ring * (0.5 + dr[:, 0])
            dv += dru * kick[:, None]
            dv += np.cross(dru, np.array([0.0, 0.0, 1.0], np.float32)) * (0.35 * kick[:, None])
            if not _no_move:
                dp += dv * np.float32(dt)
            rel = dp - center
            rel = (rel + box) % (2.0 * box) - box
            dp[:] = center + rel
            edge = np.clip((1.0 - np.abs(rel).max(axis=1) / box) / 0.4, 0.0, 1.0)
            edge = edge * edge * (3.0 - 2.0 * edge)  # smoothstep fade near the box walls
            tw2 = 0.5 + 0.5 * np.sin(t * (1.5 + 4.0 * dr[:, 1]) + 6.2832 * dr[:, 2])
            react = 1.0 + mid * P.mid_gain + pulse * P.pulse_gain * 0.5 + bass * P.bass_gain * 0.5
            react = react + ring * P.dust_push * 1.5  # dust lights up as the wave passes
            gd = P.dust_glow * (0.35 + 0.65 * tw2) * react * edge
            bk = dr[:, 3] < P.bokeh_frac
            self.glow[n:] = np.where(bk, gd * 0.8, gd)
            sd = P.dust_size * (0.5 + dr[:, 0])
            self.size[n:] = np.where(bk, P.dust_size * P.bokeh_size * (0.6 + 0.8 * dr[:, 2]), sd)
        return self.pos, self.glow, self.size


# --------------------------------------------------------------------------
# Pulse events (keys / beats / audio onsets)
# --------------------------------------------------------------------------
def eval_events(ev_frames, ev_strength, frame, decay_frames, fps):
    """-> (envelope 0..1.5, age in seconds since last event, strength of last event)."""
    if len(ev_frames) == 0:
        return 0.0, 1e9, 1.0
    hi = int(np.searchsorted(ev_frames, frame, side="right"))
    if hi == 0:
        return 0.0, 1e9, 1.0
    lo = int(np.searchsorted(ev_frames, frame - 6.0 * decay_frames, side="left"))
    env = 0.0
    if hi > lo:
        d = frame - ev_frames[lo:hi]
        env = float((ev_strength[lo:hi] * np.exp(-d / max(decay_frames, 1e-3))).sum())
    age = float(frame - ev_frames[hi - 1]) / max(fps, 1e-3)
    return min(env, 1.5), age, float(ev_strength[hi - 1])


def beat_events(bpm, offset_frames, fps, first_frame, last_frame):
    period = fps * 60.0 / max(bpm, 1.0)
    k0 = int(math.floor((first_frame - offset_frames) / period)) - 1
    k1 = int(math.ceil((last_frame - offset_frames) / period)) + 1
    fr = offset_frames + np.arange(k0, k1 + 1) * period
    return fr.astype(np.float64)


# --------------------------------------------------------------------------
# Audio analysis
# --------------------------------------------------------------------------
def load_wav(path):
    """Minimal PCM WAV loader (fallback when Blender's `aud` can't be used)."""
    with wave.open(path, "rb") as w:
        ch, sw, rate, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    if sw == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        a = v.astype(np.float32) / 8388608.0
    else:
        raise ValueError("unsupported WAV sample width %d" % sw)
    return a.reshape(-1, ch), rate


def load_audio(path):
    """-> (samples (n,ch) float32, rate). Uses Blender's `aud` module when available."""
    try:
        import aud  # bundled with Blender
        snd = aud.Sound(path)
        rate = int(snd.specs[0])
        data = np.asarray(snd.data(), dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        return data, rate
    except ImportError:
        return load_wav(path)


def analyze_audio(samples, rate, fps, onset_threshold=0.35):
    """Per-frame band energies + onset events.

    Returns dict(bass, mid, high, onset: float32 arrays (one value per video frame,
    index 0 = start of the audio), events: (m,2) float64 [frame_index, strength]).
    """
    x = np.asarray(samples, np.float32)
    mono = x if x.ndim == 1 else x.mean(axis=1)
    hop = rate / float(fps)
    total = max(int(len(mono) / hop), 1)
    w = 1 << int(math.ceil(math.log2(max(hop * 2.0, 1024.0))))
    win = np.hanning(w).astype(np.float32)
    freqs = np.fft.rfftfreq(w, 1.0 / rate)
    masks = {
        "bass": (freqs >= 30) & (freqs < 180),
        "mid": (freqs >= 180) & (freqs < 2500),
        "high": (freqs >= 2500) & (freqs < 12000),
    }
    edges = np.unique((np.geomspace(30.0, 12000.0, 49) / (rate / w)).astype(int))
    edges = edges[edges < w // 2 + 1]
    padded = np.pad(mono, (w // 2, w + w // 2))
    out = {k: np.zeros(total, np.float32) for k in masks}
    spec_bands = []
    chunk = 256
    ar = np.arange(w)[None, :]
    for c0 in range(0, total, chunk):
        c1 = min(c0 + chunk, total)
        centers = (np.arange(c0, c1) * hop).astype(np.int64)[:, None]
        seg = padded[centers + ar] * win
        sp = np.abs(np.fft.rfft(seg, axis=1)).astype(np.float32)
        for k, m in masks.items():
            if m.any():
                out[k][c0:c1] = np.sqrt((sp[:, m] ** 2).mean(axis=1))
        if len(edges) > 1:
            spec_bands.append(np.add.reduceat(sp[:, :edges[-1]], edges[:-1], axis=1))

    def norm(e):
        lo = np.percentile(e, 20)
        hi = np.percentile(e, 97)
        v = (e - lo) / max(hi - lo, 1e-9 * max(float(e.max()), 1.0), 1e-12)
        # power curve: keep the peaks, push the in-between levels down so beats really pop
        return (np.clip(v, 0.0, 1.5) ** 1.6).astype(np.float32)

    res = {k: norm(v) for k, v in out.items()}

    onset = np.zeros(total, np.float32)
    if spec_bands:
        B = np.concatenate(spec_bands, axis=0)
        ref = np.percentile(B, 90) + 1e-9
        L = np.log1p(B / ref * 4.0)
        # low/mid bands count more than the top end, so kicks and snares beat hi-hats
        wts = np.linspace(1.0, 0.15, L.shape[1]).astype(np.float32) ** 1.5
        flux = (np.maximum(L[1:] - L[:-1], 0.0) * wts).sum(axis=1)
        flux = np.concatenate([[0.0], flux])
        med = np.median(flux)
        top = np.percentile(flux, 98)
        onset = np.clip((flux - med) / max(top - med, 1e-9), 0.0, 1.5).astype(np.float32)
    res["onset"] = onset

    ev = []
    last = -10
    for i in range(1, total - 1):
        v = onset[i]
        if v >= onset_threshold and v >= onset[i - 1] and v > onset[i + 1] and i - last >= 4:
            ev.append((float(i), min(float(v), 1.0)))
            last = i
    res["events"] = np.array(ev, dtype=np.float64).reshape(-1, 2)
    res["duration"] = len(mono) / float(rate)
    return res
