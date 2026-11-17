# Glow Creatures (Blender add-on)

Animals made of thousands of glowing particles that flap, swim, trail dust, and
light up to music or when you press keys. Inspired by the particle creatures in
Tetris Effect. Written for Blender 4.0+ (4.2 LTS or newer recommended).

## Install
Edit > Preferences > Add-ons > (v) > Install from Disk... > pick `glow_creatures.zip`,
then tick "Glow Creatures". Panel: 3D Viewport > press N > "Glow Creatures" tab.

## Quick start
1. Pick an animal (Manta, Whale, Dolphin, Fish, or Custom Mesh) > **Create Glow Creature**.
   You get a particle object plus a `*_Controller` empty. Move/animate the empty and the
   particles follow with lag, trailing dust behind it. A looping demo swim path is added.
2. **Set Up Glow Scene** (black world, Standard colour, EEVEE + glare/bloom).
   Use Material Preview or Rendered viewport shading.
3. **Audio Reaction**: choose a sound file > Analyze Audio > tick "Use Audio File"
   (and "Add Audio To Timeline" to hear it). Bass swells the body and brightens it,
   mids lift the glow, highs make particles sparkle, detected beats fire a flash and a
   ripple that runs from head to tail. No music? Tick "Metronome Pulses" for a BPM grid.
4. **Key Pulses**: press "Play + Key Pulses", then tap any key while it plays. Each key
   fires a flash + ripple, and a shockwave that shoves the surrounding dust outward and lights it up ("Dust Kick"). Pulses are recorded (turn off "Record Key Presses" to just play
   with it) so the render reproduces them. ESC stops. "Pulse At Current Frame" adds one by hand.

## Custom animals
Choose Animal > Custom Mesh and pick any mesh as Source Mesh (head toward +Y). Particles
are scattered over its surface. Pick the Motion that suits it (wing flap / swim up-down /
swim sideways / still).

## Rendering
Particles are simulated live frame by frame. Render the animation from the start of a
shot (Ctrl+F12) so the trails build up. Scrubbing snaps particles to the pose (no trails).
EEVEE is recommended for speed; Cycles also works but is slower with 100k points.

## Tips
* Fewer particles = faster viewport. 60k body + 15k dust runs fine; try 150k+ for renders.
* Point Size, Emission and the Glare node strength set how "bloomy" it looks.
* Stiffness low + Tail Lag high = long dreamy trails. Swirl adds drifting noise.

## Known limits
* The particle/audio/pulse engine was tested outside Blender. The Blender-specific parts
  (node setup, compositor glare, key-capture mode) were written against the 4.x API but
  not run inside Blender yet. If "Set Up Glow Scene" reports something it couldn't set,
  add a Glare (Fog Glow) node in the Compositor yourself.
* mp3/ogg/flac need Blender's built-in audio decoder (normally fine); WAV always works.
