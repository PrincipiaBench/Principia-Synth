# Principia-Synth

Principia-Synth generates synthetic physics videos with NVIDIA Isaac Sim. Each scenario
compares two objects side by side and randomizes the scene for every sample.

| Scenario | Script | What it shows |
|---|---|---|
| Friction | `render_coeff_friction_isaac.py` | Two blocks of different sizes sliding down mirrored inclines |
| COM | `render_com_isaac.py` | Two spheres roll down mirrored inclines and hit blocks on the ground |
| MOI | `render_moi_isaac.py` | A hollow and a solid cylinder rolling down mirrored inclines |
| Pendulum | `render_pendulum_isaac.py` | Two pendulums of different lengths released from the same angle |
| Projectile | `render_projectile_isaac.py` | Two spheres launched off mirrored ramps on top of a box |
| Restitution | `render_resti_isaac.py` | Two spheres dropped from different heights, bouncing |
| Spring | `render_spring_isaac.py` | Two blocks of different mass on springs of equal stiffness |

## Requirements

- Linux with an NVIDIA RTX GPU
- NVIDIA Isaac Sim 5.1
- `ffmpeg` with `libx264` on `PATH`

## Run one scenario

From the Isaac Sim install folder:

```bash
./python.sh /path/to/Principia-Synth/render_<scenario>_isaac.py --headless --num_samples 10
```

| Flag | Default | Description |
|---|---|---|
| `--num_samples` | `100` | Number of videos to generate |
| `--output_dir` | `Principia-Synth/output/<Scenario>` | Where samples are written |
| `--duration` | `5.0` | Video length in seconds |
| `--fps` | `24` | Frame rate |
| `--width`, `--height` | `1280`, `720` | Frame size |
| `--spp` | `64` (`128` for COM) | Path-tracing samples per pixel; higher is less noisy and slower |
| `--seed` | random | Random seed for reproducible parameters |

## Run all scenarios

```bash
ISAAC_PYTHON=/path/to/isaac-sim/python.sh NUM_SAMPLES=10 bash run_all_renders.sh
```

Optional environment variables:

| Variable | Default | Description |
|---|---|---|
| `SCENARIOS` | all seven | Any of `coeff_friction com moi pendulum projectile resti spring` |
| `RENDER_GPUS` | `"0 1"` | GPUs to spread scenarios across |
| `OUTPUT_ROOT` | `Principia-Synth/output` | Parent output folder |
| `EXTRA_ARGS` | none | Flags passed to every script, e.g. `"--seed 0 --duration 3"` |

## Output

```
<output_dir>/sample_1/
  frame_0000.png …   # rendered frames
  video.mp4          # H.264 video
  metadata.json      # randomized physical parameters for this sample
```

Frame 0 shows every object at rest, at the moment of release.
