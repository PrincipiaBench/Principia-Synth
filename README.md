# Principia-Synth

Principia-Synth generates synthetic physics videos with NVIDIA Isaac Sim. Each scenario
compares two objects side by side and randomizes the scene for every sample.

| Scenario | Script | What it shows |
|---|---|---|
| Friction | `scenarios/render_friction.py` | Two blocks of different sizes sliding down mirrored inclines |
| COM | `scenarios/render_com.py` | Two spheres roll down mirrored inclines and hit blocks on the ground |
| MOI | `scenarios/render_moi.py` | A hollow and a solid cylinder rolling down mirrored inclines |
| Pendulum | `scenarios/render_pendulum.py` | Two pendulums of different lengths released from the same angle |
| Projectile | `scenarios/render_projectile.py` | Two spheres launched off mirrored ramps on top of a box |
| Restitution | `scenarios/render_restitution.py` | Two spheres dropped from different heights, bouncing |
| Spring | `scenarios/render_spring.py` | Two blocks of different mass on springs of equal stiffness |

## Repository layout

```
Principia-Synth/
├── run_all_renders.sh   # render several scenarios in parallel
├── scenarios/           # one render script per scenario
└── assets/
    ├── colours_palette.json
    └── models/          # cylinder meshes used by MOI
```

## Run one scenario

From the Isaac Sim install folder:

```bash
./python.sh /path/to/Principia-Synth/scenarios/render_<scenario>.py --headless --num_samples 10
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
| `SCENARIOS` | all seven | Any of `friction com moi pendulum projectile restitution spring` |
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
