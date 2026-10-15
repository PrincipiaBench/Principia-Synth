#!/usr/bin/env python3

"""
Principia-Synth: Spring scenario.

Two blocks of different mass oscillate on springs of equal stiffness.

Usage (from the Isaac Sim install folder):
    ./python.sh /path/to/Principia-Synth/scenarios/render_spring.py --headless [--num_samples N]

Run with --help to list all options.
"""

# ============================================================
# Imports
# ============================================================

import os
import sys
import json
import random
import argparse
import subprocess
import numpy as np

from isaacsim import SimulationApp

# ============================================================
# SimulationApp MUST COME FIRST
# ============================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="run without the Isaac Sim GUI")
parser.add_argument("--num_samples", type=int, default=100, help="number of videos to generate (default: %(default)s)")
parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "output", "Spring"), help="folder the sample_N folders are written to (default: %(default)s)")
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible parameters and colours (default: random)")
parser.add_argument("--fps", type=int, default=24, help="video frame rate (default: %(default)s)")
parser.add_argument("--duration", type=float, default=5.0, help="video length in seconds (default: %(default)s)")
parser.add_argument("--width", type=int, default=1280, help="frame width in pixels (default: %(default)s)")
parser.add_argument("--height", type=int, default=720, help="frame height in pixels (default: %(default)s)")
parser.add_argument("--spp", type=int, default=64, help="path-tracing samples per pixel; use a multiple of 16 (default: %(default)s)")
ARGS, _ = parser.parse_known_args()

if ARGS.seed is not None:
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

HEADLESS = ARGS.headless

RESOLUTION = (ARGS.width, ARGS.height)

simulation_app = SimulationApp(
    {
        "headless": HEADLESS,
        "hide_ui": HEADLESS,
        "renderer": "PathTracing",
        "width": RESOLUTION[0],
        "height": RESOLUTION[1],
    }
)

# ============================================================
# Isaac Imports
# ============================================================

import carb
import carb.settings
import omni
import omni.usd
import omni.kit.commands
import omni.replicator.core as rep
from PIL import Image
import omni.kit.viewport.utility as vp_utils

from pxr import (
    UsdGeom,
    UsdLux,
    UsdPhysics,
    Gf,
)

from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import (
    DynamicCuboid,
    FixedCuboid,
)

from isaacsim.core.prims import RigidPrim

# ============================================================
# RTX SETTINGS
# ============================================================

settings = carb.settings.get_settings()

settings.set("/rtx/rendermode", "PathTracing")
# Full accumulation per frame: each frame renders RT_SUBFRAMES subframes of
# SPP_PER_SUBFRAME samples with the timeline held, reaching TOTAL_SPP.
TOTAL_SPP = ARGS.spp
SPP_PER_SUBFRAME = min(16, TOTAL_SPP)
RT_SUBFRAMES = TOTAL_SPP // SPP_PER_SUBFRAME

settings.set("/rtx/pathtracing/spp", SPP_PER_SUBFRAME)
settings.set("/rtx/pathtracing/totalSpp", TOTAL_SPP)

# Only capture frames when rep.orchestrator.step() is called
settings.set("/omni/replicator/captureOnPlay", False)
settings.set("/rtx/pathtracing/maxBounces", 8)
settings.set("/rtx/post/aa/op", 3)
settings.set("/rtx/post/dlss/execMode", 1)

# ============================================================
# Globals
# ============================================================

FPS = ARGS.fps
DURATION = ARGS.duration
TOTAL_FRAMES = int(FPS * DURATION)

OUTPUT_DIR = ARGS.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Simulation
# ============================================================

simulation_context = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 240.0,
    rendering_dt=1.0 / FPS,
)

stage = omni.usd.get_context().get_stage()

from omni.isaac.core.materials import OmniPBR

# ============================================================
# COLORS
# ============================================================

colors_palette = json.load(
    open(
        os.path.join(REPO_ROOT, "assets", "colours_palette.json"),
        "r",
    )
)

bg_colors = []

for _, color_data in colors_palette["reference_palette"].items():
    bg_colors.append(np.array(color_data["rgb_0_1"]))

all_colors = []

for _, color_data in colors_palette["main_object_pop_colors"].items():
    all_colors.append(np.array(color_data["rgb_0_1"]))

# ============================================================
# Camera
# ============================================================

camera_path = "/World/MainCamera"

camera = UsdGeom.Camera.Define(stage, camera_path)
camera.ClearXformOpOrder()

camera.AddTranslateOp().Set(
    Gf.Vec3f(0.0, -20.0, 4)
)

camera.AddRotateXYZOp().Set(
    Gf.Vec3f(80.0, 0.0, 0.0)
)

camera.CreateFocalLengthAttr(50)

viewport = vp_utils.get_active_viewport()
viewport.set_active_camera(camera_path)

# 16:9 sensor to match the output resolution
camera.CreateHorizontalApertureAttr(20.955)
camera.CreateVerticalApertureAttr(20.955 * RESOLUTION[1] / RESOLUTION[0])

# Frames come from the render product below; don't also path-trace the viewport
if HEADLESS:
    viewport.updates_enabled = False

# Dedicated render product on the camera: captures only the rendered image,
# without the viewport HUD / UI overlay or letterboxing
render_product = rep.create.render_product(camera_path, RESOLUTION)

rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])


def capture_frame():
    # Render with the timeline held (delta_time=0.0) so the path tracer
    # accumulates the full TOTAL_SPP over RT_SUBFRAMES on a static scene
    rep.orchestrator.step(
        rt_subframes=RT_SUBFRAMES,
        delta_time=0.0,
        pause_timeline=False,
    )

    return np.asarray(rgb_annotator.get_data())[:, :, :3]


def step_physics(num_frames=1):
    # Physics-only advance of num_frames video frames. A render=True step
    # ran rendering_dt / physics_dt physics substeps per frame; keep that so
    # the motion timing is unchanged.
    substeps = max(1, round(simulation_context.get_rendering_dt() / simulation_context.get_physics_dt()))

    for _ in range(num_frames * substeps):
        simulation_context.step(render=False)


# ============================================================
# Lighting
# ============================================================

sun = UsdLux.DistantLight.Define(
    stage,
    "/World/SunLight"
)

sun.CreateIntensityAttr(3000)

sun.CreateColorAttr(
    Gf.Vec3f(1.0, 1.0, 0.98)
)

sun.AddRotateXYZOp().Set(
    Gf.Vec3f(-30.0, 40.0, -20.0)
)

sun.CreateAngleAttr(3.0)

dome = UsdLux.DomeLight.Define(
    stage,
    "/World/DomeLight"
)

dome.CreateIntensityAttr(500)

dome.CreateColorAttr(
    Gf.Vec3f(0.72, 0.78, 0.90)
)

# ============================================================
# Ground
# ============================================================

# ground = FixedCuboid(
#     prim_path="/World/Ground",
#     position=np.array([0.0, 0.0, -0.05]),
#     scale=np.array([20.0, 20.0, 0.1]),
# )

# ground_material = OmniPBR(
#     prim_path="/World/Materials/GroundMaterial",
#     color=random.choice(bg_colors),
# )
# ground.apply_visual_material(ground_material)
# ground_material.set_diffuse_texture(
#     "/Isaac/Materials/Textures/checker.png"
# )

background = FixedCuboid(
    prim_path="/World/Backdrop",
    position=np.array([0.0, 3.0, 2.0]),
    scale=np.array([12.0, 0.1, 6.0]),
)

background_material = OmniPBR(
    prim_path="/World/Materials/BackgroundMaterial",
    color=random.choice(bg_colors),
)

# ============================================================
# Helix Function
# ============================================================

def helix(
    start,
    end,
    turns=12,
    radius=0.1,
    segments=200,
):

    start = np.array(start)
    end = np.array(end)

    axis = end - start

    length = np.linalg.norm(axis)

    if length < 1e-6:
        return np.zeros((segments, 3))

    z = axis / length

    up = np.array([0, 0, 1])

    if abs(np.dot(up, z)) > 0.9:
        up = np.array([1, 0, 0])

    x = np.cross(up, z)
    x /= np.linalg.norm(x)

    y = np.cross(z, x)

    pts = []

    for i in range(segments):

        t = i / (segments - 1)

        angle = 2 * np.pi * turns * t

        offset = (
            np.cos(angle) * x
            + np.sin(angle) * y
        )

        pt = (
            start
            + z * (t * length)
            + radius * offset
        )

        pts.append(pt)

    return np.array(pts)

# ============================================================
# Visual Spring
# ============================================================

def create_visual_spring(path, segments=200):

    spring = UsdGeom.BasisCurves.Define(
        stage,
        path
    )

    spring.CreateTypeAttr("linear")

    spring.CreateWrapAttr("nonperiodic")

    spring.CreateCurveVertexCountsAttr(
        [segments]
    )

    spring.CreateWidthsAttr([0.02])

    return spring

# ============================================================
# Create Spring Joint
# ============================================================

def create_spring_joint(
    joint_path,
    ceiling_path,
    block_path,
    stiffness,
    damping,
):

    joint = UsdPhysics.PrismaticJoint.Define(
        stage,
        joint_path
    )

    joint.CreateBody0Rel().SetTargets(
        [ceiling_path]
    )

    joint.CreateBody1Rel().SetTargets(
        [block_path]
    )

    joint.CreateAxisAttr("Z")

    joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(0, 0, 0)
    )

    joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(0, 0, 0)
    )

    joint.CreateLowerLimitAttr().Set(-5.0)
    joint.CreateUpperLimitAttr().Set(5.0)

    drive = UsdPhysics.DriveAPI.Apply(
        joint.GetPrim(),
        "linear"
    )

    drive.CreateTypeAttr("force")

    drive.CreateTargetPositionAttr(0)

    drive.CreateStiffnessAttr(stiffness)

    drive.CreateDampingAttr(damping)

# ============================================================
# Create Scene
# ============================================================

def create_scene():

    color = random.choice(all_colors)
    left_anchor = FixedCuboid(
        prim_path="/World/left_anchor",
        position=np.array([-1.5, 0.0, 2.5]),
        scale=np.array([0.35, 0.35, 0.1]),
        color = color
    )

    right_anchor = FixedCuboid(
        prim_path="/World/right_anchor",
        position=np.array([1.5, 0.0, 2.5]),
        scale=np.array([0.35, 0.35, 0.1]),
        color = color
    )

    center_anchor = FixedCuboid(
        prim_path="/World/center_anchor",
        position=np.array([0.0, 0.0, 2.5]),
        scale=np.array([4, 0.35, 0.1]),
        color = color
    )

# ============================================================
# Configure Sample
# ============================================================

def configure_sample(sample_idx):

    # --------------------------------------------------------
    # delete old blocks
    # --------------------------------------------------------

    delete_paths = [
        "/World/left_block",
        "/World/right_block",
        "/World/left_joint",
        "/World/right_joint",
        "/World/left_spring_visual",
        "/World/right_spring_visual",
    ]

    existing = []

    for p in delete_paths:

        if stage.GetPrimAtPath(p).IsValid():
            existing.append(p)

    if len(existing) > 0:

        omni.kit.commands.execute(
            "DeletePrims",
            paths=existing,
        )

    COLOR_BG, COLOR_FLOOR = random.sample(bg_colors, 2)
    COLOR = random.choice(all_colors)

    background_material.set_color(COLOR_BG)
    background.apply_visual_material(background_material)

    # ground_material.set_color(COLOR_FLOOR)
    # ground.apply_visual_material(ground_material)

    # --------------------------------------------------------
    # random sizes
    # --------------------------------------------------------
    size_low = random.uniform(0.28, 0.35)
    size_high = random.uniform(0.37, 0.45)

    if random.random() < 0.5:
        size_left = size_low
        size_right = size_high
    else:
        size_left = size_high
        size_right = size_low

    # mass proportional to volume
    block_density = 250  # kg/m^3
    mass_left = block_density * (size_left ** 3)
    mass_right = block_density * (size_right ** 3)

    # same spring constant
    stiffness = 75

    damping = 0.0

    # --------------------------------------------------------
    # blocks
    # --------------------------------------------------------

    left_block = DynamicCuboid(
        prim_path="/World/left_block",
        position=np.array([
            -1.5,
            0.0,
            1.7,
        ]),
        scale=np.array([
            size_left,
            size_left,
            size_left,
        ]),
        mass=mass_left,
        color=COLOR,
    )

    right_block = DynamicCuboid(
        prim_path="/World/right_block",
        position=np.array([
            1.5,
            0.0,
            1.7,
        ]),
        scale=np.array([
            size_right,
            size_right,
            size_right,
        ]),
        mass=mass_right,
        color=COLOR,
    )

    # --------------------------------------------------------
    # spring joints
    # --------------------------------------------------------

    create_spring_joint(
        "/World/left_joint",
        "/World/left_anchor",
        "/World/left_block",
        stiffness,
        damping,
    )

    create_spring_joint(
        "/World/right_joint",
        "/World/right_anchor",
        "/World/right_block",
        stiffness,
        damping,
    )

    # --------------------------------------------------------
    # visual springs
    # --------------------------------------------------------

    left_visual = create_visual_spring(
        "/World/left_spring_visual"
    )

    right_visual = create_visual_spring(
        "/World/right_spring_visual"
    )

    return {
        "left_visual": left_visual,
        "right_visual": right_visual,
        "size_left": size_left,
        "size_right": size_right,
        "mass_left": mass_left,
        "mass_right": mass_right,
        "stiffness": stiffness,
        "damping": damping,
    }

# ============================================================
# Render Sample
# ============================================================

def render_sample(sample_idx, sample_data):

    sample_dir = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_idx}"
    )

    os.makedirs(sample_dir, exist_ok=True)

    left_visual = sample_data["left_visual"]
    right_visual = sample_data["right_visual"]

    left_rigid = RigidPrim(
        prim_paths_expr="/World/left_block"
    )

    right_rigid = RigidPrim(
        prim_paths_expr="/World/right_block"
    )

    left_anchor = np.array([
        -1.5,
        0.0,
        2.5,
    ])

    right_anchor = np.array([
        1.5,
        0.0,
        2.5,
    ])

    # --------------------------------------------------------
    # initialize springs BEFORE warmup
    # --------------------------------------------------------

    left_pos = left_rigid.get_world_poses()[0][0]
    right_pos = right_rigid.get_world_poses()[0][0]

    left_pts = helix(
        left_anchor,
        left_pos,
    )

    right_pts = helix(
        right_anchor,
        right_pos,
    )

    left_visual.GetPointsAttr().Set(
        [Gf.Vec3f(*p) for p in left_pts]
    )

    right_visual.GetPointsAttr().Set(
        [Gf.Vec3f(*p) for p in right_pts]
    )

    # --------------------------------------------------------
    # warmup
    # --------------------------------------------------------

    # Render-only passes (no physics) so materials are loaded; frame 0 is
    # then the release pose with every object at rest
    for _ in range(3):
        capture_frame()

    # --------------------------------------------------------
    # render frames
    # --------------------------------------------------------

    for frame in range(TOTAL_FRAMES):

        left_pos = left_rigid.get_world_poses()[0][0]
        right_pos = right_rigid.get_world_poses()[0][0]

        left_pts = helix(
            left_anchor,
            left_pos,
        )

        right_pts = helix(
            right_anchor,
            right_pos,
        )

        left_visual.GetPointsAttr().Set(
            [Gf.Vec3f(*p) for p in left_pts]
        )

        right_visual.GetPointsAttr().Set(
            [Gf.Vec3f(*p) for p in right_pts]
        )

        frame_path = os.path.join(
            sample_dir,
            f"frame_{frame:04d}.png",
        )

        Image.fromarray(capture_frame()).save(frame_path)

        step_physics()


    # --------------------------------------------------------
    # video
    # --------------------------------------------------------

    video_path = os.path.join(
        sample_dir,
        "video.mp4"
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        os.path.join(
            sample_dir,
            "frame_%04d.png"
        ),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        video_path,
    ]

    subprocess.run(
        ffmpeg_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # --------------------------------------------------------
    # metadata
    # --------------------------------------------------------

    metadata = {
        "resolution": list(RESOLUTION),
        "size_left": sample_data["size_left"],
        "size_right": sample_data["size_right"],
        "mass_left": sample_data["mass_left"],
        "mass_right": sample_data["mass_right"],
        "stiffness": sample_data["stiffness"],
        "damping": sample_data["damping"],
        "fps": FPS,
        "duration": DURATION,
    }

    with open(
        os.path.join(
            sample_dir,
            "metadata.json"
        ),
        "w",
    ) as f:

        json.dump(metadata, f, indent=4)

# ============================================================
# Main
# ============================================================

def main():

    n_samples = ARGS.num_samples

    create_scene()

    simulation_context.initialize_physics()

    for sample_idx in range(1, n_samples + 1):

        print(f"\nRunning sample {sample_idx}")

        simulation_context.reset()

        sample_data = configure_sample(
            sample_idx
        )

        render_sample(
            sample_idx,
            sample_data,
        )

    simulation_context.stop()

    simulation_app.close()

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()