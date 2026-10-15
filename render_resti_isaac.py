#!/usr/bin/env python3

"""
Isaac Sim - Stable Bouncing Spheres
-----------------------------------

Key Fix:
- Entire scene recreated every sample
- Prevents stale PhysX state
- Reliable restitution / bouncing

Run:
./python.sh bouncing_spheres.py --headless
"""

import os
import sys
import json
import random
import argparse
import subprocess
import numpy as np

# ============================================================
# MUST COME FIRST
# ============================================================

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "Restitution"))
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--duration", type=float, default=5.0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--spp", type=int, default=64)
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
# Imports
# ============================================================

import carb
import carb.settings
import omni
import omni.usd
import omni.kit.commands

from pxr import (
    UsdGeom,
    Gf,
    UsdLux,
)

from isaacsim.core.api import SimulationContext

from isaacsim.core.api.objects import (
    DynamicSphere,
)

from isaacsim.core.api.objects.ground_plane import (
    GroundPlane,
)

from isaacsim.core.api.materials.physics_material import (
    PhysicsMaterial,
)

import omni.kit.viewport.utility as vp_utils
import omni.replicator.core as rep
from PIL import Image
from isaacsim.core.api.objects import FixedCuboid
from omni.isaac.core.materials import OmniPBR

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

settings.set("/rtx/ambientOcclusion/enabled", True)
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/indirectDiffuse/enabled", True)
settings.set("/rtx/translucency/enabled", True)
settings.set("/rtx/shadows/enabled", True)
# settings.set("/app/renderer/resolution/width", 720)
# settings.set("/app/renderer/resolution/height", 1280)

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
    physics_dt=1.0 / (FPS*10),
    rendering_dt=1.0 / FPS,
)

stage = omni.usd.get_context().get_stage()

# ============================================================
# COLORS
# ============================================================
import json
colors_palette = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "colours_palette.json"), "r"))
bg_colors = []

for color_name, color_data in colors_palette["reference_palette"].items():

    rgb = np.array(color_data["rgb_0_1"])

    bg_colors.append(rgb)

# COLOR_BG = random.choice(bg_colors)
# COLOR_FLOOR = random.choice(bg_colors)

all_colors = []
for color_name, color_data in colors_palette["main_object_pop_colors"].items():
    rgb = np.array(color_data["rgb_0_1"])
    all_colors.append(rgb)

COLOR_INCLINE = random.choice(all_colors)
COLOR_BG = random.choice(bg_colors)
COLOR_FLOOR = random.choice(bg_colors)

wood_material = OmniPBR(
prim_path="/World/Materials/WoodMaterial",
color=COLOR_INCLINE,
)
wood_material.set_reflection_roughness(0.75)
wood_material.set_metallic_constant(0.0)

# ============================================================
# Camera
# ============================================================

camera_path = "/World/MainCamera"

camera = UsdGeom.Camera.Define(stage, camera_path)
camera.ClearXformOpOrder()

camera_pos = (0.0, -18, 2.0)
camera_rot = (90.0, 0.0, 0.0)

camera.AddTranslateOp().Set(Gf.Vec3f(*camera_pos))
camera.AddRotateXYZOp().Set(Gf.Vec3f(*camera_rot))

# wider cinematic lens
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

sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
# sun.CreateIntensityAttr(4500) # Boosted intensity
sun.CreateIntensityAttr(3000)
sun.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.98)) 
sun.AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 40, -20.0)) # Angled for dynamic side-front shadows
sun.CreateAngleAttr(3.0) # Lowered from 10.0 for sharper, cleaner shadow edges

# ------------------------------------------------------------
# Soft ambient dome light
# ------------------------------------------------------------

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(500)
dome.CreateColorAttr(Gf.Vec3f(0.72, 0.78, 0.90))

# ============================================================
# Renderer
# ============================================================


# ============================================================
# Build Entire Scene
# ============================================================

def build_scene(sample_index):

    sample_root = f"/World/Sample_{sample_index}"

    # --------------------------------------------------------
    # Delete old sample completely
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Delete all previous samples
    # --------------------------------------------------------

    paths_to_delete = []

    world_prim = stage.GetPrimAtPath("/World")

    for child in world_prim.GetChildren():

        child_path = str(child.GetPath())

        if child_path.startswith("/World/Sample_"):

            paths_to_delete.append(child_path)

    if len(paths_to_delete) > 0:

        omni.kit.commands.execute(
            "DeletePrims",
            paths=paths_to_delete,
        )

    # --------------------------------------------------------
    # Random parameters
    # --------------------------------------------------------

    restitution = float(
        np.round(random.uniform(0.5, 0.95), 2)
    )

    friction = float(
        np.round(random.uniform(0.1, 0.5), 2)
    )

    sphere_radius = float(
        np.round(random.uniform(0.2, 0.4), 2)
    )

    height1 = float(
        np.round(random.uniform(1.0, 4.0), 2)
    )

    height2 = float(
        np.round(random.uniform(1.0, 4.0), 2)
    )

    

    # --------------------------------------------------------
    # Physics materials
    # --------------------------------------------------------

    ground_phys_mat = PhysicsMaterial(
        prim_path=f"{sample_root}/GroundPhysicsMaterial",
        static_friction=friction,
        dynamic_friction=friction,
        restitution=restitution,
    )

    sphere_phys_mat = PhysicsMaterial(
        prim_path=f"{sample_root}/SpherePhysicsMaterial",
        static_friction=friction,
        dynamic_friction=friction,
        restitution=restitution,
    )

    # --------------------------------------------------------
    # Ground
    # --------------------------------------------------------

    COLOR_BG, COLOR_FLOOR = random.sample(bg_colors, 2)
    COLOR_SPHERE = random.sample(all_colors, 1)

    ground = FixedCuboid(
        prim_path=f"{sample_root}/Ground",
        position=np.array([0.0, 0.0, -0.25]),
        scale=np.array([20.0, 20.0, 0.5]),
        color=COLOR_FLOOR,
    )

    ground.apply_physics_material(
        ground_phys_mat
    )

    background = FixedCuboid(
    prim_path="/World/Backdrop",
    position=np.array([0, 3.0, 2.0]),
    scale=np.array([12.0, 0.1, 6.0]),
    color=COLOR_BG,
)

    # --------------------------------------------------------
    # Sphere 1
    # --------------------------------------------------------

    sphere1 = DynamicSphere(
        prim_path=f"{sample_root}/Sphere1",
        position=np.array([-0.8, 0.0, height1]),
        radius=sphere_radius,
        color=np.array(COLOR_SPHERE),
        mass=1.0,
    )

    sphere1.apply_physics_material(
        sphere_phys_mat
    )

    # --------------------------------------------------------
    # Sphere 2
    # --------------------------------------------------------

    sphere2 = DynamicSphere(
        prim_path=f"{sample_root}/Sphere2",
        position=np.array([0.8, 0.0, height2]),
        radius=sphere_radius,
        color=np.array(COLOR_SPHERE),
        mass=1.0,
    )

    sphere2.apply_physics_material(
        sphere_phys_mat
    )

    # --------------------------------------------------------
    # Explicit velocity reset
    # --------------------------------------------------------

    sphere1.set_linear_velocity(
        np.zeros(3)
    )

    sphere2.set_linear_velocity(
        np.zeros(3)
    )

    sphere1.set_angular_velocity(
        np.zeros(3)
    )

    sphere2.set_angular_velocity(
        np.zeros(3)
    )

    metadata = {
        "resolution": list(RESOLUTION),
        "restitution": restitution,
        "friction": friction,
        "sphere_radius": sphere_radius,
        # "sphere_color": COLOR_SPHERE,
        "height1": height1,
        "height2": height2,
        "fps": FPS,
        "duration": DURATION,
    }

    return metadata

# ============================================================
# Rendering
# ============================================================

def render_sample(sample_index):

    sample_dir = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_index}",
    )

    os.makedirs(sample_dir, exist_ok=True)

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    # Render-only passes (no physics) so materials are loaded; frame 0 is
    # then the release pose with every object at rest
    for _ in range(3):
        capture_frame()

    # --------------------------------------------------------
    # Render frames
    # --------------------------------------------------------

    for frame in range(TOTAL_FRAMES):

        frame_path = os.path.join(
            sample_dir,
            f"frame_{frame:04d}.png",
        )

        Image.fromarray(capture_frame()).save(frame_path)

        step_physics()


    return sample_dir

# ============================================================
# Video
# ============================================================

def create_video(sample_dir):

    video_path = os.path.join(
        sample_dir,
        "video.mp4",
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        os.path.join(
            sample_dir,
            "frame_%04d.png",
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

# ============================================================
# Main
# ============================================================

def main():


    simulation_context.initialize_physics()

    num_samples = ARGS.num_samples

    for i in range(1, num_samples + 1):

        print(f"\nRunning sample {i}")

        # ----------------------------------------------------
        # FULL SCENE REBUILD
        # ----------------------------------------------------

        metadata = build_scene(i)

        # ----------------------------------------------------
        # IMPORTANT:
        # reset AFTER rebuilding scene
        # ----------------------------------------------------

        simulation_context.reset()

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        sample_dir = render_sample(i)

        create_video(sample_dir)

        with open(
            os.path.join(
                sample_dir,
                "metadata.json",
            ),
            "w",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

    simulation_context.stop()

    simulation_app.close()

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()