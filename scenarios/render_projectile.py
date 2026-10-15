#!/usr/bin/env python3

"""
Principia-Synth: Projectile scenario.

Two spheres roll off mirrored ramps on top of a box and land on the ground.

Usage (from the Isaac Sim install folder):
    ./python.sh /path/to/Principia-Synth/scenarios/render_projectile.py --headless [--num_samples N]

Run with --help to list all options.
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="run without the Isaac Sim GUI")
parser.add_argument("--num_samples", type=int, default=100, help="number of videos to generate (default: %(default)s)")
parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "output", "Projectile"), help="folder the sample_N folders are written to (default: %(default)s)")
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
    UsdShade,
)
from omni.isaac.dynamic_control import _dynamic_control

from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import (
    DynamicCuboid,
    FixedCuboid,
    DynamicSphere,
)

from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial

import omni.kit.viewport.utility as vp_utils
import omni.replicator.core as rep
from PIL import Image

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
settings.set("/rtx/indirectDiffuse/enabled", True)
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/translucency/enabled", True)
settings.set("/rtx/shadows/enabled", True)

# ============================================================
# Globals
# ============================================================

FPS = ARGS.fps
DURATION = ARGS.duration
TOTAL_FRAMES = int(FPS * DURATION)

OUTPUT_DIR = ARGS.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# COLORS
import json
colors_palette = json.load(open(os.path.join(REPO_ROOT, "assets", "colours_palette.json"), "r"))
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
# Simulation
# ============================================================

simulation_context = SimulationContext(stage_units_in_meters=1.0, physics_dt=1.0 / FPS, rendering_dt=1.0 / FPS)
stage = omni.usd.get_context().get_stage()

# ============================================================
# Camera
# ============================================================

camera_path = "/World/MainCamera"

camera = UsdGeom.Camera.Define(stage, camera_path)
camera.ClearXformOpOrder()

camera_pos = (0.0, -30, 8.0)
camera_rot = (80.0, 0.0, 0.0)

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
# Ground Plane
# ============================================================

ground = FixedCuboid(
    prim_path="/World/GroundPlane",
    position=np.array([0.0, 0.0, -0.05]),
    scale=np.array([20.0, 20.0, 0.1]),
)

ground_material = OmniPBR(
prim_path="/World/Materials/GroundMaterial",
color=COLOR_FLOOR,
)

background = FixedCuboid(
    prim_path="/World/Backdrop",
    position=np.array([0, 3.0, 2.0]),
    scale=np.array([12.0, 0.1, 6.0]),
    # color=COLOR_BG,
)

bg_material = OmniPBR(
prim_path="/World/Materials/BackgroundMaterial",
color=COLOR_BG,
)

# ============================================================
# Renderer
# ============================================================


# ============================================================
# Scene Constants
# ============================================================

half_len = 1.0
plane_thick = 0.06
plane_width = 0.65

# ============================================================
# Shared Objects
# ============================================================

right_plane = None
left_plane = None

right_sphere = None
left_sphere = None

middle_box = None


# ============================================================
# Create Initial Scene
# ============================================================

def create_scene():

    global right_plane
    global left_plane

    right_plane = FixedCuboid(
        prim_path="/World/incline_right",
        position=np.array([0, 0, 0]),
        scale=np.array(
            [
                half_len * 2.0,
                plane_width,
                plane_thick * 2.0,
            ]
        ),
    )

    left_plane = FixedCuboid(
        prim_path="/World/incline_left",
        position=np.array([0, 0, 0]),
        scale=np.array(
            [
                half_len * 2.0,
                plane_width,
                plane_thick * 2.0,
            ]
        ),
    )

# ============================================================
# Configure Scene
# ============================================================

def configure_scene(
    angle_deg,
    sphere_radius,
    friction_val,
    sample_index,
):

    global right_sphere
    global left_sphere
    global middle_box
    global sphere_handle

    COLOR_BG, COLOR_FLOOR = random.sample(bg_colors, 2)
    COLOR_INCLINE, COLOR_SPHERE, COLOR_BOX = random.sample(all_colors, 3)

    ground_material.set_color(COLOR_FLOOR)
    ground.apply_visual_material(ground_material)
    bg_material.set_color(COLOR_BG)
    background.apply_visual_material(bg_material)
    wood_material.set_color(COLOR_INCLINE)
    left_plane.apply_visual_material(wood_material)
    right_plane.apply_visual_material(wood_material)

    angle_rad = np.radians(angle_deg)

    quat_right = np.array(
        [
            np.cos(angle_rad / 2.0),
            0.0,
            np.sin(angle_rad / 2.0),
            0.0,
        ]
    )

    quat_left = np.array(
        [
            np.cos(-angle_rad / 2.0),
            0.0,
            np.sin(-angle_rad / 2.0),
            0.0,
        ]
    )

    plane_x = half_len * np.cos(angle_rad)

    plane_z = (
        half_len * np.sin(angle_rad)
        + plane_thick * np.cos(angle_rad)
    )

    # ========================================================
    # Physics Material
    # ========================================================

    phys_mat = PhysicsMaterial(
        prim_path=f"/World/PhysicsMaterial_{sample_index}",
        static_friction=friction_val,
        dynamic_friction=friction_val,
        restitution=0.0,
    )

    # ========================================================
    # Middle Box
    # ========================================================

    box_height = random.uniform(2.0, 4)

    box_z = box_height / 2.0

    plane_z = plane_z + box_height

    box_width = 4 * plane_x

    # ========================================================
    # Plane Placement
    # ========================================================

    right_plane.set_world_pose(
        position=np.array(
            [
                plane_x,
                0.0,
                plane_z,
            ]
        ),
        orientation=quat_right,
    )

    left_plane.set_world_pose(
        position=np.array(
            [
                -plane_x,
                0.0,
                plane_z,
            ]
        ),
        orientation=quat_left,
    )

    right_plane.apply_physics_material(phys_mat)
    left_plane.apply_physics_material(phys_mat)

    # ========================================================
    # Sphere placement
    # ========================================================

    pos1 = random.choice([0, 0.3, 0.6, 0.9])
    pos2 = random.choice([0, 0.3, 0.6, 0.9])

    dist1 = -pos1 * half_len
    dist2 = -pos2 * half_len

    offset = (
        plane_thick
        + sphere_radius
        + 0.002
    )

    bx1 = (
        plane_x
        + dist1 * np.cos(angle_rad)
        + offset * np.sin(angle_rad)
    )

    bz1 = (
        plane_z
        - dist1 * np.sin(angle_rad)
        + offset * np.cos(angle_rad)
    )

    bx2 = (
        plane_x
        + dist2 * np.cos(angle_rad)
        + offset * np.sin(angle_rad)
    )

    bz2 = (
        plane_z
        - dist2 * np.sin(angle_rad)
        + offset * np.cos(angle_rad)
    )

    # ========================================================
    # Delete old objects
    # ========================================================

    delete_paths = [
        "/World/right_sphere",
        "/World/left_sphere",
        "/World/middle_box",
    ]

    for path in delete_paths:

        if stage.GetPrimAtPath(path).IsValid():

            omni.kit.commands.execute(
                "DeletePrims",
                paths=[path],
            )

    # ========================================================
    # Random color
    # ========================================================

    sphere_color = np.array(COLOR_SPHERE)

    # ========================================================
    # Create spheres
    # ========================================================
    mass=random.uniform(0.1, 0.5)
    print(f"Creating spheres with mass: {mass}")
    right_sphere = DynamicSphere(
        prim_path="/World/right_sphere",
        position=np.array([bx1, 0, bz1]),
        radius=sphere_radius,
        color=sphere_color,
        mass=mass,
    )

    left_sphere = DynamicSphere(
        prim_path="/World/left_sphere",
        position=np.array([-bx2, 0, bz2]),
        radius=sphere_radius,
        color=sphere_color,
        mass=mass,
    )

    if bz1 >= bz2:
        sphere_handle = right_sphere
    else:
        sphere_handle = left_sphere

    # ========================================================
    # Create middle box
    # ========================================================

    middle_box = FixedCuboid(
        prim_path="/World/middle_box",
        position=np.array(
            [
                0,
                0,
                box_z,
            ]
        ),
        scale=np.array(
            [
                box_width,
                box_width,
                box_height,
            ]
        ),
        color=np.array(COLOR_BOX),
    )

    # ========================================================
    # Physics materials
    # ========================================================

    right_sphere.apply_physics_material(
        phys_mat
    )

    left_sphere.apply_physics_material(
        phys_mat
    )

    # ========================================================
    # Reset velocities
    # ========================================================

    right_sphere.set_linear_velocity(
        np.zeros(3)
    )

    right_sphere.set_angular_velocity(
        np.zeros(3)
    )

    left_sphere.set_linear_velocity(
        np.zeros(3)
    )

    left_sphere.set_angular_velocity(
        np.zeros(3)
    )

    metadata = {
        "resolution": list(RESOLUTION),
        "angle_deg": angle_deg,
        "sphere_radius": sphere_radius,
        "friction_val": friction_val,
        "pos1": pos1,
        "pos2": pos2,
        "box_height": box_height,
        "camera_position": camera_pos,
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
        f"sample_{sample_index}"
    )

    os.makedirs(sample_dir, exist_ok=True)

    # warmup
    # Render-only passes (no physics) so materials are loaded; frame 0 is
    # then the release pose with every object at rest
    for _ in range(3):
        capture_frame()

    for frame in range(TOTAL_FRAMES):

        frame_path = os.path.join(
            sample_dir,
            f"frame_{frame:04d}.png"
        )

        Image.fromarray(capture_frame()).save(frame_path)

        step_physics()


    return sample_dir

# ============================================================
# Video Creation
# ============================================================

def create_video(sample_dir):

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

# ============================================================
# Main
# ============================================================

def main():


    create_scene()

    simulation_context.initialize_physics()

    

    num_samples = ARGS.num_samples

    for i in range(1, num_samples + 1):

        carb.log_info(
            f"Running sample {i}"
        )

        friction_val = 0.0

        sphere_radius = random.uniform(
            0.10,
            0.20,
        )
        angle_deg = random.uniform(15.0, 35.0)
        simulation_context.reset()

        metadata = configure_scene(
            angle_deg=angle_deg,
            sphere_radius=sphere_radius,
            friction_val=friction_val,
            sample_index=i,
        )

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