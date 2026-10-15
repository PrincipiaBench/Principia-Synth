#!/usr/bin/env python3

"""
Principia-Synth: COM scenario.

Two spheres roll down mirrored inclines from different starting positions and hit blocks on the ground.

Usage (from the Isaac Sim install folder):
    ./python.sh /path/to/Principia-Synth/scenarios/render_com.py --headless [--num_samples N]

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
# SimulationApp MUST come first
# ============================================================

from isaacsim import SimulationApp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="run without the Isaac Sim GUI")
parser.add_argument("--num_samples", type=int, default=100, help="number of videos to generate (default: %(default)s)")
parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "output", "COM"), help="folder the sample_N folders are written to (default: %(default)s)")
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible parameters and colours (default: random)")
parser.add_argument("--fps", type=int, default=24, help="video frame rate (default: %(default)s)")
parser.add_argument("--duration", type=float, default=5.0, help="video length in seconds (default: %(default)s)")
parser.add_argument("--width", type=int, default=1280, help="frame width in pixels (default: %(default)s)")
parser.add_argument("--height", type=int, default=720, help="frame height in pixels (default: %(default)s)")
parser.add_argument("--spp", type=int, default=128, help="path-tracing samples per pixel; use a multiple of 16 (default: %(default)s)")
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
    UsdPhysics,
)

from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import (
    DynamicCuboid,
    FixedCuboid,
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
# Camera
# ============================================================

camera_path = "/World/MainCamera"

camera = UsdGeom.Camera.Define(stage, camera_path)
camera.ClearXformOpOrder()

camera_pos = (0.0, -20, 3.5)
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


# ============================================================
# Create Static Scene
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
# Create Sphere
# ============================================================

def create_dynamic_sphere(
    prim_path,
    radius,
    position,
    color,
    mass,
    physics_material,
):

    sphere_prim = UsdGeom.Sphere.Define(
        stage,
        prim_path
    )

    sphere_prim.GetRadiusAttr().Set(radius)

    xform = UsdGeom.Xformable(
        sphere_prim.GetPrim()
    )

    xform.ClearXformOpOrder()

    xform.AddTranslateOp().Set(
        Gf.Vec3f(*position)
    )

    # ========================================================
    # Collision
    # ========================================================

    UsdPhysics.CollisionAPI.Apply(
        sphere_prim.GetPrim()
    )

    # ========================================================
    # Rigid Body
    # ========================================================

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(
        sphere_prim.GetPrim()
    )

    rigid_api.CreateRigidBodyEnabledAttr(True)

    mass_api = UsdPhysics.MassAPI.Apply(
        sphere_prim.GetPrim()
    )

    mass_api.CreateMassAttr(mass)

    # ========================================================
    # Visual Material
    # ========================================================

    sphere_material = OmniPBR(
        prim_path=f"{prim_path}_Material",
        color=color,
    )

    sphere_material.set_reflection_roughness(0.25)

    omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=prim_path,
        material_path=sphere_material.prim_path,
    )

    # ========================================================
    # Physics Material
    # ========================================================

    omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=prim_path,
        material_path=physics_material.prim_path,
        strength="strongerThanDescendants",
    )

# ============================================================
# Configure Scene
# ============================================================

def configure_scene(
    angle_deg,
    friction_val,
    sample_index,
):

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

    COLOR_BG, COLOR_FLOOR = random.sample(bg_colors, 2)
    COLOR_INCLINE, COLOR_BLOCKS, COLOR_SPHERE = random.sample(all_colors, 3)

    ground_material.set_color(COLOR_FLOOR)
    ground.apply_visual_material(ground_material)
    bg_material.set_color(COLOR_BG)
    background.apply_visual_material(bg_material)
    wood_material.set_color(COLOR_INCLINE)
    left_plane.apply_visual_material(wood_material)
    right_plane.apply_visual_material(wood_material)

    # ========================================================
    # Sphere Properties
    # ========================================================

    sphere_radius = random.uniform(0.10, 0.20)

    sphere_mass = random.uniform(0.5, 1)

    sphere_color = COLOR_SPHERE

    # ========================================================
    # Ground Block Properties
    # ========================================================

    block_size = sphere_radius * 2

    block_density = 600

    ground_block_color = COLOR_BLOCKS

    # ========================================================
    # Incline Positions
    # ========================================================

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
    # Inclines
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

    right_plane.apply_physics_material(
        phys_mat
    )

    left_plane.apply_physics_material(
        phys_mat
    )

    # ========================================================
    # Random sphere positions
    # ========================================================

    len1 = random.choice(
        [-0.25, 0.25, 0.75]
    )

    len2 = random.choice(
        [-0.25, 0.25, 0.75]
    )

    dist1 = -len1 * half_len
    dist2 = -len2 * half_len

    offset_z = (
        plane_thick
        + sphere_radius
        + 0.002
    )

    # ========================================================
    # Right sphere
    # ========================================================

    bx1 = (
        plane_x
        + dist1 * np.cos(angle_rad)
        + offset_z * np.sin(angle_rad)
    )

    bz1 = (
        plane_z
        - dist1 * np.sin(angle_rad)
        + offset_z * np.cos(angle_rad)
    )

    # ========================================================
    # Left sphere
    # ========================================================

    bx2 = (
        plane_x
        + dist2 * np.cos(angle_rad)
        + offset_z * np.sin(angle_rad)
    )

    bz2 = (
        plane_z
        - dist2 * np.sin(angle_rad)
        + offset_z * np.cos(angle_rad)
    )

    # ========================================================
    # Delete old objects
    # ========================================================

    prims_to_delete = [
        "/World/right_sphere",
        "/World/left_sphere",
        "/World/ground_block_right",
        "/World/ground_block_left",
    ]

    for prim_path in prims_to_delete:

        if stage.GetPrimAtPath(prim_path).IsValid():

            omni.kit.commands.execute(
                "DeletePrims",
                paths=[prim_path],
            )

    # ========================================================
    # Create Right Sphere
    # ========================================================

    create_dynamic_sphere(
        prim_path="/World/right_sphere",
        radius=sphere_radius,
        position=np.array([bx1, 0.0, bz1]),
        color=sphere_color,
        mass=sphere_mass,
        physics_material=phys_mat,
    )

    # ========================================================
    # Create Left Sphere
    # ========================================================

    create_dynamic_sphere(
        prim_path="/World/left_sphere",
        radius=sphere_radius,
        position=np.array([-bx2, 0.0, bz2]),
        color=sphere_color,
        mass=sphere_mass,
        physics_material=phys_mat,
    )

    # ========================================================
    # Ground Blocks
    # ========================================================

    block_right = DynamicCuboid(
        prim_path="/World/ground_block_right",
        position=np.array(
            [
                plane_x + half_len + block_size + 0.5,
                0.0,
                block_size / 2.0,
            ]
        ),
        size=block_size ,
        color=ground_block_color,
        mass=sphere_mass / 5
    )

    block_left = DynamicCuboid(
        prim_path="/World/ground_block_left",
        position=np.array(
            [
                -(plane_x + half_len + block_size + 0.5),
                0.0,
                block_size / 2.0,
            ]
        ),
        size=block_size,
        color=ground_block_color,
        mass=sphere_mass / 5
    )

    block_right.apply_physics_material(
        phys_mat
    )

    block_left.apply_physics_material(
        phys_mat
    )

    print("Configured sample:", sample_index)

    metadata = {
        "resolution": list(RESOLUTION),
        "angle_deg": angle_deg,
        "friction_val": friction_val,
        "sphere_radius": sphere_radius,
        "sphere_mass": sphere_mass,
        "block_size": block_size,

        "pos1": len1,
        "pos2": len2,
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

    print("Warmup frames...")

    # Render-only passes (no physics) so materials are loaded; frame 0 is
    # then the release pose with every object at rest
    for _ in range(3):
        capture_frame()

    print("Rendering...")

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


    create_scene()

    simulation_context.initialize_physics()

    num_samples = ARGS.num_samples

    for i in range(1, num_samples + 1):

        carb.log_info(
            f"Running sample {i}"
        )

        angle_deg = random.uniform(25.0, 45.0)

        # friction_val = random.uniform(
        #     0.0, 0.2
        # )
        friction_val = 0.0

        simulation_context.reset()

        metadata = configure_scene(
            angle_deg=angle_deg,
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
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()