#!/usr/bin/env python3

"""
Isaac Sim - Dual Pendulum Dataset Generator
-------------------------------------------

Creates:
- two ceiling anchors
- two pendulum spheres (same colour)
- two distance joint constraints
- randomized pendulum lengths
- randomized sphere sizes
- opposite randomized initial angles
- path-traced rendering (fully accumulated per frame)
- clean 1280x720 frame saving via a Replicator render product (no viewport UI)
- multiple samples

Run:
./python.sh render_pendulum_isaac.py --headless [--num_samples 100] [--output_dir /mnt/data/varun/Self-Refinement-Training/Pendulum]
"""

import os
import sys
import math
import json
import random
import argparse
import subprocess
import numpy as np

# ============================================================
# SimulationApp MUST come first
# ============================================================

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "Pendulum"))
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
    UsdPhysics,
    PhysxSchema,
    Sdf,
)

from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane

import omni.kit.viewport.utility as vp_utils
import omni.replicator.core as rep
from omni.isaac.core.materials import OmniPBR
from PIL import Image
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
settings.set("/rtx/pathtracing/maxBounces", 8)

settings.set("/rtx/post/aa/op", 3)

# Only capture frames when rep.orchestrator.step() is called
settings.set("/omni/replicator/captureOnPlay", False)

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
    physics_dt=1.0 / (FPS),
    rendering_dt=1.0 / FPS,
)

stage = omni.usd.get_context().get_stage()

# ============================================================
# COLORS
# Extended palette (same style as scripts/colours_palette.json, more colours)
PALETTE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colours_palette.json")
colors_palette = json.load(open(PALETTE_PATH, "r"))
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

camera_pos = (0.0, -9.5, 2.5)
camera_rot = (80.0, 0.0, 0.0)

camera.AddTranslateOp().Set(Gf.Vec3f(*camera_pos))
camera.AddRotateXYZOp().Set(Gf.Vec3f(*camera_rot))

# wider cinematic lens
camera.CreateFocalLengthAttr(50)

# 16:9 sensor to match the output resolution
camera.CreateHorizontalApertureAttr(20.955)
camera.CreateVerticalApertureAttr(20.955 * RESOLUTION[1] / RESOLUTION[0])

viewport = vp_utils.get_active_viewport()
viewport.set_active_camera(camera_path)

# Frames come from the render product below; don't also path-trace the viewport
if HEADLESS:
    viewport.updates_enabled = False

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

# Dedicated render product on the camera: captures only the rendered image,
# without the viewport HUD / UI overlay or letterboxing
render_product = rep.create.render_product(camera_path, RESOLUTION)

rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])

# ============================================================
# Globals
# ============================================================

LEFT_ANCHOR = np.array([-1.2, 0.0, 1.5])
RIGHT_ANCHOR = np.array([1.2, 0.0, 1.5])

# ============================================================
# Create Pendulums
# ============================================================

def create_pendulums():

    left_ceiling = FixedCuboid(
        prim_path="/World/LeftCeiling",
        position=LEFT_ANCHOR,
        scale=np.array([0.4, 0.4, 0.05]),
        color=np.array([0.8, 0.8, 0.8]),
    )

    right_ceiling = FixedCuboid(
        prim_path="/World/RightCeiling",
        position=RIGHT_ANCHOR,
        scale=np.array([0.4, 0.4, 0.05]),
        color=np.array([0.8, 0.8, 0.8]),
    )

# ============================================================
# Configure Sample
# ============================================================

def configure_sample(
    sample_index,
    length_left,
    length_right,
    radius_left,
    radius_right,
    angle_deg,
):
    
    COLOR_BG, COLOR_FLOOR = random.sample(bg_colors, 2)
    # Both spheres share one colour
    COLOR_SPHERE = random.choice(all_colors)

    ground_material.set_color(COLOR_FLOOR)
    ground.apply_visual_material(ground_material)
    bg_material.set_color(COLOR_BG)
    background.apply_visual_material(bg_material)
    wood_material.set_color(COLOR_INCLINE)

    # ========================================================
    # Delete old pendulums
    # ========================================================

    delete_paths = [
        "/World/LeftSphere",
        "/World/RightSphere",
        "/World/LeftJoint",
        "/World/RightJoint",
        "/World/LeftRope",
        "/World/RightRope",
    ]

    valid_paths = []

    for p in delete_paths:

        if stage.GetPrimAtPath(p).IsValid():
            valid_paths.append(p)

    if len(valid_paths) > 0:

        omni.kit.commands.execute(
            "DeletePrims",
            paths=valid_paths,
        )

    # ========================================================
    # Angles
    # ========================================================

    theta_left = math.radians(-angle_deg)
    theta_right = math.radians(-angle_deg)

    # ========================================================
    # Sphere positions
    # ========================================================

    left_sphere_pos = np.array([
        LEFT_ANCHOR[0] + length_left * math.sin(theta_left),
        0.0,
        LEFT_ANCHOR[2] - length_left * math.cos(theta_left),
    ])

    right_sphere_pos = np.array([
        RIGHT_ANCHOR[0] + length_right * math.sin(theta_right),
        0.0,
        RIGHT_ANCHOR[2] - length_right * math.cos(theta_right),
    ])

    print(f"Left sphere position: {left_sphere_pos}")
    print(f"Right sphere position: {right_sphere_pos}")

    # ========================================================
    # Left sphere
    # ========================================================

    left_sphere = DynamicSphere(
        prim_path="/World/LeftSphere",
        position=left_sphere_pos,
        radius=radius_left,
        mass=5.0,
        color=np.array(COLOR_SPHERE),
    )

    # ========================================================
    # Right sphere
    # ========================================================

    right_sphere = DynamicSphere(
        prim_path="/World/RightSphere",
        position=right_sphere_pos,
        radius=radius_right,
        mass=5.0,
        color=np.array(COLOR_SPHERE),
    )

    # ========================================================
    # Left Joint
    # ========================================================

    left_joint = UsdPhysics.DistanceJoint.Define(
        stage,
        "/World/LeftJoint",
    )

    left_joint.CreateBody0Rel().SetTargets(
        [Sdf.Path("/World/LeftCeiling")]
    )

    left_joint.CreateBody1Rel().SetTargets(
        [Sdf.Path("/World/LeftSphere")]
    )

    left_joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )

    left_joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )

    left_joint.CreateMinDistanceAttr().Set(
        length_left
    )

    left_joint.CreateMaxDistanceAttr().Set(
        length_left
    )

    # ========================================================
    # Right Joint
    # ========================================================

    right_joint = UsdPhysics.DistanceJoint.Define(
        stage,
        "/World/RightJoint",
    )

    right_joint.CreateBody0Rel().SetTargets(
        [Sdf.Path("/World/RightCeiling")]
    )

    right_joint.CreateBody1Rel().SetTargets(
        [Sdf.Path("/World/RightSphere")]
    )

    right_joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )

    right_joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )

    right_joint.CreateMinDistanceAttr().Set(
        length_right
    )

    right_joint.CreateMaxDistanceAttr().Set(
        length_right
    )

    # ========================================================
    # Spring APIs
    # ========================================================

    left_drive_api = (
        PhysxSchema.PhysxPhysicsDistanceJointAPI.Apply(
            left_joint.GetPrim()
        )
    )

    right_drive_api = (
        PhysxSchema.PhysxPhysicsDistanceJointAPI.Apply(
            right_joint.GetPrim()
        )
    )

    # ========================================================
    # Left Rope
    # ========================================================

    left_curve = UsdGeom.BasisCurves.Define(
        stage,
        "/World/LeftRope",
    )

    left_curve.CreateTypeAttr("linear")

    left_curve.CreateCurveVertexCountsAttr([2])

    left_curve.CreatePointsAttr([
        Gf.Vec3f(*LEFT_ANCHOR),
        Gf.Vec3f(*left_sphere_pos),
    ])

    left_curve.CreateWidthsAttr([0.02, 0.02])

    # ========================================================
    # Right Rope
    # ========================================================

    right_curve = UsdGeom.BasisCurves.Define(
        stage,
        "/World/RightRope",
    )

    right_curve.CreateTypeAttr("linear")

    right_curve.CreateCurveVertexCountsAttr([2])

    right_curve.CreatePointsAttr([
        Gf.Vec3f(*RIGHT_ANCHOR),
        Gf.Vec3f(*right_sphere_pos),
    ])

    right_curve.CreateWidthsAttr([0.02, 0.02])

    return {
        "sphere": np.asarray(COLOR_SPHERE).tolist(),
        "background": np.asarray(COLOR_BG).tolist(),
        "floor": np.asarray(COLOR_FLOOR).tolist(),
    }

# ============================================================
# Update Ropes
# ============================================================

def update_ropes(frame):

    # ========================================================
    # Left rope
    # ========================================================

    left_rope = UsdGeom.BasisCurves(
        stage.GetPrimAtPath("/World/LeftRope")
    )

    left_sphere_prim = stage.GetPrimAtPath(
        "/World/LeftSphere"
    )

    left_xform = omni.usd.get_world_transform_matrix(
        left_sphere_prim
    )

    left_pos = left_xform.ExtractTranslation()

    # ========================================================
    # Right rope
    # ========================================================

    right_rope = UsdGeom.BasisCurves(
        stage.GetPrimAtPath("/World/RightRope")
    )

    right_sphere_prim = stage.GetPrimAtPath(
        "/World/RightSphere"
    )

    right_xform = omni.usd.get_world_transform_matrix(
        right_sphere_prim
    )

    right_pos = right_xform.ExtractTranslation()

    # print(f"Frame {frame}")
    # print(f"Left sphere: {left_pos}")
    # print(f"Right sphere: {right_pos}")

    left_rope.GetPointsAttr().Set([
        Gf.Vec3f(*LEFT_ANCHOR),
        Gf.Vec3f(left_pos[0], left_pos[1], left_pos[2]),
    ])

    right_rope.GetPointsAttr().Set([
        Gf.Vec3f(*RIGHT_ANCHOR),
        Gf.Vec3f(right_pos[0], right_pos[1], right_pos[2]),
    ])

# ============================================================
# Render
# ============================================================

def capture_frame():
    # Render with the timeline held (delta_time=0.0) so the path tracer
    # accumulates the full TOTAL_SPP over RT_SUBFRAMES on a static scene
    rep.orchestrator.step(
        rt_subframes=RT_SUBFRAMES,
        delta_time=0.0,
        pause_timeline=False,
    )

    rgba = rgb_annotator.get_data()

    return np.asarray(rgba)[:, :, :3]


def render_frames(sample_dir):

    # Render-only warmup (no physics) so materials/textures are loaded;
    # frame 0 is then the release pose
    print("Warmup...")

    update_ropes(0)

    for _ in range(3):
        capture_frame()

    print("Rendering...")

    for frame in range(TOTAL_FRAMES):

        frame_path = os.path.join(
            sample_dir,
            f"frame_{frame:04d}.png",
        )

        update_ropes(frame)

        Image.fromarray(capture_frame()).save(frame_path)

        # Advance physics by one frame (physics_dt = 1 / FPS)
        simulation_context.step(render=False)

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

    subprocess.run(ffmpeg_cmd)

# ============================================================
# Main
# ============================================================

def main():

    create_pendulums()

    simulation_context.initialize_physics()

    num_samples = ARGS.num_samples

    for i in range(1, num_samples + 1):

        print(f"Running sample {i}")

        sample_dir = os.path.join(
            OUTPUT_DIR,
            f"sample_{i}"
        )

        os.makedirs(sample_dir, exist_ok=True)

        # ====================================================
        # Randomized parameters
        # ====================================================

        length_left = random.uniform(0.3, 1.0)
        length_right = random.uniform(0.3, 1.0)

        radius = random.uniform(0.07, 0.12)

        angle_deg = random.uniform(20.0, 45.0)

        # ====================================================
        # Reset simulation
        # ====================================================

        simulation_context.reset()

        colors = configure_sample(
            sample_index=i,
            length_left=length_left,
            length_right=length_right,
            radius_left=radius,
            radius_right=radius,
            angle_deg=angle_deg,
        )

        render_frames(sample_dir)

        create_video(sample_dir)

        metadata = {
            "length_left": length_left,
            "length_right": length_right,
            "radius_left": radius,
            "radius_right": radius,
            "angle_deg": angle_deg,
            "fps": FPS,
            "duration": DURATION,
            "resolution": list(RESOLUTION),
            "colors_rgb_0_1": colors,
        }

        with open(
            os.path.join(sample_dir, "metadata.json"),
            "w",
        ) as f:

            json.dump(metadata, f, indent=4)

    simulation_context.stop()

    simulation_app.close()

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()