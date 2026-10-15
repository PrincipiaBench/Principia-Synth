"""
Principia-Synth: MOI scenario.

A hollow and a solid cylinder of equal mass and size roll down mirrored inclines.

Usage (from the Isaac Sim install folder):
    ./python.sh /path/to/Principia-Synth/scenarios/render_moi.py --headless [--num_samples N]

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
parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "output", "MOI"), help="folder the sample_N folders are written to (default: %(default)s)")
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
    UsdPhysics,
    PhysxSchema,
)

from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import (
    PhysicsMaterial,
)

from isaacsim.core.prims import XFormPrim

import omni.kit.viewport.utility as vp_utils
import omni.replicator.core as rep
from PIL import Image

from omni.isaac.core.materials import OmniPBR
from pxr import Gf, UsdGeom
from pxr import Sdf
from omni.isaac.core.prims import RigidPrim
from omni.isaac.dynamic_control import _dynamic_control
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
# Colors
# ============================================================

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

simulation_context = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / (FPS*10),
    rendering_dt=1.0 / FPS,
)

stage = omni.usd.get_context().get_stage()

# ============================================================
# Camera
# ============================================================

camera_path = "/World/MainCamera"

camera = UsdGeom.Camera.Define(stage, camera_path)
camera.ClearXformOpOrder()

camera_pos = (0.0, -25, 7.0)
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

half_len = 3.0
plane_thick = 0.06
plane_width = 1.0

# ============================================================
# Shared Objects
# ============================================================

right_plane = None
left_plane = None

# ============================================================
# Create Scene
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

    right_plane.apply_visual_material(
        wood_material
    )

    left_plane.apply_visual_material(
        wood_material
    )

def compute_cylinder_inertia(mass, radius, height, is_hollow):

    if is_hollow:
        Ixx = 0.5 * mass * radius**2 + (1/12) * mass * height**2
        Iyy = Ixx
        Izz = mass * radius**2
    else:
        Ixx = 0.25 * mass * radius**2 + (1/12) * mass * height**2
        Iyy = Ixx
        Izz = 0.5 * mass * radius**2

    return Ixx, Iyy, Izz

# ============================================================
# Configure Scene
# ============================================================

def configure_scene(sample_index):

    COLOR_BG, COLOR_FLOOR = random.sample(bg_colors, 2)
    COLOR_INCLINE, COLOR_CYLINDER1, COLOR_CYLINDER2 = random.sample(all_colors, 3)

    ground_material.set_color(COLOR_FLOOR)
    ground.apply_visual_material(ground_material)
    bg_material.set_color(COLOR_BG)
    background.apply_visual_material(bg_material)
    wood_material.set_color(COLOR_INCLINE)
    left_plane.apply_visual_material(wood_material)
    right_plane.apply_visual_material(wood_material)
    left_mat = OmniPBR(
    prim_path="/World/Materials/LeftCylinderMat",
    color=COLOR_CYLINDER1,)
    right_mat = OmniPBR(
        prim_path="/World/Materials/RightCylinderMat",
        color=COLOR_CYLINDER2,
    )

    angle_deg = random.uniform(10, 30)

    angle_rad = np.radians(angle_deg)

    # d_scale = random.uniform(0.30, 0.50)

    # h_scale = (
    #     random.uniform(1.0, 1.5)
    #     * d_scale
    # )

    h_scale = d_scale = random.uniform(0.5, 1.0)

    radius = 0.5 * d_scale

    mass = random.uniform(5, 10)

    friction = 1.0

    # ========================================================
    # Plane transforms
    # ========================================================

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

    plane_x = (
        half_len
        * np.cos(angle_rad)
    )

    plane_z = (
        half_len * np.sin(angle_rad)
        + plane_thick * np.cos(angle_rad)
    )

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

    # ========================================================
    # Physics Material
    # ========================================================

    mat_path = (
        f"/World/PhysicsMaterial_{sample_index}"
    )

    phys_mat = PhysicsMaterial(
        prim_path=mat_path,
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )

    right_plane.apply_physics_material(
        phys_mat
    )

    left_plane.apply_physics_material(
        phys_mat
    )

    # ========================================================
    # Cylinder placement
    # ========================================================

    dist = -0.85 * half_len

    offset = (
        plane_thick
        + radius
        + 0.002
    )

    bx = (
        plane_x
        + dist * np.cos(angle_rad)
        + offset * np.sin(angle_rad)
    )

    bz = (
        plane_z
        - dist * np.sin(angle_rad)
        + offset * np.cos(angle_rad)
    )

    # ========================================================
    # Delete previous cylinders
    # ========================================================

    paths_to_delete = []

    if stage.GetPrimAtPath(
        "/World/right_cylinder"
    ).IsValid():

        paths_to_delete.append(
            "/World/right_cylinder"
        )

    if stage.GetPrimAtPath(
        "/World/left_cylinder"
    ).IsValid():

        paths_to_delete.append(
            "/World/left_cylinder"
        )

    if len(paths_to_delete) > 0:

        omni.kit.commands.execute(
            "DeletePrims",
            paths=paths_to_delete,
        )

    # ========================================================
    # Assets
    # ========================================================

    ASSET_DIR = os.path.join(
        REPO_ROOT,
        "assets",
        "models",
    )

    for asset_name in ("hollow_cylinder.usdc", "solid_cylinder.usdc"):
        if not os.path.exists(os.path.join(ASSET_DIR, asset_name)):
            # Fail loudly: a missing reference renders an empty scene without error
            raise FileNotFoundError(os.path.join(ASSET_DIR, asset_name))

    hollow_asset = os.path.join(
        ASSET_DIR,
        "hollow_cylinder.usdc",
    )

    solid_asset = os.path.join(
        ASSET_DIR,
        "solid_cylinder.usdc",
    )

    print("HOLLOW ASSET:", hollow_asset)
    print("SOLID ASSET :", solid_asset)

    hollow_is_left = random.choice(
        [True, False]
    )

    if hollow_is_left:

        left_asset = hollow_asset
        right_asset = solid_asset

    else:

        left_asset = solid_asset
        right_asset = hollow_asset

    # ========================================================
    # LEFT CYLINDER
    # ========================================================

    left_path = "/World/left_cylinder"

    left_prim = stage.DefinePrim(
        left_path,
        "Xform",
    )

    left_prim.GetReferences().AddReference(
        left_asset
    )

    left_xform = UsdGeom.Xformable(left_prim)

    left_xform.ClearXformOpOrder()

    left_xform.AddTranslateOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(
        Gf.Vec3d(
            -bx,
            0.0,
            bz,
        )
    )

    rot_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), 90)
    rot_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)
    rot_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)

    rot_0 = Gf.Rotation(Gf.Vec3d(0, 0, 1), 0)

    combined = rot_y*rot_z   # order matters!

    left_xform.AddOrientOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(
        combined.GetQuat()
    )

    left_xform.AddScaleOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(
        Gf.Vec3d(
            h_scale,
            h_scale,
            h_scale,
        )
    )

    # ========================================================
    # RIGHT CYLINDER
    # ========================================================


    right_path = "/World/right_cylinder"

    right_prim = stage.DefinePrim(
        right_path,
        "Xform",
    )

    right_prim.GetReferences().AddReference(
        right_asset
    )

    right_xform = UsdGeom.Xformable(right_prim)

    right_xform.ClearXformOpOrder()

    right_xform.AddTranslateOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(
        Gf.Vec3d(
            bx,
            0.0,
            bz,
        )
    )

    

    

    right_xform.AddOrientOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(
        combined.GetQuat()
    )

    right_xform.AddScaleOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(
        Gf.Vec3d(
            h_scale,
            h_scale,
            h_scale,
        )
    )

    omni.kit.commands.execute(
    "BindMaterialCommand",
    prim_path="/World/left_cylinder",
    material_path=left_mat.prim_path,)

    omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path="/World/right_cylinder",
        material_path=right_mat.prim_path,
    )

    # ========================================================
    # Physics
    # ========================================================

    for path, is_hollow in [
    ("/World/left_cylinder", hollow_is_left),
    ("/World/right_cylinder", not hollow_is_left),
]:

        prim = stage.GetPrimAtPath(path)

        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)

        prim.CreateAttribute(
    "physics:approximation",
    Sdf.ValueTypeNames.Token
).Set("convexHull")


        mass_api = UsdPhysics.MassAPI.Apply(prim)

        mass_api.CreateMassAttr(mass)

        height = h_scale

        Ixx, Iyy, Izz = compute_cylinder_inertia(
            mass=mass,
            radius=radius,
            height=height,
            is_hollow=is_hollow,
        )

        print(path, Ixx, Iyy, Izz)

        mass_api.CreateDiagonalInertiaAttr(
            Gf.Vec3f(Ixx, Izz, Iyy)
        )

        mass_api.CreateCenterOfMassAttr(
            Gf.Vec3f(0, 0, 0)
        )

        # optional but recommended (identity principal axes)
        # prim.CreateAttribute("physx:principalAxes", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))

        # # ALSO force PhysX to respect custom inertia
        # mass_api = UsdPhysics.MassAPI.Apply(prim)
        # mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))

    metadata = {
        "resolution": list(RESOLUTION),
        "angle_deg": angle_deg,
        "mass": mass,
        "radius": radius,
        "height_scale": h_scale,
        "hollow_is_left": hollow_is_left,
        "fps": FPS,
        "duration": DURATION,
    }

    return metadata

# ============================================================
# Rendering
# ============================================================

def render_sample(sample_index, dc):

    left_rb = RigidPrim("/World/left_cylinder")
    right_rb = RigidPrim("/World/right_cylinder")

    sample_dir = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_index}",
    )

    os.makedirs(sample_dir, exist_ok=True)

    print("Warmup...")

    # Render-only passes (no physics) so materials are loaded; frame 0 is
    # then the release pose with every object at rest
    for _ in range(3):
        capture_frame()

    print("Rendering...")

    for frame in range(TOTAL_FRAMES):

        left_w = left_rb.get_angular_velocity()
        right_w = right_rb.get_angular_velocity()

        print("LEFT :", left_w)
        print("RIGHT:", right_w)

        frame_path = os.path.join(
            sample_dir,
            f"frame_{frame:04d}.png",
        )

        Image.fromarray(capture_frame()).save(frame_path)

        step_physics()

    # Queried after the physics has stepped, so PhysX has created the bodies
    left_handle = dc.get_rigid_body("/World/left_cylinder")
    right_handle = dc.get_rigid_body("/World/right_cylinder")

    left_props = dc.get_rigid_body_properties(left_handle)
    right_props = dc.get_rigid_body_properties(right_handle)

    print("\nLEFT CYLINDER")
    print("Mass:", left_props.mass)
    print("Inertia:", left_props.moment)

    print("\nRIGHT CYLINDER")
    print("Mass:", right_props.mass)
    print("Inertia:", right_props.moment)

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


    create_scene()

    simulation_context.initialize_physics()

    dc = _dynamic_control.acquire_dynamic_control_interface()

    num_samples = ARGS.num_samples

    for i in range(1, num_samples + 1):

        print(f"\nRunning sample {i}")

        simulation_context.reset()

        metadata = configure_scene(i)

        sample_dir = render_sample(i, dc)

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