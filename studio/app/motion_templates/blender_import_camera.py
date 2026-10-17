# From Motion Previs Studio 4.1.0 by Sam Wasserman (https://github.com/wassermanproductions/motion-previs-studio),
# Apache-2.0, Copyright 2026 Sam Wasserman. This is the Blender import script its production pack ships, taken from
# electron/main.cjs; only this header and the version in the last message were changed. See LICENSES/ in the Studio repo.
import json
import math
import os
import bpy

def load_camera_motion():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "camera_motion.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    data = load_camera_motion()
    frames = data.get("frames", [])
    fps = int(data.get("fps", 12))

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = max(1, len(frames))
    bpy.context.scene.render.fps = fps

    camera_data = bpy.data.cameras.new("Motion Previs Solved Camera")
    camera = bpy.data.objects.new("Motion Previs Solved Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    camera.location = (0, -6, 1.6)
    camera.rotation_euler = (math.radians(78), 0, 0)
    camera_data.lens = 35

    for index, item in enumerate(frames, start=1):
        move = item.get("cameraMove", {})
        pan = float(move.get("pan", 0.0))
        tilt = float(move.get("tilt", 0.0))
        zoom = max(0.2, float(move.get("dollyZoom", 1.0)))
        roll = float(move.get("roll", 0.0))

        bpy.context.scene.frame_set(index)
        camera.location.x = pan * 3.5
        camera.location.z = 1.6 + tilt * 2.2
        camera.location.y = -6 / zoom
        camera.rotation_euler = (math.radians(78) + tilt * 0.22, pan * 0.18, roll)
        camera_data.lens = 35 * zoom
        camera.keyframe_insert(data_path="location", frame=index)
        camera.keyframe_insert(data_path="rotation_euler", frame=index)
        camera_data.keyframe_insert(data_path="lens", frame=index)

    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "BEZIER"

    if camera_data.animation_data and camera_data.animation_data.action:
        for fcurve in camera_data.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "BEZIER"

    print(f"Imported {len(frames)} camera move frames from Motion Previs Studio 4.1.0 (Python port in Wasserman Slate AI Studio).")

if __name__ == "__main__":
    main()
