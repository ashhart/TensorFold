# From Motion Previs Studio 4.1.0 by Sam Wasserman (https://github.com/wassermanproductions/motion-previs-studio),
# Apache-2.0, Copyright 2026 Sam Wasserman. This is the Blender import script its production pack ships, taken from
# electron/main.cjs; only this header and the version in the last message were changed. See LICENSES/ in the Studio repo.
import json
import math
import os
import bpy

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index"
]

def load_pose():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "pose_landmarks.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def clear_collection(name):
    existing = bpy.data.collections.get(name)
    if existing:
        for obj in list(existing.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(existing)

def make_joint(collection, name):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.035)
    obj = bpy.context.object
    obj.name = name
    collection.objects.link(obj)
    try:
        bpy.context.collection.objects.unlink(obj)
    except Exception:
        pass
    return obj

def main():
    data = load_pose()
    frames = data.get("frames", [])
    fps = data.get("fps", 12)
    clear_collection("Motion Previs Pose")
    collection = bpy.data.collections.new("Motion Previs Pose")
    bpy.context.scene.collection.children.link(collection)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = max(1, len(frames))
    bpy.context.scene.render.fps = int(fps)

    joints = [make_joint(collection, f"{i:02d}_{LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else 'joint'}") for i in range(33)]

    for frame_index, frame in enumerate(frames, start=1):
        points = frame.get("worldLandmarks") or frame.get("landmarks") or []
        bpy.context.scene.frame_set(frame_index)
        for idx, obj in enumerate(joints):
            if idx >= len(points) or not points[idx]:
                continue
            point = points[idx]
            x = float(point.get("x", 0))
            y = float(point.get("y", 0))
            z = float(point.get("z", 0))
            obj.location = (x * 2.5, -z * 2.5, -y * 2.5)
            obj.keyframe_insert(data_path="location", frame=frame_index)

    for obj in joints:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"

    print(f"Imported {len(frames)} pose frames from Motion Previs Studio 4.1.0 (Python port in Wasserman Slate AI Studio).")

if __name__ == "__main__":
    main()
