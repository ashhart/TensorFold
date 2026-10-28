# From Motion Previs Studio 4.1.0 by Sam Wasserman (https://github.com/wassermanproductions/motion-previs-studio),
# Apache-2.0, Copyright 2026 Sam Wasserman. This is the Blender import script its production pack ships, taken from
# electron/main.cjs; only this header and the version in the last message were changed. See LICENSES/ in the Studio repo.
import json
import math
import os
import runpy
import bpy

FOLDER = os.path.dirname(os.path.abspath(__file__))

def run_if_exists(name):
    path = os.path.join(FOLDER, name)
    if os.path.exists(path):
        runpy.run_path(path, run_name="__main__")

def make_movie_plane(name, movie_file, location, scale=(3.2, 1.8, 1.0)):
    path = os.path.join(FOLDER, movie_file)
    if not os.path.exists(path):
        return None
    bpy.ops.mesh.primitive_plane_add(size=1, location=location, rotation=(math.radians(90), 0, 0))
    plane = bpy.context.object
    plane.name = name
    plane.scale = scale
    material = bpy.data.materials.new(name + " Material")
    material.use_nodes = True
    image = bpy.data.images.load(path)
    image.source = "MOVIE"
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    material.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    plane.data.materials.append(material)
    return plane

def main():
    run_if_exists("blender_import_camera.py")
    run_if_exists("blender_import_pose.py")
    make_movie_plane("Reference Plate", "reference.mp4", (0, 1.8, 1.0), (3.2, 1.8, 1.0))
    make_movie_plane("Depth Plate", "depth.mp4", (-3.5, 2.0, 1.0), (1.25, 1.25, 1.0))
    make_movie_plane("Edges Plate", "edges.mp4", (3.5, 2.0, 1.0), (1.25, 1.25, 1.0))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(FOLDER, "motion_previs_scene.blend"))
    print("Built Motion Previs Studio 4.1.0 (Python port in Wasserman Slate AI Studio) Blender scene with camera, pose, and reference plates.")

if __name__ == "__main__":
    main()
