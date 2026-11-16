The frozen Metal file is TensorFold's own lane kernel from main
fb8fdf3d8501255058a3221452164c56cb9a57ca. It is copied without modification from
zig/kernels/metal/core/lane_projection.metal and retains the repository's Apache-2.0 license.

core_lane_long.zig compiles this kernel with each tested shape's defines and uses
16-row dispatches as the byte reference for the current whole-prompt kernel.
The fixture must stay frozen when the production kernel changes.
