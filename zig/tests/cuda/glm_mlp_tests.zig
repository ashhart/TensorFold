//! glm.layer.mlp: a real RED-SNOW dense-MLP KDA layer at TP=2 rank 0 vs the Python engine's layer_forward, bit for bit.
const glm_tests = @import("glm_tests.zig");
const layer_tests = @import("glm_layer_tests.zig");

const id = "glm.layer.mlp";

pub fn run(r: *glm_tests.Rig, state: *glm_tests.Run, dir: []const u8) !void {
    layer_tests.kdaLayer(r, dir, "mlp", id) catch |e| state.fail(e, id);
}
