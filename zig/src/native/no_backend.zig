//! The refusal when no native backend serves a model_type: Python 0.6.6 is named only for the types it serves.
const std = @import("std");

/// The model types the Python engine 0.6.6 serves, as its own refusal lists them.
pub const python_066 = [_][]const u8{
    "deepseek_v4",
    "gemma4",
    "gemma4_text",
    "glm5_next",
    "nemotron_h",
    "prism_hadamard_qwen35",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_8_flash_next",
    "qwen4_exp",
};

/// Whether the Python engine 0.6.6 serves `model_type`.
pub fn pythonServes(model_type: []const u8) bool {
    for (python_066) |t| if (std.mem.eql(u8, t, model_type)) return true;
    return false;
}

/// What `open` says when no native backend serves `model_type`.
pub fn message(a: std.mem.Allocator, model_type: []const u8) ![]const u8 {
    if (pythonServes(model_type)) return std.fmt.allocPrint(a, "the native engine has no backend for {s} checkpoints yet; the Python engine 0.6.6 may serve them: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6", .{model_type});
    return std.fmt.allocPrint(a, "no TensorFold release serves {s} checkpoints yet. To run one, write a recipe with the recipe book (https://github.com/ashhart/TensorFold/blob/main/docs/recipes/README.md), and read the runbook first (https://github.com/ashhart/TensorFold/blob/main/RUNBOOK.md)", .{model_type});
}

test "no backend: every model type 0.6.6 serves still gets its install hint, word for word" {
    const a = std.testing.allocator;
    for (python_066) |t| {
        const got = try message(a, t);
        defer a.free(got);
        const want = try std.fmt.allocPrint(a, "the native engine has no backend for {s} checkpoints yet; the Python engine 0.6.6 may serve them: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6", .{t});
        defer a.free(want);
        try std.testing.expectEqualStrings(want, got);
    }
}

test "no backend: hy_v4_flash, which 0.6.6 also refuses, is pointed at the recipe book and runbook, not at 0.6.6" {
    const a = std.testing.allocator;
    const got = try message(a, "hy_v4_flash");
    defer a.free(got);
    try std.testing.expect(std.mem.startsWith(u8, got, "no TensorFold release serves hy_v4_flash checkpoints yet."));
    try std.testing.expect(std.mem.indexOf(u8, got, "0.6.6") == null);
    try std.testing.expect(std.mem.indexOf(u8, got, "pip install") == null);
    try std.testing.expect(std.mem.indexOf(u8, got, "docs/recipes/README.md") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "RUNBOOK.md") != null);
}
