//! Which CUDA family opens a config's model_type: its own name first, then a family's `cuda_also_model_types`.
//! `also_model_types` (HIP's aliases) stays out of it: nemotron_h_omni has a family of its own on CUDA.
const std = @import("std");
const registry = @import("cuda_families.zig").registry;

/// The registry index of the family that opens `model_type`, or null.
pub fn which(model_type: []const u8) ?usize {
    inline for (registry, 0..) |F, i| if (std.mem.eql(u8, model_type, F.model_type)) return i;
    inline for (registry, 0..) |F, i| if (@hasDecl(F, "cuda_also_model_types")) {
        for (F.cuda_also_model_types) |t| if (std.mem.eql(u8, model_type, t)) return i;
    };
    return null;
}

fn name(i: ?usize) []const u8 {
    inline for (registry, 0..) |F, j| if (i == j) return F.model_type;
    return "none";
}

test "kolibri1_1 opens Kolibri-1's family; nemotron's wrappers open as before (omni its own, Transcribe none)" {
    try std.testing.expectEqualStrings("kolibri1", name(which("kolibri1_1")));
    try std.testing.expectEqualStrings("kolibri1", name(which("kolibri1")));
    try std.testing.expectEqualStrings("nemotron_h_omni", name(which("nemotron_h_omni")));
    try std.testing.expectEqualStrings("nemotron_h", name(which("nemotron_h")));
    try std.testing.expectEqualStrings("none", name(which("nemo_speechlm")));
    try std.testing.expectEqualStrings("none", name(which("kolibri2")));
}
