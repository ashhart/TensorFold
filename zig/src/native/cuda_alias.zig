//! Which CUDA family opens a model_type: its own name first, then `cuda_also_model_types` (not HIP's aliases).
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

test "kolibri1_1 opens Kolibri-1's family; nemotron's wrappers open as before (omni its own)" {
    try std.testing.expectEqualStrings("kolibri1", name(which("kolibri1_1")));
    try std.testing.expectEqualStrings("kolibri1", name(which("kolibri1")));
    try std.testing.expectEqualStrings("nemotron_h_omni", name(which("nemotron_h_omni")));
    try std.testing.expectEqualStrings("nemotron_h", name(which("nemotron_h")));
    try std.testing.expectEqualStrings("nemo_speechlm", name(which("nemo_speechlm")));
    try std.testing.expectEqualStrings("none", name(which("kolibri2")));
}

/// The dispatch before `cuda_also_model_types`: a family by its own model_type only.
fn exact(model_type: []const u8) ?usize {
    inline for (registry, 0..) |F, i| if (std.mem.eql(u8, model_type, F.model_type)) return i;
    return null;
}

test "every model_type the registry knows opens where it opened before: only cuda_also_model_types moved" {
    inline for (registry) |F| {
        try std.testing.expectEqual(exact(F.model_type), which(F.model_type));
        if (@hasDecl(F, "also_model_types")) for (F.also_model_types) |t| try std.testing.expectEqual(exact(t), which(t));
    }
}

/// Whether `F` takes this checkpoint of one of its other model_types, by its own format check (`claims`).
pub fn claims(comptime F: type, a: std.mem.Allocator, io: std.Io, model_type: []const u8, dir: []const u8) bool {
    if (comptime !@hasDecl(F, "claims")) return false;
    if (std.mem.eql(u8, model_type, F.model_type)) return F.claims(a, io, dir);
    if (@hasDecl(F, "also_model_types")) for (F.also_model_types) |t| if (std.mem.eql(u8, model_type, t)) return F.claims(a, io, dir);
    return false;
}

/// The registry index that opens this checkpoint: a family claiming it by its own check first, then its model_type's.
pub fn pick(comptime families: anytype, a: std.mem.Allocator, io: std.Io, model_type: []const u8, dir: []const u8) ?usize {
    inline for (families, 0..) |F, i| if (claims(F, a, io, model_type, dir)) return i;
    inline for (families, 0..) |F, i| if (std.mem.eql(u8, model_type, F.model_type)) return i;
    return null;
}

test {
    _ = @import("cuda_alias_test.zig");
}
