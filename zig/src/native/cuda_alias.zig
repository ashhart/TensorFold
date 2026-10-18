//! The config model_types a CUDA family reads: its own, then its wrappers' (as Bolt's nemotron_h_omni).
const std = @import("std");

/// The other config model_types a family reads (`also_model_types`).
pub fn aliases(comptime F: type) []const []const u8 {
    return if (@hasDecl(F, "also_model_types")) F.also_model_types else &.{};
}

pub fn reads(comptime F: type, model_type: []const u8) bool {
    if (std.mem.eql(u8, model_type, F.model_type)) return true;
    for (aliases(F)) |t| if (std.mem.eql(u8, model_type, t)) return true;
    return false;
}
