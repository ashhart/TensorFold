//! Inkling 2 Mini f32 top-k rule samplers matching lanes.gpu_rule.sample.
const guard = @import("../threadgroup_source.zig");
pub const sample = guard.normalize(@embedFile("sample.metal"));
