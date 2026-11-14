//! The device draw the Qwen CUDA lanes ask of a forward, from a stream's sampling rule.
const cuda = @import("cuda");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");

/// The device draw of a stream's rule at `position`: sample.cu's keyed rule, greedy at temperature 0 or without one.
pub fn pick(sampling: ?lanes.Sampling, position: u64) E.Draw {
    const s = sampling orelse return .{ .rule = null, .position = position };
    if (s.temperature <= 0) return .{ .rule = null, .position = position };
    return .{ .rule = cuda.sample.Rule.from(lanes.gpu_rule.rule(s)), .position = position };
}
