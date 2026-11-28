//! The dots.mocr shapes the dense engine serves stay on the BF16 prompt tile (issue #3754).
const std = @import("std");
const cuda = @import("cuda");
const q = cuda.qmmf;

test "the dots.mocr prompt projections the dense engine serves are prompt-eligible" {
    // Issue #3754's measured shapes (QKV, out, gate/up, down) at 1024 rows.
    const f: cuda.Function = .{ .d = undefined, .handle = @ptrFromInt(1) };
    var lane: q.Lane = .{ .mod = undefined, .fns = undefined, .f32_fns = undefined, .reduce = undefined, .half = undefined, .prompt = @splat(@splat(f)), .prompt_sms = 48, .clusters = false };
    for ([_]struct { n: u32, k: u32, sk: u32 }{ .{ .n = 2048, .k = 1536, .sk = 2 }, .{ .n = 1536, .k = 1536, .sk = 2 }, .{ .n = 17920, .k = 1536, .sk = 1 }, .{ .n = 1536, .k = 8960, .sk = 4 } }) |s| {
        const w = try q.pinned(q.Weight{ .mode = .bf16, .codes = 0, .scales = 0, .n = s.n, .k = s.k, .npad = s.n }, s.sk);
        try std.testing.expect(q.promptEligible(1024, w) and lane.usesPrompt(1024, w) and !q.promptEligible(q.prompt_rows - 1, w));
    }
    lane.prompt = null; // no loadPrompt (TF_DENSE_PROMPT_BF16=0): lane dispatch
    const w = try q.pinned(q.Weight{ .mode = .bf16, .codes = 0, .scales = 0, .n = 2048, .k = 1536, .npad = 2048 }, 2);
    try std.testing.expect(!lane.usesPrompt(1024, w));
}
