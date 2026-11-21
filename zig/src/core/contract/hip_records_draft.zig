//! Gate 5's records of the DFlash2 drafter's kernels (draft.hip's tf_draft_*: gathers, ring writes, conv, attention).
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

/// A draft.hip launch and its (VGPRs, SGPRs, LDS bytes) on each chip, wave32 with no scratch.
fn drafter(comptime name: []const u8, threads: u32, u: [limits.rdna.len][3]u32) Record {
    var usage: [limits.rdna.len]U = undefined;
    for (&usage, u) |*x, v| x.* = .{ .vgprs = v[0], .sgprs = v[1], .lds = v[2], .bound = threads };
    const symbol = "tf_draft_" ++ name;
    return .{ .launch = .{ .name = "drafter " ++ symbol, .symbol = symbol, .image = "draft", .threads = threads }, .usage = usage };
}

pub const records = [_]Record{
    drafter("gather", 256, .{ .{ 5, 23, 0 }, .{ 5, 23, 0 }, .{ 5, 23, 0 }, .{ 5, 24, 0 } }),
    drafter("ring_put", 256, .{ .{ 12, 23, 0 }, .{ 11, 23, 0 }, .{ 11, 23, 0 }, .{ 11, 27, 0 } }),
    drafter("dyn_conv", 256, .{ .{ 12, 38, 0 }, .{ 11, 38, 0 }, .{ 11, 38, 0 }, .{ 11, 38, 0 } }),
    drafter("mask_rows", 256, .{ .{ 5, 24, 0 }, .{ 6, 24, 0 }, .{ 6, 24, 0 }, .{ 6, 28, 0 } }),
    drafter("attn_partial", 128, .{ .{ 213, 51, 3616 }, .{ 213, 51, 3616 }, .{ 213, 51, 3616 }, .{ 212, 56, 3616 } }),
    drafter("attn_merge", 32, .{ .{ 21, 19, 0 }, .{ 21, 19, 0 }, .{ 19, 20, 0 }, .{ 19, 20, 0 } }),
};
