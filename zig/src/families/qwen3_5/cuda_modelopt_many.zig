//! ModelOpt NVFP4 prompt rows in one launch on the screened Qwen3.8 GB10 geometries.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");

const Engine = engine.Engine;
const Weight = cuda.qlinear.Weight;
const modelopt = @import("cuda_modelopt.zig");

/// The screened rows: at or past the large-prompt regime the GB10 A/B ran (2k, 8k and 32k prompts).
pub const min_rows: usize = 2048;
/// prompt_many.cu's part count.
pub const max_parts: usize = cuda.nvfp4.parts_max;

/// Every member's outputs, as the dense MLP's gate and up share the layer input on
/// nvidia/Qwen3.8-27B-NVFP4 (MLPs NVFP4): k 5120, 17408 outputs, two to eight of them.
pub fn admitted(e: *const Engine, ws: []const Weight, rows: usize) bool {
    if (!e.affine.pdl or rows < min_rows or ws.len < 2 or ws.len > max_parts) return false;
    if (e.mo == null) return false;
    for (ws) |w| {
        if (w != .nvfp4) return false;
        const q = w.nvfp4;
        if (q.k != 5120 or q.n != 17408) return false;
    }
    return true;
}

/// The fused launch when `admitted` holds; false, without touching the stream, otherwise.
pub fn pair(e: *Engine, gate: Weight, up: Weight, x: u64, gate_out: u64, up_out: u64, rows: usize) !bool {
    if (!admitted(e, &.{ gate, up }, rows)) return false;
    return group(e, &.{ gate, up }, x, &.{ gate_out, up_out }, rows);
}

/// `pair` over an already gathered group; callers check `admitted` through `pair`.
pub fn group(e: *Engine, ws: []const Weight, x: u64, outs: []const u64, rows: usize) !bool {
    if (!admitted(e, ws, rows)) return false;
    const k = if (e.mo) |*m| m else return error.FormatNotLoaded;
    const l = k.linear();
    var members: [max_parts]cuda.nvfp4.Prompt.ManyMember = undefined;
    for (ws, outs, members[0..ws.len]) |w, out, *m| m.* = .{ .w = w.nvfp4, .out = out };
    try l.nvfp4_prompt.?.matmulMany(e.stream, x, 5120, rows, members[0..ws.len], .bf16);
    return true;
}

test "the screened admission holds the 27B MLP pair on GB10 and refuses other shapes" {
    const nvfp4 = cuda.qlinear.Weight{ .nvfp4 = cuda.nvfp4.weight(1, 2, 1.0, 17408, 5120) };
    const wide = cuda.qlinear.Weight{ .nvfp4 = cuda.nvfp4.weight(1, 2, 1.0, 6144, 5120) };
    const deep = cuda.qlinear.Weight{ .nvfp4 = cuda.nvfp4.weight(1, 2, 1.0, 5120, 17408) };
    var e: Engine = undefined;
    e.affine.pdl = true;
    e.mo = null;
    try std.testing.expect(!admitted(&e, &.{ nvfp4, nvfp4 }, min_rows)); // no ModelOpt kernels
    const k: modelopt.Kernels = undefined;
    e.mo = k;
    defer e.mo = null;
    try std.testing.expect(admitted(&e, &.{ nvfp4, nvfp4 }, min_rows));
    try std.testing.expect(admitted(&e, &.{ nvfp4, nvfp4, nvfp4, nvfp4 }, min_rows + 4096));
    try std.testing.expect(!admitted(&e, &.{nvfp4}, min_rows)); // one projection: no group
    try std.testing.expect(!admitted(&e, &.{ nvfp4, nvfp4 }, min_rows - 1)); // rows below the screen
    try std.testing.expect(!admitted(&e, &.{ wide, nvfp4 }, min_rows)); // not the MLP width
    try std.testing.expect(!admitted(&e, &.{ deep, deep }, min_rows)); // not the MLP depth
    e.affine.pdl = false;
    try std.testing.expect(!admitted(&e, &.{ nvfp4, nvfp4 }, min_rows)); // not the screened GPU class
}
