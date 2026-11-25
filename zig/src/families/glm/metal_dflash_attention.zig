//! GLM's DFlash2 drafter attention launch (glm_draft_attention.metal): block rows over the K/V ring, a key a lane.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// Keys a threadgroup takes, the most streams a pass and rows a block, threads a threadgroup (one simdgroup).
pub const chunk = 128;
pub const max_streams = 8;
pub const max_block = 16;
pub const threads = 32;

/// glm_draft_args: `streams` blocks of `block` rows; mask rows `mask_stride` apart; `chunks` the longest stream's.
pub const Args = extern struct { rows: u32, block: u32, streams: u32, mask_stride: u32, scale_bits: u32, window: u32, chunks: u32, pad: u32 = 0 };
/// glm_draft_stream: a stream's keys, [0, context) at its ring's slots (at + key) % window, the rest its own rows.
pub const Stream = extern struct { keys: u32, context: u32, at: u32, pad: u32 = 0 };

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
/// A stream's rolling K/V for the layer.
pub const Ring = struct { k: Ref, v: Ref };

/// The query and KV heads and head dim the kernel is compiled for.
pub const Shape = struct { heads: u32, kv_heads: u32, dim: u32 };

pub fn chunks(keys: u32) u32 {
    return (keys + chunk - 1) / chunk;
}

/// A pass's arguments for `table`'s streams of `block` rows each, or the refusal of a table the kernel cannot read.
pub fn plan(table: []const Stream, block: u32, window: u32, scale: f32) !Args {
    if (table.len == 0 or table.len > max_streams or block == 0 or block > max_block or window == 0) return error.InvalidDraftAttention;
    var most: u32 = 0;
    for (table) |t| {
        if (t.context > window or t.keys != t.context + block or t.at >= window) return error.InvalidDraftAttention;
        most = @max(most, t.keys);
    }
    return .{ .rows = @intCast(table.len * block), .block = block, .streams = @intCast(table.len), .mask_stride = most, .scale_bits = @bitCast(scale), .window = window, .chunks = chunks(most) };
}

/// fp32 partials the attend step leaves for the merge: [rows][kv heads][chunks][heads a kv head][dim + 4].
pub fn partBytes(s: Shape, args: Args) usize {
    return @as(usize, args.rows) * s.heads * args.chunks * (s.dim + 4) * 4;
}

pub fn source(a: std.mem.Allocator, s: Shape) ![]u8 {
    if (s.kv_heads == 0 or s.heads % s.kv_heads != 0 or (s.dim != 128 and s.dim != 64) or s.heads / s.kv_heads != 4) return error.InvalidDraftAttentionShape;
    return std.fmt.allocPrint(a, "#define TF_QH {d}\n#define TF_KVH {d}\n#define TF_D {d}\n{s}", .{ s.heads, s.kv_heads, s.dim, sources.glm_draft_attention });
}

pub const Pipes = struct {
    shape: Shape,
    attend: mtl.Pipeline,
    merge: mtl.Pipeline,

    /// `edit` replaces one text of the source (a contract control's planted fault), else null.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, s: Shape, edit: ?[2][]const u8) !Pipes {
        const plain = try source(a, s);
        defer a.free(plain);
        const text = if (edit) |x| blk: {
            if (std.mem.indexOf(u8, plain, x[0]) == null) return error.ControlEditMissing;
            break :blk try std.mem.replaceOwned(u8, a, plain, x[0], x[1]);
        } else try a.dupe(u8, plain);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        const attend = try mtl.Pipeline.init(device, lib, "glm_draft_attend", false);
        errdefer attend.deinit();
        const merge = try mtl.Pipeline.init(device, lib, "glm_draft_attend_merge", false);
        errdefer merge.deinit();
        if (attend.maxThreads() < threads or attend.simdWidth() != 32) return error.UnsupportedDraftAttention;
        return .{ .shape = s, .attend = attend, .merge = merge };
    }

    pub fn deinit(p: Pipes) void {
        p.attend.deinit();
        p.merge.deinit();
    }

    /// Each stream's rows of `q` over its ring's context and own `ko`/`vo` rows into `out`, whatever streams share it.
    pub fn encode(p: Pipes, e: mtl.ComputeEncoder, args: Args, table: []const Stream, rings: []const Ring, q: Ref, ko: Ref, vo: Ref, mask: Ref, part: Ref, out: Ref) !void {
        if (table.len != args.streams or rings.len != table.len or args.rows != args.streams * args.block) return error.InvalidDraftAttention;
        var t: [max_streams]Stream = @splat(.{ .keys = 0, .context = 0, .at = 0 });
        @memcpy(t[0..table.len], table);
        e.setPipeline(p.attend);
        for ([_]Ref{ q, ko, vo, mask }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(args, 4);
        e.setValue(t, 5);
        e.setBuffer(part.buf, part.off, 6);
        // unused ring slots repeat stream 0's: bound, never read
        for (0..max_streams) |s| {
            const r = rings[if (s < rings.len) s else 0];
            e.setBuffer(r.k.buf, r.k.off, 8 + s);
            e.setBuffer(r.v.buf, r.v.off, 16 + s);
        }
        e.dispatchGroups(mtl.Size.of(p.shape.kv_heads, args.rows, args.chunks), mtl.Size.of(threads, 1, 1));
        e.barrier();
        e.setPipeline(p.merge);
        e.setBuffer(part.buf, part.off, 0);
        e.setValue(args, 1);
        e.setBuffer(out.buf, out.off, 2);
        e.setValue(t, 3);
        e.dispatchGroups(mtl.Size.of(p.shape.heads, args.rows, 1), mtl.Size.of(32, 1, 1));
    }
};

test "a pass's rows, mask stride and chunks follow its longest stream; a table the kernel cannot read is refused" {
    const s = Shape{ .heads = 32, .kv_heads = 8, .dim = 128 };
    const table = [_]Stream{ .{ .keys = 2063, .context = 2047, .at = 5 }, .{ .keys = 316, .context = 300, .at = 0 }, .{ .keys = 16, .context = 0, .at = 0 } };
    const args = try plan(&table, 16, 2048, 0.088);
    try std.testing.expectEqual(@as(u32, 48), args.rows);
    try std.testing.expectEqual(@as(u32, 2063), args.mask_stride);
    try std.testing.expectEqual(@as(u32, 17), args.chunks);
    try std.testing.expectEqual(@as(usize, 48 * 32 * 17 * 132 * 4), partBytes(s, args));
    try std.testing.expectError(error.InvalidDraftAttention, plan(&.{.{ .keys = 20, .context = 2, .at = 0 }}, 16, 2048, 1));
    try std.testing.expectError(error.InvalidDraftAttention, plan(&table, 17, 2048, 1));
    try std.testing.expectEqual(@as(u32, 17), chunks(2047 + 16));
    try std.testing.expectEqual(@as(u32, 1), chunks(1));
    try std.testing.expectEqual(@as(u32, 2), chunks(129));
}
