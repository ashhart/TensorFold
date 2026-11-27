//! One DFlash frame for all selected streams, with separate rings, positions, convolution boundaries and trees.
const std = @import("std");
const cuda = @import("cuda");
const dr = @import("lanes").drafter;
const draft = @import("cuda_draft.zig");
const tree = @import("cuda_draft_tree.zig");
const layers = @import("cuda_layers.zig");
const sig = @import("cuda_signatures.zig");
pub const ql_span = @import("cuda_draft_ql_span.zig");
pub const max_streams = @import("cuda_engine.zig").max_segs;
pub const max_rows = max_streams * 16;
const lp_words = 43;
/// DFlash2's draft vocabulary as the Metal drafter and qwen27 engine read it: the head's first 98,304 rows, last 288.
const spans = [2][2]usize{ .{ 0, 98304 }, .{ 248032, 248320 } };
const span_width = spans[0][1] + spans[1][1] - spans[1][0];
/// The tail span's 64-row tile: projected from row 248,000, its first 32 columns dropped.
const tail_tile = spans[1][0] / 64 * 64;
pub const Segment = extern struct { ring: u64, begin: i64, position: i64 };
pub const segments_at = max_rows * 4;
pub const words_at = segments_at + max_streams * @sizeOf(Segment);
pub const hp_at = words_at + max_rows * lp_words * 4;
pub fn stagedBytes(rank: usize) !usize {
    return std.math.add(usize, hp_at, try std.math.mul(usize, try std.math.mul(usize, max_rows, rank), 2));
}
pub fn stagedLen(rank: usize) usize {
    return stagedBytes(rank) catch unreachable;
}
pub fn segments(st: []u8) []Segment {
    return @alignCast(std.mem.bytesAsSlice(Segment, st[segments_at..words_at]));
}
pub fn stage(c: draft.Config, items: []const dr.Hold, contexts: []const *draft.Ctx, st: []u8) !void {
    if (items.len == 0 or items.len > max_streams or contexts.len != items.len or (c.block != 8 and c.block != 16) or st.len < try stagedBytes(c.rank) or @intFromPtr(st.ptr) % 8 != 0) return error.BadDraftFrame;
    for (items, contexts, 0..) |it, ctx, index| {
        if (it.position == 0 or it.position > std.math.maxInt(i64) - c.block or ctx.end != it.position - 1 or ctx.begin > ctx.end) return error.DraftContextBehind;
        // a chain drafts a row a depth; a grown tree's depth is its node budget (tree.nodes(), up to the window)
        const most = if (c.block == 16 and it.lanes == null) @max(c.block - 1, try tree.nodes()) else c.block - 1;
        if (it.pending >= c.vocab or it.depth > most) return error.BadDraftFrame;
        for (contexts[0..index]) |prior| if (ctx == prior or ctx.ring.ptr == prior.ring.ptr) return error.OverlappingDraftContexts;
        if (it.lanes) |shape| {
            if (shape.parents.len > ctx.held.len or shape.parents.len != shape.depths.len or shape.parents.len != shape.ranks.len) return error.BadLanes;
            for (shape.parents, shape.depths, shape.ranks, 0..) |parent, depth, rank, row| {
                if (parent < -1 or parent >= @as(i32, @intCast(row)) or depth >= c.block - 1 or rank >= c.topk) return error.BadLanes;
            }
        }
    }
    const ids = std.mem.bytesAsSlice(u32, st[0 .. c.block * items.len * 4]);
    const sg = segments(st);
    for (items, contexts, 0..) |it, ctx, i| {
        ids[i * c.block] = it.pending;
        @memset(ids[i * c.block + 1 ..][0 .. c.block - 1], c.mask);
        sg[i] = .{ .ring = ctx.ring.ptr, .begin = @intCast(ctx.begin), .position = @intCast(ctx.end) };
    }
}
/// A launch whose arguments are checked against the kernel's .cu signature: a wrong count or kind does not build.
fn launch(d: anytype, f: cuda.Function, comptime s: sig.Sig, grid: [3]usize, block: [3]usize, args: anytype) !void {
    sig.check(s, @TypeOf(args));
    return d.e.run(f, grid, block, args);
}
fn conv(d: anytype, x: u64, base: u64, res: u64, out: u64, rows: usize, branch: c_int) !void {
    try d.e.run(d.fns.dyn_conv_batch, .{ rows, (d.c.hidden + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ x, d.dyn, base, res, out, @as(c_int, @intCast(d.c.hidden)), @as(c_int, @intCast(d.c.group)), branch, @as(c_int, @intCast(d.c.block)) });
}
pub fn rope(d: anytype, x: u64, w: u64, out: u64, rows: usize, heads: usize) !void {
    try d.e.run(d.fns.norm_rope_batch, .{ rows * heads, 1, 1 }, .{ 32, 1, 1 }, .{ x, w, out, @as(c_int, @intCast(heads)), @as(c_int, @intCast(heads * 128)), d.segments, @as(c_int, @intCast(d.c.block)), d.c.eps, d.c.theta });
}
pub fn attention(d: anytype, rows: usize, layer: usize) !void {
    const c = d.c;
    // causal: DSpine's block rows see the rows up to their own; DFlash2's see the whole block
    const partial = .{ d.q, d.kb, d.vb, d.segments, @as(c_int, @intCast(c.block)), @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(c.ringRows())), @as(c_int, @intCast(c.window)), @as(c_int, @intCast(layer)), @as(c_int, @intCast(c.chunks())), @as(c_int, @intFromBool(c.kind == .dspine)), d.pm, d.pl, d.po };
    try launch(d, d.fns.attn_partial, sig.draft_partial, .{ rows, c.heads, c.chunks() }, .{ 128, 1, 1 }, partial);
    const merge = .{ d.pm, d.pl, d.po, @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.chunks())), d.att };
    try launch(d, d.fns.attn_merge, sig.draft_merge, .{ rows, c.heads, 1 }, .{ 32, 1, 1 }, merge);
}
pub fn publish(d: anytype, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    defer d.selection_base = 0;
    for (items, contexts, 0..) |it, ctx, i| {
        d.selection_base = i * (d.c.block - 1);
        try d.selectHold(ctx, it);
    }
}
pub fn forward(d: anytype, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    const c = d.c;
    const e = d.e;
    const B = items.len * c.block;
    const st = d.staged.bytes;
    try stage(c, items, contexts, st);
    for (contexts) |ctx| {
        ctx.held_n = 0;
        ctx.grown = 0;
    }
    errdefer cuda.completion.drain(e.d, e.stream, "a failed DFlash frame did not drain before its pinned staging was reused");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.ids, st.ptr, B * 4, e.stream.handle), "cuMemcpyHtoDAsync");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.segments, st.ptr + segments_at, items.len * @sizeOf(Segment), e.stream.handle), "cuMemcpyHtoDAsync");
    const rows = items.len * (c.block - 1);
    const mixed = mixedHeadEligible(e.head, contexts);
    // The graph key does not include the mixed-width row mask; mixed batches must run eagerly.
    const spanned = if (mixed) try kernels(d, items.len, contexts) else try draft.graphs.hold(d, items.len, contexts, narrow(contexts), kernels);
    if (!mixed) try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + words_at, d.words, rows * lp_words * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + hp_at, d.hp, rows * c.rank * 2, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    if (spanned != null or mixed) for (contexts, 0..) |ctx, i| if (spanned != null or !ctx.wide) {
        for (0..c.block - 1) |r| {
            const row = i * (c.block - 1) + r;
            const w = std.mem.bytesAsSlice(u32, st[words_at + row * lp_words * 4 ..][0 .. lp_words * 4]);
            for (w[2..][0..c.topk]) |*id| id.* = columnId(id.*);
        }
    };
    try publish(d, items, contexts);
}

/// A hold frame's kernels, from the embedding to the selector rows; the packed logits when spanned.
fn kernels(d: anytype, streams: usize, contexts: []const *draft.Ctx) !?u64 {
    const c = d.c;
    const e = d.e;
    const B = streams * c.block;
    try e.embed(d.ids, d.h, B, 0);
    for (d.w.layers, 0..) |l, i| {
        try d.norm(d.h, l.input_norm, d.x, B);
        try d.projAgain(l.attn_conv, d.x, d.dyn, B);
        try conv(d, d.x, l.attn_base, 0, d.xin, B, 0);
        try d.projMany(&.{ l.q, l.k, l.v }, d.xin, &.{ d.qraw, d.kraw, d.vb }, B);
        try rope(d, d.qraw, l.q_norm, d.q, B, c.heads);
        try rope(d, d.kraw, l.k_norm, d.kb, B, c.kv_heads);
        try attention(d, B, i);
        try d.proj(l.o, d.att, d.o, B);
        try conv(d, d.o, l.attn_base, d.h, d.h2, B, 1);
        try d.norm(d.h2, l.post_norm, d.x, B);
        try d.projAgain(l.mlp_conv, d.x, d.dyn, B);
        try conv(d, d.x, l.mlp_base, 0, d.xin, B, 0);
        try d.projManySums(&.{ l.gate, l.up }, d.xin, &.{ d.gate, d.up }, B, true);
        try e.run(e.fns.mlp_act, .{ (B * c.intermediate + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ d.gate, d.up, d.act, @as(c_longlong, @intCast(B * c.intermediate)), @as(u64, 0), @as(c_int, @intCast(c.intermediate)), d.xs });
        try d.projAgain(l.down, d.act, d.o, B);
        try conv(d, d.o, l.mlp_base, d.h2, d.h, B, 1);
    }
    const rows = streams * (c.block - 1);
    var masked = d.h + c.hidden * 2;
    if (streams > 1) {
        try e.run(d.fns.mask_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.h, d.h2, @as(c_int, @intCast(c.hidden)), @as(c_int, @intCast(c.block)) });
        masked = d.h2;
    }
    try d.norm(masked, d.w.final_norm, d.x, rows);
    const mixed = mixedHeadEligible(e.head, contexts);
    const spanned = if (mixed) blk: {
        try mixedHead(d, contexts, d.staged.bytes);
        break :blk null;
    } else try logitsOf(d, contexts, d.x, d.logits, rows);
    if (!mixed) try e.draws.topkRows(e.stream, spanned orelse d.logits, if (spanned != null) span_width else c.vocab, @intCast(c.topk), d.parts, d.words, rows);
    try d.proj(d.w.selector, d.x, d.hp, rows);
    return spanned;
}

/// Mixed contexts use the same head-width class as an otherwise identical homogeneous frame.
pub fn mixedHeadEligible(h: layers.Proj, contexts: []const *draft.Ctx) bool {
    const supported = switch (h) {
        .a4 => |q| q.n == spans[1][1] and q.k % 64 == 0,
        .x3 => |q| q.n == spans[1][1] and q.k % 128 == 0,
        .an, .ql, .tfq3, .bf => false,
    };
    if (!supported) return false;
    var wide = false;
    var has_narrow = false;
    for (contexts) |ctx| if (ctx.wide) {
        wide = true;
    } else {
        has_narrow = true;
    };
    return wide and has_narrow;
}

/// Stream indices in their original order for one vocabulary-width class.
pub fn headGroup(contexts: []const *draft.Ctx, wide: bool, out: *[max_streams]usize) []const usize {
    var n: usize = 0;
    for (contexts, 0..) |ctx, i| if (ctx.wide == wide) {
        out[n] = i;
        n += 1;
    };
    return out[0..n];
}

/// Rebuild affine-4 group sums when a frame compacts rows; EXL3 projections do not consume `xs`.
pub fn rebuildProjectionSums(d: anytype, x: u64, rows: usize) !void {
    switch (d.e.head) {
        .a4 => {},
        else => return,
    }
    const k = d.c.hidden;
    try d.e.run(d.e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(k)), @as(u64, 0) });
}

/// Project and pick each width class separately, returning words to their original staging rows.
pub fn mixedHead(d: anytype, contexts: []const *draft.Ctx, st: []u8) !void {
    const e = d.e;
    const c = d.c;
    const per = c.block - 1;
    const input_bytes = per * c.hidden * 2;
    const word_bytes = per * lp_words * 4;
    for ([_]bool{ false, true }) |wide| {
        var indices: [max_streams]usize = undefined;
        const group = headGroup(contexts, wide, &indices);
        for (group, 0..) |i, at| try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(d.h2 + at * input_bytes, d.x + i * input_bytes, input_bytes, e.stream.handle), "cuMemcpyDtoDAsync");
        const rows = group.len * per;
        try rebuildProjectionSums(d, d.h2, rows);
        const spanned = if (wide) try full(d, d.h2, d.logits, rows) else try head(d, d.h2, d.logits, rows);
        try e.draws.topkRows(e.stream, spanned orelse d.logits, if (spanned != null) span_width else c.vocab, @intCast(c.topk), d.parts, d.words, rows);
        for (group, 0..) |i, at| try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + words_at + i * word_bytes, d.words + at * word_bytes, word_bytes, e.stream.handle), "cuMemcpyDtoHAsync");
    }
}

test "DFlash mixed EXL3 head projects both spans and restores reversed stream order" {
    const Trace = struct {
        const Copy = struct { dst: u64, src: u64, bytes: usize };
        const Projection = struct { n: usize, k: usize, rows: usize, x: u64, out: u64 };
        const Topk = struct { logits: u64, columns: usize, rows: usize, out: u64 };
        var device: [2]Copy = undefined;
        var device_count: usize = 0;
        var host: [2]Copy = undefined;
        var host_count: usize = 0;
        var projections: [3]Projection = undefined;
        var projection_count: usize = 0;
        var topks: [2]Topk = undefined;
        var topk_count: usize = 0;
        var copies2d: usize = 0;

        fn copyDevice(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            device[device_count] = .{ .dst = dst, .src = src, .bytes = bytes };
            device_count += 1;
            return cuda.abi.success;
        }
        fn copyHost(dst: ?*anyopaque, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            host[host_count] = .{ .dst = @intFromPtr(dst.?), .src = src, .bytes = bytes };
            host_count += 1;
            return cuda.abi.success;
        }
        fn copy2d(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            copies2d += 1;
            return cuda.abi.success;
        }
    };
    const Draws = struct {
        fn topkRows(_: *@This(), _: cuda.Stream, logits: u64, column_count: usize, _: u8, _: u64, out: u64, rows: usize) !void {
            Trace.topks[Trace.topk_count] = .{ .logits = logits, .columns = column_count, .rows = rows, .out = out };
            Trace.topk_count += 1;
        }
    };
    const E = struct {
        d: *cuda.Driver,
        stream: cuda.Stream,
        head: layers.Proj,
        affine: struct { pdl: bool },
        fns: struct { sums: cuda.Function },
        draws: Draws,

        pub fn run(_: *@This(), _: cuda.Function, _: [3]usize, _: [3]usize, _: anytype) !void {}

        pub fn projSums(_: *@This(), l: layers.Proj, x: u64, _: u64, out: u64, rows: usize, _: bool, _: u64) !void {
            const q = switch (l) {
                .x3 => |weight| weight,
                else => unreachable,
            };
            Trace.projections[Trace.projection_count] = .{ .n = q.n, .k = q.k, .rows = rows, .x = x, .out = out };
            Trace.projection_count += 1;
        }

        pub fn projManySums(_: *@This(), _: []const layers.Proj, _: u64, _: u64, _: []const u64, _: usize, _: bool, _: u64) !void {}
    };
    const D = struct {
        e: *E,
        c: struct { block: usize, hidden: usize, vocab: usize, topk: usize },
        x: u64,
        h2: u64,
        logits: u64,
        xs: u64,
        parts: u64,
        words: u64,
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoDAsync_v2 = Trace.copyDevice;
    driver.api.cuMemcpyDtoHAsync_v2 = Trace.copyHost;
    driver.api.cuMemcpy2DAsync_v2 = Trace.copy2d;
    const q = cuda.exl3;
    const st = q.strides(5120, 248320, 12, false);
    const weight: q.Weight = .{ .trellis = 0x8000, .suh = 0x9000, .svh = 0xa000, .k = 5120, .n = 248320, .halves = 12, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    var e = E{ .d = &driver, .stream = .{ .d = &driver, .handle = null }, .head = .{ .x3 = weight }, .affine = .{ .pdl = false }, .fns = .{ .sums = .{ .d = &driver, .handle = null } }, .draws = .{} };
    const d = D{ .e = &e, .c = .{ .block = 8, .hidden = 64, .vocab = 248320, .topk = 2 }, .x = 0x100000, .h2 = 0x200000, .logits = 0x300000, .xs = 0x400000, .parts = 0x500000, .words = 0x600000 };
    var contexts: [2]draft.Ctx = .{ .{ .ring = undefined, .wide = true }, .{ .ring = undefined } };
    const refs = [_]*draft.Ctx{ &contexts[0], &contexts[1] };
    var storage: [stagedLen(2)]u8 align(8) = undefined;
    Trace.device_count = 0;
    Trace.host_count = 0;
    Trace.projection_count = 0;
    Trace.topk_count = 0;
    Trace.copies2d = 0;
    try mixedHead(&d, &refs, &storage);

    const rows = d.c.block - 1;
    const tail_n = spans[1][1] - x3_tail_at;
    const span_logits = d.logits + rows * (spans[0][1] + tail_n) * 2;
    const input_bytes = rows * d.c.hidden * 2;
    const word_bytes = rows * 43 * 4;
    const views = try x3Views(weight);
    try std.testing.expectEqual(@as(usize, 2), Trace.device_count);
    try std.testing.expectEqual(@as(usize, 2), Trace.host_count);
    try std.testing.expectEqual(@as(usize, 3), Trace.projection_count);
    try std.testing.expectEqual(@as(usize, 2), Trace.topk_count);
    try std.testing.expectEqual(@as(usize, 2), Trace.copies2d);
    try std.testing.expectEqual(spans[0][1], Trace.projections[0].n);
    try std.testing.expectEqual(views.first.k, Trace.projections[0].k);
    try std.testing.expectEqual(views.tail.n, Trace.projections[1].n);
    try std.testing.expectEqual(views.tail.k, Trace.projections[1].k);
    try std.testing.expectEqual(weight.n, Trace.projections[2].n);
    try std.testing.expectEqual(d.logits, Trace.projections[0].out);
    try std.testing.expectEqual(d.logits + rows * spans[0][1] * 2, Trace.projections[1].out);
    try std.testing.expectEqual(d.logits, Trace.projections[2].out);
    for (Trace.projections) |projection| {
        try std.testing.expectEqual(rows, projection.rows);
        try std.testing.expectEqual(d.h2, projection.x);
    }
    try std.testing.expectEqual(span_logits, Trace.topks[0].logits);
    try std.testing.expectEqual(span_width, Trace.topks[0].columns);
    try std.testing.expectEqual(d.logits, Trace.topks[1].logits);
    try std.testing.expectEqual(d.c.vocab, Trace.topks[1].columns);
    for (Trace.topks) |topk| {
        try std.testing.expectEqual(rows, topk.rows);
        try std.testing.expectEqual(d.words, topk.out);
    }
    try std.testing.expectEqual(d.h2, Trace.device[0].dst);
    try std.testing.expectEqual(d.x + input_bytes, Trace.device[0].src);
    try std.testing.expectEqual(input_bytes, Trace.device[0].bytes);
    try std.testing.expectEqual(d.h2, Trace.device[1].dst);
    try std.testing.expectEqual(d.x, Trace.device[1].src);
    try std.testing.expectEqual(input_bytes, Trace.device[1].bytes);
    for ([_]usize{ 1, 0 }, 0..) |stream, group| {
        try std.testing.expectEqual(d.words, Trace.host[group].src);
        try std.testing.expectEqual(@intFromPtr(&storage) + words_at + stream * word_bytes, Trace.host[group].dst);
        try std.testing.expectEqual(word_bytes, Trace.host[group].bytes);
    }
}

/// A lane stream's prompt and reply tokens since the last scan: one outside the spans marks the stream wide.
pub fn scan(ctx: *draft.Ctx, toks: []const u32) void {
    if (!ctx.wide) for (toks[@min(ctx.scanned, toks.len)..]) |t| if (t >= spans[0][1] and t < spans[1][0]) {
        ctx.wide = true;
        break;
    };
    ctx.scanned = toks.len;
}

/// The rows' draft logits of x into `logits`: over the spans when every stream keeps to them (the packed rows).
pub fn logitsOf(d: anytype, contexts: []const *draft.Ctx, x: u64, logits: u64, rows: usize) !?u64 {
    return if (narrow(contexts)) try head(d, x, logits, rows) else try full(d, x, logits, rows);
}

/// A top-k column's token id: past the first span, the tail's.
pub fn columnId(column: u32) u32 {
    return if (column >= spans[0][1]) column + @as(u32, @intCast(spans[1][0] - spans[0][1])) else column;
}

/// The column split and id shift a kernel maps spanned columns with (0: the whole head, ids are columns).
pub fn split(spanned: bool) [2]c_int {
    return if (spanned) .{ @intCast(spans[0][1]), @intCast(spans[1][0] - spans[0][1]) } else .{ 0, 0 };
}

/// The spans' packed width, or the whole vocabulary.
pub fn columns(spanned: bool, vocab: usize) usize {
    return if (spanned) span_width else vocab;
}

/// Whether every stream's tokens so far keep to the spans (a warmup's or standalone hold's context reads none).
fn narrow(contexts: []const *draft.Ctx) bool {
    for (contexts) |ctx| if (ctx.wide) return false;
    return true;
}

/// An EXL3 head's tail view: the last wide_blocks blocks (from row 240,128), so it plans as the whole head does.
pub const x3_tail_at = spans[1][1] - cuda.exl3.wide_blocks * 128;

/// An EXL3 head's first span and the blocks holding its tail span, as views that keep the whole head's bits.
pub fn x3Views(q: cuda.exl3.Weight) !struct { first: cuda.exl3.Weight, tail: cuda.exl3.Weight } {
    return .{ .first = try q.columns(0, spans[0][1]), .tail = try q.columns(x3_tail_at, spans[1][1] - x3_tail_at) };
}

/// The rows' draft logits over the spans' columns, packed in d.logits after the two projections (null: not this head).
fn head(d: anytype, x: u64, logits: u64, rows: usize) !?u64 {
    const e = d.e;
    // the head's first span and the whole tiles holding its tail span, as views of the stored matrix
    const first: layers.Proj, const tail: layers.Proj, const tail_at: usize = switch (e.head) {
        .a4 => |q| blk: {
            if (q.n != spans[1][1] or q.k % 64 != 0) return full(d, x, logits, rows);
            break :blk .{
                .{ .a4 = .{ .w = q.w, .s = q.s, .b = q.b, .n = spans[0][1], .k = q.k, .npad = q.npad } },
                .{ .a4 = .{ .w = q.w + tail_tile * q.k / 2, .s = q.s + tail_tile * 2, .b = q.b + tail_tile * 2, .n = q.n - tail_tile, .k = q.k, .npad = q.npad } },
                tail_tile,
            };
        },
        .x3 => |q| blk: {
            if (q.n != spans[1][1] or q.k % 128 != 0) return full(d, x, logits, rows);
            const v = try x3Views(q);
            break :blk .{ .{ .x3 = v.first }, .{ .x3 = v.tail }, x3_tail_at };
        },
        .ql => |w| blk: {
            if (!ql_span.enabled()) return full(d, x, logits, rows);
            const v = ql_span.views(w, spans[0][1], tail_tile, spans[1][1]) orelse return full(d, x, logits, rows);
            break :blk .{ .{ .ql = v.first }, .{ .ql = v.tail }, tail_tile };
        },
        .tfq3 => |q| blk: {
            if (q.n != spans[1][1] or q.k % 128 != 0) return full(d, x, logits, rows);
            const tfq3_tail_at: u32 = @intCast(spans[1][0] / 128 * 128);
            const first_w = q.sliceRows(0, @intCast(spans[0][1])) catch return full(d, x, logits, rows);
            const tail_w = q.sliceRows(tfq3_tail_at, @intCast(spans[1][1] - tfq3_tail_at)) catch return full(d, x, logits, rows);
            break :blk .{ .{ .tfq3 = first_w }, .{ .tfq3 = tail_w }, tfq3_tail_at };
        },
        .an, .bf => return full(d, x, logits, rows),
    };
    const tail_n = spans[1][1] - tail_at;
    const a = logits; // rows x 98,304, then rows x the tail's columns, then the packed rows x 98,592
    const b = a + rows * spans[0][1] * 2;
    const out = b + rows * tail_n * 2;
    if (e.affine.pdl and first == .a4 and tail == .a4) {
        try e.projManySums(&.{ first, tail }, x, d.xs, &.{ a, b }, rows, false, 0);
    } else {
        try draft.precision.project(e, first, x, d.xs, a, rows, .proposal, draft.precision.dflash_projection_prompt_cutoff);
        try draft.precision.project(e, tail, x, d.xs, b, rows, .proposal, draft.precision.dflash_projection_prompt_cutoff);
    }
    const skip = spans[1][0] - tail_at;
    const copies = [2]cuda.abi.Memcpy2D{
        .{ .src = a, .src_pitch = spans[0][1] * 2, .dst = out, .dst_pitch = span_width * 2, .width = spans[0][1] * 2, .height = rows },
        .{ .src = b + skip * 2, .src_pitch = tail_n * 2, .dst = out + spans[0][1] * 2, .dst_pitch = span_width * 2, .width = (tail_n - skip) * 2, .height = rows },
    };
    for (&copies) |*m| try e.d.check(e.d.api.cuMemcpy2DAsync_v2(m, e.stream.handle), "cuMemcpy2DAsync");
    return out;
}

/// The rows' draft logits over the whole vocabulary, in `logits`.
fn full(d: anytype, x: u64, logits: u64, rows: usize) !?u64 {
    try draft.precision.project(d.e, d.e.head, x, d.xs, logits, rows, .proposal, draft.precision.dflash_projection_prompt_cutoff);
    return null;
}
