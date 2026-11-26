//! DSpine's draft frame: block-causal layers each closed by the code bridge, then the exact-sender table per row.
const std = @import("std");
const cuda = @import("cuda");
const dr = @import("lanes").drafter;
const draft = @import("cuda_draft.zig");
const frame = @import("cuda_draft_frame.zig");
const layer_mod = @import("cuda_layers.zig");
const bridge = @import("cuda_dspine_weights.zig");
const Config = @import("cuda_draft_config.zig").Config;

const lp_words = 43;
pub const Fns = struct { bridge_read: cuda.Function, bridge_code: cuda.Function, bridge_message: cuda.Function, bridge_write: cuda.Function, senders: cuda.Function, sender_message: cuda.Function, table: cuda.Function, transpose: cuda.Function };

/// The bridge's weights and kernels beside the drafter's.
pub const Spine = struct {
    fns: Fns,
    w: bridge.Bridge,
};

pub fn init(gpa: std.mem.Allocator, io: std.Io, e: anytype, dir: []const u8, c: Config, mod: cuda.Module) !*Spine {
    const s = try gpa.create(Spine);
    errdefer gpa.destroy(s);
    inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(s.fns, name) = try mod.function("tf_q35d_" ++ name);
    s.w = try bridge.load(gpa, io, e, dir, c, s.fns.transpose);
    return s;
}

pub fn deinit(gpa: std.mem.Allocator, s: *Spine) void {
    s.w.deinit();
    gpa.destroy(s);
}

/// A frame's bridge rows: rms (f32), read codes, codes, gates, sender terms, messages, writes, head rows and logits.
const Bufs = struct { rms: u64, p: u64, code: u64, q: u64, sv: u64, msg: u64, wm: u64, hx: u64, logits: u64, sids: u64, table: u64, end: u64 };

fn bufs(c: Config, streams: usize, base: u64) Bufs {
    const B = c.block * streams;
    const rows = (c.block - 1) * streams;
    const P = rows * c.topk;
    const W = @max(B, P);
    var at = base;
    const sizes = [_]usize{ B * 4, B * c.code * 2, W * c.code * 2, B * c.rank * 2, W * c.rank * 4, W * c.rank * 2, P * c.hidden * 2, 2 * rows * c.hidden * 2, 2 * rows * c.vocab * 2, P * 4, P * c.topk * 4 };
    var out: [sizes.len + 1]u64 = undefined;
    for (sizes, 0..) |n, i| {
        out[i] = at;
        at = std.mem.alignForward(u64, at + n, 256);
    }
    out[sizes.len] = at;
    return .{ .rms = out[0], .p = out[1], .code = out[2], .q = out[3], .sv = out[4], .msg = out[5], .wm = out[6], .hx = out[7], .logits = out[8], .sids = out[9], .table = out[10], .end = out[11] };
}

/// Scratch bytes of a frame of `streams` (none for DFlash2).
pub fn bytes(c: Config, streams: usize) usize {
    return if (c.kind == .dspine) bufs(c, streams, 0).end else 0;
}

fn int(x: usize) c_int {
    return @intCast(x);
}

/// The block's layers: attention as DFlash2's (causal here), a plain residual MLP, and the bridge after each.
fn layers(d: anytype, s: *Spine, b: Bufs, B: usize, rows: usize) !void {
    const c = d.c;
    const e = d.e;
    const w = &s.w;
    const sqrt_c: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(c.code))));
    try e.embed(d.ids, d.h, B, 0);
    try d.norm(d.h, d.w.layers[0].input_norm, d.x, B);
    for (d.w.layers, 0..) |l, i| {
        try d.projMany(&.{ l.q, l.k, l.v }, d.x, &.{ d.qraw, d.kraw, d.vb }, B);
        try frame.rope(d, d.qraw, l.q_norm, d.q, B, c.heads);
        try frame.rope(d, d.kraw, l.k_norm, d.kb, B, c.kv_heads);
        try frame.attention(d, B, i);
        try d.proj(l.o, d.att, d.o, B);
        try e.run(e.fns.norm, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ d.o, d.h, l.post_norm, d.x, c.eps, int(c.hidden), @as(u64, 0), d.xs });
        try d.projManySums(&.{ l.gate, l.up }, d.x, &.{ d.gate, d.up }, B, false);
        try e.run(e.fns.mlp_act, .{ (B * c.intermediate + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ d.gate, d.up, d.act, @as(c_longlong, @intCast(B * c.intermediate)), @as(u64, 0), int(c.intermediate), d.xs });
        try d.projAgain(l.down, d.act, d.h2, B);
        try e.run(s.fns.bridge_read, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ d.h2, d.o, d.xin, d.x, b.rms, int(c.hidden), c.bridge_eps });
        try d.proj(w.read[i], d.x, b.p, B);
        try e.run(s.fns.bridge_code, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ b.p, w.codebook, d.ids, b.code, int(c.code), int(c.block), sqrt_c, c.bridge_eps });
        try d.projMany(&.{ w.gate, w.pv }, b.code, &.{ b.q, b.sv }, B);
        try e.run(s.fns.bridge_message, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ b.q, b.sv, w.gate_b, b.msg, int(c.rank), int(c.block) });
        try d.proj(w.write[i], b.msg, d.o, B);
        const last = i + 1 == c.layers;
        const next = if (last) d.w.final_norm else d.w.layers[i + 1].input_norm;
        const xn = if (last) b.hx else d.x;
        try e.run(s.fns.bridge_write, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ d.xin, d.o, b.rms, next, d.h, xn, if (last) b.hx else @as(u64, 0), int(c.hidden), int(c.block), int(rows), c.eps });
    }
}

/// The second pass: each head row's candidates scored after each of the row before's (its anchor for the first).
const TableRows = struct { ids: u64, raw: u64, rms: u64, q: u64, words: u64, out: u64 };

fn table(d: anytype, s: *Spine, b: Bufs, rows: usize, logits: u64, spanned: bool, input: TableRows, logits_offset: usize) !void {
    const c = d.c;
    const e = d.e;
    const w = &s.w;
    const P = rows * c.topk;
    const sp = frame.split(spanned);
    const sqrt_c: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(c.code))));
    try e.run(s.fns.senders, .{ rows, 1, 1 }, .{ 32, 1, 1 }, .{ input.words, input.ids, b.sids, int(c.topk), int(c.block), int(lp_words), sp[0], sp[1] });
    try e.run(s.fns.bridge_code, .{ P, 1, 1 }, .{ 256, 1, 1 }, .{ @as(u64, 0), w.codebook, b.sids, b.code, int(c.code), int(1), sqrt_c, c.bridge_eps });
    try d.proj(w.pv, b.code, b.sv, P);
    try e.run(s.fns.sender_message, .{ P, 1, 1 }, .{ 256, 1, 1 }, .{ input.q, b.sv, w.gate_b, b.msg, int(c.rank), int(c.topk), int(c.block) });
    try d.proj(w.write[c.layers - 1], b.msg, b.wm, P);
    const cols = frame.columns(spanned, c.vocab);
    try e.run(s.fns.table, .{ P, 1, 1 }, .{ 256, 1, 1 }, .{ input.raw, input.rms, b.msg, b.wm, w.tg, logits, input.words, input.out, int(c.hidden), int(c.rank), int(c.topk), int(c.block), int(logits_offset), int(cols), int(lp_words), sp[0], sp[1], c.eps });
}

const MixedScratch = struct { ids: u64, raw: u64, rms: u64, q: u64, words: u64, table: u64 };

fn scratchAt(at: *usize, size: usize) !usize {
    const start = (try std.math.add(usize, at.*, 255)) & ~@as(usize, 255);
    at.* = try std.math.add(usize, start, size);
    return start;
}

fn mixedScratch(b: Bufs, c: Config, streams: usize) !MixedScratch {
    const rows = try std.math.mul(usize, streams, c.block - 1);
    const blocks = try std.math.mul(usize, streams, c.block);
    var at: usize = 0;
    const ids = try scratchAt(&at, try std.math.mul(usize, blocks, 4));
    const raw = try scratchAt(&at, try std.math.mul(usize, try std.math.mul(usize, blocks, c.hidden), 2));
    const rms = try scratchAt(&at, try std.math.mul(usize, blocks, 4));
    const q = try scratchAt(&at, try std.math.mul(usize, try std.math.mul(usize, blocks, c.rank), 2));
    const words = try scratchAt(&at, try std.math.mul(usize, try std.math.mul(usize, rows, lp_words), 4));
    const table_bytes = try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, rows, c.topk), c.topk), 4);
    const table_offset = try scratchAt(&at, table_bytes);
    const capacity = try std.math.mul(usize, try std.math.mul(usize, rows, c.vocab), 4);
    if (at > capacity or b.end - b.logits < capacity) return error.BadDraftFrame;
    return .{ .ids = b.logits + ids, .raw = b.logits + raw, .rms = b.logits + rms, .q = b.logits + q, .words = b.logits + words, .table = b.logits + table_offset };
}

fn groupLogits(d: anytype, contexts: []const *draft.Ctx, group: []const usize, source: u64) !?u64 {
    const c = d.c;
    const per = c.block - 1;
    const row_bytes = c.hidden * 2;
    const stream_bytes = per * row_bytes;
    var refs: [frame.max_streams]*draft.Ctx = undefined;
    for (group, 0..) |stream, at| {
        refs[at] = contexts[stream];
        try d.e.d.check(d.e.d.api.cuMemcpyDtoDAsync_v2(d.h2 + at * stream_bytes, source + stream * stream_bytes, stream_bytes, d.e.stream.handle), "cuMemcpyDtoDAsync");
    }
    try frame.rebuildProjectionSums(d, d.h2, group.len * per);
    return frame.logitsOf(d, refs[0..group.len], d.h2, d.logits, group.len * per);
}

fn packTableRows(d: anytype, b: Bufs, scratch: MixedScratch, group: []const usize) !TableRows {
    const c = d.c;
    const id_bytes = c.block * 4;
    const raw_bytes = c.block * c.hidden * 2;
    const rms_bytes = c.block * 4;
    const q_bytes = c.block * c.rank * 2;
    for (group, 0..) |stream, at| {
        try d.e.d.check(d.e.d.api.cuMemcpyDtoDAsync_v2(scratch.ids + at * id_bytes, d.ids + stream * id_bytes, id_bytes, d.e.stream.handle), "cuMemcpyDtoDAsync");
        try d.e.d.check(d.e.d.api.cuMemcpyDtoDAsync_v2(scratch.raw + at * raw_bytes, d.xin + stream * raw_bytes, raw_bytes, d.e.stream.handle), "cuMemcpyDtoDAsync");
        try d.e.d.check(d.e.d.api.cuMemcpyDtoDAsync_v2(scratch.rms + at * rms_bytes, b.rms + stream * rms_bytes, rms_bytes, d.e.stream.handle), "cuMemcpyDtoDAsync");
        try d.e.d.check(d.e.d.api.cuMemcpyDtoDAsync_v2(scratch.q + at * q_bytes, b.q + stream * q_bytes, q_bytes, d.e.stream.handle), "cuMemcpyDtoDAsync");
    }
    return .{ .ids = scratch.ids, .raw = scratch.raw, .rms = scratch.rms, .q = scratch.q, .words = d.words, .out = b.table };
}

/// Run the head and second pass by width class, then restore every row to its original stream slot.
fn mixedHead(d: anytype, s: *Spine, b: Bufs, contexts: []const *draft.Ctx, rows: usize) !void {
    const c = d.c;
    const e = d.e;
    const per = c.block - 1;
    const scratch = try mixedScratch(b, c, contexts.len);
    var group_start: usize = 0;
    for ([_]bool{ false, true }) |wide| {
        var indices: [frame.max_streams]usize = undefined;
        const group = frame.headGroup(contexts, wide, &indices);
        if (group.len == 0) continue;
        const group_rows = group.len * per;
        const first = try groupLogits(d, contexts, group, b.hx);
        try e.draws.topkRows(e.stream, first orelse d.logits, frame.columns(first != null, c.vocab), @intCast(c.topk), d.parts, d.words, group_rows);
        const input = try packTableRows(d, b, scratch, group);
        const second = try groupLogits(d, contexts, group, b.hx + rows * c.hidden * 2);
        // Each group's second-plane logits start at row zero in the compact projection buffer.
        try table(d, s, b, group_rows, second orelse d.logits, second != null, input, 0);
        const row_at = group_start * per;
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(scratch.words + row_at * lp_words * 4, d.words, group_rows * lp_words * 4, e.stream.handle), "cuMemcpyDtoDAsync");
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(scratch.table + row_at * c.topk * c.topk * 4, b.table, group_rows * c.topk * c.topk * 4, e.stream.handle), "cuMemcpyDtoDAsync");
        group_start += group.len;
    }
    group_start = 0;
    for ([_]bool{ false, true }) |wide| {
        var indices: [frame.max_streams]usize = undefined;
        const group = frame.headGroup(contexts, wide, &indices);
        for (group, 0..) |stream, at| {
            const source_row = (group_start + at) * per;
            const target_row = stream * per;
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(d.words + target_row * lp_words * 4, scratch.words + source_row * lp_words * 4, per * lp_words * 4, e.stream.handle), "cuMemcpyDtoDAsync");
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(b.table + target_row * c.topk * c.topk * 4, scratch.table + source_row * c.topk * c.topk * 4, per * c.topk * c.topk * 4, e.stream.handle), "cuMemcpyDtoDAsync");
        }
        group_start += group.len;
    }
}

test "DSpine mixed width: both head passes, table groups, and scatter keep stream rows" {
    const Trace = struct {
        const Copy = struct { dst: u64, src: u64, bytes: usize };
        const Projection = struct { columns: usize, rows: usize, x: u64, out: u64 };
        const Sum = struct { x: u64, xs: u64, k: c_int, rows: usize, grid: [3]usize };
        const Event = enum { compact, sums, projection };
        const Topk = struct { logits: u64, columns: usize, rows: usize, out: u64 };
        const Table = struct { logits: u64, words: u64, out: u64, grid: usize, offset: c_int, columns: c_int };
        var copies: [48]Copy = undefined;
        var copy_count: usize = 0;
        var projections: [8]Projection = undefined;
        var projection_count: usize = 0;
        var many_projection_calls: usize = 0;
        var sums: [4]Sum = undefined;
        var sum_count: usize = 0;
        var events: [32]Event = undefined;
        var event_count: usize = 0;
        var compact_base: u64 = 0;
        var compact_stream_bytes: usize = 0;
        var topks: [4]Topk = undefined;
        var topk_count: usize = 0;
        var tables: [4]Table = undefined;
        var table_count: usize = 0;
        fn copy(dst: u64, src: u64, copy_bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            copies[copy_count] = .{ .dst = dst, .src = src, .bytes = copy_bytes };
            copy_count += 1;
            if (dst >= compact_base and dst < compact_base + 2 * compact_stream_bytes and copy_bytes == compact_stream_bytes) record(.compact);
            return cuda.abi.success;
        }
        fn copy2d(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn record(event: Event) void {
            events[event_count] = event;
            event_count += 1;
        }
        fn projection(l: layer_mod.Proj, x: u64, out: u64, rows: usize) void {
            const columns = switch (l) {
                .a4 => |q| q.n,
                .x3 => |q| q.n,
                else => unreachable,
            };
            projections[projection_count] = .{ .columns = columns, .rows = rows, .x = x, .out = out };
            projection_count += 1;
            record(.projection);
        }
    };
    const Draws = struct {
        fn topkRows(_: *@This(), _: cuda.Stream, logits: u64, columns: usize, _: u8, _: u64, out: u64, rows: usize) !void {
            Trace.topks[Trace.topk_count] = .{ .logits = logits, .columns = columns, .rows = rows, .out = out };
            Trace.topk_count += 1;
        }
    };
    const E = struct {
        d: *cuda.Driver,
        stream: cuda.Stream,
        head: layer_mod.Proj,
        affine: struct { pdl: bool },
        fns: struct { sums: cuda.Function },
        draws: Draws,
        pub fn projSums(_: *@This(), l: layer_mod.Proj, x: u64, _: u64, out: u64, rows: usize, _: bool, _: u64) !void {
            Trace.projection(l, x, out, rows);
        }
        pub fn projManySums(_: *@This(), ls: []const layer_mod.Proj, x: u64, _: u64, outs: []const u64, rows: usize, _: bool, _: u64) !void {
            Trace.many_projection_calls += 1;
            for (ls, outs) |l, out| Trace.projection(l, x, out, rows);
        }
        pub fn run(_: *@This(), f: cuda.Function, grid: [3]usize, _: [3]usize, args: anytype) !void {
            if (f.handle) |handle| if (@intFromPtr(handle) == 9) {
                Trace.sums[Trace.sum_count] = .{ .x = args[0], .xs = args[1], .k = @intCast(args[2]), .rows = grid[0], .grid = grid };
                Trace.sum_count += 1;
                Trace.record(.sums);
                return;
            };
            if (comptime @typeInfo(@TypeOf(args)).@"struct".field_names.len >= 14) {
                if (f.handle) |handle| if (@intFromPtr(handle) == 7) {
                    Trace.tables[Trace.table_count] = .{
                        .logits = args[5],
                        .words = args[6],
                        .out = args[7],
                        .grid = grid[0],
                        .offset = args[12],
                        .columns = args[13],
                    };
                    Trace.table_count += 1;
                };
            }
        }
    };
    const D = struct {
        e: *E,
        c: Config,
        h2: u64,
        logits: u64,
        xs: u64,
        parts: u64,
        words: u64,
        ids: u64,
        xin: u64,
        fn proj(_: *const @This(), _: cuda.qlinear.Affine4, _: u64, _: u64, _: usize) !void {}
    };
    const c: Config = .{
        .hidden = 128,
        .intermediate = 128,
        .layers = 1,
        .vocab = 248320,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 128,
        .block = 16,
        .mask = 0,
        .window = 16,
        .group = 16,
        .rank = 8,
        .topk = 2,
        .eps = 1e-6,
        .theta = 1,
        .taps = @splat(0),
        .n_taps = 1,
        .kind = .dspine,
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoDAsync_v2 = Trace.copy;
    driver.api.cuMemcpy2DAsync_v2 = Trace.copy2d;
    var e = E{ .d = &driver, .stream = .{ .d = &driver, .handle = null }, .head = undefined, .affine = .{ .pdl = true }, .fns = .{ .sums = .{ .d = &driver, .handle = @ptrFromInt(9) } }, .draws = .{} };
    const head: cuda.qlinear.Affine4 = .{ .w = 1, .s = 2, .b = 3, .n = c.vocab, .k = c.hidden, .npad = c.vocab };
    e.head = .{ .a4 = head };
    const function = struct {
        fn get(d: *cuda.Driver, id: usize) cuda.Function {
            return .{ .d = d, .handle = @ptrFromInt(id) };
        }
    }.get;
    var s: Spine = undefined;
    s.fns = .{
        .bridge_read = function(&driver, 1),
        .bridge_code = function(&driver, 2),
        .bridge_message = function(&driver, 3),
        .bridge_write = function(&driver, 4),
        .senders = function(&driver, 5),
        .sender_message = function(&driver, 6),
        .table = function(&driver, 7),
        .transpose = function(&driver, 8),
    };
    s.w.pv = head;
    s.w.write[0] = head;
    s.w.codebook = 0x900000;
    s.w.gate_b = 0xa00000;
    s.w.tg = 0xb00000;
    const rows = 4 * (c.block - 1);
    const capacity = rows * c.vocab * 4;
    // Draft.layout allocates d.logits before its DSpine `ds` region, where b.logits lives.
    const projection_base: u64 = 0x1000000;
    const projection_capacity = 4 * c.block * c.vocab * 2;
    const base: u64 = 0x3000000;
    const b: Bufs = .{
        .rms = 0x200000,
        .p = 0x210000,
        .code = 0x220000,
        .q = 0x230000,
        .sv = 0x240000,
        .msg = 0x250000,
        .wm = 0x260000,
        .hx = 0x270000,
        .logits = base,
        .sids = base + capacity,
        .table = base + capacity + 0x10000,
        .end = base + capacity + 0x20000,
    };
    const d = D{ .e = &e, .c = c, .h2 = 0x300000, .logits = projection_base, .xs = 0x310000, .parts = 0x320000, .words = 0x330000, .ids = 0x340000, .xin = 0x350000 };
    Trace.compact_base = d.h2;
    Trace.compact_stream_bytes = (c.block - 1) * c.hidden * 2;
    const scratch = try mixedScratch(b, c, 4);
    const projection_end = d.logits + projection_capacity;
    const scratch_end = b.logits + capacity;
    // `d.logits` is the earlier Draft.layout allocation; mixed scratch occupies `b.logits` in `ds`.
    try std.testing.expect(projection_end <= b.logits);
    const scratch_sizes = [_]usize{
        4 * c.block * 4,
        4 * c.block * c.hidden * 2,
        4 * c.block * 4,
        4 * c.block * c.rank * 2,
        rows * lp_words * 4,
        rows * c.topk * c.topk * 4,
    };
    const scratch_starts = [_]u64{ scratch.ids, scratch.raw, scratch.rms, scratch.q, scratch.words, scratch.table };
    for (scratch_starts, scratch_sizes) |start, size| {
        try std.testing.expect(start >= b.logits);
        try std.testing.expect(start + size <= scratch_end);
    }
    var contexts: [4]draft.Ctx = undefined;
    var refs: [4]*draft.Ctx = undefined;
    for (&contexts, &refs, 0..) |*ctx, *ref, i| {
        ctx.* = .{ .ring = undefined, .wide = i % 2 == 1 };
        ref.* = ctx;
    }
    Trace.copy_count = 0;
    Trace.projection_count = 0;
    Trace.many_projection_calls = 0;
    Trace.sum_count = 0;
    Trace.event_count = 0;
    Trace.topk_count = 0;
    Trace.table_count = 0;
    try mixedHead(&d, &s, b, &refs, rows);
    const per = c.block - 1;
    const group_rows = 2 * per;
    const stream_bytes = per * c.hidden * 2;
    const narrow_logits = d.logits + group_rows * (98304 + 320) * 2;
    try std.testing.expectEqual(@as(usize, 36), Trace.copy_count);
    const input_groups = [_]struct { start: usize, streams: [2]usize }{
        .{ .start = 2, .streams = .{ 0, 2 } },
        .{ .start = 16, .streams = .{ 1, 3 } },
    };
    const input_sources = [_]u64{ d.ids, d.xin, b.rms, b.q };
    const input_targets = [_]u64{ scratch.ids, scratch.raw, scratch.rms, scratch.q };
    const input_sizes = [_]usize{ c.block * 4, c.block * c.hidden * 2, c.block * 4, c.block * c.rank * 2 };
    const compact_groups = [_]struct { start: usize, source: u64, streams: [2]usize }{
        .{ .start = 0, .source = b.hx, .streams = .{ 0, 2 } },
        .{ .start = 10, .source = b.hx + rows * c.hidden * 2, .streams = .{ 0, 2 } },
        .{ .start = 14, .source = b.hx, .streams = .{ 1, 3 } },
        .{ .start = 24, .source = b.hx + rows * c.hidden * 2, .streams = .{ 1, 3 } },
    };
    for (compact_groups) |group| {
        for (group.streams, 0..) |stream, at| {
            const copy = Trace.copies[group.start + at];
            try std.testing.expectEqual(d.h2 + at * stream_bytes, copy.dst);
            try std.testing.expectEqual(group.source + stream * stream_bytes, copy.src);
            try std.testing.expectEqual(stream_bytes, copy.bytes);
        }
    }
    for (input_groups) |group| {
        for (group.streams, 0..) |stream, at| {
            for (input_sources, input_targets, input_sizes, 0..) |src, dst, copy_len, plane| {
                const copy = Trace.copies[group.start + at * 4 + plane];
                try std.testing.expectEqual(dst + at * copy_len, copy.dst);
                try std.testing.expectEqual(src + stream * copy_len, copy.src);
                try std.testing.expectEqual(copy_len, copy.bytes);
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 6), Trace.projection_count);
    try std.testing.expectEqual(@as(usize, 2), Trace.many_projection_calls);
    try std.testing.expectEqual(@as(usize, 4), Trace.sum_count);
    for (Trace.sums) |sum| {
        try std.testing.expectEqual(d.h2, sum.x);
        try std.testing.expectEqual(d.xs, sum.xs);
        try std.testing.expectEqual(@as(c_int, @intCast(c.hidden)), sum.k);
        try std.testing.expectEqual(group_rows, sum.rows);
        try std.testing.expectEqual([_]usize{ group_rows, 1, 1 }, sum.grid);
    }
    const expected_events = [_]Trace.Event{
        .compact, .compact, .sums,       .projection, .projection,
        .compact, .compact, .sums,       .projection, .projection,
        .compact, .compact, .sums,       .projection, .compact,
        .compact, .sums,    .projection,
    };
    try std.testing.expectEqual(expected_events.len, Trace.event_count);
    try std.testing.expectEqualSlices(Trace.Event, &expected_events, Trace.events[0..Trace.event_count]);
    try std.testing.expectEqual(@as(usize, 2), Trace.topk_count);
    try std.testing.expectEqual(@as(usize, 2), Trace.table_count);
    for (Trace.projections[0..Trace.projection_count], [_]usize{ 98304, 320, 98304, 320, c.vocab, c.vocab }) |p, columns| {
        try std.testing.expectEqual(columns, p.columns);
        try std.testing.expectEqual(group_rows, p.rows);
        try std.testing.expectEqual(d.h2, p.x);
        const bytes_written = p.rows * p.columns * 2;
        try std.testing.expect(p.out >= d.logits);
        try std.testing.expect(p.out + bytes_written <= projection_end);
        try std.testing.expect(p.out + bytes_written <= b.logits);
    }
    try std.testing.expectEqual(d.logits, Trace.projections[0].out);
    try std.testing.expectEqual(d.logits + group_rows * 98304 * 2, Trace.projections[1].out);
    for (Trace.topks[0..Trace.topk_count], [_]usize{ frame.columns(true, c.vocab), c.vocab }) |topk, columns| {
        try std.testing.expectEqual(columns, topk.columns);
        try std.testing.expectEqual(group_rows, topk.rows);
        try std.testing.expectEqual(d.words, topk.out);
        const bytes_read = topk.rows * topk.columns * 2;
        try std.testing.expect(topk.logits >= d.logits);
        try std.testing.expect(topk.logits + bytes_read <= projection_end);
        try std.testing.expect(topk.logits + bytes_read <= b.logits);
    }
    try std.testing.expectEqual(narrow_logits, Trace.topks[0].logits);
    try std.testing.expectEqual(d.logits, Trace.topks[1].logits);
    for (Trace.tables[0..Trace.table_count], [_]usize{ frame.columns(true, c.vocab), c.vocab }) |call, columns| {
        try std.testing.expectEqual(group_rows * c.topk, call.grid);
        try std.testing.expectEqual(d.words, call.words);
        try std.testing.expectEqual(b.table, call.out);
        try std.testing.expectEqual(@as(c_int, 0), call.offset);
        try std.testing.expectEqual(@as(c_int, @intCast(columns)), call.columns);
    }
    try std.testing.expectEqual(narrow_logits, Trace.tables[0].logits);
    try std.testing.expectEqual(d.logits, Trace.tables[1].logits);
    for ([_]usize{ 0, 2 }, 0..) |stream, at| {
        for ([_]usize{ 0, 10 }, 0..) |copy_base, plane| {
            const copy_at = copy_base + at;
            const source = b.hx + plane * rows * c.hidden * 2 + stream * stream_bytes;
            try std.testing.expectEqual(d.h2 + at * stream_bytes, Trace.copies[copy_at].dst);
            try std.testing.expectEqual(source, Trace.copies[copy_at].src);
            try std.testing.expectEqual(stream_bytes, Trace.copies[copy_at].bytes);
        }
    }
    for ([_]usize{ 1, 3 }, 0..) |stream, at| {
        for ([_]usize{ 14, 24 }, 0..) |copy_base, plane| {
            const copy_at = copy_base + at;
            const source = b.hx + plane * rows * c.hidden * 2 + stream * stream_bytes;
            try std.testing.expectEqual(d.h2 + at * stream_bytes, Trace.copies[copy_at].dst);
            try std.testing.expectEqual(source, Trace.copies[copy_at].src);
            try std.testing.expectEqual(stream_bytes, Trace.copies[copy_at].bytes);
        }
    }
    try std.testing.expectEqual(scratch.words, Trace.copies[12].dst);
    try std.testing.expectEqual(d.words, Trace.copies[12].src);
    try std.testing.expectEqual(group_rows * lp_words * 4, Trace.copies[12].bytes);
    try std.testing.expectEqual(scratch.table, Trace.copies[13].dst);
    try std.testing.expectEqual(b.table, Trace.copies[13].src);
    try std.testing.expectEqual(group_rows * c.topk * c.topk * 4, Trace.copies[13].bytes);
    const wide_row_at = group_rows;
    try std.testing.expectEqual(scratch.words + wide_row_at * lp_words * 4, Trace.copies[26].dst);
    try std.testing.expectEqual(d.words, Trace.copies[26].src);
    try std.testing.expectEqual(scratch.table + wide_row_at * c.topk * c.topk * 4, Trace.copies[27].dst);
    try std.testing.expectEqual(b.table, Trace.copies[27].src);
    for ([_]usize{ 0, 2, 1, 3 }, 0..) |stream, at| {
        const scatter_at = 28 + at * 2;
        try std.testing.expectEqual(d.words + stream * per * lp_words * 4, Trace.copies[scatter_at].dst);
        try std.testing.expectEqual(scratch.words + at * per * lp_words * 4, Trace.copies[scatter_at].src);
        try std.testing.expectEqual(per * lp_words * 4, Trace.copies[scatter_at].bytes);
        try std.testing.expectEqual(b.table + stream * per * c.topk * c.topk * 4, Trace.copies[scatter_at + 1].dst);
        try std.testing.expectEqual(scratch.table + at * per * c.topk * c.topk * 4, Trace.copies[scatter_at + 1].src);
        try std.testing.expectEqual(per * c.topk * c.topk * 4, Trace.copies[scatter_at + 1].bytes);
    }
    // EXL3 stages its tail through the same d.logits allocation; its ranges must stay clear of `ds` scratch too.
    const x3 = cuda.exl3;
    const x3_strides = x3.strides(c.hidden, c.vocab, 12, false);
    const x3_weight: x3.Weight = .{ .trellis = 0xc00000, .suh = 0xd00000, .svh = 0xe00000, .k = c.hidden, .n = c.vocab, .halves = 12, .codebook = .mul1, .stride_k = x3_strides.k, .stride_nb = x3_strides.nb };
    e.head = .{ .x3 = x3_weight };
    Trace.copy_count = 0;
    Trace.projection_count = 0;
    Trace.topk_count = 0;
    Trace.table_count = 0;
    try mixedHead(&d, &s, b, &refs, rows);
    const x3_views = try frame.x3Views(x3_weight);
    const x3_columns = [_]usize{ 98304, x3_views.tail.n, 98304, x3_views.tail.n, c.vocab, c.vocab };
    try std.testing.expectEqual(@as(usize, 6), Trace.projection_count);
    try std.testing.expectEqual(@as(usize, 2), Trace.topk_count);
    try std.testing.expectEqual(@as(usize, 2), Trace.table_count);
    for (Trace.projections[0..Trace.projection_count], x3_columns) |p, columns| {
        try std.testing.expectEqual(columns, p.columns);
        try std.testing.expectEqual(group_rows, p.rows);
        const bytes_written = p.rows * p.columns * 2;
        try std.testing.expect(p.out >= d.logits);
        try std.testing.expect(p.out + bytes_written <= projection_end);
        try std.testing.expect(p.out + bytes_written <= b.logits);
    }
    for (Trace.topks[0..Trace.topk_count], [_]usize{ frame.columns(true, c.vocab), c.vocab }) |topk, columns| {
        const bytes_read = topk.rows * topk.columns * 2;
        try std.testing.expectEqual(columns, topk.columns);
        try std.testing.expectEqual(group_rows, topk.rows);
        try std.testing.expect(topk.logits >= d.logits);
        try std.testing.expect(topk.logits + bytes_read <= projection_end);
        try std.testing.expect(topk.logits + bytes_read <= b.logits);
    }
}
/// One DSpine frame for `items`: block, bridge, head over the drafted rows and final_norm * raw rows, table, selection.
pub fn forward(d: anytype, s: *Spine, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    const attribution = if (comptime @import("profile").round_attribution.enabled) @import("cuda").round_attribution.scope(.candidate_frame) else {};
    defer if (comptime @import("profile").round_attribution.enabled) attribution.end();
    const c = d.c;
    const e = d.e;
    const B = items.len * c.block;
    const rows = items.len * (c.block - 1);
    const b = bufs(c, d.frame_streams, d.ds);
    const st = d.staged.bytes;
    try frame.stage(c, items, contexts, st);
    for (contexts) |ctx| {
        ctx.held_n = 0;
        ctx.grown = 0;
    }
    errdefer cuda.completion.drain(e.d, e.stream, "a failed DSpine frame did not drain before its pinned staging was reused");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.ids, st.ptr, B * 4, e.stream.handle), "cuMemcpyHtoDAsync");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.segments, st.ptr + frame.segments_at, items.len * @sizeOf(frame.Segment), e.stream.handle), "cuMemcpyHtoDAsync");
    try layers(d, s, b, B, rows);
    try e.run(e.fns.sums, .{ 2 * rows, (c.hidden / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ b.hx, d.xs, int(c.hidden), @as(u64, 0) });
    const mixed = frame.mixedHeadEligible(e.head, contexts);
    var spanned: ?u64 = null;
    if (mixed) {
        try mixedHead(d, s, b, contexts, rows);
    } else {
        spanned = try frame.logitsOf(d, contexts, b.hx, b.logits, 2 * rows);
        const logits = spanned orelse b.logits;
        try e.draws.topkRows(e.stream, logits, frame.columns(spanned != null, c.vocab), @intCast(c.topk), d.parts, d.words, rows);
        try table(d, s, b, rows, logits, spanned != null, .{ .ids = d.ids, .raw = d.xin, .rms = b.rms, .q = b.q, .words = d.words, .out = b.table }, rows);
    }
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + frame.words_at, d.words, rows * lp_words * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + frame.hp_at, b.table, rows * c.topk * c.topk * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    for (contexts, 0..) |ctx, i| if (spanned != null or (mixed and !ctx.wide)) {
        for (0..c.block - 1) |r| {
            const row = i * (c.block - 1) + r;
            const words = std.mem.bytesAsSlice(u32, st[frame.words_at + row * lp_words * 4 ..][0 .. lp_words * 4]);
            for (words[2..][0..c.topk]) |*id| id.* = frame.columnId(id.*);
        }
    };
    try frame.publish(d, items, contexts);
}
