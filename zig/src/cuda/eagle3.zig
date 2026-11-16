//! An EAGLE-3 drafter's passes on shared operators only (no kernel of its own): taps and a token in, draft picks out.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const Function = @import("module.zig").Function;
const launch_ = @import("launch.zig");
const abi = @import("abi.zig");
const DenseRows = @import("dense_rows.zig").DenseRows;
const Norms = @import("norms.zig").Norms;
const QkRows = @import("qk_rows.zig").QkRows;
const ActRows = @import("act_rows.zig").ActRows;
const EmbedRows = @import("embed_rows.zig").EmbedRows;
const wa = @import("window_attention.zig");

/// Rows one pass takes: an absorb goes in parts of this many, a hold's step takes one row a stream.
pub const pass_rows = 256;

pub const Dims = struct {
    hidden: usize,
    heads: usize, // key heads too: every query head has its own
    head_dim: usize,
    ffn: usize,
    layers: usize,
    taps: usize, // tapped rows a context row reads, each `hidden` wide
    vocab: usize, // the draft vocabulary's rows of the head
    window: usize, // keys back a row attends to, its own included
    chain: usize, // a hold's steps at most
    eps: f32,

    fn att(d: Dims) usize {
        return d.heads * d.head_dim;
    }

    /// Ring rows a stream keeps: a pass's rows and a chain written before the first row reads its window.
    pub fn ring(d: Dims) usize {
        return (d.window + pass_rows + d.chain + 63) / 64 * 64;
    }

    /// Device bytes one stream's keys and values take (every layer's) and its last output.
    pub fn streamBytes(d: Dims) usize {
        return 2 * d.layers * d.ring() * d.att() * 2 + d.hidden * 2;
    }
};

/// One layer: projections bf16 [out, in] (qkv and gate|up stacked), norms fp32.
pub const Layer = struct { qkv: u64, o: u64, gate_up: u64, down: u64, n1: u64, n2: u64 };

/// The drafter's weights and the target's: `embed` its table [V, hidden], `head` its rows for the draft vocabulary.
pub const Weights = struct { fuse: u64, fc: u64, layers: []const Layer, out_norm: u64, inv: u64, embed: u64, head: u64 };

/// The shared operators the passes launch, loaded by the caller (a target's family loads most of them already).
pub const Ops = struct {
    dense: *const DenseRows,
    norms: *const Norms,
    qk: *const QkRows,
    win: *const wa.WindowAttention,
    embed: *const EmbedRows,
    act: *const ActRows,
    argmax: Function, // torch's first-maximum argmax over fp32 rows into int32
    to_bf16: Function, // fp32 to bf16, round to nearest even
};

/// One row of a pass: the stream's slot, its position, the token after it; `src` its tapped row or prior output.
pub const Row = struct { slot: u32, pos: u32, token: u32 };

pub const Eagle3 = struct {
    gpa: std.mem.Allocator,
    d: *const Driver,
    s: Stream,
    dims: Dims,
    w: Weights,
    ops: Ops,
    slots: usize = 0,
    kc: u64 = 0, // [slots, ring, heads, head_dim] bf16
    vc: u64 = 0,
    out: u64 = 0, // [slots, hidden] bf16: each stream's last row's normed output (its next draft's state)
    x: Scratch,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    caches: std.ArrayList(DeviceBuffer) = .empty,
    host: []i32, // [3, pass_rows] staging: tokens, positions, slots

    const Scratch = struct { ids: u64, pos: u64, slot: u64, picks: u64, cat: u64, e: u64, st: u64, f: u64, z: u64, a: u64, qkv: u64, q: u64, k: u64, v: u64, att: u64, gu: u64, act: u64, part: u64, logits: u64 };

    pub fn init(gpa: std.mem.Allocator, d: *const Driver, s: Stream, dims: Dims, w: Weights, ops: Ops) !Eagle3 {
        if (dims.head_dim != 128 or dims.hidden % 64 != 0 or dims.ffn % 64 != 0 or dims.chain == 0) return error.Eagle3Shape;
        var x: Eagle3 = .{ .gpa = gpa, .d = d, .s = s, .dims = dims, .w = w, .ops = ops, .x = undefined, .host = try gpa.alloc(i32, 3 * pass_rows) };
        errdefer x.deinit();
        const n = pass_rows;
        const h = dims.hidden;
        const wide = @max(dims.taps * h, @max(3 * dims.att(), 2 * dims.ffn));
        x.x = .{
            .ids = try x.alloc(n * 4),
            .pos = try x.alloc(n * 4),
            .slot = try x.alloc(n * 4),
            .picks = try x.alloc(n * dims.chain * 4),
            .cat = try x.alloc(n * 2 * h * 2),
            .e = try x.alloc(n * h * 2),
            .st = try x.alloc(n * h * 2),
            .f = try x.alloc(n * wide * 4),
            .z = try x.alloc(n * h * 4),
            .a = try x.alloc(n * h * 2),
            .qkv = try x.alloc(n * 3 * dims.att() * 2),
            .q = try x.alloc(n * dims.att() * 2),
            .k = try x.alloc(n * dims.att() * 2),
            .v = try x.alloc(n * dims.att() * 2),
            .att = try x.alloc(n * dims.att() * 2),
            .gu = try x.alloc(n * 2 * dims.ffn * 2),
            .act = try x.alloc(n * dims.ffn * 2),
            .part = try x.alloc(wa.WindowAttention.scratchBytes(x.window(), n)),
            .logits = try x.alloc(n * dims.vocab * 4),
        };
        return x;
    }

    pub fn deinit(x: *Eagle3) void {
        for (x.buffers.items) |*b| b.free();
        x.buffers.deinit(x.gpa);
        for (x.caches.items) |*b| b.free();
        x.caches.deinit(x.gpa);
        x.gpa.free(x.host);
    }

    fn alloc(x: *Eagle3, bytes: usize) !u64 {
        const b = try DeviceBuffer.alloc(x.d, @max(bytes, 256));
        errdefer {
            var y = b;
            y.free();
        }
        try x.buffers.append(x.gpa, b);
        return b.ptr;
    }

    /// Caches for `slots` streams (made once; a second call with no more slots keeps them).
    pub fn reserve(x: *Eagle3, slots: usize) !void {
        if (slots <= x.slots) return;
        try x.s.synchronize();
        for (x.caches.items) |*b| b.free();
        x.caches.clearRetainingCapacity();
        x.slots = 0;
        const per = x.dims.layers * x.dims.ring() * x.dims.att() * 2;
        var bufs: [3]u64 = undefined;
        for (&bufs, [_]usize{ slots * per, slots * per, slots * x.dims.hidden * 2 }) |*p, bytes| {
            const b = try DeviceBuffer.alloc(x.d, bytes);
            x.caches.append(x.gpa, b) catch |e| {
                var y = b;
                y.free();
                return e;
            };
            p.* = b.ptr;
        }
        x.kc, x.vc, x.out = bufs;
        x.slots = slots;
    }

    fn window(x: *const Eagle3) wa.Window {
        const dh: f32 = @floatFromInt(x.dims.head_dim);
        return .{ .h = x.dims.heads, .hk = x.dims.heads, .d = x.dims.head_dim, .win = x.dims.window, .ring = x.dims.ring(), .scale = 1.0 / @sqrt(dh) };
    }

    fn copy2d(x: *Eagle3, dst: u64, dst_pitch: usize, src: u64, src_pitch: usize, width: usize, rows: usize) !void {
        const p: abi.Memcpy2D = .{ .src = src, .src_pitch = src_pitch, .dst = dst, .dst_pitch = dst_pitch, .width = width, .height = rows };
        try x.d.check(x.d.api.cuMemcpy2DAsync_v2(&p, x.s.handle), "cuMemcpy2DAsync");
    }

    fn toBf16(x: *Eagle3, src: u64, dst: u64, n: usize) !void {
        var a: launch_.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(u64, n));
        try launch_.launch(x.ops.to_bf16, .{ .grid = .{ .x = @intCast(@max(1, @min((n + 255) / 256, 65535))) }, .block = .{ .x = 256 } }, x.s, &a);
    }

    /// bf16 rows [m, k] times weight rows [n, k] into bf16 `out` [m, n], through the fp32 scratch.
    fn linear(x: *Eagle3, in: u64, in_stride: usize, w: u64, out: u64, m: usize, n: usize, k: usize) !void {
        try x.ops.dense.rows(x.s, in, in_stride, w, x.x.f, n, m, n, k);
        try x.toBf16(x.x.f, out, m * n);
    }

    /// The rows' tokens, positions and slots on the device (ids: the token embedded beside each row's state).
    fn stage(x: *Eagle3, rows: []const Row) !void {
        const n = rows.len;
        for (rows, 0..) |r, i| {
            x.host[i] = @intCast(r.token);
            x.host[pass_rows + i] = @intCast(r.pos);
            x.host[2 * pass_rows + i] = @intCast(r.slot);
        }
        for ([_]u64{ x.x.ids, x.x.pos, x.x.slot }, 0..) |dst, j| try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(dst, &x.host[j * pass_rows], n * 4, x.s.handle), "cuMemcpyHtoDAsync");
    }

    /// Staged rows: fc(cat(embed(token), state)) through every layer into `a`; `fused`: state is taps, fused first.
    fn pass(x: *Eagle3, n: usize, state: u64, state_ld: usize, fused: bool) !void {
        const dm = x.dims;
        const h = dm.hidden;
        const sx = x.x;
        try x.ops.embed.bf16(x.s, sx.ids, x.w.embed, sx.e, n, h);
        try x.copy2d(sx.cat, 4 * h, sx.e, 2 * h, 2 * h, n);
        if (fused) {
            try x.linear(state, state_ld, x.w.fuse, sx.a, n, h, dm.taps * h);
            try x.copy2d(sx.cat + 2 * h, 4 * h, sx.a, 2 * h, 2 * h, n);
        } else try x.copy2d(sx.cat + 2 * h, 4 * h, state, 2 * state_ld, 2 * h, n);
        try x.ops.dense.rows(x.s, sx.cat, 2 * h, x.w.fc, sx.z, h, n, h, 2 * h);
        try x.ops.norms.rms(x.s, sx.z, .f32, x.w.layers[0].n1, sx.a, .bf16, n, h, dm.eps);
        const att = dm.att();
        const win = x.window();
        for (x.w.layers, 0..) |l, i| {
            try x.linear(sx.a, h, l.qkv, sx.qkv, n, 3 * att, h);
            const shape: @import("qk_rows.zig").Shape = .{ .inv = x.w.inv, .rot = dm.head_dim };
            const kv_at = i * x.slots * dm.ring() * att * 2;
            const out: @import("qk_rows.zig").Out = .{ .q = sx.q, .k = sx.k, .v = sx.v, .kc = x.kc + kv_at, .vc = x.vc + kv_at, .len = dm.ring() };
            try x.ops.qk.rows(x.s, sx.qkv, .{ .h = dm.heads, .hk = dm.heads, .dim = dm.head_dim }, shape, sx.pos, sx.slot, out, n);
            try x.ops.win.decode(x.s, sx.q, x.kc + kv_at, x.vc + kv_at, sx.att, sx.pos, sx.slot, win, sx.part, n);
            try x.ops.dense.rows(x.s, sx.att, att, l.o, sx.f, h, n, h, att);
            try x.ops.norms.addRms(x.s, sx.f, .f32, sx.z, null, l.n2, sx.a, .bf16, n, h, dm.eps);
            try x.linear(sx.a, h, l.gate_up, sx.gu, n, 2 * dm.ffn, h);
            try x.ops.act.swiglu(x.s, sx.gu, 2 * dm.ffn, sx.gu + dm.ffn * 2, 2 * dm.ffn, sx.act, null, n, dm.ffn);
            try x.ops.dense.rows(x.s, sx.act, dm.ffn, l.down, sx.f, h, n, h, dm.ffn);
            const next = if (i + 1 < x.w.layers.len) x.w.layers[i + 1].n1 else x.w.out_norm;
            try x.ops.norms.addRms(x.s, sx.f, .f32, sx.z, null, next, sx.a, .bf16, n, h, dm.eps);
        }
    }

    /// Context rows from tapped rows `taps` [rows.len, taps * hidden] bf16; each slot's last row's output kept.
    pub fn absorb(x: *Eagle3, rows: []const Row, taps: u64) !void {
        const dm = x.dims;
        if (x.slots == 0) return error.NoCaches;
        var at: usize = 0;
        while (at < rows.len) {
            const n = @min(pass_rows, rows.len - at);
            const part = rows[at..][0..n];
            for (part) |r| if (r.slot >= x.slots) return error.NoSuchDraftSlot;
            try x.stage(part);
            try x.pass(n, taps + at * dm.taps * dm.hidden * 2, dm.taps * dm.hidden, true);
            for (part, 0..) |r, i| if (i + 1 == n or part[i + 1].slot != r.slot)
                try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(x.out + r.slot * dm.hidden * 2, x.x.a + i * dm.hidden * 2, dm.hidden * 2, x.s.handle), "cuMemcpyDtoDAsync");
            at += n;
        }
    }

    /// `steps` draft-vocabulary indices for each item (slot, last absorbed position) into out[item]; a sync a step.
    pub fn chain(x: *Eagle3, items: []const Row, steps: usize, map: []const u32, out: [][]u32) !void {
        const dm = x.dims;
        const n = items.len;
        if (n > pass_rows or steps > dm.chain) return error.TooManyDraftRows;
        const h = dm.hidden;
        var rows: [pass_rows]Row = undefined;
        for (items, 0..) |it, i| {
            if (it.slot >= x.slots) return error.NoSuchDraftSlot;
            try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(x.x.a + i * h * 2, x.out + it.slot * h * 2, h * 2, x.s.handle), "cuMemcpyDtoDAsync");
        }
        var picks: [pass_rows]i32 = undefined;
        for (0..steps) |j| {
            if (j > 0) { // the step's state is the last step's output; its keys sit past the context, uncommitted
                for (items, rows[0..n], 0..) |it, *r, i| r.* = .{ .slot = it.slot, .pos = it.pos + @as(u32, @intCast(j)), .token = map[out[i][j - 1]] };
                try x.stage(rows[0..n]);
                try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(x.x.st, x.x.a, n * h * 2, x.s.handle), "cuMemcpyDtoDAsync");
                try x.pass(n, x.x.st, h, false);
            }
            try x.ops.dense.rows(x.s, x.x.a, h, x.w.head, x.x.logits, dm.vocab, n, dm.vocab, h);
            var a: launch_.Args = .{};
            a.add(x.x.logits);
            a.add(x.x.picks);
            for ([_]usize{ n, dm.vocab, dm.vocab }) |v| a.add(@as(i64, @intCast(v)));
            a.add(@as(i32, 1)); // fp32 rows
            try launch_.launch(x.ops.argmax, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 1024 } }, x.s, &a);
            try x.s.synchronize();
            try @import("memory.zig").download(x.d, x.x.picks, std.mem.sliceAsBytes(picks[0..n]));
            for (out[0..n], picks[0..n]) |o, p| {
                if (p < 0 or p >= dm.vocab) return error.DraftPickOutOfRange;
                o[j] = @intCast(p);
            }
        }
    }
};

test "a stream's ring holds the window, a pass and a chain in whole key tiles" {
    const d: Dims = .{ .hidden = 2560, .heads = 20, .head_dim = 128, .ffn = 4096, .layers = 1, .taps = 3, .vocab = 32768, .window = 2048, .chain = 8, .eps = 1e-6 };
    try std.testing.expectEqual(@as(usize, 2368), d.ring());
    try std.testing.expect(d.ring() >= d.window + pass_rows + d.chain and d.ring() % 64 == 0);
    try std.testing.expectEqual(@as(usize, 2 * 2368 * 2560 * 2 + 2560 * 2), d.streamBytes());
}
