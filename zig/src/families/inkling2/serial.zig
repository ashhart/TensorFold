//! The serial reference (test tools only): a world-two engine's two cuts on one GPU, rank 0's partials then rank 1's.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("pack_weights.zig");
const fw = @import("forward.zig");
const ops = @import("ops.zig");
const shard = @import("shard.zig");
const engine = @import("engine.zig");

const Buf = cuda.DeviceBuffer;

/// Rank 1's cut beside rank 0's engine: its weights, its caches and its logits rows of prompt and decode calls.
pub const Serial = struct {
    w: weights.Weights,
    st: ?fw.State = null,
    logits_p: Buf,
    logits_d: Buf,

    /// Rank 1 of world two loaded beside `e` (rank 0 of world two), whose forwards then run both ranks' partials.
    pub fn attach(e: *engine.Engine, dir: []const u8) !void {
        if (e.w.share.world != 2 or e.w.share.rank != 0 or e.serial != null or e.st != null) return error.Inkling2SerialSetup;
        const x = try e.gpa.create(Serial);
        errdefer e.gpa.destroy(x);
        x.* = .{ .w = try weights.load(e.gpa, e.io, e.d, &e.m.dense, e.stream, dir, e.c, 1, 2), .logits_p = undefined, .logits_d = undefined };
        errdefer x.w.deinit();
        // key heads split evenly, so rank 1's caches and scratch take rank 0's shapes and one staging serves both
        if (x.w.share.heads.len() != e.shape_p.heads or x.w.share.kv.len() != e.shape_p.kv) return error.Inkling2SerialSetup;
        for (x.w.layers) |l| for ([_]cuda.exl3.Weight{ l.attn.q, l.attn.k, l.attn.v, l.attn.r, l.attn.gate, l.attn.o }) |y| try e.m.dense.reserve(e.stream, y);
        try e.m.dense.reserve(e.stream, x.w.head);
        for (x.w.layers) |l| for ([_]cuda.exl3.Weight{ l.attn.k, l.attn.v, l.attn.r }) |y| try e.m.side.dense.reserve(e.stream, y);
        const nv = x.w.share.vocab.len();
        x.logits_p = try Buf.alloc(e.d, e.shape_p.rows * nv * 4);
        errdefer x.logits_p.free();
        x.logits_d = try Buf.alloc(e.d, e.shape_d.rows * nv * 4);
        errdefer x.logits_d.free();
        try e.stream.synchronize();
        e.serial = x;
        e.exchange = .serial;
    }

    pub fn deinit(x: *Serial, gpa: std.mem.Allocator) void {
        if (x.st) |*s| s.deinit();
        x.logits_d.free();
        x.logits_p.free();
        x.w.deinit();
        gpa.destroy(x);
    }

    /// Rank 1's caches for `slots` streams, beside rank 0's.
    pub fn reserve(x: *Serial, e: *const engine.Engine, slots: usize) !void {
        x.st = try fw.State.init(e.gpa, e.d, &e.c, e.shape_d, slots);
    }

    /// Rank 1 for a forward on prompt (`prompt`) or decode scratch.
    pub fn peer(x: *Serial, prompt: bool) fw.Peer {
        return .{ .w = &x.w, .st = &x.st.?, .logits = x.logits(prompt) };
    }

    fn logits(x: *const Serial, prompt: bool) u64 {
        return if (prompt) x.logits_p.ptr else x.logits_d.ptr;
    }

    /// Rank 1's greedy picks of `rows` rows from `row0`, into its slot of the picks, as rank 1 makes them at two ranks.
    pub fn pick(x: *const Serial, e: *engine.Engine, prompt: bool, row0: usize, rows: usize, vals: u64) !void {
        const sh = x.w.share;
        const n = sh.vocab.len();
        const real = @min(n, @as(usize, e.c.real_vocab) -| sh.vocab.lo);
        try ops.top1(&e.m.ops, e.stream, x.logits(prompt) + row0 * n * 4, rows, n, real, sh.vocab.lo, vals, vals + rows * 4);
    }
};

/// The widest rank's padded vocabulary span: each rank's part of the whole-row gather.
pub fn widest(spans: []const shard.Span) usize {
    var w: usize = 0;
    for (spans) |s| w = @max(w, s.len());
    return w;
}

/// The whole row from every rank's part (`parts`: [world][widest] fp32): each span at its own columns, padding dropped.
pub fn assemble(out: []u8, parts: []const u8, spans: []const shard.Span) void {
    const w = widest(spans);
    for (spans, 0..) |s, q| @memcpy(out[s.lo * 4 ..][0 .. s.len() * 4], parts[q * w * 4 ..][0 .. s.len() * 4]);
}

/// A call's logits row over the whole vocabulary on rank 0 (`out`); every rank calls it, gathering through `scratch`.
pub fn rebuild(e: *engine.Engine, prompt: bool, row: usize, scratch: Buf, host: []u8, out: []u8) !void {
    const b = if (prompt) &e.bp else &e.bd;
    const sh = e.w.share;
    const nv = sh.vocab.len();
    const w = widest(e.spans);
    try e.stream.synchronize();
    switch (e.exchange) {
        .local => try b.logits.download(row * nv * 4, out[0 .. nv * 4]),
        .serial => {
            const x = e.serial.?;
            const s1 = x.w.share.vocab;
            try b.logits.download(row * nv * 4, out[sh.vocab.lo * 4 ..][0 .. nv * 4]);
            const view: Buf = .{ .d = e.d, .ptr = x.logits(prompt), .len = (row + 1) * s1.len() * 4, .borrowed = true };
            try view.download(row * s1.len() * 4, out[s1.lo * 4 ..][0 .. s1.len() * 4]);
        },
        .nccl, .pair => {
            const own = scratch.ptr + sh.rank * w * 4;
            const view: Buf = .{ .d = e.d, .ptr = own, .len = w * 4, .borrowed = true };
            try view.copyFrom(0, b.logits.ptr + row * nv * 4, nv * 4, e.stream.handle);
            switch (e.exchange) {
                .nccl => |y| try y.lib.check(y.lib.api.ncclAllGather(own, scratch.ptr, w * 4, .u8, y.comm, e.stream.handle), "ncclAllGather"),
                .pair => |y| try y.p.allGather(y.rank, e.stream, scratch.ptr, w * 4),
                else => unreachable,
            }
            try e.stream.synchronize();
            if (sh.rank != 0) return;
            try scratch.download(0, host[0 .. sh.world * w * 4]);
            assemble(out, host, e.spans);
        },
    }
}

test "serial: rank 0 rebuilds the whole row from two unequal padded spans, each at its own columns" {
    const c: @import("shape.zig").Config = .{};
    const spans = [_]shard.Span{ (try shard.Share.of(&c, 201088, 0, 2)).vocab, (try shard.Share.of(&c, 201088, 1, 2)).vocab };
    try std.testing.expectEqual(@as(usize, 786 * 128), widest(&spans));
    const gpa = std.testing.allocator;
    const w = widest(&spans);
    const parts = try gpa.alloc(u8, 2 * w * 4);
    defer gpa.free(parts);
    const vals = std.mem.bytesAsSlice(u32, parts);
    for (0..w) |j| {
        vals[j] = @intCast(j); // rank 0's column j is vocabulary row j
        vals[w + j] = if (j < spans[1].len()) @intCast(spans[1].lo + j) else 0xdead; // rank 1's last slot is gather padding
    }
    const out = try gpa.alloc(u8, 201088 * 4);
    defer gpa.free(out);
    @memset(out, 0xff);
    assemble(out, parts, &spans);
    for (std.mem.bytesAsSlice(u32, out), 0..) |v, j| try std.testing.expectEqual(@as(u32, @intCast(j)), v);
}

test "serial: rank 1's partial lands where two ranks put it, the part the combine reads second" {
    const gathered: u64 = 1 << 20;
    try std.testing.expectEqual(gathered, fw.partAt(gathered, 0, 7, 2304));
    try std.testing.expectEqual(gathered + 7 * 2304 * 4, fw.partAt(gathered, 1, 7, 2304));
}
