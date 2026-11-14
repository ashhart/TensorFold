//! Flash Next's DeltaNet through linear() on the GPU: the fused prompt prep against the unfused chain, bit for bit.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const config = @import("../config.zig");
const model = @import("model.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const mixers = @import("mixers.zig");

const quant = core.quant;
const gpa = std.testing.allocator;
/// Segments a case takes at most.
const most_segs = 2;

/// The device, a launcher with the merged launches on and one with them off, and the scratch linear() takes.
pub const Gpu = struct {
    r: hip.Runtime,
    ctx: hip.Context,
    stream: hip.Stream,
    on: hip.Launcher,
    off: hip.Launcher,
    arena: hip.Arena,
    act: model.Kind,

    /// Skipped with the inert kernels the host tests build; a GPU build fails when its device does not open.
    pub fn open(g: *Gpu, arena_bytes: usize) !void {
        if (hip.kernels.arch.len == 0) return error.SkipZigTest;
        g.r = try hip.Runtime.open();
        errdefer g.r.close();
        g.ctx = try hip.Context.init(&g.r, 0);
        errdefer g.ctx.deinit();
        const caps = try hip.Device.kernelCaps(&g.r, 0);
        g.act = if (caps.act == .f16) .f16 else .bf16;
        g.stream = try hip.Stream.init(&g.r);
        errdefer g.stream.deinit();
        g.on = try hip.Launcher.load(&g.r, .{}, hip.kernels.images);
        errdefer g.on.unload();
        g.off = try hip.Launcher.load(&g.r, .{ .fuse = false }, hip.kernels.images);
        errdefer g.off.unload();
        g.arena = try hip.Arena.init(&g.r, arena_bytes);
    }

    pub fn close(g: *Gpu) void {
        g.arena.deinit();
        g.off.unload();
        g.on.unload();
        g.stream.deinit();
        g.ctx.deinit();
        g.r.close();
    }
};

/// Uniform on [-1, 1).
fn unit(rng: std.Random) f32 {
    return rng.float(f32) * 2 - 1;
}

/// `v` as the activation's 16 bits.
fn bits(act: model.Kind, v: f32) u16 {
    return if (act == .f16) @bitCast(@as(f16, @floatCast(v))) else @intCast(@as(u32, @bitCast(v)) >> 16);
}

/// `n` values `offset + scale u`, u uniform on [-1, 1).
fn randoms(rng: std.Random, n: usize, scale: f32, offset: f32) ![]f32 {
    const v = try gpa.alloc(f32, n);
    for (v) |*x| x.* = offset + scale * unit(rng);
    return v;
}

/// fp32 values as a host tensor of `shape`.
fn tensor(v: []const f32, shape: []const usize) quant.Tensor {
    var t: quant.Tensor = .{ .dtype = .f32, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = std.mem.sliceAsBytes(v) };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

/// A synthetic DeltaNet layer at the fixture's dimensions on one rank: dense fp32 products, fp32 vectors, the qk norms.
pub const Layer = struct {
    buffers: std.ArrayList(hip.DeviceBuffer),
    m: model.Model,
    l: model.Linear,

    pub fn init(w: *Layer, g: *Gpu, rng: std.Random) !void {
        var c = try config.parse(gpa, @embedFile("../fixtures/config.json"));
        defer c.deinit();
        const d = try model.Dims.of(&c, 1);
        w.buffers = .empty;
        errdefer w.deinit();
        const up: hip.Upload = .{ .r = &g.r, .buffers = &w.buffers, .gpa = gpa };
        const kw = d.key_heads * d.key_dim;
        const vw = d.value_heads * d.value_dim;
        const qk = mixers.qkConstants(d.key_dim);
        w.m = .{
            .dims = d,
            .qk = .{ .q = try fill(&up, d.key_dim, qk.q), .k = try fill(&up, d.key_dim, qk.k), .eps = qk.eps },
            .act = g.act,
            .embed = undefined,
            .layers = &.{},
            .mix = undefined,
            .head = undefined,
            .ple = undefined,
        };
        w.l = .{
            .qkv = try dense(&up, rng, 2 * kw + vw, d.hidden),
            .z = try dense(&up, rng, vw, d.hidden),
            .a = try dense(&up, rng, d.value_heads, d.hidden),
            .b = try dense(&up, rng, d.value_heads, d.hidden),
            .out = try dense(&up, rng, d.hidden, vw),
            .conv = try vector(&up, rng, &.{ d.convWidth(), d.conv }, 0.5, 0),
            .a_log = try vector(&up, rng, &.{d.value_heads}, 1, 0),
            .dt_bias = try vector(&up, rng, &.{d.value_heads}, 1, 0),
            .norm = try vector(&up, rng, &.{d.value_dim}, 0.25, 1),
        };
    }

    pub fn deinit(w: *Layer) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(gpa);
    }

    /// A dense fp32 projection (n, k) whose rows sum to values near unit scale.
    fn dense(up: *const hip.Upload, rng: std.Random, n: usize, k: usize) !model.Projection {
        const v = try randoms(rng, n * k, 1 / @sqrt(@as(f32, @floatFromInt(k)) / 3), 0);
        defer gpa.free(v);
        return quant.view(try quant.upload(up.uploader(), .{ .dense = .{ .weight = tensor(v, &.{ n, k }) } }), false);
    }

    fn vector(up: *const hip.Upload, rng: std.Random, shape: []const usize, scale: f32, offset: f32) !u64 {
        var n: usize = 1;
        for (shape) |s| n *= s;
        const v = try randoms(rng, n, scale, offset);
        defer gpa.free(v);
        return (try up.uploader().tensor(tensor(v, shape))).ptr;
    }

    fn fill(up: *const hip.Upload, n: usize, value: f32) !u64 {
        const v = try gpa.alloc(f32, n);
        defer gpa.free(v);
        @memset(v, value);
        return (try up.uploader().tensor(tensor(v, &.{n}))).ptr;
    }
};

/// One buffer linear() leaves, by name and segment, as bytes on the host.
const Part = struct { name: []const u8, seg: ?usize = null, bytes: []u8 };

fn freeParts(parts: *std.ArrayList(Part)) void {
    for (parts.items) |p| gpa.free(p.bytes);
    parts.deinit(gpa);
}

/// q, k, v, gate and beta's places: the takes linear() makes before its branch, replayed on a copy of the arena.
fn places(arena: hip.Arena, d: model.Dims, act: model.Kind, rows: usize) ![5]u64 {
    var a = arena;
    for ([_]usize{ d.convWidth(), d.value_heads * d.value_dim, d.value_heads, d.value_heads }) |n| _ = try a.take(rows * n * act.size());
    var at: [5]u64 = undefined;
    for (&at, widths(d)) |*p, n| p.* = try a.of(f32, rows * n);
    return at;
}

/// q, k, v, gate and beta's fp32 values a row.
fn widths(d: model.Dims) [5]usize {
    const kw = d.key_heads * d.key_dim;
    return .{ kw, kw, d.value_heads * d.value_dim, d.value_heads, d.value_heads };
}

/// The scratch only the unfused chain takes, its mixed, qc and kc, replayed on a counting arena to the next take.
fn chainOnly(d: model.Dims, rows: usize) !usize {
    const kw = d.key_heads * d.key_dim;
    var a = hip.Arena.counting();
    for ([_]usize{ d.convWidth(), kw, kw }) |n| _ = try a.of(f32, rows * n);
    _ = try a.take(0);
    return a.used;
}

/// linear() on `l` over `segs` from the conv tails and states in `starts`; the arena holds `junk`, `taken` its scratch.
fn run(g: *Gpu, w: *const Layer, l: *const hip.Launcher, x: hip.ops.Tensor, segs: []const forward.Seg, starts: []const []f32, junk: u8, taken: *usize) !std.ArrayList(Part) {
    const d = w.m.dims;
    var rows: usize = 0;
    for (segs, starts) |s, st| {
        rows += s.tokens.len;
        const view = s.c.linearAt(0);
        const taps = s.c.layout.conv_bytes / 4;
        try s.c.mem.upload(view.conv - s.c.mem.base(), std.mem.sliceAsBytes(st[0..taps]));
        try s.c.mem.upload(view.state - s.c.mem.base(), std.mem.sliceAsBytes(st[taps..]));
    }
    try g.arena.buf.?.fill8(junk);
    const mark = g.arena.mark();
    defer g.arena.release(mark);
    const o: hip.ops.Ops = .{ .l = l, .bf16 = g.act == .bf16, .stream = g.stream.handle, .arena = &g.arena, .prefill = true };
    const mid = try places(g.arena, d, g.act, rows);
    const y = try mixers.linear(o, &w.m, w.l, segs, 0, x, rows, false);
    taken.* = g.arena.used - mark;
    try g.stream.synchronize();
    var parts: std.ArrayList(Part) = .empty;
    errdefer freeParts(&parts);
    try parts.append(gpa, .{ .name = "output", .bytes = try fetch(g.arena.buf.?, y.ptr, rows * d.hidden * y.kind.size()) });
    for (mid, widths(d), [_][]const u8{ "q", "k", "v", "gate", "beta" }) |at, n, name| {
        try parts.append(gpa, .{ .name = name, .bytes = try fetch(g.arena.buf.?, at, rows * n * 4) });
    }
    for (segs, 0..) |s, i| {
        const view = s.c.linearAt(0);
        try parts.append(gpa, .{ .name = "conv state", .seg = i, .bytes = try fetch(s.c.mem, view.conv, s.c.layout.conv_bytes) });
        try parts.append(gpa, .{ .name = "recurrent state", .seg = i, .bytes = try fetch(s.c.mem, view.state, s.c.layout.state_bytes) });
    }
    return parts;
}

/// `n` bytes at device address `at` inside `buf`.
fn fetch(buf: hip.DeviceBuffer, at: u64, n: usize) ![]u8 {
    const out = try gpa.alloc(u8, n);
    errdefer gpa.free(out);
    try buf.download(at - buf.base(), out);
    return out;
}

/// Whether every activation value is finite and one is nonzero.
fn live(act: model.Kind, bytes: []const u8) bool {
    const exp: u16 = if (act == .f16) 0x7c00 else 0x7f80;
    var any = false;
    for (std.mem.bytesAsSlice(u16, bytes)) |b| {
        if (b & exp == exp) return false;
        any = any or b & 0x7fff != 0;
    }
    return any;
}

/// The fused prompt prep against the unfused chain on segments of `lens` rows: every part byte for byte.
fn check(lens: []const usize) !void {
    var g: Gpu = undefined;
    try g.open(64 << 20);
    defer g.close();
    var prng: std.Random.DefaultPrng = .init(0x2582);
    const rng = prng.random();
    var w: Layer = undefined;
    try w.init(&g, rng);
    defer w.deinit();
    const d = w.m.dims;
    var rows: usize = 0;
    for (lens) |n| rows += n;
    const hx = try gpa.alloc(u16, rows * d.hidden);
    defer gpa.free(hx);
    for (hx) |*v| v.* = bits(g.act, unit(rng));
    var dx = try hip.DeviceBuffer.fromHost(&g.r, std.mem.sliceAsBytes(hx));
    defer dx.free();
    const x: hip.ops.Tensor = .{ .ptr = dx.base(), .kind = g.act };
    const tokens = try gpa.alloc(u32, rows);
    defer gpa.free(tokens);
    @memset(tokens, 0);
    var caches: [most_segs]cache.Caches = undefined;
    var made: usize = 0;
    defer for (caches[0..made]) |*c| c.deinit(gpa);
    for (caches[0..lens.len]) |*c| {
        c.* = try cache.Caches.init(gpa, &g.r, d, &.{true}, 64, g.act, 0);
        made += 1;
    }
    // nonzero conv tails and states, each segment its own
    var starts: [most_segs][]f32 = undefined;
    var drawn: usize = 0;
    defer for (starts[0..drawn]) |st| gpa.free(st);
    for (starts[0..lens.len], caches[0..lens.len]) |*st, c| {
        st.* = try gpa.alloc(f32, (c.layout.conv_bytes + c.layout.state_bytes) / 4);
        drawn += 1;
        const taps = c.layout.conv_bytes / 4;
        for (st.*[0..taps]) |*v| v.* = unit(rng);
        for (st.*[taps..]) |*v| v.* = 0.1 * unit(rng);
    }
    var segs: [most_segs]forward.Seg = undefined;
    var at: usize = 0;
    for (segs[0..lens.len], caches[0..lens.len], lens) |*s, *c, n| {
        s.* = .{ .c = c, .tokens = tokens[at..][0..n], .pos = 0 };
        at += n;
    }
    var fused_taken: usize = 0;
    var fused = try run(&g, &w, &g.on, x, segs[0..lens.len], starts[0..lens.len], 0xa5, &fused_taken);
    defer freeParts(&fused);
    var chain_taken: usize = 0;
    var chain = try run(&g, &w, &g.off, x, segs[0..lens.len], starts[0..lens.len], 0x5a, &chain_taken);
    defer freeParts(&chain);
    // the fused branch ran: it holds the chain's scratch less the mixed, qc and kc it never takes
    try std.testing.expectEqual(chain_taken, fused_taken + try chainOnly(d, rows));
    try std.testing.expect(live(g.act, chain.items[0].bytes));
    var differ = false;
    for (fused.items, chain.items) |a, b| {
        const i = std.mem.indexOfDiff(u8, a.bytes, b.bytes) orelse continue;
        differ = true;
        if (a.seg) |s| {
            std.debug.print("segment {d}'s {s} differs from the unfused chain's at byte {d} of {d}\n", .{ s, a.name, i, a.bytes.len });
        } else std.debug.print("{s} differs from the unfused chain's at byte {d} of {d}\n", .{ a.name, i, a.bytes.len });
    }
    try std.testing.expect(!differ);
}

test "Flash Next fused prompt prep: one segment through linear() leaves the unfused chain's bytes" {
    try check(&.{45});
}

test "Flash Next fused prompt prep: two segments through linear() leave the unfused chain's bytes" {
    try check(&.{ 45, 19 });
}
