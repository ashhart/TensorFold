//! LFM2's shared-round kernels (lfm2_segments.metal): every stream's conv or attention rows in one dispatch a kernel.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const controls = @import("metal_lfm2_controls.zig");

const d = 2048;
const qh = 32;
const kvh = 8;
const hd = 64;
const subs = 8; // sub-spans a 4,096-key chunk for a stream of at most 8 rows (state.subs)
const chunks = 32;
const chunk_keys = 4096;
const most = 32; // a shared round's rows (state.batch_rows)
const part = qh * chunks * subs; // partials a row owns (the kernels' SEG_PART)
pub const Which = enum { attention, conv };
const names = [_][]const u8{ "metal lfm2 stream attention", "metal lfm2 stream conv" };

/// rows: a round's rows over its streams (sizes from `sizes` in turn); k: the first stream's cached keys.
const attn_shapes = [_]c.Shape{ .{ .rows = 2, .n = qh * hd, .k = 5 }, .{ .rows = 13, .n = qh * hd, .k = 300 }, .{ .rows = 32, .n = qh * hd, .k = 4100 } };
const conv_shapes = [_]c.Shape{ .{ .rows = 2, .n = d, .k = 3 }, .{ .rows = 13, .n = d, .k = 3 }, .{ .rows = 32, .n = d, .k = 3 } };
const sizes = [_]usize{ 1, 9, 3, 16, 2, 8, 5 };

fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .attention => .{ .name = names[0], .accum = .none, .depth = none, .out = .bf16, .served = &attn_shapes, .replaces = "lfm2_qk, lfm2_attn_mma and lfm2_attn_merge a stream", .nan = .{ .stated = "a NaN key of one stream's kv head makes that stream's heads of it NaN, every other output stays finite" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "the per-stream kernels' bits; the check bounds outputs by float64 attention" } },
        .conv => .{ .name = names[1], .accum = .none, .depth = none, .out = .bf16, .served = &conv_shapes, .replaces = "lfm2_conv a stream", .nan = .{ .stated = "outputs equal the host's replay of the kernel's bf16 steps, NaN for NaN" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "three fp32 products a channel, replayed exactly on the host" } },
    };
}

/// lfm2_segments.metal's lfm2_seg.
const Seg = extern struct { keys: u64 = 0, values: u64 = 0, conv: u64 = 0, base: u32, rows: u32, len: u32, cap: u32, subs: u32, pad: u32 = 0 };

fn value(i: usize, seed: u64, scale: f32) u16 {
    const h: u64 = (@as(u64, i) *% 0x9e3779b97f4a7c15 +% seed *% 0xbf58476d1ce4e5b9) >> 40;
    return c.toBf16((@as(f32, @floatFromInt(h % 2048)) / 1024 - 1) * scale);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: controls.Mutation,
    pipes: [8]mtl.Pipeline,
    made: usize = 0,
    rows: usize = 0,
    nan: bool = false,
    first_len: usize = 0,
    streams: [most]usize = undefined, // rows a stream
    count_streams: usize = 0,
    host: []u16 = &.{}, // attention: the cached keys and values as prepared; conv: bcx, then the states
    weights: []u16 = &.{},
    args: [6]rig_.Ref = undefined,
    caches: [2][most]rig_.Ref = undefined, // attention: each stream's [keys | values], batched and per stream
    y: rig_.Ref = undefined,
    queries: rig_.Ref = undefined,
    work: [9]rig_.Ref = undefined, // segs, rowseg, tiles, pm, pl, po; the per-stream queries, y and partials' base
    golden: []u8 = &.{},
    snap: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, mutation: controls.Mutation) !Probe {
        const list: []const [:0]const u8 = switch (w) {
            .attention => &.{ "lfm2_qk_segs", "lfm2_attn_segs", "lfm2_attn_merge_segs", "lfm2_qk", "lfm2_attn_mma", "lfm2_attn_mma_r2", "lfm2_attn_merge" },
            .conv => &.{ "lfm2_conv_segs", "lfm2_conv" },
        };
        var p: Probe = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .mutation = mutation, .pipes = undefined };
        const text = try controls.source(a, mutation);
        defer if (mutation != .none) a.free(text);
        errdefer for (p.pipes[0..p.made]) |x| x.deinit();
        for (list) |n| {
            p.pipes[p.made] = try rig_.pipeline(device, text, n);
            p.made += 1;
        }
        return p;
    }

    fn drop(p: *Probe) void {
        p.a.free(p.golden);
        p.golden = &.{};
        p.a.free(p.host);
        p.a.free(p.weights);
        p.host = &.{};
        p.weights = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        for (p.pipes[0..p.made]) |x| x.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec(p.which);
        if (p.mutation != .none) s.name = p.mutation.name();
        return .{ .ctx = p, .vt = &vt, .spec = s, .control = p.mutation.gate() };
    }

    fn len(p: *const Probe, j: usize) usize {
        return p.first_len + 37 * j;
    }

    fn cap(p: *const Probe, j: usize) usize {
        return (p.len(j) + p.streams[j] + 63) / 64 * 64 + 32;
    }

    fn base(p: *const Probe, j: usize) usize {
        var b: usize = 0;
        for (p.streams[0..j]) |r| b += r;
        return b;
    }

    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        if (s.rows < 2 or s.rows > most or s.n != @as(usize, if (p.which == .attention) qh * hd else d)) return false;
        if (p.which == .attention and (s.k == 0 or s.k + 37 * most + most > chunks * chunk_keys)) return false;
        p.drop();
        p.rows = s.rows;
        p.first_len = s.k;
        p.nan = f == .nonfinite;
        p.count_streams = 0;
        var left = s.rows;
        while (left > 0) : (p.count_streams += 1) {
            const r = @min(left, sizes[p.count_streams % sizes.len]);
            p.streams[p.count_streams] = r;
            left -= r;
        }
        return if (p.which == .attention) p.prepareAttention(f, seed) else p.prepareConv(f, seed);
    }

    fn prepareAttention(p: *Probe, f: g.Fill, seed: u64) !bool {
        const n = p.rows;
        var q = try p.a.alloc(u16, n * (qh + 2 * kvh) * hd);
        defer p.a.free(q);
        for (q, 0..) |*x, i| x.* = value(i, seed, if (f == .stress) 4 else 1);
        p.weights = try p.a.alloc(u16, 2 * hd);
        for (p.weights, 0..) |*x, i| x.* = value(i, seed +% 9, 1.5);
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(q[0 .. n * qh * hd]));
        p.args[1] = try p.rig.input(std.mem.sliceAsBytes(q[n * qh * hd ..][0 .. n * kvh * hd]));
        p.args[2] = try p.rig.input(std.mem.sliceAsBytes(q[n * (qh + kvh) * hd ..][0 .. n * kvh * hd]));
        p.args[3] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
        var total: usize = 0;
        for (0..p.count_streams) |j| total += 2 * kvh * p.cap(j) * hd;
        p.host = try p.a.alloc(u16, total);
        var at: usize = 0;
        for (0..p.count_streams) |j| {
            const kv = p.host[at..][0 .. 2 * kvh * p.cap(j) * hd];
            for (kv, 0..) |*x, i| x.* = value(i, seed +% 3 +% j, 1);
            if (p.nan and j == 0) kv[(0 * p.cap(j) + p.len(j) / 2) * hd + 7] = 0x7fc0; // stream 0, kv head 0
            for (0..2) |copy| p.caches[copy][j] = try p.rig.scratch(kv.len * 2);
            at += kv.len;
        }
        p.y = try p.rig.output(n * d * 2);
        p.queries = try p.rig.scratch(n * d * 2);
        p.work[0] = try p.rig.scratch(most * @sizeOf(Seg));
        p.work[1] = try p.rig.scratch(most * 4);
        p.work[2] = try p.rig.scratch(most * 4);
        p.work[3] = try p.rig.scratch(n * part * 4);
        p.work[4] = try p.rig.scratch(n * part * 4);
        p.work[5] = try p.rig.scratch(n * part * hd * 4);
        p.work[6] = try p.rig.scratch(n * d * 2);
        p.work[7] = try p.rig.scratch(n * d * 2);
        return true;
    }

    fn prepareConv(p: *Probe, f: g.Fill, seed: u64) !bool {
        const n = p.rows;
        p.host = try p.a.alloc(u16, n * 3 * d + p.count_streams * 2 * d);
        for (p.host, 0..) |*x, i| x.* = value(i, seed, if (f == .stress) 8 else 1);
        if (p.nan) {
            p.host[1 * 3 * d + 11] = 0x7fc0; // row 1's B at channel 11
            p.host[n * 3 * d + 2 * d + 5] = 0x7f80; // stream 1's state at channel 5
        }
        p.weights = try p.a.alloc(u16, d * 3);
        for (p.weights, 0..) |*x, i| x.* = value(i, seed +% 5, 1);
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(p.host[0 .. n * 3 * d]));
        p.args[1] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
        for (0..p.count_streams) |j| p.caches[0][j] = try p.rig.input(std.mem.sliceAsBytes(p.host[n * 3 * d + j * 2 * d ..][0 .. 2 * d]));
        p.y = try p.rig.output(n * d * 2);
        p.snap = try p.rig.output(n * 2 * d * 2);
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.rows * d * @as(usize, if (p.which == .conv) 3 else 1);
    }

    /// Each stream's cache as prepared (its cached keys before the call's).
    fn fill(p: *Probe, copy: usize) void {
        var at: usize = 0;
        for (0..p.count_streams) |j| {
            const n = 2 * kvh * p.cap(j) * hd;
            @memcpy(std.mem.bytesAsSlice(u16, p.rig.bytes(p.caches[copy][j], n * 2)), p.host[at..][0..n]);
            at += n;
        }
    }

    fn table(p: *Probe, copy: usize) ![]Seg {
        const t = try p.a.alloc(Seg, most);
        for (0..p.count_streams) |j| {
            const r = p.caches[copy][j];
            const addr = r.buf.gpuAddress() + r.off;
            const wide = p.streams[j] > 8;
            t[j] = .{ .base = @intCast(p.base(j)), .rows = @intCast(p.streams[j]), .len = @intCast(p.len(j)), .cap = @intCast(p.cap(j)), .subs = if (wide) 1 else subs };
            if (p.which == .attention) {
                t[j].keys = addr;
                t[j].values = addr + kvh * p.cap(j) * hd * 2;
            } else t[j].conv = addr;
        }
        return t;
    }

    pub fn run(p: *Probe) !void {
        if (p.which == .conv) return p.runConv(p.y, p.snap, false);
        p.fill(0);
        const n = p.rows;
        const t = try p.table(0);
        defer p.a.free(t);
        var rowseg: [most]u32 = undefined;
        var tiles: [most]u32 = undefined;
        var count_tiles: u32 = 0;
        var used: u32 = 0;
        for (0..p.count_streams) |j| {
            @memset(rowseg[p.base(j)..][0..p.streams[j]], @intCast(j));
            for (0..(p.streams[j] + 7) / 8) |i| {
                tiles[count_tiles] = @intCast((j << 16) | i);
                count_tiles += 1;
            }
            const s: usize = if (p.streams[j] > 8) 1 else subs;
            used = @max(used, @as(u32, @intCast((p.len(j) + p.streams[j] + chunk_keys - 1) / chunk_keys * s)));
        }
        const segs = p.work[0];
        @memcpy(p.rig.bytes(segs, most * @sizeOf(Seg)), std.mem.sliceAsBytes(t));
        const rs = p.work[1];
        @memcpy(p.rig.bytes(rs, most * 4), std.mem.asBytes(&rowseg));
        const ts = p.work[2];
        @memcpy(p.rig.bytes(ts, most * 4), std.mem.asBytes(&tiles));
        const queries = p.queries;
        const pm = p.work[3];
        const pl = p.work[4];
        const po = p.work[5];
        const pass = p.rig.begin();
        const e = pass.e;
        for (p.caches[0][0..p.count_streams]) |r| e.useResource(r.buf, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        e.setPipeline(p.pipes[0]);
        for (p.args[0..3], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBuffer(p.args[3].buf, p.args[3].off, 3);
        e.setBuffer(p.args[3].buf, p.args[3].off + hd * 2, 4);
        const qk = [4]u32{ 0, 0, @bitCast(@as(f32, 1e-5)), @bitCast(std.math.log2(@as(f32, 1e6))) };
        e.setBytes(std.mem.asBytes(&qk), 5);
        e.setBuffer(queries.buf, queries.off, 6);
        e.setBuffer(segs.buf, segs.off, 7);
        e.setBuffer(rs.buf, rs.off, 8);
        e.dispatchGroups(mtl.Size.of(qh + kvh, n, 1), mtl.Size.of(32, 1, 1));
        const args = [4]u32{ 0, 0, @bitCast(1.0 / @sqrt(@as(f32, hd))), 0 };
        e.setPipeline(p.pipes[1]);
        e.setBuffer(queries.buf, queries.off, 0);
        e.setBuffer(segs.buf, segs.off, 1);
        e.setBuffer(ts.buf, ts.off, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        for ([_]rig_.Ref{ pm, pl, po }, 4..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.dispatchGroups(mtl.Size.of(count_tiles, kvh, used), mtl.Size.of(128, 1, 1));
        e.setPipeline(p.pipes[2]);
        for ([_]rig_.Ref{ pm, pl, po }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(p.y.buf, p.y.off, 4);
        e.setBuffer(segs.buf, segs.off, 5);
        e.setBuffer(rs.buf, rs.off, 6);
        e.dispatchGroups(mtl.Size.of(qh, n, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }

    /// conv: the batched kernel, or (`alone`) lfm2_conv a stream, into y and snap.
    fn runConv(p: *Probe, y: rig_.Ref, snap: rig_.Ref, alone: bool) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        if (alone) {
            e.setPipeline(p.pipes[1]);
            for (0..p.count_streams) |j| {
                const b = p.base(j);
                e.setBuffer(p.args[0].buf, p.args[0].off + b * 3 * d * 2, 0);
                e.setBuffer(p.caches[0][j].buf, p.caches[0][j].off, 1);
                e.setBuffer(p.args[1].buf, p.args[1].off, 2);
                e.setBuffer(y.buf, y.off + b * d * 2, 3);
                e.setBuffer(snap.buf, snap.off + b * 2 * d * 2, 4);
                e.setValue(@as(u32, 0), 5);
                e.dispatchThreads(mtl.Size.of(d, p.streams[j], 1), mtl.Size.of(256, 1, 1));
            }
            return p.rig.finish(pass);
        }
        const t = try p.table(0);
        defer p.a.free(t);
        var rowseg: [most]u32 = undefined;
        for (0..p.count_streams) |j| @memset(rowseg[p.base(j)..][0..p.streams[j]], @intCast(j));
        const segs = try p.rig.scratch(most * @sizeOf(Seg));
        @memcpy(p.rig.bytes(segs, most * @sizeOf(Seg)), std.mem.sliceAsBytes(t));
        const rs = try p.rig.scratch(most * 4);
        @memcpy(p.rig.bytes(rs, most * 4), std.mem.asBytes(&rowseg));
        for (p.caches[0][0..p.count_streams]) |r| e.useResource(r.buf, mtl.ResourceUsage.read);
        e.setPipeline(p.pipes[0]);
        e.setBuffer(p.args[0].buf, p.args[0].off, 0);
        e.setBuffer(segs.buf, segs.off, 1);
        e.setBuffer(p.args[1].buf, p.args[1].off, 2);
        e.setBuffer(y.buf, y.off, 3);
        e.setBuffer(snap.buf, snap.off, 4);
        e.setBuffer(rs.buf, rs.off, 5);
        e.dispatchThreads(mtl.Size.of(d, p.rows, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    /// The per-stream kernels' bytes on the same inputs.
    pub fn replaced(p: *Probe) ![]const u8 {
        const n = p.rows;
        if (p.which == .conv) {
            const y = try p.rig.scratch(n * d * 2);
            const snap = try p.rig.scratch(n * 2 * d * 2);
            try p.runConv(y, snap, true);
            const out = try p.a.alloc(u8, n * 3 * d * 2);
            @memcpy(out[0 .. n * d * 2], p.rig.bytes(y, n * d * 2));
            @memcpy(out[n * d * 2 ..], p.rig.bytes(snap, n * 2 * d * 2));
            return p.keep(out);
        }
        p.fill(1);
        const queries = p.work[6];
        const y = p.work[7];
        const pm = p.work[3];
        const pl = p.work[4];
        const po = p.work[5];
        const pass = p.rig.begin();
        const e = pass.e;
        for (0..p.count_streams) |j| {
            const b = p.base(j);
            const rows = p.streams[j];
            const cache = p.caches[1][j];
            const kv_bytes = kvh * p.cap(j) * hd * 2;
            e.setPipeline(p.pipes[3]);
            e.setBuffer(p.args[0].buf, p.args[0].off + b * qh * hd * 2, 0);
            e.setBuffer(p.args[1].buf, p.args[1].off + b * kvh * hd * 2, 1);
            e.setBuffer(p.args[2].buf, p.args[2].off + b * kvh * hd * 2, 2);
            e.setBuffer(p.args[3].buf, p.args[3].off, 3);
            e.setBuffer(p.args[3].buf, p.args[3].off + hd * 2, 4);
            const qk = [4]u32{ @intCast(p.len(j)), @intCast(p.cap(j)), @bitCast(@as(f32, 1e-5)), @bitCast(std.math.log2(@as(f32, 1e6))) };
            e.setBytes(std.mem.asBytes(&qk), 5);
            e.setBuffer(queries.buf, queries.off + b * d * 2, 6);
            e.setBuffer(cache.buf, cache.off, 7);
            e.setBuffer(cache.buf, cache.off + kv_bytes, 8);
            e.dispatchGroups(mtl.Size.of(qh + kvh, rows, 1), mtl.Size.of(32, 1, 1));
            const wide = rows > 8;
            const s: u32 = if (wide) 1 else subs;
            const args = [4]u32{ @intCast(p.len(j)), @intCast(p.cap(j)), @bitCast(1.0 / @sqrt(@as(f32, hd))), s };
            e.setPipeline(p.pipes[if (wide) 5 else 4]);
            e.setBuffer(queries.buf, queries.off + b * d * 2, 0);
            e.setBuffer(cache.buf, cache.off, 1);
            e.setBuffer(cache.buf, cache.off + kv_bytes, 2);
            e.setBytes(std.mem.asBytes(&args), 3);
            for ([_]rig_.Ref{ pm, pl, po }, 4..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setValue(@as(u32, @intCast(rows)), 7);
            const per: usize = if (wide) 16 else 8;
            const used = (p.len(j) + rows + chunk_keys - 1) / chunk_keys * s;
            e.dispatchGroups(mtl.Size.of((rows + per - 1) / per, kvh, used), mtl.Size.of(128, 1, 1));
            e.setPipeline(p.pipes[6]);
            for ([_]rig_.Ref{ pm, pl, po }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setBytes(std.mem.asBytes(&args), 3);
            e.setBuffer(y.buf, y.off + b * d * 2, 4);
            e.dispatchGroups(mtl.Size.of(qh, rows, 1), mtl.Size.of(32, 1, 1));
        }
        try p.rig.finish(pass);
        return p.keep(try p.a.dupe(u8, p.rig.bytes(y, n * d * 2)));
    }

    fn keep(p: *Probe, bytes: []u8) []const u8 {
        p.a.free(p.golden);
        p.golden = bytes;
        return bytes;
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0); // the check computes its own references
        @memset(mag, 0);
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        return if (p.which == .conv) p.checkConv() else p.checkAttention();
    }

    /// Each output against float64 attention over its stream's keys, from the queries and cache the call left.
    fn checkAttention(p: *Probe) !?[]const u8 {
        const out = std.mem.bytesAsSlice(u16, p.rig.bytes(p.y, p.rows * d * 2));
        const queries = std.mem.bytesAsSlice(u16, p.rig.bytes(p.queries, p.rows * d * 2));
        const w = try p.a.alloc(f64, chunks * chunk_keys);
        defer p.a.free(w);
        for (0..p.count_streams) |j| {
            const cp = p.cap(j);
            const kv = std.mem.bytesAsSlice(u16, p.rig.bytes(p.caches[0][j], 2 * kvh * cp * hd * 2));
            for (0..p.streams[j]) |r| for (0..qh) |h| {
                const row = p.base(j) + r;
                const kh = h / (qh / kvh);
                const keys = p.len(j) + r + 1;
                var top: f64 = -std.math.inf(f64);
                for (0..keys) |k| {
                    var dot: f64 = 0;
                    for (0..hd) |i| dot += c.fromBf16(queries[(row * qh + h) * hd + i]) * c.fromBf16(kv[(kh * cp + k) * hd + i]);
                    w[k] = dot / @sqrt(@as(f64, hd));
                    top = @max(top, w[k]);
                }
                var sum: f64 = 0;
                for (w[0..keys]) |*x| {
                    x.* = @exp(x.* - top);
                    sum += x.*;
                }
                for (0..hd) |i| {
                    var o: f64 = 0;
                    var big: f64 = 0;
                    for (0..keys) |k| {
                        const v: f64 = c.fromBf16(kv[((kvh + kh) * cp + k) * hd + i]);
                        o += w[k] / sum * v;
                        big = @max(big, @abs(v));
                    }
                    const got: f64 = c.fromBf16(out[(row * qh + h) * hd + i]);
                    if (p.nan and j == 0 and kh == 0) {
                        if (!std.math.isNan(got)) return "a head behind its stream's NaN key gave a number";
                        continue;
                    }
                    if (!std.math.isFinite(got)) return "a finite head gave a non-finite output";
                    if (@abs(got - o) > big * 0x1p-7 + @abs(o) * 0x1p-8) return "an output is past its float64 bound";
                }
            };
        }
        return null;
    }

    /// Each output against the host's replay of lfm2_conv's steps (bf16 products, fp32 taps, the bf16 gate).
    fn checkConv(p: *Probe) !?[]const u8 {
        const n = p.rows;
        const y = std.mem.bytesAsSlice(u16, p.rig.bytes(p.y, n * d * 2));
        const snap = std.mem.bytesAsSlice(u16, p.rig.bytes(p.snap, n * 2 * d * 2));
        for (0..p.count_streams) |j| {
            const b = p.base(j);
            const state = p.host[n * 3 * d + j * 2 * d ..][0 .. 2 * d];
            for (0..p.streams[j]) |r| for (0..d) |ch| {
                var bx: [3]u16 = undefined;
                for (0..3) |tap| {
                    const src = @as(isize, @intCast(r + tap)) - 2;
                    if (src >= 0) {
                        const row = p.host[(b + @as(usize, @intCast(src))) * 3 * d ..];
                        bx[tap] = c.toBf16(c.fromBf16(row[ch]) * c.fromBf16(row[2 * d + ch]));
                    } else bx[tap] = state[@as(usize, @intCast(src + 2)) * d + ch];
                }
                // The taps' sum as products then adds, or as fused steps: the compiler may contract either way.
                var acc: f32 = 0;
                var fused: f32 = 0;
                for (0..3) |tap| {
                    acc += c.fromBf16(bx[tap]) * c.fromBf16(p.weights[ch * 3 + tap]);
                    fused = @mulAdd(f32, c.fromBf16(bx[tap]), c.fromBf16(p.weights[ch * 3 + tap]), fused);
                }
                const gate = c.fromBf16(p.host[(b + r) * 3 * d + d + ch]);
                const got = y[(b + r) * d + ch];
                if (!same(got, c.toBf16(gate * c.fromBf16(c.toBf16(acc)))) and !same(got, c.toBf16(gate * c.fromBf16(c.toBf16(fused)))))
                    return "a conv output differs from the host's replay";
                if (!same(snap[((b + r) * 2) * d + ch], bx[1]) or !same(snap[((b + r) * 2 + 1) * d + ch], bx[2])) return "a kept state differs from the host's replay";
            };
        }
        return null;
    }
};

fn same(x: u16, y: u16) bool {
    const nx = std.math.isNan(c.fromBf16(x));
    return if (nx) std.math.isNan(c.fromBf16(y)) else x == y;
}
