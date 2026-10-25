//! cos3_qk under the contract: interleaved M-RoPE [24, 20, 20] on queries and keys, keys and values into cache slots.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const ref = @import("cosmos3_ref.zig");

const c = contract.spec;
const g = contract.gates;

const qh = 16;
const kvh = 8;
const hd = 128;
const log2_theta: f32 = 26.575424759098897; // log2(1e8), Cosmos3-Edge's rope_theta

pub const Which = enum { qk, ctl_axes };

const fault = [2][]const u8{ "if (j % 3 == 1 && j < 3 * SEC_H) return 1;", "if (j % 3 == 1 && j < 3 * SEC_H) return 2;" };

fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec(w: Which) c.Spec {
    const name = if (w == .qk) "metal cosmos3 cos3_qk (interleaved M-RoPE, keys and values to cache slots)" else "control: cosmos3 M-RoPE turning h pairs at the w axis's position";
    return .{ .name = name, .accum = .none, .depth = none, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 128, .k = 1 }, .{ .rows = 64, .n = 128, .k = 1 } } };
}

/// The axis of rotary pair j (0 t, 1 h, 2 w), as cos3_axis.
fn axis(j: usize) usize {
    if (j % 3 == 1 and j < 60) return 1;
    if (j % 3 == 2 and j < 60) return 2;
    return 0;
}

const Args = extern struct { cap: u32, log2_theta: f32 };

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    which: Which,
    rows: usize = 0,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    pos: []i32 = &.{},
    slots: []u32 = &.{},
    point: []f64 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    in: [5]rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which) !Probe {
        const src = if (which == .qk) try a.dupe(u8, sources.cosmos3) else try std.mem.replaceOwned(u8, a, sources.cosmos3, fault[0], fault[1]);
        defer a.free(src);
        if (which != .qk and std.mem.indexOf(u8, sources.cosmos3, fault[0]) == null) return error.ControlLineMissing;
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, "cos3_qk"), .which = which };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.pipe.deinit();
        p.arena.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.which), .ctx = p, .vt = &vt, .control = if (p.which == .qk) null else .shapes };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const ar = p.arena.allocator();
        const r = shape.rows;
        p.rows = r;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.q = try ar.alloc(u16, r * qh * hd);
        p.k = try ar.alloc(u16, r * kvh * hd);
        p.v = try ar.alloc(u16, r * kvh * hd);
        for ([_][]u16{ p.q, p.k, p.v }) |t| for (t) |*x| {
            x.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        };
        p.pos = try ar.alloc(i32, 3 * r);
        p.slots = try ar.alloc(u32, r);
        const first: i32 = if (fill == .nonfinite) 0 else 3000 + @as(i32, @intCast(seed % 97));
        for (0..r) |i| {
            const t = first + @as(i32, @intCast(i));
            const image = i % 2 == 1; // image rows: t held, h and w on the grid
            p.pos[3 * i ..][0..3].* = if (image) .{ t, t + @as(i32, @intCast(i % 7)), t + @as(i32, @intCast(i % 5)) + 2 } else .{ t, t, t };
            p.slots[i] = @intCast(r - 1 - i);
        }
        if (fill == .nonfinite) {
            const nan = c.toBf16(std.math.nan(f32));
            const inf = c.toBf16(std.math.inf(f32));
            const ninf = c.toBf16(-std.math.inf(f32));
            p.q[3] = nan; // row 0, head 0
            if (r > 1) p.k[kvh * hd + 5] = inf; // row 1, key head 0
            if (r > 2) p.v[2 * kvh * hd + 9] = ninf;
            if (r > 3) {
                p.q[3 * qh * hd + hd + 7] = inf;
                p.q[3 * qh * hd + hd + 7 + 64] = ninf;
            }
        }
        const bytes = std.mem.sliceAsBytes;
        p.in[0] = try p.rig.input(bytes(p.q));
        p.in[1] = try p.rig.input(bytes(p.k));
        p.in[2] = try p.rig.input(bytes(p.v));
        p.in[3] = try p.rig.input(bytes(p.pos));
        p.in[4] = try p.rig.input(bytes(p.slots));
        p.out[0] = try p.rig.output(p.q.len * 2);
        p.out[1] = try p.rig.output(p.k.len * 2);
        p.out[2] = try p.rig.output(p.v.len * 2);
        try p.compute();
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |x, i| e.setBuffer(x.buf, x.off, i);
        e.setValue(Args{ .cap = @intCast(p.rows), .log2_theta = log2_theta }, 5);
        for (p.out, 6..) |x, i| e.setBuffer(x.buf, x.off, i);
        e.dispatchGroups(mtl.Size.of(qh + kvh, p.rows, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.q.len + p.k.len + p.v.len;
    }

    /// One head's 128 values at a row's positions: rotated pairs (j, j + 64) with their bf16 ends.
    fn turn(p: *Probe, src: []const u16, row: usize, dst: usize) void {
        for (0..64) |j| {
            var o: [2]f64 = undefined;
            var h: [2]f64 = undefined;
            ref.rotate(c.fromBf16(src[j]), c.fromBf16(src[j + 64]), p.pos[3 * row + axis(j)], j, log2_theta, &o, &h);
            for ([2]usize{ j, j + 64 }, o, h) |at, y, half| {
                p.point[dst + at] = y;
                ref.ends(y, half, &p.lo[dst + at], &p.hi[dst + at]);
            }
        }
    }

    fn compute(p: *Probe) !void {
        const ar = p.arena.allocator();
        const n = p.count();
        p.point = try ar.alloc(f64, n);
        p.lo = try ar.alloc(f64, n);
        p.hi = try ar.alloc(f64, n);
        const kv0 = p.q.len;
        const v0 = kv0 + p.k.len;
        for (0..p.rows) |r| {
            for (0..qh) |h| p.turn(p.q[(r * qh + h) * hd ..][0..hd], r, (r * qh + h) * hd);
            for (0..kvh) |h| {
                const dst = (h * p.rows + p.slots[r]) * hd;
                p.turn(p.k[(r * kvh + h) * hd ..][0..hd], r, kv0 + dst);
                for (0..hd) |d| {
                    const x = c.fromBf16(p.v[(r * kvh + h) * hd + d]);
                    p.point[v0 + dst + d] = x;
                    p.lo[v0 + dst + d] = x;
                    p.hi[v0 + dst + d] = x;
                }
            }
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        @memcpy(want, p.point);
        @memset(mag, 0);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, p.lo, p.hi) |*l, w, lo, hi| l.* = @max(@abs(lo - w), @abs(hi - w));
    }

    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        @memcpy(low, p.lo);
        @memcpy(high, p.hi);
    }
};
