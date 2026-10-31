//! LFM2's block attention (lfm2_attn_block, then lfm2_attn_block_merge): every row of a drafter's block sees all keys.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const controls = @import("metal_lfm2_controls.zig");

const qh = 32;
const kvh = 8;
const hd = 64;
const subs = 8; // the drafter's sub-spans a 4,096-key chunk (state.subs)
const chunks = 32;
const chunk_keys = 4096;
pub const name = "metal lfm2 block attention";

/// rows: a block's rows; k: the keys every row sees (a context and the block past it).
const shapes = [_]c.Shape{ .{ .rows = 1, .n = qh * hd, .k = 9 }, .{ .rows = 3, .n = qh * hd, .k = 700 }, .{ .rows = 9, .n = qh * hd, .k = 4500 } };

fn depth(s: c.Shape) usize {
    return s.k;
}

pub fn spec() c.Spec {
    return .{ .name = name, .accum = .none, .depth = depth, .out = .bf16, .served = &shapes, .nan = .{ .stated = "a NaN key of a kv head makes its query heads' outputs NaN, the others stay finite" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "the check bounds each output by its bf16 probabilities' rounding, not a sum's bits" } };
}

fn value(i: usize, seed: u64, scale: f32) u16 {
    const h: u64 = (@as(u64, i) *% 0x9e3779b97f4a7c15 +% seed *% 0xbf58476d1ce4e5b9) >> 40;
    return c.toBf16((@as(f32, @floatFromInt(h % 2048)) / 1024 - 1) * scale);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    mutation: controls.Mutation,
    pipes: [2]mtl.Pipeline,
    shape: c.Shape = undefined,
    cap: usize = 0,
    nan: bool = false,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    args: [5]rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mutation: controls.Mutation) !Probe {
        const text = try controls.source(a, mutation);
        defer if (mutation != .none) a.free(text);
        const first = try rig_.pipeline(device, text, "lfm2_attn_block");
        errdefer first.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = .{ first, try rig_.pipeline(device, text, "lfm2_attn_block_merge") }, .mutation = mutation };
    }

    fn drop(p: *Probe) void {
        inline for (.{ "q", "k", "v" }) |f| {
            p.a.free(@field(p, f));
            @field(p, f) = &.{};
        }
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        for (p.pipes) |x| x.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec();
        if (p.mutation != .none) s.name = p.mutation.name();
        return .{ .ctx = p, .vt = &vt, .spec = s, .control = p.mutation.gate() };
    }

    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        if (s.n != qh * hd or s.rows == 0 or s.rows > 16 or s.k < s.rows or s.k > chunks * chunk_keys) return false;
        p.drop();
        p.shape = s;
        p.nan = f == .nonfinite;
        p.cap = (s.k + 63) / 64 * 64 + 32; // keys past `k` hold finite values, as a lane's cache does
        const padded = (s.rows + 7) / 8 * 8 + 8; // the kernel reads a tile's 8 rows whatever its row count
        p.q = try p.a.alloc(u16, padded * qh * hd);
        p.k = try p.a.alloc(u16, kvh * p.cap * hd);
        p.v = try p.a.alloc(u16, kvh * p.cap * hd);
        for (p.q, 0..) |*x, i| x.* = value(i, seed, 1);
        for (p.k, 0..) |*x, i| x.* = value(i, seed +% 1, if (f == .stress) 4 else 1);
        for (p.v, 0..) |*x, i| x.* = value(i, seed +% 2, 2);
        if (p.nan) p.k[(0 * p.cap + s.k / 2) * hd + 5] = 0x7fc0; // kv head 0, a visible key
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.args[1] = try p.rig.input(std.mem.sliceAsBytes(p.k));
        p.args[2] = try p.rig.input(std.mem.sliceAsBytes(p.v));
        const Args = extern struct { len: u32, cap: u32, scale: f32, subs: u32 };
        const args = Args{ .len = @intCast(s.k - 1), .cap = @intCast(p.cap), .scale = 1.0 / @sqrt(@as(f32, hd)), .subs = subs };
        p.args[3] = try p.rig.input(std.mem.asBytes(&args));
        const n: u32 = @intCast(s.rows);
        p.args[4] = try p.rig.input(std.mem.asBytes(&n));
        p.y = try p.rig.output(p.count() * 2);
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.shape.rows * qh * hd;
    }

    pub fn run(p: *Probe) !void {
        const rows = p.shape.rows;
        const partials = @max(rows, 8) * subs * qh * chunks;
        const pm = try p.rig.scratch(partials * 4);
        const pl = try p.rig.scratch(partials * 4);
        const po = try p.rig.scratch(partials * hd * 4);
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipes[0]);
        for (p.args[0..4], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        for ([_]rig_.Ref{ pm, pl, po, p.args[4] }, 4..) |r, i| e.setBuffer(r.buf, r.off, i);
        const used = (p.shape.k + chunk_keys - 1) / chunk_keys * subs;
        e.dispatchGroups(mtl.Size.of((rows + 7) / 8, kvh, used), mtl.Size.of(128, 1, 1));
        e.setPipeline(p.pipes[1]);
        for ([_]rig_.Ref{ pm, pl, po, p.args[3], p.y }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.dispatchGroups(mtl.Size.of(qh, rows, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0); // the check computes its own references
        @memset(mag, 0);
    }

    /// Every output against float64 attention: within 2^-7 of its values' largest magnitude and 2^-8 of itself.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const s = p.shape;
        const out = std.mem.bytesAsSlice(u16, p.rig.bytes(p.y, p.count() * 2));
        const w = try p.a.alloc(f64, s.k);
        defer p.a.free(w);
        for (0..s.rows) |r| for (0..qh) |h| {
            const kv = h / (qh / kvh);
            const nan_head = p.nan and kv == 0;
            var top: f64 = -std.math.inf(f64);
            for (0..s.k) |j| {
                var dot: f64 = 0;
                for (0..hd) |d| dot += c.fromBf16(p.q[(r * qh + h) * hd + d]) * c.fromBf16(p.k[(kv * p.cap + j) * hd + d]);
                w[j] = dot / @sqrt(@as(f64, hd));
                top = @max(top, w[j]);
            }
            var sum: f64 = 0;
            for (w) |*x| {
                x.* = @exp(x.* - top);
                sum += x.*;
            }
            for (0..hd) |d| {
                var o: f64 = 0;
                var big: f64 = 0;
                for (0..s.k) |j| {
                    const vv: f64 = c.fromBf16(p.v[(kv * p.cap + j) * hd + d]);
                    o += w[j] / sum * vv;
                    big = @max(big, @abs(vv));
                }
                const got: f64 = c.fromBf16(out[(r * qh + h) * hd + d]);
                if (nan_head) {
                    if (!std.math.isNan(got)) return "a head behind a NaN key gave a number";
                    continue;
                }
                if (!std.math.isFinite(got)) return "a finite head gave a non-finite output";
                if (@abs(got - o) > big * 0x1p-7 + @abs(o) * 0x1p-8) return "an output is past its float64 bound";
            }
        };
        return null;
    }
};
