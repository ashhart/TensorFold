//! LFM2's greedy pick of every verify row (lfm2_argmax): the lowest id of a row's largest bf16 logit, and a flag.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const controls = @import("metal_lfm2_controls.zig");

const vocab = 128000;
pub const name = "metal lfm2 argmax";

/// rows: a verify's head rows, one to a shared round's 32.
const shapes = [_]c.Shape{ .{ .rows = 1, .n = vocab, .k = 1 }, .{ .rows = 9, .n = vocab, .k = 1 }, .{ .rows = 32, .n = vocab, .k = 1 } };

fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec() c.Spec {
    return .{ .name = name, .accum = .none, .depth = none, .out = .raw, .served = &shapes, .nan = .{ .stated = "a row with a NaN or an infinity is flagged nonfinite" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "picks compare values: no sums" } };
}

/// A bf16 logit for element `i`: few distinct values (ties in every row); the nonfinite fill plants NaN and infinities.
fn value(i: usize, seed: u64, fill: g.Fill) u16 {
    const h: u64 = (@as(u64, i) *% 0x9e3779b97f4a7c15 +% seed *% 0xbf58476d1ce4e5b9) >> 40;
    if (fill == .nonfinite and h % 50021 < 3) return ([3]u16{ 0x7fc0, 0x7f80, 0xff80 })[h % 50021];
    const v = @as(f32, @floatFromInt(h % 61)) / 8 - 4; // -4 to 3.5 in eighths, signed zeros among them
    return if (v == 0 and (h >> 20) & 1 == 1) 0x8000 else c.toBf16(v);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mutation: controls.Mutation,
    rows: usize = 0,
    x: []u16 = &.{},
    args: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mutation: controls.Mutation) !Probe {
        const text = try controls.source(a, mutation);
        defer if (mutation != .none) a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, text, "lfm2_argmax"), .mutation = mutation };
    }

    fn drop(p: *Probe) void {
        p.a.free(p.x);
        p.x = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec();
        if (p.mutation != .none) s.name = p.mutation.name();
        return .{ .ctx = p, .vt = &vt, .spec = s, .control = p.mutation.gate() };
    }

    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        p.drop();
        if (s.n != vocab or s.k != 1 or s.rows == 0 or s.rows > 32) return false;
        p.rows = s.rows;
        p.x = try p.a.alloc(u16, s.rows * vocab);
        for (p.x, 0..) |*v, i| v.* = value(i, seed, f);
        // Row 0's best value sits at two ids: the pick is the lower one.
        p.x[0 * vocab + 70001] = c.toBf16(5);
        p.x[0 * vocab + 1234] = c.toBf16(5);
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        const n: u32 = vocab;
        p.args[1] = try p.rig.input(std.mem.asBytes(&n));
        p.out = try p.rig.output(p.count());
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.rows * 8;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.args[0].buf, p.args[0].off, 0);
        e.setBuffer(p.args[1].buf, p.args[1].off, 1);
        e.setBuffer(p.out.buf, p.out.off, 2);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0); // the picks check themselves (check)
        @memset(mag, 0);
    }

    /// Each row's pick against the host's greedy draw: the first id of the largest value, or the row flagged.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const out = p.rig.bytes(p.out, p.count());
        for (0..p.rows) |r| {
            const token = std.mem.readInt(u32, out[8 * r ..][0..4], .little);
            const flag = std.mem.readInt(u32, out[8 * r + 4 ..][0..4], .little);
            var best: ?usize = null;
            var bad = false;
            for (p.x[r * vocab ..][0..vocab], 0..) |w, id| {
                const v = c.fromBf16(w);
                if (!std.math.isFinite(v)) bad = true else if (best == null or v > c.fromBf16(p.x[r * vocab + best.?])) best = id;
            }
            if (bad) {
                if (flag != 1) return "a row with a nonfinite logit is not flagged";
                continue;
            }
            if (flag != 0) return "a finite row is flagged nonfinite";
            if (token != best.?) return "a pick differs from the first id of the row's largest logit";
        }
        return null;
    }
};
