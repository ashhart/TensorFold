//! Bolt's legal-row projection compared byte for byte with its full head and the same grammar mask.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const bolt = @import("metal_bolt.zig");
const rig = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;

pub fn spec() c.Spec {
    return .{
        .name = "metal bolt gemv tf_gemv_bf16_bm8_subset",
        .replaces = bolt.gemvSpec(.bm8).name,
        .accum = .f32,
        .depth = bolt.gemvDepth,
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = 32, .step = 4, .min = 4 }, .{ .size = 128, .step = 1, .min = 1 } },
        .served = &.{
            .{ .rows = 4, .n = 64, .k = 4096 },
            .{ .rows = 1, .n = 4, .k = 4096 },
            .{ .rows = 1, .n = 2048, .k = 4096 },
            .{ .rows = 1, .n = 131072, .k = 4096 },
        },
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    base: bolt.GemvProbe,
    rig: *rig.Rig = undefined,
    sparse: mtl.Pipeline,
    mask: mtl.Pipeline,
    ids: rig.Ref = undefined,
    bits: rig.Ref = undefined,
    oracle: rig.Ref = undefined,
    legal: usize = 0,
    words: usize = 0,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Probe {
        var base = try bolt.GemvProbe.init(a, device, queue, .bm8, null);
        errdefer base.deinit();
        const sparse = try rig.pipeline(device, bolt.opSource("gemv_ops"), "tf_gemv_bf16_bm8_subset");
        errdefer sparse.deinit();
        const mask = try rig.pipeline(device, sources.core_grammar_mask, "tf_grammar_mask");
        return .{ .a = a, .base = base, .sparse = sparse, .mask = mask };
    }

    pub fn deinit(p: *Probe) void {
        p.mask.deinit();
        p.sparse.deinit();
        p.base.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        p.rig = &p.base.rig;
        return .{ .spec = spec(), .ctx = p, .vt = &vt };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try p.base.prepare(shape, fill, seed)) return false;
        p.legal = if (shape.n < 8) 1 else if (shape.n < 1024) 7 else 512;
        p.words = (shape.n + 31) / 32;
        var ids: [512]u32 = undefined;
        const words = try p.a.alloc(u32, shape.rows * p.words);
        defer p.a.free(words);
        @memset(words, 0);
        for (ids[0..p.legal], 0..) |*id, i| {
            const t = shape.n - 1 - i * (shape.n / p.legal);
            id.* = @intCast(t);
            for (0..shape.rows) |r| words[r * p.words + t / 32] |= @as(u32, 1) << @as(u5, @truncate(t));
        }
        p.ids = try p.base.rig.input(std.mem.sliceAsBytes(ids[0..p.legal]));
        p.bits = try p.base.rig.input(std.mem.sliceAsBytes(words));
        p.oracle = try p.base.rig.scratch(p.count() * 2);
        return true;
    }

    fn applyMask(p: *Probe, e: mtl.ComputeEncoder, dst: rig.Ref) void {
        const s = p.base.d.?.shape;
        e.setPipeline(p.mask);
        e.setBuffer(dst.buf, dst.off, 0);
        e.setBuffer(p.bits.buf, p.bits.off, 1);
        e.setValue([4]u32{ @intCast(s.n), @intCast(p.words), @intCast(s.rows), 0 }, 2);
        e.dispatchThreads(mtl.Size.of(p.words, s.rows, 1), mtl.Size.of(@min(p.words, 256), 1, 1));
    }

    pub fn run(p: *Probe) !void {
        const s = p.base.d.?.shape;
        const pass = p.base.rig.begin();
        const e = pass.e;
        e.setPipeline(p.sparse);
        e.setBuffer(p.base.w.buf, p.base.w.off, 0);
        e.setValue([3]i32{ @intCast(s.k), @intCast(p.legal), @intCast(s.k) }, 2);
        e.setBuffer(p.ids.buf, p.ids.off, 4);
        for (0..s.rows) |r| {
            e.setBuffer(p.base.x.buf, p.base.x.off + r * s.k * 2, 1);
            e.setBuffer(p.base.out.buf, p.base.out.off + r * s.n * 2, 3);
            e.dispatchGroups(mtl.Size.of((p.legal + 31) / 32, 1, 1), mtl.Size.of(32, 1, 8));
        }
        p.applyMask(e, p.base.out);
        try p.base.rig.finish(pass);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        const pass = p.base.rig.begin();
        p.base.encode(pass.e, p.oracle);
        p.applyMask(pass.e, p.oracle);
        try p.base.rig.finish(pass);
        return p.base.rig.bytes(p.oracle, p.count() * 2);
    }

    pub fn count(p: *Probe) usize {
        return p.base.count();
    }

    fn maskReference(p: *Probe, vals: []f64) void {
        const s = p.base.d.?.shape;
        const bits = p.bits.buf.slice(u32, p.bits.off / 4 + p.words * s.rows)[p.bits.off / 4 ..];
        for (vals, 0..) |*v, i| {
            const t = i % s.n;
            if (bits[(i / s.n) * p.words + t / 32] & (@as(u32, 1) << @as(u5, @truncate(t))) == 0)
                v.* = -std.math.inf(f64);
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.base.reference(want, mag);
        p.maskReference(want);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.base.simulate(acc, order, out);
        p.maskReference(out);
    }
};
