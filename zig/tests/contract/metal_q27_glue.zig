//! Flat GDN preparation and MLP rows compare every output byte with their original production kernels.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const Params = contract.delta_slots_ref.Params;
pub const Kind = enum { pre, mlp, mlp_xs };
fn none(_: c.Shape) usize {
    return 0;
}
const pre_shapes = [_]c.Shape{ .{ .rows = 1, .n = 128, .k = 0 }, .{ .rows = 3, .n = 128, .k = 96 }, .{ .rows = 16, .n = 128, .k = 0 }, .{ .rows = 2048, .n = 128, .k = 96 } };
const mlp_shapes = [_]c.Shape{ .{ .rows = 1, .n = 17408, .k = 1 }, .{ .rows = 15, .n = 17408, .k = 1 }, .{ .rows = 17, .n = 64, .k = 1 }, .{ .rows = 128, .n = 64, .k = 1 }, .{ .rows = 2048, .n = 64, .k = 1 } };
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    bad: bool,
    name: []const u8,
    pipe: ?mtl.Pipeline = null,
    old: ?mtl.Pipeline = null,
    shape: c.Shape = undefined,
    p: Params = undefined,
    ins: std.ArrayList(rig_.Ref) = .empty,
    lens: std.ArrayList(usize) = .empty,
    outs: std.ArrayList(rig_.Ref) = .empty,
    bases: std.ArrayList(rig_.Ref) = .empty,
    joined: std.ArrayList(u8) = .empty,
    baseline: std.ArrayList(u8) = .empty,

    pub fn init(a: std.mem.Allocator, d: mtl.Device, q: mtl.Queue, kind: Kind, bad: bool, name: []const u8) Probe {
        return .{ .a = a, .rig = .{ .a = a, .device = d, .queue = q }, .kind = kind, .bad = bad, .name = name };
    }
    fn reset(p: *Probe) void {
        p.rig.reset();
        p.ins.clearRetainingCapacity();
        p.outs.clearRetainingCapacity();
        p.bases.clearRetainingCapacity();
        p.lens.clearRetainingCapacity();
        if (p.pipe) |v| v.deinit();
        if (p.old) |v| v.deinit();
        p.pipe = null;
        p.old = null;
    }
    pub fn deinit(p: *Probe) void {
        p.reset();
        p.ins.deinit(p.a);
        p.outs.deinit(p.a);
        p.bases.deinit(p.a);
        p.lens.deinit(p.a);
        p.joined.deinit(p.a);
        p.baseline.deinit(p.a);
        p.rig.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = .{
            .name = p.name,
            .accum = .none,
            .out = .raw,
            .depth = none,
            .served = if (p.kind == .pre) &pre_shapes else &mlp_shapes,
            .tiles = if (p.kind == .pre) .{ .{ .size = 16, .min = 1 }, .{}, .{} } else .{ .{ .size = 16, .min = 1 }, .{ .size = 64, .step = 64, .min = 64 }, .{} },
            .replaces = if (p.kind == .pre) "tf_delta_gdn_pre" else "custom_kernel_q27_mlp_xs with identical Q27_XS",
            .nan = .{ .stated = "all bytes, including planted non-finite outputs and untouched XS when off, equal the original" },
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "replacement bytes preserve the original rounding and sums; no independent stress claim" },
        }, .ctx = p, .vt = &vtable, .control = if (p.bad) .shapes else null };
    }
    const vtable = blk: {
        var vt = g.hooks(Probe);
        vt.output = output;
        break :blk vt;
    };
    fn input(p: *Probe, bytes: []const u8) !void {
        try p.ins.append(p.a, try p.rig.input(bytes));
    }
    fn bf(p: *Probe, n: usize, fill: g.Fill, seed: u64, plant: bool) !void {
        const data = try p.a.alloc(u16, n);
        defer p.a.free(data);
        var rng = std.Random.DefaultPrng.init(seed);
        for (data) |*x| x.* = c.toBf16(rng.random().float(f32) * 2 - 1);
        if (fill == .nonfinite and plant and n > 128) {
            data[0] = c.toBf16(std.math.nan(f32));
            data[128] = c.toBf16(std.math.inf(f32));
        }
        try p.input(std.mem.sliceAsBytes(data));
    }
    fn out(p: *Probe, bytes: usize) !void {
        try p.lens.append(p.a, bytes);
        try p.outs.append(p.a, try p.rig.output(bytes));
        try p.bases.append(p.a, try p.rig.scratch(bytes));
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > 2048) return false;
        if (p.kind == .pre and (s.n != 128 or (s.k != 0 and s.k != 96))) return false;
        if (p.kind != .pre and (s.n == 0 or s.n % 64 != 0)) return false;
        p.reset();
        p.shape = s;
        if (p.kind == .pre) {
            p.p = .{ .rows = @intCast(s.rows), .zba_stride = @intCast(s.k) };
            const channels = 80 * 128;
            try p.bf(s.rows * channels, fill, seed, true);
            const gates = s.rows * @max(s.k, 48);
            try p.bf(gates, fill, seed +% 1, true);
            try p.bf(gates, fill, seed +% 2, true);
            try p.bf(3 * 3 * channels, fill, seed +% 3, false);
            try p.bf(channels * 4, fill, seed +% 4, false);
            try p.bf(48, fill, seed +% 5, false);
            try p.bf(48, fill, seed +% 6, false);
            const windows = try p.a.alloc(u32, s.rows * 4);
            defer p.a.free(windows);
            for (windows, 0..) |*v, i| v.* = @intCast(i / 4 + i % 4);
            try p.input(std.mem.sliceAsBytes(windows));
            const slots = try p.a.alloc(u32, s.rows);
            defer p.a.free(slots);
            for (slots, 0..) |*v, i| v.* = @intCast(i % 3);
            try p.input(std.mem.sliceAsBytes(slots));
            for ([_]usize{ s.rows * 16 * 128 * 2, s.rows * 16 * 128 * 2, s.rows * 48 * 128 * 2, s.rows * 48 * 4, s.rows * 48 * 2 }) |bytes| try p.out(bytes);
            const text = if (p.bad) try std.mem.replaceOwned(u8, p.a, sources.core_deltanet, "value[(row * p.nv + output_head) * 32 + lane] =", "value[(row * p.nv + output_head) * 32 + lane] = vec<bfloat, 4>(0) *") else try p.a.dupe(u8, sources.core_deltanet);
            defer p.a.free(text);
            p.pipe = try rig_.pipeline(p.rig.device, text, "tf_delta_gdn_pre_vectors");
            p.old = try rig_.pipeline(p.rig.device, sources.core_deltanet, "tf_delta_gdn_pre");
        } else {
            try p.bf(s.rows * 2 * s.n, fill, seed, true);
            const dims = [2]i32{ @intCast(s.rows), @intCast((s.rows + 15) / 16 * 16) };
            try p.input(std.mem.asBytes(&dims));
            try p.out(s.rows * s.n * 2);
            try p.out(s.n / 64 * @as(usize, @intCast(dims[1])) * 4);
            const text = try tf.qwen27.quant_gpu.specializedSource(p.a, .{ .kind = .mlp_rows, .k = @intCast(s.n), .n = @intCast(s.n), .xs = p.kind == .mlp_xs });
            defer p.a.free(text);
            const edit = if (p.bad) try std.mem.replaceOwned(u8, p.a, text, "HOUT[m * (width / 4) + four] = rounded;", "HOUT[m * (width / 4) + four] = vec<bfloat, 4>(0);") else try p.a.dupe(u8, text);
            defer p.a.free(edit);
            p.pipe = try rig_.pipeline(p.rig.device, edit, "q27_mlp_rows");
            p.old = try rig_.pipeline(p.rig.device, text, "custom_kernel_q27_mlp_xs_bfloat16_t_bfloat16_t_int32_t_bfloat16_t_float");
        }
        return true;
    }
    fn launch(p: *Probe, old: bool) !void {
        const targets = if (old) p.bases.items else p.outs.items;
        for (targets, p.lens.items) |b, n| @memset(p.rig.bytes(b, n), 0xff);
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(if (old) p.old.? else p.pipe.?);
        if (p.kind == .pre) {
            for (p.ins.items, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
            e.setValue(p.p, 9);
            for (targets, 10..) |b, i| e.setBuffer(b.buf, b.off, i);
            e.dispatchGroups(mtl.Size.of(1, 80, p.shape.rows), mtl.Size.of(32, 1, 1));
        } else {
            const fused = p.ins.items[0];
            e.setBuffer(fused.buf, fused.off, 0);
            e.setBuffer(fused.buf, fused.off + p.shape.n * 2, 1);
            const dims = p.ins.items[1];
            e.setBuffer(dims.buf, dims.off, 2);
            for (targets, 3..) |b, i| e.setBuffer(b.buf, b.off, i);
            const padded = (p.shape.rows + 15) / 16 * 16;
            if (!old) e.setValue(@as(u32, @intCast(p.shape.n)), 5);
            e.dispatchGroups(mtl.Size.of(p.shape.n / 64, if (!old) padded / 16 else if (p.kind == .mlp_xs) padded else p.shape.rows, 1), mtl.Size.of(if (old) 64 else 256, 1, 1));
        }
        try p.rig.finish(pass);
    }
    pub fn run(p: *Probe) !void {
        try p.launch(false);
    }
    fn gather(p: *Probe, refs: []const rig_.Ref, list: *std.ArrayList(u8)) ![]const u8 {
        list.clearRetainingCapacity();
        for (refs, p.lens.items) |b, n| try list.appendSlice(p.a, p.rig.bytes(b, n));
        return list.items;
    }
    fn output(ctx: *anyopaque) ![]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        return p.gather(p.outs.items, &p.joined);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        try p.launch(true);
        return p.gather(p.bases.items, &p.baseline);
    }
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        if (!std.mem.eql(u8, try output(p), try p.replaced())) return "flat glue differs from original output bytes";
        return null;
    }
    pub fn count(p: *Probe) usize {
        var n: usize = 0;
        for (p.lens.items) |bytes| n += bytes;
        return n;
    }
    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
};
