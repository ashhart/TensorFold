//! Inkling 2 Mini fused 6-projection matvec under FP64 references, bit-identity vs sp_matvec, and planted controls.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const dense_in = contract.dense;
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_dense_controls.zig");
const host = @import("tensorfold").inkling2_metal_host.dense_fuse;
const sources = @import("inkling2_metal");
const all_sources = @import("kernel_sources");

const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

fn depth(s: c.Shape) usize {
    return s.k / 32 + 8;
}

fn kernelLabel(tier: u32) []const u8 {
    return switch (tier) {
        1 => "metal inkling2 dense mv6_1 tf_ink2_mv6_1",
        2 => "metal inkling2 dense mv6_2 tf_ink2_mv6_2",
        4 => "metal inkling2 dense mv6_4 tf_ink2_mv6_4",
        8 => "metal inkling2 dense mv6_8 tf_ink2_mv6_8",
        16 => "metal inkling2 dense mv6_16 tf_ink2_mv6_16",
        32 => "metal inkling2 dense mv6_32 tf_ink2_mv6_32",
        else => unreachable,
    };
}

fn spSymbol(tier: u32) [:0]const u8 {
    return switch (tier) {
        1 => "sp_matvec_1",
        2 => "sp_matvec_2",
        4 => "sp_matvec_4",
        8 => "sp_matvec_8",
        16 => "sp_matvec_16",
        32 => "sp_matvec_32",
        else => unreachable,
    };
}

fn projDims(n: usize) [6]usize {
    if (n == 3072) return .{ 3072, 1024, 1024, 384, 24, 3072 };
    return .{ n, @max(1, n / 2 + 1), n + 3, @max(1, n / 3 + 2), @max(1, n / 4 + 1), n };
}

fn fillStressX(d: dense_in.Dense, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const h = d.shape.k / 2;
    for (0..d.shape.rows) |r| {
        const row = d.x[r * d.shape.k ..][0..d.shape.k];
        row[0] = c.toBf16(1.0);
        row[h] = c.toBf16(2.0);
        for (1..h) |i| {
            const v: f32 = @floatFromInt(64 * (1 + rng.uintLessThan(u32, 24)));
            const bits = c.toBf16(v);
            row[i] = bits;
            row[h + i] = bits;
        }
    }
}

fn fillWeights(w16: []u16, d: dense_in.Dense, s: c.Shape, fill: g.Fill, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const h = s.k / 2;
    for (0..s.n) |j| for (0..s.k) |i| {
        const v: f32 = if (fill != .stress)
            rng.float(f32) * 2 - 1
        else if (i < h)
            @floatFromInt(1 + (i * 7 + j) % 6)
        else
            -@as(f32, @floatFromInt(1 + ((i - h) * 7 + j) % 6));
        w16[j * s.k + i] = c.toBf16(v);
        d.w[j * s.k + i] = c.fromBf16(w16[j * s.k + i]);
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    tier: u32,
    control: bool,
    pipes: host.Pipelines,
    sp_pipe: ?mtl.Pipeline = null,
    served: [3]c.Shape,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    dims: [6]usize = @splat(0),
    d: [6]?dense_in.Dense = @splat(null),
    w16: [6][]u16 = @splat(&.{}),
    x_ref: [3]rig_.Ref = undefined,
    w_ref: [6]rig_.Ref = undefined,
    y_ref: [6]rig_.Ref = undefined,
    base_ref: [6]rig_.Ref = undefined,
    base_bytes: std.ArrayList(u8) = .empty,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, tier: u32, control: bool) !Probe {
        const pipes = if (control) blk: {
            const text = try controls.source(a, sources.dense_fuse, tier);
            defer a.free(text);
            break :blk try host.Pipelines.loadSource(device, text);
        } else try host.Pipelines.load(device);
        errdefer {
            var p = pipes;
            p.deinit();
        }
        const sp_pipe: ?mtl.Pipeline = if (control) null else blk: {
            const lib = try mtl.Library.fromSource(device, all_sources.spark25_glue, .{
                .language = mtl.CompileOptions.mlx().language,
                .math_functions = .precise,
            });
            defer lib.deinit();
            break :blk try mtl.Pipeline.init(device, lib, spSymbol(tier), false);
        };
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .tier = tier,
            .control = control,
            .pipes = pipes,
            .sp_pipe = sp_pipe,
            .served = .{
                .{ .rows = tier, .n = 3072, .k = 2304 },
                .{ .rows = @max(1, tier / 2), .n = 13, .k = 512 },
                .{ .rows = 1, .n = 25, .k = 64 },
            },
        };
    }

    fn drop(p: *Probe) void {
        for (&p.d) |*maybe| {
            if (maybe.*) |d| d.deinit();
            maybe.* = null;
        }
        for (&p.w16) |*w| {
            p.a.free(w.*);
            w.* = &.{};
        }
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.base_bytes.deinit(p.a);
        if (p.sp_pipe) |sp| sp.deinit();
        p.pipes.deinit();
        p.rig.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn spec(p: *const Probe) c.Spec {
        return .{
            .name = if (p.control) controls.name(p.tier) else kernelLabel(p.tier),
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .replaces = if (p.control) null else "six separate sp_matvec launches",
            .tiles = .{ .{ .size = p.tier, .step = 1, .min = 1 }, .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 256, .step = 8, .min = 8 } },
            .served = &p.served,
        };
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{
            .spec = p.spec(),
            .ctx = p,
            .vt = &vt,
            .control = if (p.control) .shapes else null,
        };
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.k == 0 or s.k % 8 != 0) return false;
        p.drop();
        p.shape = s;
        p.dims = projDims(s.n);
        for (p.dims, 0..) |n_i, idx| {
            const sub = c.Shape{ .rows = s.rows, .n = n_i, .k = s.k };
            const d = try dense_in.Dense.init(p.a, sub, 1);
            p.d[idx] = d;
            switch (idx) {
                0, 1, 2 => if (fill == .stress)
                    fillStressX(d, seed ^ (@as(u64, idx) * 0x101))
                else
                    d.fillX(fill, seed ^ (@as(u64, idx) * 0x101), 2),
                else => @memcpy(d.x, p.d[0].?.x),
            }
            p.w16[idx] = try p.a.alloc(u16, n_i * s.k);
            fillWeights(p.w16[idx], d, sub, fill, seed ^ (0x5151 + @as(u64, idx) * 0x111));
        }
        for (0..3) |xi| p.x_ref[xi] = try p.rig.input(std.mem.sliceAsBytes(p.d[xi].?.x));
        for (0..6) |idx| {
            p.w_ref[idx] = try p.rig.input(std.mem.sliceAsBytes(p.w16[idx]));
            p.y_ref[idx] = try p.rig.output(s.rows * p.dims[idx] * 2);
            if (!p.control) p.base_ref[idx] = try p.rig.scratch(s.rows * p.dims[idx] * 2);
        }
        return true;
    }

    pub fn count(p: *Probe) usize {
        var total: usize = 0;
        for (p.d) |maybe| total += maybe.?.outputs();
        return total;
    }

    fn hr(ref: rig_.Ref, extra: usize) host.Ref {
        return .{ .buf = ref.buf, .off = ref.off + extra };
    }

    pub fn run(p: *Probe) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        var at: usize = 0;
        while (at < s.rows) : (at += p.tier) {
            const here: u32 = @intCast(@min(p.tier, s.rows - at));
            const x_off = at * s.k * 2;
            try host.encode6(
                &p.pipes,
                e,
                hr(p.x_ref[0], x_off),
                hr(p.x_ref[1], x_off),
                hr(p.x_ref[2], x_off),
                .{ .w = hr(p.w_ref[0], 0), .y = hr(p.y_ref[0], at * p.dims[0] * 2), .n = @intCast(p.dims[0]) },
                .{ .w = hr(p.w_ref[1], 0), .y = hr(p.y_ref[1], at * p.dims[1] * 2), .n = @intCast(p.dims[1]) },
                .{ .w = hr(p.w_ref[2], 0), .y = hr(p.y_ref[2], at * p.dims[2] * 2), .n = @intCast(p.dims[2]) },
                .{ .w = hr(p.w_ref[3], 0), .y = hr(p.y_ref[3], at * p.dims[3] * 2), .n = @intCast(p.dims[3]) },
                .{ .w = hr(p.w_ref[4], 0), .y = hr(p.y_ref[4], at * p.dims[4] * 2), .n = @intCast(p.dims[4]) },
                .{ .w = hr(p.w_ref[5], 0), .y = hr(p.y_ref[5], at * p.dims[5] * 2), .n = @intCast(p.dims[5]) },
                here,
                @intCast(s.k),
            );
        }
        try p.rig.finish(pass);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        const s = p.shape;
        const x_map = [6]usize{ 0, 1, 2, 0, 0, 0 };
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.sp_pipe.?);
        var at: usize = 0;
        while (at < s.rows) : (at += p.tier) {
            const here: u32 = @intCast(@min(p.tier, s.rows - at));
            const x_off = at * s.k * 2;
            for (0..6) |idx| {
                const xr = p.x_ref[x_map[idx]];
                const n_i = p.dims[idx];
                e.setBuffer(xr.buf, xr.off + x_off, 0);
                e.setBuffer(p.w_ref[idx].buf, p.w_ref[idx].off, 1);
                e.setValue(DenseArgs{ .rows = here, .n = @intCast(n_i), .k = @intCast(s.k), .ldy = @intCast(n_i) }, 2);
                e.setBuffer(p.base_ref[idx].buf, p.base_ref[idx].off + at * n_i * 2, 3);
                e.dispatchGroups(mtl.Size.of((n_i + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            }
        }
        try p.rig.finish(pass);
        p.base_bytes.clearRetainingCapacity();
        for (0..6) |idx| {
            try p.base_bytes.appendSlice(p.a, p.rig.bytes(p.base_ref[idx], s.rows * p.dims[idx] * 2));
        }
        const total = p.count();
        const want = try p.a.alloc(f64, total);
        defer p.a.free(want);
        const mag = try p.a.alloc(f64, total);
        defer p.a.free(mag);
        try p.reference(want, mag);
        const sp = p.spec();
        for (want, mag, 0..) |w, m, i| {
            const got = c.read(.bf16, p.base_bytes.items, i);
            const limit = c.bound(sp, s, w, m);
            if (@abs(got - w) > limit) return error.SpMatvecExceededFp64Bound;
        }
        return p.base_bytes.items;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        var at: usize = 0;
        for (p.d) |maybe| {
            const d = maybe.?;
            const n = d.outputs();
            try d.reference(want[at..][0..n], mag[at..][0..n]);
            at += n;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        var at: usize = 0;
        for (p.d) |maybe| {
            const d = maybe.?;
            const n = d.outputs();
            try d.simulate(acc, order, out[at..][0..n]);
            at += n;
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (controls.tiers) |tier| for ([_]bool{ false, true }) |control| {
            const label = if (control) controls.name(tier) else kernelLabel(tier);
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, tier, control);
            errdefer p.deinit();
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        };
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
