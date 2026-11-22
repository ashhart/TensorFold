//! Inkling 2 Mini fused 6-projection matvec under FP64 references, bit-identity vs mv/mv_mma, and planted controls.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const dense_in = contract.dense;
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_dense_controls.zig");
const host = tf.inkling2_metal_host.dense_fuse;
const mv = tf.dsv41_metal.mv;
const mv_mma = tf.dsv41_metal.mv_mma;
const sources = @import("inkling2_metal");

fn depth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 6;
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
    mv_pipes: ?mv.Pipelines = null,
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
        const mv_pipes: ?mv.Pipelines = if (control) null else try mv.Pipelines.load(device);
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .tier = tier,
            .control = control,
            .pipes = pipes,
            .mv_pipes = mv_pipes,
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
        if (p.mv_pipes) |*mp| mp.deinit();
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
            .replaces = if (p.control) null else "six separate tf_dsv41_mv launches",
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
        var at: usize = 0;
        while (at < s.rows) : (at += p.tier) {
            const here: u32 = @intCast(@min(p.tier, s.rows - at));
            const x_off = at * s.k * 2;
            for (0..6) |idx| {
                const xr = p.x_ref[x_map[idx]];
                const n_i = p.dims[idx];
                try mv.encode(&p.mv_pipes.?, e, .{ .buf = xr.buf, .off = xr.off + x_off }, .{ .buf = p.w_ref[idx].buf, .off = p.w_ref[idx].off }, .{ .buf = p.base_ref[idx].buf, .off = p.base_ref[idx].off + at * n_i * 2 }, .{ .k = @intCast(s.k), .n = @intCast(n_i), .rows = here, .out = .bf16 });
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
            if (@abs(got - w) > limit) return error.MvExceededFp64Bound;
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

    /// Verify rows=1..33 across every mv6 tier and the rows=32->33 mv_mma seam with zero byte divergence.
    pub fn widthSeam(s: *Suite, seed: u64) !usize {
        var prod: ?*Probe = null;
        for (s.probes.items) |p| if (!p.control) {
            prod = p;
            break;
        };
        const p = prod orelse return 0;
        var mma_pipes = try mv_mma.Pipelines.load(s.device);
        defer mma_pipes.deinit();
        var bad: usize = 0;
        var cases: usize = 0;
        for (1..34) |rows| {
            if (!try p.prepare(.{ .rows = rows, .n = 64, .k = 512 }, .random, seed + rows)) return error.Refused;
            const pass = p.rig.begin();
            const e = pass.e;
            var at: usize = 0;
            while (at < rows) : (at += 32) {
                const here: u32 = @intCast(@min(32, rows - at));
                const x_off = at * 512 * 2;
                try host.encode6(
                    &p.pipes,
                    e,
                    Probe.hr(p.x_ref[0], x_off),
                    Probe.hr(p.x_ref[1], x_off),
                    Probe.hr(p.x_ref[2], x_off),
                    .{ .w = Probe.hr(p.w_ref[0], 0), .y = Probe.hr(p.y_ref[0], at * p.dims[0] * 2), .n = @intCast(p.dims[0]) },
                    .{ .w = Probe.hr(p.w_ref[1], 0), .y = Probe.hr(p.y_ref[1], at * p.dims[1] * 2), .n = @intCast(p.dims[1]) },
                    .{ .w = Probe.hr(p.w_ref[2], 0), .y = Probe.hr(p.y_ref[2], at * p.dims[2] * 2), .n = @intCast(p.dims[2]) },
                    .{ .w = Probe.hr(p.w_ref[3], 0), .y = Probe.hr(p.y_ref[3], at * p.dims[3] * 2), .n = @intCast(p.dims[3]) },
                    .{ .w = Probe.hr(p.w_ref[4], 0), .y = Probe.hr(p.y_ref[4], at * p.dims[4] * 2), .n = @intCast(p.dims[4]) },
                    .{ .w = Probe.hr(p.w_ref[5], 0), .y = Probe.hr(p.y_ref[5], at * p.dims[5] * 2), .n = @intCast(p.dims[5]) },
                    here,
                    512,
                );
            }
            const x_map = [6]usize{ 0, 1, 2, 0, 0, 0 };
            for (0..6) |idx| {
                const xr = p.x_ref[x_map[idx]];
                try mv_mma.encode(&mma_pipes, e, .{ .buf = xr.buf, .off = xr.off }, .{ .buf = p.w_ref[idx].buf, .off = p.w_ref[idx].off }, .{ .buf = p.base_ref[idx].buf, .off = p.base_ref[idx].off }, .{ .k = 512, .n = @intCast(p.dims[idx]), .rows = @intCast(rows), .out = .bf16 });
            }
            try p.rig.finish(pass);
            for (0..6) |idx| {
                const nb = rows * p.dims[idx] * 2;
                if (std.mem.indexOfDiff(u8, p.rig.bytes(p.y_ref[idx], nb), p.rig.bytes(p.base_ref[idx], nb))) |diff| {
                    bad += 1;
                    std.debug.print("FAIL metal inkling2 dense width seam rows={d} proj={d}: first diff at byte {d}\n", .{ rows, idx, diff });
                }
            }
            cases += 1;
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} width seam metal inkling2 dense: rows=1..33 ({d} cases) byte-equal to mv_mma, {d} differ\n", .{ word, cases, bad });
        return bad;
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
