//! Two-stream mHC contract suite for SenseNova-Loop: Gates 1-5, planted controls, row invariance and xsum parity.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const limits_mod = @import("metal_limits.zig");
const offline_mod = @import("metal_offline.zig");

const c = contract.spec;
const g = contract.gates;
const mhc = tf.sensenova_loop.mhc_metal;
const ref = tf.sensenova_loop.mhc_ref;
const base = "sensenova mhc ";

pub const Kind = enum { enter, pre_norm, post_merge, post_merge_exit, post_and_pre_norm };

const hidden_max = ref.max_hidden;
const served_shapes = [_]c.Shape{
    .{ .rows = 16, .n = 1, .k = hidden_max },
    .{ .rows = 1, .n = 1, .k = hidden_max },
    .{ .rows = 7, .n = 1, .k = hidden_max },
};

fn oneDepth(_: c.Shape) usize {
    return 1;
}

fn postDepth(_: c.Shape) usize {
    return 4;
}

fn preNormDepth(s: c.Shape) usize {
    return 2 * s.k / 256 + 16;
}

pub fn spec(k: Kind, name: []const u8) c.Spec {
    const t = [_]c.Tile{ .{ .size = 8, .step = 1, .min = 1 }, .{}, .{ .size = 256, .step = 1, .min = 1 } };
    return switch (k) {
        .enter => .{ .name = name, .accum = .none, .depth = oneDepth, .out = .bf16, .tiles = t, .served = &served_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = "elementwise stream expansion has no reduction sum" } },
        .pre_norm => .{ .name = name, .accum = .f32, .depth = preNormDepth, .ops = 260, .mids = 2, .out = .bf16, .tiles = t, .served = &served_shapes, .sums = .{ .why = "fp32 sqrsum, matvec and branch RMSNorm" } },
        .post_merge => .{ .name = name, .accum = .f32, .depth = postDepth, .ops = 2, .mids = 1, .out = .bf16, .tiles = t, .served = &served_shapes, .sums = .{ .why = "fp32 residual stream mix and post-weighted branch sum" } },
        .post_merge_exit => .{ .name = name, .accum = .f32, .depth = postDepth, .ops = 2, .mids = 2, .out = .bf16, .tiles = t, .served = &served_shapes, .sums = .{ .why = "fp32 residual mix, branch sum and exit stream mean" } },
        .post_and_pre_norm => .{ .name = name, .accum = .f32, .depth = preNormDepth, .ops = 260, .mids = 2, .out = .bf16, .tiles = t, .served = &served_shapes, .sums = .{ .why = "fused fp32 post-merge + log-Sinkhorn + collapsed RMSNorm" }, .replaces = "sn_mhc_post_merge + sn_mhc_pre_norm" },
    };
}

fn mRef(r: rig_.Ref, extra: usize) mhc.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

fn hasPre(k: Kind) bool {
    return k == .pre_norm or k == .post_and_pre_norm;
}

fn hasPost(k: Kind) bool {
    return k == .post_merge or k == .post_merge_exit or k == .post_and_pre_norm;
}

const vt = g.hooks(Proj);
const plain_vt = blk: {
    var v = g.hooks(Proj);
    v.replaced = null;
    break :blk v;
};

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: Kind,
    ctl: u8,
    name: []const u8,
    pipes: mhc.Pipelines,
    shape: c.Shape = undefined,
    h_in: []u16 = &.{},
    r_in: []u16 = &.{},
    y_in: []u16 = &.{},
    coeff_in: []u16 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    sub_w: []u16 = &.{},
    in_a: rig_.Ref = undefined,
    in_b: rig_.Ref = undefined,
    in_c: rig_.Ref = undefined,
    in_w: rig_.Ref = undefined,
    in_bias: rig_.Ref = undefined,
    in_sub: rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined,
    out_len: [3]usize = .{ 0, 0, 0 },
    n_outs: usize = 0,
    xs_group: u32 = 0,
    xs_stride: u32 = 0,
    shift_norm: bool = true,
    pre_scale: f32 = 1.0,
    has_pending: bool = false,
    write_h: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: Kind, ctl: u8, name: []const u8) !Proj {
        const pipes = if (ctl == 0) try mhc.Pipelines.load(device) else blk: {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.core_sensenova_mhc });
            defer a.free(text);
            break :blk try mhc.Pipelines.loadSource(device, text);
        };
        return .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .kind = k, .ctl = ctl, .name = name, .pipes = pipes };
    }

    pub fn deinit(p: *Proj) void {
        p.pipes.deinit();
        p.rig.deinit();
        p.arena.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.ctl) {
            1 => .nonfinite,
            3 => .sums,
            2, 4, 5 => .shapes,
            else => null,
        };
        var s = spec(p.kind, p.name);
        if (p.ctl == 0) return .{ .spec = s, .ctx = p, .vt = &vt, .control = null };
        s.replaces = null;
        return .{ .spec = s, .ctx = p, .vt = &plain_vt, .control = control };
    }

    pub fn count(p: *Proj) usize {
        const rows = p.shape.rows;
        const h = p.shape.k;
        return switch (p.kind) {
            .enter => rows * (2 * h + if (p.write_h) h else 0),
            .pre_norm => rows * (ref.coeff_width + h),
            .post_merge => rows * 2 * h,
            .post_merge_exit => rows * h,
            .post_and_pre_norm => rows * (2 * h + ref.coeff_width + h),
        };
    }

    fn allocHost(p: *Proj, rows: usize, h: usize) !void {
        const aa = p.arena.allocator();
        p.h_in = if (p.kind == .enter) try aa.alloc(u16, rows * h) else &.{};
        p.r_in = if (p.kind != .enter) try aa.alloc(u16, rows * 2 * h) else &.{};
        p.y_in = if (hasPost(p.kind) or p.has_pending) try aa.alloc(u16, rows * h) else &.{};
        p.coeff_in = if (hasPost(p.kind)) try aa.alloc(u16, rows * ref.coeff_width) else &.{};
        p.w = if (hasPre(p.kind)) try aa.alloc(f32, 8 * 2 * h) else &.{};
        p.b = if (hasPre(p.kind)) try aa.alloc(f32, 8) else &.{};
        p.sub_w = if (hasPre(p.kind)) try aa.alloc(u16, h) else &.{};
    }

    fn fillRandom(p: *Proj, rows: usize, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.h_in) |*v| v.* = ref.toBf16((rng.float(f32) * 2 - 1) * 4);
        for (p.r_in) |*v| v.* = ref.toBf16((rng.float(f32) * 2 - 1) * 4);
        for (p.y_in) |*v| v.* = ref.toBf16((rng.float(f32) * 2 - 1) * 4);
        for (p.w) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.04;
        for (p.b) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.5;
        for (p.sub_w) |*v| v.* = ref.toBf16((rng.float(f32) * 2 - 1) * 0.5);
        for (0..if (hasPost(p.kind)) rows else 0) |r| {
            for (p.coeff_in[r * ref.coeff_width ..][0..ref.coeff_width]) |*v| v.* = ref.toBf16(0.25 + rng.float(f32));
        }
    }

    fn fillNonfinite(p: *Proj, rows: usize, h: usize) void {
        const nan16 = ref.toBf16(std.math.nan(f32));
        const pinf16 = ref.toBf16(std.math.inf(f32));
        const ninf16 = ref.toBf16(-std.math.inf(f32));
        @memset(p.h_in, ref.toBf16(1.0));
        @memset(p.r_in, ref.toBf16(1.0));
        @memset(p.y_in, ref.toBf16(1.0));
        @memset(p.w, 0.01);
        @memset(p.b, 0.0);
        @memset(p.sub_w, ref.toBf16(0.5));
        for (0..if (hasPost(p.kind)) rows else 0) |r| {
            p.coeff_in[r * ref.coeff_width ..][0..ref.coeff_width].* = .{ ref.toBf16(1.0), ref.toBf16(1.0), ref.toBf16(0.5), ref.toBf16(0.5), ref.toBf16(0.5), ref.toBf16(0.5) };
        }
        switch (p.kind) {
            .enter => {
                p.h_in[0] = nan16;
                if (rows > 1) p.h_in[h] = pinf16;
                if (rows > 2) p.h_in[2 * h] = ninf16;
            },
            .pre_norm => {
                p.r_in[0] = nan16;
                p.sub_w[0] = pinf16;
                if (h > 1) p.sub_w[1] = ninf16;
            },
            .post_merge, .post_merge_exit => {
                p.y_in[0] = nan16;
                if (rows > 1) p.y_in[h] = pinf16;
                if (rows > 2) p.y_in[2 * h] = ninf16;
            },
            .post_and_pre_norm => {
                p.y_in[0] = nan16;
                if (rows > 1) p.y_in[h + @intFromBool(h > 1)] = pinf16;
                if (rows > 2) p.y_in[2 * h + (if (h > 2) @as(usize, 2) else 0)] = ninf16;
                if (h > 3) p.sub_w[3] = pinf16;
                if (h > 4) p.sub_w[4] = ninf16;
            },
        }
    }

    fn fillStressPre(p: *Proj, rows: usize, h: usize) void {
        const two_h = 2 * h;
        const half_h = h / 2;
        var ss: f64 = 0;
        for (0..h) |d| {
            const v: f32 = @floatFromInt(16 + (d % 8));
            const vb = ref.toBf16(v);
            const v1 = if (d < half_h) ref.toBf16(-v) else vb;
            for (0..rows) |r| {
                p.r_in[(r * 2 + 0) * h + d] = vb;
                p.r_in[(r * 2 + 1) * h + d] = v1;
            }
            ss += 2.0 * @as(f64, ref.fromBf16(vb)) * @as(f64, ref.fromBf16(vb));
        }
        const inv_rms = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(two_h)) + 1e-6);
        var nudge: f64 = 0;
        for (0..@min(@as(usize, 8), h)) |d| {
            const v: f64 = @floatFromInt(16 + (d % 8));
            nudge += if (d < half_h) -v else v;
        }
        for (0..8) |m| {
            const wm = p.w[m * two_h ..][0..two_h];
            for (0..h) |d| {
                const bw: f32 = @floatFromInt(1 + ((d + m) % 7));
                wm[d] = bw;
                wm[h + d] = (if (d < half_h) bw else -bw) + @as(f32, @floatFromInt(@intFromBool(d < 8)));
            }
            p.b[m] = @floatCast(-nudge * inv_rms + if (m == 2) @as(f64, -8.0) else 0.0);
        }
        for (0..two_h) |j| p.w[two_h + j] = -p.w[j];
        p.b[1] = -p.b[0];
        @memset(p.sub_w, ref.toBf16(0.5));
    }

    fn fillStressPost(p: *Proj, rows: usize, h: usize, fused: bool) void {
        for (0..rows) |r| {
            p.coeff_in[r * ref.coeff_width ..][0..ref.coeff_width].* = .{ ref.toBf16(1.0), ref.toBf16(1.0), ref.toBf16(19.0 / 16.0), ref.toBf16(-1.0), ref.toBf16(19.0 / 16.0), ref.toBf16(-1.0) };
            for (0..h) |d| {
                p.r_in[(r * 2 + 0) * h + d] = ref.toBf16(16768.0);
                p.r_in[(r * 2 + 1) * h + d] = ref.toBf16(19840.0);
                const target: f32 = if (!fused or d == 0) 0.5 else @floatFromInt(16 + (d % 8));
                p.y_in[r * h + d] = ref.toBf16(-(72.0 - target));
            }
        }
        if (fused) {
            const two_h = 2 * h;
            var ss: f64 = 0.25 * 2.0;
            for (1..h) |d| {
                const v: f64 = @floatFromInt(16 + (d % 8));
                ss += 2.0 * v * v;
            }
            const inv_rms = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(two_h)) + 1e-6);
            var nudge: f64 = 0;
            for (1..@min(@as(usize, 9), h)) |d| nudge += @floatFromInt(16 + (d % 8));
            for (0..8) |m| {
                const wm = p.w[m * two_h ..][0..two_h];
                for (0..h) |d| {
                    const bw: f32 = @floatFromInt(1 + ((d + m) % 7));
                    wm[d] = bw;
                    wm[h + d] = -bw + @as(f32, @floatFromInt(@intFromBool(d >= 1 and d < 9)));
                }
                p.b[m] = @floatCast(-nudge * inv_rms + if (m == 2) @as(f64, -8.0) else 0.0);
            }
            @memset(p.sub_w, ref.toBf16(0.5));
        }
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.k > hidden_max) return false;
        if (p.xs_group != 0 and ((p.xs_group != 64 and p.xs_group != 128) or shape.k % p.xs_group != 0)) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        p.shape = shape;
        const n = shape.rows;
        const h = shape.k;
        try p.allocHost(n, h);
        switch (fill) {
            .random => p.fillRandom(n, seed),
            .nonfinite => p.fillNonfinite(n, h),
            .stress => switch (p.kind) {
                .enter => p.fillRandom(n, seed),
                .pre_norm => p.fillStressPre(n, h),
                .post_merge, .post_merge_exit => p.fillStressPost(n, h, false),
                .post_and_pre_norm => p.fillStressPost(n, h, true),
            },
        }
        try p.bindBuffers(n, h);
        return true;
    }

    fn bindBuffers(p: *Proj, n: usize, h: usize) !void {
        p.in_a = try p.rig.input(std.mem.sliceAsBytes(if (p.kind == .enter) p.h_in else p.r_in));
        p.in_b = if (hasPost(p.kind) or p.has_pending) try p.rig.input(std.mem.sliceAsBytes(p.y_in)) else p.in_a;
        p.in_c = if (hasPost(p.kind)) try p.rig.input(std.mem.sliceAsBytes(p.coeff_in)) else p.in_a;
        p.in_w = if (hasPre(p.kind)) try p.rig.input(std.mem.sliceAsBytes(p.w)) else p.in_a;
        p.in_bias = if (hasPre(p.kind)) try p.rig.input(std.mem.sliceAsBytes(p.b)) else p.in_a;
        p.in_sub = if (hasPre(p.kind)) try p.rig.input(std.mem.sliceAsBytes(p.sub_w)) else p.in_a;
        const lens: [3]usize = switch (p.kind) {
            .enter => .{ n * 2 * h * 2, if (p.write_h) n * h * 2 else 0, 0 },
            .pre_norm => .{ n * ref.coeff_width * 2, n * h * 2, 0 },
            .post_merge => .{ n * 2 * h * 2, 0, 0 },
            .post_merge_exit => .{ n * h * 2, 0, 0 },
            .post_and_pre_norm => .{ n * 2 * h * 2, n * ref.coeff_width * 2, n * h * 2 },
        };
        p.out_len = lens;
        p.n_outs = 0;
        for (lens, 0..) |len, i| if (len > 0) {
            p.out[i] = try p.rig.output(len);
            p.n_outs = i + 1;
        };
    }

    pub fn encode(p: *Proj, e: anytype, r0: usize, n: usize, xs_out: ?rig_.Ref) void {
        const rows: u32 = @intCast(n);
        const h: u32 = @intCast(p.shape.k);
        const call = mhc.PreNormCall{ .iters = 20, .shift_norm = p.shift_norm, .xs_group = p.xs_group, .xs_stride = if (p.xs_group != 0) p.xs_stride else 0, .pre_scale = p.pre_scale, .eps = 1e-6 };
        const xs_ref: ?mhc.Ref = if (xs_out) |xr| mRef(xr, r0 * 4) else null;
        switch (p.kind) {
            .enter => mhc.enter(&p.pipes, e, mRef(p.in_a, r0 * h * 2), if (p.has_pending) mRef(p.in_b, r0 * h * 2) else null, mRef(p.out[0], r0 * 2 * h * 2), if (p.write_h) mRef(p.out[1], r0 * h * 2) else null, rows, h) catch unreachable,
            .pre_norm => mhc.preNorm(&p.pipes, e, mRef(p.in_a, r0 * 2 * h * 2), mRef(p.in_w, 0), mRef(p.in_bias, 0), mRef(p.in_sub, 0), mRef(p.out[0], r0 * ref.coeff_width * 2), mRef(p.out[1], r0 * h * 2), xs_ref, rows, h, call) catch unreachable,
            .post_merge, .post_merge_exit => mhc.postMerge(&p.pipes, e, mRef(p.in_a, r0 * 2 * h * 2), mRef(p.in_b, r0 * h * 2), mRef(p.in_c, r0 * ref.coeff_width * 2), mRef(p.out[0], r0 * (if (p.kind == .post_merge_exit) h else 2 * h) * 2), rows, h, p.kind == .post_merge_exit) catch unreachable,
            .post_and_pre_norm => mhc.postAndPreNorm(&p.pipes, e, mRef(p.in_a, r0 * 2 * h * 2), mRef(p.in_b, r0 * h * 2), mRef(p.in_c, r0 * ref.coeff_width * 2), mRef(p.in_w, 0), mRef(p.in_bias, 0), mRef(p.in_sub, 0), mRef(p.out[0], r0 * 2 * h * 2), mRef(p.out[1], r0 * ref.coeff_width * 2), mRef(p.out[2], r0 * h * 2), xs_ref, rows, h, call) catch unreachable,
        }
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        p.encode(pass.e, 0, p.shape.rows, null);
        try p.rig.finish(pass);
    }

    pub fn replaced(p: *Proj) ![]const u8 {
        const rows: u32 = @intCast(p.shape.rows);
        const h: u32 = @intCast(p.shape.k);
        const call = mhc.PreNormCall{ .shift_norm = p.shift_norm, .pre_scale = p.pre_scale, .eps = 1e-6 };
        const pass = p.rig.begin();
        try mhc.postMerge(&p.pipes, pass.e, mRef(p.in_a, 0), mRef(p.in_b, 0), mRef(p.in_c, 0), mRef(p.out[0], 0), rows, h, false);
        try mhc.preNorm(&p.pipes, pass.e, mRef(p.out[0], 0), mRef(p.in_w, 0), mRef(p.in_bias, 0), mRef(p.in_sub, 0), mRef(p.out[1], 0), mRef(p.out[2], 0), null, rows, h, call);
        try p.rig.finish(pass);
        return p.rig.outputBytes();
    }

    fn computeModel(p: *Proj, acc: ref.Accum, order: ref.Order, want: []f64, mag: []f64) !void {
        const rows = p.shape.rows;
        const h = p.shape.k;
        const params = ref.Params{ .hidden = h, .iterations = 20, .pre_scale = p.pre_scale, .eps = 1e-6, .shift_norm = p.shift_norm };
        const aa = p.arena.allocator();
        switch (p.kind) {
            .enter => ref.enterRef(p.h_in, if (p.has_pending) p.y_in else null, rows, h, want[0 .. rows * 2 * h], mag[0 .. rows * 2 * h], if (p.write_h) want[rows * 2 * h ..] else null, if (p.write_h) mag[rows * 2 * h ..] else null),
            .pre_norm => {
                const c_n = rows * ref.coeff_width;
                try ref.preNormRef(aa, .{ .r_in = p.r_in, .w = p.w, .b = p.b, .sub_w = p.sub_w }, rows, params, acc, order, .{ .coeff = want[0..c_n], .mag_coeff = mag[0..c_n], .norm = want[c_n..], .mag_norm = mag[c_n..] });
            },
            .post_merge, .post_merge_exit => ref.postMergeRef(p.r_in, p.y_in, p.coeff_in, rows, h, p.kind == .post_merge_exit, acc, order, want, mag),
            .post_and_pre_norm => {
                const r_n = rows * 2 * h;
                const c_n = rows * ref.coeff_width;
                try ref.postAndPreNormRef(aa, p.r_in, p.y_in, p.coeff_in, p.w, p.b, p.sub_w, rows, params, acc, order, want[0..r_n], mag[0..r_n], .{ .coeff = want[r_n .. r_n + c_n], .mag_coeff = mag[r_n .. r_n + c_n], .norm = want[r_n + c_n ..], .mag_norm = mag[r_n + c_n ..] });
            },
        }
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        try p.computeModel(.none, .sequential, want, mag);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const m_acc: ref.Accum = switch (acc) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            .none => .none,
        };
        const m_ord: ref.Order = switch (order) {
            .sequential => .sequential,
            .pairwise => .pairwise,
            .strided => .strided,
        };
        const mag = try p.arena.allocator().alloc(f64, p.count());
        try p.computeModel(m_acc, m_ord, out, mag);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { k: Kind, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .k = .enter, .ctl = 0, .name = base ++ "enter sn_mhc_enter" },
        .{ .k = .pre_norm, .ctl = 0, .name = base ++ "pre_norm sn_mhc_pre_norm" },
        .{ .k = .post_merge, .ctl = 0, .name = base ++ "post_merge sn_mhc_post_merge" },
        .{ .k = .post_merge_exit, .ctl = 0, .name = base ++ "post_merge_exit sn_mhc_post_merge" },
        .{ .k = .post_and_pre_norm, .ctl = 0, .name = base ++ "post_and_pre_norm sn_mhc_post_and_pre_norm" },
        .{ .k = .enter, .ctl = 1, .name = base ++ "control: enter non-finite flushed to zero" },
        .{ .k = .enter, .ctl = 2, .name = base ++ "control: enter zeroes second stream" },
        .{ .k = .enter, .ctl = 4, .name = base ++ "control: enter skips tail element" },
        .{ .k = .pre_norm, .ctl = 1, .name = base ++ "control: pre_norm non-finite flushed to zero" },
        .{ .k = .pre_norm, .ctl = 2, .name = base ++ "control: pre_norm skips Sinkhorn" },
        .{ .k = .pre_norm, .ctl = 3, .name = base ++ "control: pre_norm fp16 sum" },
        .{ .k = .pre_norm, .ctl = 4, .name = base ++ "control: pre_norm skips tail element" },
        .{ .k = .pre_norm, .ctl = 5, .name = base ++ "control: pre_norm transposes W" },
        .{ .k = .post_merge, .ctl = 1, .name = base ++ "control: post_merge non-finite flushed to zero" },
        .{ .k = .post_merge, .ctl = 2, .name = base ++ "control: post_merge drops post weight" },
        .{ .k = .post_merge, .ctl = 3, .name = base ++ "control: post_merge fp16 sum" },
        .{ .k = .post_merge, .ctl = 4, .name = base ++ "control: post_merge skips tail element" },
        .{ .k = .post_merge, .ctl = 5, .name = base ++ "control: post_merge transposes mix" },
        .{ .k = .post_merge_exit, .ctl = 1, .name = base ++ "control: post_merge_exit non-finite flushed to zero" },
        .{ .k = .post_merge_exit, .ctl = 2, .name = base ++ "control: post_merge_exit drops post weight" },
        .{ .k = .post_merge_exit, .ctl = 3, .name = base ++ "control: post_merge_exit fp16 sum" },
        .{ .k = .post_merge_exit, .ctl = 4, .name = base ++ "control: post_merge_exit skips tail element" },
        .{ .k = .post_merge_exit, .ctl = 5, .name = base ++ "control: post_merge_exit transposes mix" },
        .{ .k = .post_and_pre_norm, .ctl = 1, .name = base ++ "control: post_and_pre_norm non-finite flushed to zero" },
        .{ .k = .post_and_pre_norm, .ctl = 2, .name = base ++ "control: post_and_pre_norm drops post or Sinkhorn" },
        .{ .k = .post_and_pre_norm, .ctl = 3, .name = base ++ "control: post_and_pre_norm fp16 sum" },
        .{ .k = .post_and_pre_norm, .ctl = 4, .name = base ++ "control: post_and_pre_norm skips tail element" },
        .{ .k = .post_and_pre_norm, .ctl = 5, .name = base ++ "control: post_and_pre_norm transposes mix" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.k, entry.ctl, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    /// Launches of 1, 7 and 11 rows/launch equal one launch at rows = 33, plus xs_group and pending checks.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var kernels: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            kernels += 1;
            const shape = c.Shape{ .rows = 33, .n = 1, .k = hidden_max };
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            try p.run();
            const whole = try s.a.dupe(u8, try p.rig.outputBytes());
            defer s.a.free(whole);
            for ([_]usize{ 1, 7, 11 }) |step| {
                for (p.out[0..p.n_outs], p.out_len[0..p.n_outs]) |o, len| @memset(p.rig.bytes(o, len), 0x55);
                const pass = p.rig.begin();
                var r0: usize = 0;
                while (r0 < shape.rows) : (r0 += step) {
                    p.encode(pass.e, r0, @min(step, shape.rows - r0), null);
                    launches += 1;
                }
                try p.rig.finish(pass);
                if (!std.mem.eql(u8, whole, try p.rig.outputBytes())) {
                    bad += 1;
                    std.debug.print("FAIL row invariance {s}: {d} rows/launch differs\n", .{ p.name, step });
                }
            }
        }
        if (kernels > 0) bad += try s.checkExtras(seed);
        if (kernels == 0) return 0;
        std.debug.print("{s} row invariance {s}: {d} kernels at 1, 7 and 11 rows ({d} launches) + xsum/pending, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }

    fn checkExtras(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var ep = try Proj.init(s.a, s.device, s.queue, .enter, 0, base ++ "enter_pending");
        defer ep.deinit();
        ep.has_pending = true;
        ep.write_h = true;
        if (!try ep.prepare(.{ .rows = 9, .n = 1, .k = 257 }, .random, seed + 1)) return 1;
        try ep.run();
        const e_sim = try s.a.alloc(u16, ep.count());
        defer s.a.free(e_sim);
        ref.enterSim(ep.h_in, ep.y_in, 9, 257, e_sim[0 .. 9 * 2 * 257], e_sim[9 * 2 * 257 ..]);
        if (!std.mem.eql(u8, try ep.rig.outputBytes(), std.mem.sliceAsBytes(e_sim))) bad += 1;
        for ([_]u32{ 64, 128 }) |xg| bad += try s.checkXsumAndFused(xg, seed);
        return bad;
    }

    fn checkXsumAndFused(s: *Suite, xs_group: u32, seed: u64) !usize {
        var bad: usize = 0;
        const rows: usize = 5;
        const h: usize = hidden_max;
        const groups = h / xs_group;
        var fp = try Proj.init(s.a, s.device, s.queue, .post_and_pre_norm, 0, base ++ "xs_fused");
        defer fp.deinit();
        fp.shift_norm = false;
        fp.pre_scale = 0.5;
        fp.xs_group = xs_group;
        fp.xs_stride = rows;
        if (!try fp.prepare(.{ .rows = rows, .n = 1, .k = h }, .random, seed + xs_group)) return 1;
        const xs_bytes = groups * rows * 4;
        const xs_fused = try fp.rig.scratch(xs_bytes);
        const xs_seq = try fp.rig.scratch(xs_bytes);
        const pass1 = fp.rig.begin();
        fp.encode(pass1.e, 0, rows, xs_fused);
        try fp.rig.finish(pass1);
        const fused_out = try s.a.dupe(u8, try fp.rig.outputBytes());
        defer s.a.free(fused_out);
        const fused_xs = try s.a.dupe(u8, fp.rig.bytes(xs_fused, xs_bytes));
        defer s.a.free(fused_xs);

        const call = mhc.PreNormCall{ .shift_norm = false, .xs_group = xs_group, .xs_stride = rows, .pre_scale = 0.5, .eps = 1e-6 };
        const pass2 = fp.rig.begin();
        try mhc.postMerge(&fp.pipes, pass2.e, mRef(fp.in_a, 0), mRef(fp.in_b, 0), mRef(fp.in_c, 0), mRef(fp.out[0], 0), rows, h, false);
        try mhc.preNorm(&fp.pipes, pass2.e, mRef(fp.out[0], 0), mRef(fp.in_w, 0), mRef(fp.in_bias, 0), mRef(fp.in_sub, 0), mRef(fp.out[1], 0), mRef(fp.out[2], 0), mRef(xs_seq, 0), rows, h, call);
        try fp.rig.finish(pass2);
        if (!std.mem.eql(u8, fused_out, try fp.rig.outputBytes())) {
            std.debug.print("FAIL xsum fused_out != seq_out (xg={d})\n", .{xs_group});
            bad += 1;
        }
        if (!std.mem.eql(u8, fused_xs, fp.rig.bytes(xs_seq, xs_bytes))) {
            std.debug.print("FAIL xsum fused_xs != seq_xs (xg={d})\n", .{xs_group});
            bad += 1;
        }

        const norm_bytes = fp.rig.bytes(fp.out[2], rows * h * 2);
        for (0..groups) |g_idx| {
            for (0..rows) |r_idx| {
                var sum_gpu: f32 = 0;
                var sum_abs: f32 = 0;
                for (0..xs_group) |i| {
                    const b16 = std.mem.readInt(u16, norm_bytes[2 * (r_idx * h + g_idx * xs_group + i) ..][0..2], .little);
                    const fv = ref.fromBf16(b16);
                    sum_gpu += fv;
                    sum_abs += @abs(fv);
                }
                const xs_idx = g_idx * rows + r_idx;
                const g_v: f32 = @bitCast(std.mem.readInt(u32, fused_xs[4 * xs_idx ..][0..4], .little));
                if (@abs(g_v - sum_gpu) > sum_abs * 1e-5 + 1e-6) {
                    std.debug.print("FAIL xsum sum mismatch xg={d} g={d} r={d}\n", .{ xs_group, g_idx, r_idx });
                    bad += 1;
                }
            }
        }

        const r_sim = try s.a.alloc(u16, rows * 2 * h);
        defer s.a.free(r_sim);
        const c_sim = try s.a.alloc(u16, rows * ref.coeff_width);
        defer s.a.free(c_sim);
        const n_sim = try s.a.alloc(u16, rows * h);
        defer s.a.free(n_sim);
        const xs_sim = try s.a.alloc(f32, groups * rows);
        defer s.a.free(xs_sim);
        const params = ref.Params{ .hidden = h, .pre_scale = 0.5, .shift_norm = false, .xs_group = xs_group, .xs_stride = rows };
        try ref.postAndPreNormSim(s.a, fp.r_in, fp.y_in, fp.coeff_in, fp.w, fp.b, fp.sub_w, rows, params, r_sim, .{ .coeff = c_sim, .norm = n_sim, .xs = xs_sim });
        for (0..groups * rows) |i| {
            const g_v: f32 = @bitCast(std.mem.readInt(u32, fused_xs[4 * i ..][0..4], .little));
            if (@abs(g_v - xs_sim[i]) > 0.05) {
                std.debug.print("FAIL xsum sim mismatch xg={d} i={d} gpu={d} sim={d}\n", .{ xs_group, i, g_v, xs_sim[i] });
                bad += 1;
            }
        }
        return bad;
    }
};

fn launchOf(_: std.mem.Allocator, _: mtl.Device, d: contract.limits.MetalLaunch) !limits_mod.Launch {
    const fun = d.name[std.mem.lastIndexOfScalar(u8, d.name, ' ').? + 1 ..];
    return .{ .decl = d, .function = fun, .here = sources.core_sensenova_mhc, .pre_m5 = sources.core_sensenova_mhc };
}

pub fn limits(a: std.mem.Allocator, io: std.Io, device: mtl.Device, offline: ?[]const u8, only: ?[]const u8) !usize {
    var list: std.ArrayList(contract.limits.MetalLaunch) = .empty;
    defer list.deinit(a);
    for (contract.registry.sensenova_mhc.records) |d| {
        if (only) |needle| if (std.mem.indexOf(u8, d.name, needle) == null) continue;
        try list.append(a, d);
    }
    if (list.items.len == 0) return 0;
    var n = try contract.registry.checkMetal(a, list.items, false);
    for (list.items) |d| n += try limits_mod.onThisGpu(a, device, try launchOf(a, device, d), false);
    if (offline) |dir| n += try offline_mod.run(a, io, dir, list.items, launchOf, device);
    return n;
}
