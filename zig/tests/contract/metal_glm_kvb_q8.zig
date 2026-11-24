//! GLM intact 8-bit/64 kv_b decode/verify absorb and unabsorb kernels under gates 1-4 and controls.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const glm_k = tf.glm.kernels;

pub const Kind = enum { absorb, unabsorb };
const Variant = struct { kind: Kind, ctl: u8 = 0 };

const abs_served = [_]c.Shape{ .{ .rows = 4, .n = 128, .k = 128 }, .{ .rows = 1, .n = 64, .k = 64 }, .{ .rows = 8, .n = 512, .k = 256 }, .{ .rows = 16, .n = 64, .k = 96 } };
const unabs_served = [_]c.Shape{ .{ .rows = 4, .n = 12, .k = 576 }, .{ .rows = 1, .n = 64, .k = 512 }, .{ .rows = 8, .n = 256, .k = 512 }, .{ .rows = 16, .n = 10, .k = 192 } };

fn depth(s: c.Shape) usize {
    return 16 + 3 * ((s.k + 511) / 512) + 8;
}

pub fn entryName(v: Variant) []const u8 {
    return switch (v.ctl) {
        0 => switch (v.kind) {
            .absorb => "glm kvb8 glm_absorb_8b",
            .unabsorb => "glm kvb8 glm_unabsorb_8b",
        },
        1 => switch (v.kind) {
            .absorb => "control: glm kvb8 glm_absorb_8b writes into input",
            .unabsorb => "control: glm kvb8 glm_unabsorb_8b writes into input",
        },
        else => switch (v.kind) {
            .absorb => "control: glm kvb8 glm_absorb_8b drops tail cols",
            .unabsorb => "control: glm kvb8 glm_unabsorb_8b drops tail rows",
        },
    };
}

fn spec(v: Variant) c.Spec {
    const k_step: usize = if (v.kind == .absorb) 32 else 64;
    return .{
        .name = entryName(v),
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .nan = .propagate,
        .tiles = .{
            .{ .size = 16 },
            .{ .size = 64, .step = if (v.kind == .absorb) 64 else 1, .min = if (v.kind == .absorb) 64 else 1 },
            .{ .size = 512, .step = k_step, .min = k_step },
        },
        .served = switch (v.kind) {
            .absorb => &abs_served,
            .unabsorb => &unabs_served,
        },
    };
}

const heads: usize = 2;

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    unabs: [heads]?contract.affine.Affine = @splat(null),
    abs_x: []u16 = &.{},
    abs_q: []u8 = &.{},
    abs_s: []u16 = &.{},
    abs_b: []u16 = &.{},
    x_ref: rig_.Ref = undefined,
    w_ref: rig_.Ref = undefined,
    s_ref: rig_.Ref = undefined,
    b_ref: rig_.Ref = undefined,
    y_ref: rig_.Ref = undefined,
    stride: u32 = 0,
    per_head: u32 = 0,
    row_off: u32 = 0,

    fn drop(p: *Probe) void {
        for (&p.unabs) |*u| if (u.*) |f| {
            f.deinit();
            u.* = null;
        };
        for (&[_]*[]u16{ &p.abs_x, &p.abs_s, &p.abs_b }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.a.free(p.abs_q);
        p.abs_q = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.pipe.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        const ctl: ?g.Gate = switch (p.v.ctl) {
            1 => .repeat,
            2 => .shapes,
            else => null,
        };
        return .{ .spec = spec(p.v), .ctx = p, .vt = &vt, .control = ctl };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.rows == 0 or shape.rows > 16 or shape.n == 0 or shape.k == 0) return false;
        if (p.v.kind == .absorb and (shape.n % 64 != 0 or shape.k % 32 != 0)) return false;
        if (p.v.kind == .unabsorb and shape.k % 64 != 0) return false;
        p.shape = shape;
        switch (p.v.kind) {
            .absorb => try p.prepareAbsorb(shape, fill, seed),
            .unabsorb => try p.prepareUnabsorb(shape, fill, seed),
        }
        return true;
    }

    fn prepareAbsorb(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        p.per_head = @intCast(shape.k + 16);
        p.row_off = 16;
        p.stride = @intCast(heads * shape.k + 16);
        const w_rows = heads * p.per_head;
        const ng = shape.n / 64;
        p.abs_x = try p.a.alloc(u16, shape.rows * p.stride);
        p.abs_q = try p.a.alloc(u8, w_rows * shape.n);
        p.abs_s = try p.a.alloc(u16, w_rows * ng);
        p.abs_b = try p.a.alloc(u16, w_rows * ng);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(p.abs_x, c.toBf16(1.0));
        for (p.abs_q) |*q| q.* = rng.int(u8);
        for (p.abs_s) |*s| s.* = c.toBf16(rng.float(f32) * 0.02 + 0.001);
        for (p.abs_b) |*b| b.* = c.toBf16((rng.float(f32) - 0.66) * 0.15);
        for (0..shape.rows) |r| for (0..heads) |h| for (0..shape.k) |i| {
            p.abs_x[r * p.stride + h * shape.k + i] = c.toBf16(rng.float(f32) * 4.0 - 2.0);
        };
        if (fill == .nonfinite) {
            p.abs_x[0] = c.toBf16(std.math.nan(f32));
            p.abs_x[1] = c.toBf16(std.math.inf(f32));
            if (shape.k > 2) p.abs_x[2] = c.toBf16(-std.math.inf(f32));
        } else if (fill == .stress) {
            const half_k = shape.k / 2;
            const xmax: u64 = @min(255, ((1 << 24) - 1) / (256 * shape.k));
            @memset(p.abs_s, c.toBf16(1.0));
            @memset(p.abs_b, c.toBf16(-128.0));
            for (0..heads) |h| for (0..half_k) |i| {
                const wr0 = (h * p.per_head + p.row_off + i) * shape.n;
                const wr1 = (h * p.per_head + p.row_off + half_k + i) * shape.n;
                for (0..shape.n) |j| {
                    const qv = 129 + rng.uintLessThan(u8, 127);
                    p.abs_q[wr0 + j] = qv;
                    p.abs_q[wr1 + j] = @intCast(256 - @as(u16, qv));
                }
            };
            for (0..shape.rows) |r| for (0..heads) |h| for (0..half_k) |i| {
                const v = 1 + rng.uintLessThan(u64, xmax);
                p.abs_x[r * p.stride + h * shape.k + i] = c.toBf16(@floatFromInt(v));
                p.abs_x[r * p.stride + h * shape.k + half_k + i] = c.toBf16(@floatFromInt(v + @intFromBool(i < 8)));
            };
        }
        p.x_ref = try p.rig.input(std.mem.sliceAsBytes(p.abs_x));
        p.w_ref = try p.rig.input(p.abs_q);
        p.s_ref = try p.rig.input(std.mem.sliceAsBytes(p.abs_s));
        p.b_ref = try p.rig.input(std.mem.sliceAsBytes(p.abs_b));
        p.y_ref = try p.rig.output(shape.rows * heads * shape.n * 2);
    }

    fn prepareUnabsorb(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        p.per_head = @intCast(shape.n + 16);
        p.row_off = 16;
        p.stride = @intCast(heads * shape.k + 16);
        const w_rows = heads * p.per_head;
        const ng = shape.k / 64;
        const x_all = try p.a.alloc(u16, shape.rows * p.stride);
        defer p.a.free(x_all);
        @memset(x_all, 0x3c00);
        const words = try p.a.alloc(u32, w_rows * shape.k / 4);
        defer p.a.free(words);
        @memset(words, 0);
        const sc = try p.a.alloc(u16, w_rows * ng);
        defer p.a.free(sc);
        @memset(sc, c.toBf16(1.0));
        const bi = try p.a.alloc(u16, w_rows * ng);
        defer p.a.free(bi);
        @memset(bi, 0);
        for (0..heads) |h| {
            var f = try contract.affine.Affine.init(p.a, shape, 8, 64);
            try f.fill(if (fill == .nonfinite and h > 0) .random else fill, seed ^ (h * 0x9e37));
            p.unabs[h] = f;
            for (0..shape.rows) |r| @memcpy(x_all[r * p.stride + h * shape.k ..][0..shape.k], f.x[r * shape.k ..][0..shape.k]);
            const dst_row = h * p.per_head + p.row_off;
            f.words(words[dst_row * (shape.k / 4) ..][0 .. shape.n * (shape.k / 4)]);
            @memcpy(sc[dst_row * ng ..][0 .. shape.n * ng], f.scales);
            @memcpy(bi[dst_row * ng ..][0 .. shape.n * ng], f.biases);
        }
        p.x_ref = try p.rig.input(std.mem.sliceAsBytes(x_all));
        p.w_ref = try p.rig.input(std.mem.sliceAsBytes(words));
        p.s_ref = try p.rig.input(std.mem.sliceAsBytes(sc));
        p.b_ref = try p.rig.input(std.mem.sliceAsBytes(bi));
        p.y_ref = try p.rig.output(shape.rows * heads * shape.n * 2);
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        inline for (.{ p.w_ref, p.s_ref, p.b_ref, p.x_ref, p.y_ref }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(glm_k.Kv8Args{ .stride = p.stride, .heads = heads, .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.k), .n = @intCast(p.shape.n), .per_head = p.per_head, .row_off = p.row_off }, 5);
        const groups = if (p.v.kind == .absorb) (p.shape.n + 31) / 32 else (p.shape.n + 3) / 4;
        e.dispatchGroups(mtl.Size.of(1, groups, heads), mtl.Size.of(32 * p.shape.rows, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.shape.rows * p.shape.n * heads;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        switch (p.v.kind) {
            .unabsorb => {
                const R = p.shape.rows;
                const N = p.shape.n;
                const hw = try p.a.alloc(f64, R * N);
                defer p.a.free(hw);
                const hm = try p.a.alloc(f64, R * N);
                defer p.a.free(hm);
                for (0..heads) |h| {
                    try p.unabs[h].?.reference(hw, hm);
                    for (0..R) |r| {
                        @memcpy(want[(r * heads + h) * N ..][0..N], hw[r * N ..][0..N]);
                        @memcpy(mag[(r * heads + h) * N ..][0..N], hm[r * N ..][0..N]);
                    }
                }
            },
            .absorb => {
                const N = p.shape.n;
                const K = p.shape.k;
                const ng = N / 64;
                for (0..p.shape.rows) |r| for (0..heads) |h| for (0..N) |j| {
                    var sum: f64 = 0;
                    var m: f64 = 0;
                    for (0..K) |i| {
                        const wr = h * p.per_head + p.row_off + i;
                        const w = c.fromBf16(p.abs_s[wr * ng + j / 64]) * @as(f64, @floatFromInt(p.abs_q[wr * N + j])) + c.fromBf16(p.abs_b[wr * ng + j / 64]);
                        const xv = c.fromBf16(p.abs_x[r * p.stride + h * K + i]);
                        sum += w * xv;
                        m += @abs(w * xv);
                    }
                    want[(r * heads + h) * N + j] = sum;
                    mag[(r * heads + h) * N + j] = m;
                };
            },
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        switch (p.v.kind) {
            .unabsorb => {
                const R = p.shape.rows;
                const N = p.shape.n;
                const ho = try p.a.alloc(f64, R * N);
                defer p.a.free(ho);
                for (0..heads) |h| {
                    try p.unabs[h].?.simulate(acc, order, ho);
                    for (0..R) |r| @memcpy(out[(r * heads + h) * N ..][0..N], ho[r * N ..][0..N]);
                }
            },
            .absorb => {
                const N = p.shape.n;
                const K = p.shape.k;
                const ng = N / 64;
                const terms = try p.a.alloc(f64, K);
                defer p.a.free(terms);
                for (0..p.shape.rows) |r| for (0..heads) |h| for (0..N) |j| {
                    for (0..K) |i| {
                        const wr = h * p.per_head + p.row_off + i;
                        const w = c.fromBf16(p.abs_s[wr * ng + j / 64]) * @as(f64, @floatFromInt(p.abs_q[wr * N + j])) + c.fromBf16(p.abs_b[wr * ng + j / 64]);
                        terms[i] = w * c.fromBf16(p.abs_x[r * p.stride + h * K + i]);
                    }
                    out[(r * heads + h) * N + j] = contract.rows.sumIn(acc, order, terms);
                };
            },
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Probe) = .empty,

    const list = [_]Variant{
        .{ .kind = .absorb },
        .{ .kind = .unabsorb },
        .{ .kind = .absorb, .ctl = 1 },
        .{ .kind = .absorb, .ctl = 2 },
        .{ .kind = .unabsorb, .ctl = 1 },
        .{ .kind = .unabsorb, .ctl = 2 },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        var libs: [3]?mtl.Library = @splat(null);
        defer for (libs) |l| if (l) |lib| lib.deinit();
        for (list) |v| {
            if (only) |text| if (std.mem.indexOf(u8, entryName(v), text) == null) continue;
            if (libs[v.ctl] == null) {
                const txt = try std.fmt.allocPrint(a, "#define TF_CTL_GLM_KVB8 {d}\n{s}", .{ v.ctl, sources.glm_kvb_q8 });
                defer a.free(txt);
                libs[v.ctl] = try mtl.Library.fromSource(device, txt, mtl.CompileOptions.mlx());
            }
            const sym = if (v.kind == .absorb) "glm_absorb_8b" else "glm_unabsorb_8b";
            const pipe = try mtl.Pipeline.init(device, libs[v.ctl].?, sym, false);
            errdefer pipe.deinit();
            const p = try a.create(Probe);
            p.* = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .pipe = pipe };
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
};
