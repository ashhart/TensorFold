//! Gates 1-4 contract probes and red controls for DSpark Q2/Q4 pack, proj, screen, refine, pick, and sample.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_dspark_host;
const controls = @import("metal_inkling2_dspark_screen_controls.zig");

const H = host.hidden;
const RANK = host.rank;
const NGROUPS = H / 64;

pub const Kind = enum { q2_pack, q4_pack, q2_proj, q2_proj_units, q4_units, q2_screen, q2_refine, q2_pick, q2_sample };

fn mutation(k: Kind) controls.Mutation {
    return switch (k) {
        .q2_pack => .q2_pack,
        .q4_pack => .q4_pack,
        .q2_proj => .q2_proj,
        .q2_proj_units => .q2_proj_units,
        .q4_units => .q4_units,
        .q2_screen => .q2_screen,
        .q2_refine => .q2_refine,
        .q2_pick => .q2_pick,
        .q2_sample => .q2_sample,
    };
}

fn noSums(_: c.Shape) usize {
    return 1;
}
fn hr(b: rig_.Ref) host.Ref {
    return .{ .buf = b.buf, .off = b.off };
}
fn mix64(x_in: u64) u64 {
    var x = x_in;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

fn spec(k: Kind, control: bool) c.Spec {
    const label = if (control) mutation(k).name() else switch (k) {
        .q2_pack => "metal inkling2 dspark chain_q2_pack tf_ink2_dspark_q2_pack",
        .q4_pack => "metal inkling2 dspark chain_q4_pack tf_ink2_dspark_q4_pack",
        .q2_proj => "metal inkling2 dspark chain_q2_proj tf_ink2_dspark_q2_proj",
        .q2_proj_units => "metal inkling2 units tf_ink2_dspark_q2_proj_units",
        .q4_units => "metal inkling2 units tf_ink2_dspark_q4_units",
        .q2_screen => "metal inkling2 dspark chain_q2_screen tf_ink2_dspark_q2_screen",
        .q2_refine => "metal inkling2 dspark chain_q2_refine tf_ink2_dspark_q2_refine",
        .q2_pick => "metal inkling2 dspark chain_q2_pick tf_ink2_dspark_q2_pick",
        .q2_sample => "metal inkling2 dspark chain_q2_sample tf_ink2_dspark_q2_sample",
    };
    return switch (k) {
        .q2_pack => .{ .name = label, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "2-bit affine pack reproduces group min/scale and quantized codes" }, .served = &.{.{ .rows = 16, .n = H, .k = 0 }} },
        .q4_pack => .{ .name = label, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "4-bit affine pack reproduces group min/scale and quantized codes" }, .served = &.{.{ .rows = 16, .n = H, .k = 0 }} },
        .q2_proj => .{ .name = label, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "2-bit factored half4 unembed matches dequantized FP64 reference" }, .served = &.{.{ .rows = 3, .n = 64, .k = H }} },
        .q2_proj_units => .{ .name = label, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "2-bit NAX unembed matches dequantized FP64 reference" }, .served = &.{.{ .rows = 3, .n = 64, .k = H }} },
        .q4_units => .{ .name = label, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "4-bit NAX 3-way split projection matches dequantized FP64 reference" }, .served = &.{.{ .rows = 8, .n = 128, .k = H }} },
        .q2_screen => .{ .name = label, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "2-bit shortlist screen picks exact top-2 per partition" }, .served = &.{.{ .rows = 1, .n = 512, .k = RANK }} },
        .q2_refine => .{ .name = label, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "exact BF16 refinement matches unembed + Phi + beta + softcap30" }, .served = &.{.{ .rows = 1, .n = 257, .k = H }} },
        .q2_pick => .{ .name = label, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "NaN at slot 256 picks id 0; else ties go to lower candidate id" }, .served = &.{ .{ .rows = 1, .n = 257, .k = 0 }, .{ .rows = 1, .n = 257, .k = 1 } } },
        .q2_sample => .{ .name = label, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "SplitMix64 Gumbel-max bitonic sampler matches CPU reference" }, .served = &.{ .{ .rows = 1, .n = 256, .k = 0 }, .{ .rows = 1, .n = 256, .k = 1 } } },
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    screen: host.Screen,
    kind: Kind,
    ctl: controls.Mutation,
    shape: c.Shape = undefined,
    refs: [8]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    bf_a: []u16 = &.{},
    bf_b: []u16 = &.{},
    bf_c: []u16 = &.{},
    bf_d: []u16 = &.{},
    bf_e: []u16 = &.{},
    u32_a: []u32 = &.{},
    u16_h: []u16 = &.{},
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const mut = if (control) mutation(kind) else .none;
        const dtext = try controls.applyDspark(a, mut);
        defer a.free(dtext);
        const utext = try controls.applyUnits(a, mut);
        defer a.free(utext);
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .screen = try host.Screen.loadSources(device, dtext, utext), .kind = kind, .ctl = mut };
    }
    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.arena.deinit();
        p.screen.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = p.ctl.gate(), .spec = spec(p.kind, p.ctl != .none) };
    }
    const vt = g.hooks(Probe);

    fn randBf(p: *Probe, n: usize, rng: std.Random, scale: f32) ![]u16 {
        const out = try p.arena.allocator().alloc(u16, n);
        for (out) |*v| v.* = c.toBf16((rng.float(f32) * 2.0 - 1.0) * scale);
        return out;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (p.kind) {
            .q2_pack, .q4_pack => {
                if ((shape.rows != 16 and fill != .nonfinite) or shape.n != H) return false;
                p.bf_a = try p.randBf(16 * H, rng, 2.0);
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.bf_a));
                const wpg: usize = if (p.kind == .q2_pack) 4 else 8;
                p.outs[0] = try p.rig.output(16 * NGROUPS * wpg * 4);
                p.outs[1] = try p.rig.output(16 * NGROUPS * 4);
            },
            .q2_proj, .q2_proj_units => {
                if ((shape.rows != 3 and fill != .nonfinite) or shape.n != 64 or shape.k != H) return false;
                p.bf_a = try p.randBf(64 * H, rng, 0.5);
                p.bf_b = try p.randBf(3 * H, rng, 0.5);
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.bf_a));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.bf_b));
                p.refs[2] = try p.rig.scratch(64 * NGROUPS * 4 * 4);
                p.refs[3] = try p.rig.scratch(64 * NGROUPS * 4);
                p.outs[0] = try p.rig.output(3 * 64 * 2);
            },
            .q4_units => {
                if ((shape.rows != 8 and fill != .nonfinite) or shape.n != 128 or shape.k != H) return false;
                p.bf_a = try p.randBf(128 * H, rng, 0.5);
                p.bf_b = try p.randBf(8 * H, rng, 0.5);
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.bf_a));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.bf_b));
                p.refs[2] = try p.rig.scratch(128 * NGROUPS * 8 * 4);
                p.refs[3] = try p.rig.scratch(128 * NGROUPS * 4);
                p.outs[0] = try p.rig.output(8 * 64 * 2);
                p.outs[1] = try p.rig.output(8 * 32 * 2);
                p.outs[2] = try p.rig.output(8 * 32 * 2);
            },
            .q2_screen => {
                if ((shape.rows != 1 and fill != .nonfinite) or shape.n != 512 or shape.k != RANK) return false;
                const vocab: usize = 512;
                p.u16_h = try p.arena.allocator().alloc(u16, vocab);
                for (p.u16_h) |*v| v.* = @bitCast(@as(f16, @floatCast((rng.float(f32) * 2.0 - 1.0) * 0.5)));
                p.bf_a = try p.randBf(vocab * RANK, rng, 0.2);
                p.bf_b = try p.randBf(vocab, rng, 4.0);
                p.bf_c = try p.randBf(RANK, rng, 0.2);
                p.bf_b[2] = c.toBf16(100.0);
                p.bf_b[0] = c.toBf16(80.0);
                p.bf_b[510] = c.toBf16(90.0);
                p.bf_b[509] = c.toBf16(70.0);
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.u16_h));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.bf_a));
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.bf_b));
                p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.bf_c));
                p.refs[4] = try p.rig.scratch(vocab * (RANK / 64) * 4 * 4);
                p.refs[5] = try p.rig.scratch(vocab * (RANK / 64) * 4);
                p.outs[0] = try p.rig.output(256 * 4);
            },
            .q2_refine => {
                if ((shape.rows != 1 and fill != .nonfinite) or shape.n != 257 or shape.k != H) return false;
                const vocab: usize = 260;
                p.u32_a = try p.arena.allocator().alloc(u32, 256);
                for (p.u32_a, 0..) |*id, i| id.* = @intCast((i * 7 + 1) % vocab);
                p.bf_a = try p.randBf(vocab * H, rng, 0.3);
                p.bf_b = try p.randBf(H, rng, 0.3);
                p.bf_c = try p.randBf(vocab * RANK, rng, 0.3);
                p.bf_d = try p.randBf(vocab, rng, 2.0);
                p.bf_e = try p.randBf(RANK, rng, 0.3);
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.u32_a));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.bf_a));
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.bf_b));
                p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.bf_c));
                p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.bf_d));
                p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(p.bf_e));
                p.outs[0] = try p.rig.output(257 * 2);
                p.outs[1] = try p.rig.output(257 * 4);
            },
            .q2_pick => {
                if ((shape.rows != 1 and fill != .nonfinite) or shape.n != 257 or shape.k > 1) return false;
                p.bf_a = try p.arena.allocator().alloc(u16, 257);
                p.u32_a = try p.arena.allocator().alloc(u32, 257);
                for (p.bf_a, p.u32_a, 0..) |*v, *id, i| {
                    v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 8))) - 4.0);
                    id.* = if (i == 256) 0 else @intCast(1000 - i);
                }
                p.bf_a[10] = c.toBf16(12.0);
                p.bf_a[20] = c.toBf16(12.0);
                if (fill == .nonfinite) p.bf_a[5] = c.toBf16(std.math.nan(f32));
                if (shape.k == 1) p.bf_a[256] = c.toBf16(std.math.nan(f32));
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.bf_a));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.u32_a));
                p.outs[0] = try p.rig.output(8 * 4);
            },
            .q2_sample => {
                if ((shape.rows != 1 and fill != .nonfinite) or shape.n != 256 or shape.k > 1) return false;
                p.bf_a = try p.arena.allocator().alloc(u16, 256);
                p.u32_a = try p.arena.allocator().alloc(u32, 256);
                for (p.bf_a, p.u32_a, 0..) |*v, *id, i| {
                    v.* = c.toBf16((rng.float(f32) * 2.0 - 1.0) * 1.5);
                    id.* = @intCast(500 + i * 3);
                }
                if (fill == .nonfinite) p.bf_a[3] = c.toBf16(std.math.nan(f32));
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.bf_a));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.u32_a));
                p.outs[0] = try p.rig.output(8 * 4);
            },
        }
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        switch (p.kind) {
            .q2_pack => host.q2Pack(&p.screen, e, hr(p.refs[0]), hr(p.outs[0]), hr(p.outs[1]), 16, NGROUPS),
            .q4_pack => host.q4Pack(&p.screen, e, hr(p.refs[0]), hr(p.outs[0]), hr(p.outs[1]), 16, NGROUPS),
            .q2_proj => {
                host.q2Pack(&p.screen, e, hr(p.refs[0]), hr(p.refs[2]), hr(p.refs[3]), 64, NGROUPS);
                e.barrier();
                e.setPipeline(p.screen.proj);
                for ([_]rig_.Ref{ p.refs[2], p.refs[3], p.refs[1], p.outs[0] }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
                e.setValue(host.BaseArgs{ .rows = 3, .vocab = 64 }, 4);
                e.dispatchGroups(mtl.Size.of(8, 1, 1), mtl.Size.of(256, 1, 1));
            },
            .q2_proj_units => {
                host.q2Pack(&p.screen, e, hr(p.refs[0]), hr(p.refs[2]), hr(p.refs[3]), 64, NGROUPS);
                e.barrier();
                host.q2Proj(&p.screen, e, hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[1]), hr(p.outs[0]), 3, 64);
            },
            .q4_units => {
                host.q4Pack(&p.screen, e, hr(p.refs[0]), hr(p.refs[2]), hr(p.refs[3]), 128, NGROUPS);
                e.barrier();
                host.q4Units(&p.screen, e, hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[1]), hr(p.outs[0]), hr(p.outs[1]), hr(p.outs[2]), H, 64, 32, 32);
            },
            .q2_screen => {
                host.q2Pack(&p.screen, e, hr(p.refs[1]), hr(p.refs[4]), hr(p.refs[5]), 512, RANK / 64);
                e.barrier();
                host.q2ScreenCount(&p.screen, e, hr(p.refs[0]), hr(p.refs[4]), hr(p.refs[5]), hr(p.refs[2]), hr(p.refs[3]), hr(p.outs[0]), 512);
            },
            .q2_refine => host.q2Refine(&p.screen, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), hr(p.refs[5]), hr(p.outs[0]), hr(p.outs[1])),
            .q2_pick => host.q2Pick(&p.screen, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.outs[0]), 2),
            .q2_sample => host.q2Sample(&p.screen, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.outs[0]), .{
                .step = 2,
                .position = if (p.shape.k == 0) 17 else 99,
                .seed_lo = if (p.shape.k == 0) 42 else 7,
                .seed_hi = 0,
                .top_k = if (p.shape.k == 0) 20 else 8,
                .inv_t = if (p.shape.k == 0) 1.0 else 2.0,
                .top_p = if (p.shape.k == 0) 0.95 else 0.8,
                .min_log = if (p.shape.k == 0) -std.math.inf(f32) else @log(@as(f32, 0.1)),
            }),
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return switch (p.kind) {
            .q2_pack => 16 * NGROUPS * 20,
            .q4_pack => 16 * NGROUPS * 36,
            .q2_proj, .q2_proj_units => 3 * 64 * 2,
            .q4_units => 8 * 128 * 2,
            .q2_screen => 256 * 4,
            .q2_refine => 257 * 6,
            .q2_pick, .q2_sample => 8 * 4,
        };
    }
    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    fn dequantMat(p: *Probe, rows: usize, k_dim: usize, bits: u5) ![]f64 {
        const ng = k_dim / 64;
        const wpg: usize = if (bits == 2) 4 else 8;
        const codes = std.mem.bytesAsSlice(u32, p.rig.bytes(p.refs[2], rows * ng * wpg * 4));
        const params = std.mem.bytesAsSlice(u16, p.rig.bytes(p.refs[3], rows * ng * 4));
        const mat = try p.arena.allocator().alloc(f64, rows * k_dim);
        const mask: u32 = if (bits == 2) 3 else 15;
        const per_w: usize = if (bits == 2) 16 else 8;
        for (0..rows) |r| for (0..ng) |gr| {
            const gi = r * ng + gr;
            const sc: f64 = @floatCast(@as(f16, @bitCast(params[2 * gi])));
            const lo: f64 = @floatCast(@as(f16, @bitCast(params[2 * gi + 1])));
            for (0..wpg) |wi| {
                const word = codes[gi * wpg + wi];
                for (0..per_w) |j| {
                    const q: f64 = @floatFromInt((word >> @intCast(j * bits)) & mask);
                    mat[r * k_dim + gr * 64 + wi * per_w + j] = lo + sc * q;
                }
            }
        };
        return mat;
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        switch (p.kind) {
            .q2_pack, .q4_pack => {
                const max_q: f32 = if (p.kind == .q2_pack) 3.0 else 15.0;
                const wpg: usize = if (p.kind == .q2_pack) 4 else 8;
                const bits: u5 = if (p.kind == .q2_pack) 2 else 4;
                const mask: u32 = if (bits == 2) 3 else 15;
                const per_w: usize = if (bits == 2) 16 else 8;
                const codes = std.mem.bytesAsSlice(u32, p.rig.bytes(p.outs[0], 16 * NGROUPS * wpg * 4));
                const params = std.mem.bytesAsSlice(u16, p.rig.bytes(p.outs[1], 16 * NGROUPS * 4));
                for (0..16 * NGROUPS) |gi| {
                    const src_g = p.bf_a[gi * 64 ..][0..64];
                    var lo = c.fromBf16(src_g[0]);
                    var hi = lo;
                    for (src_g[1..]) |b| {
                        const v = c.fromBf16(b);
                        lo = @min(lo, v);
                        hi = @max(hi, v);
                    }
                    const sc: f32 = @floatCast(@as(f16, @bitCast(params[2 * gi])));
                    const p_lo: f32 = @floatCast(@as(f16, @bitCast(params[2 * gi + 1])));
                    const want_sc = (hi - lo) / max_q;
                    if (@abs(sc - want_sc) > 0.02 * @max(1.0, @abs(want_sc)) or @abs(p_lo - lo) > 0.02 * @max(1.0, @abs(lo)))
                        return try std.fmt.bufPrint(&p.msg, "group {d} scale/lo mismatch: got ({d},{d}) want ({d},{d})", .{ gi, sc, p_lo, want_sc, lo });
                    for (0..64) |j| {
                        const q = (codes[gi * wpg + j / per_w] >> @intCast((j % per_w) * bits)) & mask;
                        const rec = p_lo + sc * @as(f32, @floatFromInt(q));
                        if (@abs(rec - c.fromBf16(src_g[j])) > 0.65 * @max(want_sc, 0.01))
                            return try std.fmt.bufPrint(&p.msg, "group {d} elem {d} quant error", .{ gi, j });
                    }
                }
            },
            .q2_proj, .q2_proj_units => {
                const w = try p.dequantMat(64, H, 2);
                const got_u16 = std.mem.bytesAsSlice(u16, p.rig.bytes(p.outs[0], 3 * 64 * 2));
                for (0..3) |r| for (0..64) |v| {
                    var dot: f64 = 0;
                    for (0..H) |d| dot += w[v * H + d] * @as(f64, c.fromBf16(p.bf_b[r * H + d]));
                    const got: f64 = @floatCast(@as(f16, @bitCast(got_u16[r * 64 + v])));
                    if (@abs(got - dot) > 0.05 * @max(1.0, @abs(dot)))
                        return try std.fmt.bufPrint(&p.msg, "q2_proj ({d},{d}) got {d} want {d}", .{ r, v, got, dot });
                };
            },
            .q4_units => {
                const w = try p.dequantMat(128, H, 4);
                const outs = [3][]align(1) const u16{
                    std.mem.bytesAsSlice(u16, p.rig.bytes(p.outs[0], 8 * 64 * 2)),
                    std.mem.bytesAsSlice(u16, p.rig.bytes(p.outs[1], 8 * 32 * 2)),
                    std.mem.bytesAsSlice(u16, p.rig.bytes(p.outs[2], 8 * 32 * 2)),
                };
                const ns = [3]usize{ 64, 32, 32 };
                var col_base: usize = 0;
                for (outs, ns, 0..) |out_s, nn, seg| {
                    for (0..8) |r| for (0..nn) |j| {
                        var dot: f64 = 0;
                        for (0..H) |d| dot += w[(col_base + j) * H + d] * @as(f64, c.fromBf16(p.bf_b[r * H + d]));
                        const got: f64 = c.fromBf16(out_s[r * nn + j]);
                        if (@abs(got - dot) > 0.04 * @max(1.0, @abs(dot)))
                            return try std.fmt.bufPrint(&p.msg, "q4_units seg {d} ({d},{d}) got {d} want {d}", .{ seg, r, j, got, dot });
                    };
                    col_base += nn;
                }
            },
            .q2_screen => {
                const cands = std.mem.bytesAsSlice(u32, p.rig.bytes(p.outs[0], 256 * 4));
                if (cands[0] != 2 or cands[1] != 0 or cands[254] != 510 or cands[255] != 509)
                    return try std.fmt.bufPrint(&p.msg, "screen top2 mismatch: [{d},{d}] .. [{d},{d}]", .{ cands[0], cands[1], cands[254], cands[255] });
            },
            .q2_refine => {
                const got_l = std.mem.bytesAsSlice(u16, p.rig.bytes(p.outs[0], 257 * 2));
                const got_i = std.mem.bytesAsSlice(u32, p.rig.bytes(p.outs[1], 257 * 4));
                for (0..257) |ci| {
                    const id: usize = if (ci == 256) 0 else p.u32_a[ci];
                    if (got_i[ci] != id) return try std.fmt.bufPrint(&p.msg, "cand_id[{d}] got {d} want {d}", .{ ci, got_i[ci], id });
                    var bsum: f64 = 0;
                    for (0..H) |d| bsum += @as(f64, c.fromBf16(p.bf_a[id * H + d])) * c.fromBf16(p.bf_b[d]);
                    var pdot: f64 = 0;
                    for (0..RANK) |d| pdot += @as(f64, c.fromBf16(p.bf_c[id * RANK + d])) * c.fromBf16(p.bf_e[d]);
                    const rbase: f64 = c.fromBf16(c.toBf16(@floatCast(bsum)));
                    const rbias: f64 = c.fromBf16(c.toBf16(@floatCast(@as(f64, c.fromBf16(p.bf_d[id])) + pdot)));
                    const rlogit: f64 = c.fromBf16(c.toBf16(@floatCast(rbase + rbias)));
                    const want: f64 = c.fromBf16(c.toBf16(@floatCast(30.0 * std.math.tanh(rlogit / 30.0))));
                    const got: f64 = c.fromBf16(got_l[ci]);
                    if (@abs(got - want) > 0.03 * @max(1.0, @abs(want)))
                        return try std.fmt.bufPrint(&p.msg, "refine[{d}] got {d} want {d}", .{ ci, got, want });
                }
            },
            .q2_pick => {
                const held = std.mem.bytesAsSlice(u32, p.rig.bytes(p.outs[0], 8 * 4));
                var want: u32 = 0;
                if (!std.math.isNan(c.fromBf16(p.bf_a[256]))) {
                    var best: f32 = -std.math.inf(f32);
                    want = std.math.maxInt(u32);
                    for (0..257) |i| {
                        const v = c.fromBf16(p.bf_a[i]);
                        if (!std.math.isNan(v) and (v > best or (v == best and p.u32_a[i] < want))) {
                            best = v;
                            want = p.u32_a[i];
                        }
                    }
                }
                if (held[3] != want) return try std.fmt.bufPrint(&p.msg, "q2_pick held[3]={d} want {d}", .{ held[3], want });
            },
            .q2_sample => {
                const held = std.mem.bytesAsSlice(u32, p.rig.bytes(p.outs[0], 8 * 4));
                const inv_t: f32 = if (p.shape.k == 0) 1.0 else 2.0;
                const top_k: usize = if (p.shape.k == 0) 20 else 8;
                const top_p: f32 = if (p.shape.k == 0) 0.95 else 0.8;
                const min_log: f32 = if (p.shape.k == 0) -std.math.inf(f32) else @log(@as(f32, 0.1));
                const seed: u64 = if (p.shape.k == 0) 42 else 7;
                const pos: u64 = if (p.shape.k == 0) 17 else 99;
                const Item = struct { v: f32, id: u32 };
                var items: [256]Item = undefined;
                for (&items, 0..) |*it, i| {
                    const r = c.fromBf16(p.bf_a[i]);
                    it.* = .{ .v = if (std.math.isNan(r)) -std.math.inf(f32) else r * inv_t, .id = p.u32_a[i] };
                }
                std.mem.sort(Item, &items, {}, struct {
                    fn less(_: void, a: Item, b: Item) bool {
                        return if (a.v != b.v) a.v > b.v else a.id < b.id;
                    }
                }.less);
                const m = items[0].v;
                var norm: f32 = 0;
                for (items[0..top_k]) |it| norm += @exp(it.v - m);
                var keep: usize = top_k;
                if (top_p > 0 and top_p < 1 and norm > 0) {
                    var cum: f32 = 0;
                    for (items[0..top_k], 0..) |it, j| {
                        cum += @exp(it.v - m) / norm;
                        if (cum >= top_p) {
                            keep = j + 1;
                            break;
                        }
                    }
                }
                if (min_log > -std.math.inf(f32)) {
                    var j: usize = 0;
                    while (j < keep and items[j].v >= m + min_log) : (j += 1) {}
                    keep = @max(j, 1);
                }
                const base = mix64(mix64(seed +% 0x9E3779B97F4A7C15) ^ (pos *% 0xD1B54A32D192ED03));
                var best_id = items[0].id;
                var best_g: f32 = -std.math.inf(f32);
                for (items[0..keep]) |it| {
                    const u: f32 = (@as(f32, @floatFromInt(@as(u32, @truncate(mix64(base ^ it.id) >> 40)))) + 0.5) * (1.0 / 16777216.0);
                    const gs = it.v - @log(-@log(u));
                    if (gs > best_g) {
                        best_g = gs;
                        best_id = it.id;
                    }
                }
                if (held[3] != best_id) return try std.fmt.bufPrint(&p.msg, "q2_sample held[3]={d} want {d}", .{ held[3], best_id });
            },
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [18]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Kind)) |kind| {
            if ((kind == .q2_proj_units or kind == .q4_units) and !s.device.tensorUnits()) continue;
            for ([_]bool{ false, true }) |control| {
                const label = if (control) mutation(kind).name() else spec(kind, false).name;
                if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
                s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, kind, control);
                s.made += 1;
                try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
            }
        }
    }
};
