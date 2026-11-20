//! GLM mixed 4/64 routed + 8/64 shared fused MoE kernels under gates 1-4, controls, and bit-exact predecessor check.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const kernels = tf.glm.kernels;
const moe_route = tf.moe_route;

const E = 10;
const D = 4096;
const N = 2048;
const TOPK = 8;
const MAXR = 16;
const MAX_EXPERTS = kernels.route_shape.experts;
const swiglu_limit: f32 = 10.0;
const routed_scale: f32 = 2.5;
const pick_slot = "const int u = PART == 1 ? MAXU : int(threadgroup_position_in_grid.z);";

pub const Kind = enum { batch, pick };
const Variant = struct { kind: Kind, control: bool = false };

const batch_served = [_]c.Shape{ .{ .rows = 4, .n = N, .k = D }, .{ .rows = 1, .n = N, .k = D }, .{ .rows = 2, .n = N, .k = D }, .{ .rows = 5, .n = N, .k = D }, .{ .rows = 8, .n = N, .k = D } };
const pick_served = [_]c.Shape{.{ .rows = 1, .n = N, .k = D }};

pub fn entryName(v: Variant) []const u8 {
    return switch (v.kind) {
        .batch => if (v.control) "control: glm sh8 fused MoE takes shared slot one early" else "glm sh8 four/two glm_moe_gateup_0_sh8 and glm_moe_down_0_sh8",
        .pick => if (v.control) "control: glm sh8 pick takes wrong routed expert" else "glm sh8 pick glm_moe_gateup_0_sh8_pick and glm_moe_down_0_sh8_pick",
    };
}

pub fn buildSources(a: std.mem.Allocator) ![8][]u8 {
    const gu = comptime kernels.generatedSource("moe_gateup_2") ++ kernels.instance("moe_gateup_2", "4096, 2048, 4, 8, 16, 128, 0, 4, 16, 8", "glm_moe_gateup_0");
    const dn = comptime kernels.generatedSource("moe_down_2") ++ kernels.instance("moe_down_2", "2048, 4096, 4, 8, 16, 128, 0, 4, 16, 8", "glm_moe_down_0");
    const gu0 = try std.mem.replaceOwned(u8, a, gu, pick_slot, "const int u = PART == 1 ? MAXU : (int(threadgroup_position_in_grid.z) == int(X_shape[0]) * TOPK ? MAXU : int(threadgroup_position_in_grid.z));");
    defer a.free(gu0);
    const dn0 = try std.mem.replaceOwned(u8, a, dn, pick_slot, "const int u = PART == 1 ? MAXU : (int(threadgroup_position_in_grid.z) == int(ACT_shape[0]) * TOPK ? MAXU : int(threadgroup_position_in_grid.z));");
    defer a.free(dn0);
    const gu0_one = try std.mem.replaceOwned(u8, a, gu0, "<4096, 2048, 4, 8, 16, 128, 0, 4, 16, 8>", "<4096, 2048, 2, 8, 16, 128, 0, 4, 16, 8>");
    defer a.free(gu0_one);
    const dn0_one = try std.mem.replaceOwned(u8, a, dn0, "<2048, 4096, 4, 8, 16, 128, 0, 4, 16, 8>", "<2048, 4096, 2, 8, 16, 128, 0, 4, 16, 8>");
    defer a.free(dn0_one);
    const member = "  if (!shared && u >= UCOUNT[0]) return;\n  const int pick = shared ? (m < R ? m * TOPK : -1) : UMEM[u * MAXR + m];";
    const by_pick = "  const int pick = shared ? (m < R ? m * TOPK : -1) : u;";
    const rc = try moe_route.source(a, kernels.route_shape);
    defer a.free(rc);
    const g1 = try std.mem.replaceOwned(u8, a, gu0_one, member, by_pick);
    defer a.free(g1);
    const g2 = try std.mem.replaceOwned(u8, a, g1, "  const size_t e = shared ? 0 : size_t(UIDS[u]);", "  const size_t e = shared ? 0 : size_t(tf_route_pick_at(RLOGITS, RBIAS, u, lane, u == TOPK - 1 && threadgroup_position_in_grid.y == 0, RPICK, RWTS, RSCALE));");
    defer a.free(g2);
    const g3 = try std.mem.replaceOwned(u8, a, g2, "  const device int32_t* UIDS [[buffer(11)]],\n  const device int32_t* UMEM [[buffer(12)]],\n  const device int32_t* UCOUNT [[buffer(13)]],", "  const device float* RLOGITS [[buffer(11)]],\n  const device float* RBIAS [[buffer(12)]],\n  device int32_t* RPICK [[buffer(13)]],");
    defer a.free(g3);
    const g4 = try std.mem.replaceOwned(u8, a, g3, "  device bfloat16_t* ACT [[buffer(15)]],", "  device bfloat16_t* ACT [[buffer(15)]],\n  device float* RWTS [[buffer(16)]],\n  constant float& RSCALE [[buffer(17)]],");
    defer a.free(g4);
    const p_gu = try std.mem.concat(a, u8, &.{ rc, g4 });
    defer a.free(p_gu);
    const p_dn = try std.mem.replaceOwned(u8, a, dn0_one, member, by_pick);
    defer a.free(p_dn);
    const gu_from = "4096, 2048, 4, 8, 16, 128, 0, 4, 16, 8";
    const dn_from = "2048, 4096, 4, 8, 16, 128, 0, 4, 16, 8";
    const gu_one_from = "4096, 2048, 2, 8, 16, 128, 0, 4, 16, 8";
    const dn_one_from = "2048, 4096, 2, 8, 16, 128, 0, 4, 16, 8";
    const s0 = try kernels.sh8Source(a, gu0, true, gu_from, "4096, 2048, 4, 8, 16, 128, 0, 8, 16, 16", "glm_moe_gateup_0_sh8");
    errdefer a.free(s0);
    const s1 = try kernels.sh8Source(a, dn0, false, dn_from, "2048, 4096, 4, 8, 16, 128, 0, 8, 16, 16", "glm_moe_down_0_sh8");
    errdefer a.free(s1);
    const s2 = try kernels.sh8Source(a, gu0, true, gu_from, "4096, 2048, 2, 8, 16, 128, 0, 8, 16, 16", "glm_moe_gateup_0_sh8_one");
    errdefer a.free(s2);
    const s3 = try kernels.sh8Source(a, dn0, false, dn_from, "2048, 4096, 2, 8, 16, 128, 0, 8, 16, 16", "glm_moe_down_0_sh8_one");
    errdefer a.free(s3);
    const s4 = try kernels.sh8Source(a, p_gu, true, gu_one_from, "4096, 2048, 2, 8, 16, 128, 0, 8, 16, 16", "glm_moe_gateup_0_sh8_pick");
    errdefer a.free(s4);
    const s5 = try kernels.sh8Source(a, p_dn, false, dn_one_from, "2048, 4096, 2, 8, 16, 128, 0, 8, 16, 16", "glm_moe_down_0_sh8_pick");
    errdefer a.free(s5);
    const comb = comptime kernels.generatedSource("moe_combine") ++ "";
    const ca = try std.mem.replaceOwned(u8, a, comb, "const device bfloat* y = Y + size_t(r) * TOPK * D + d;", "const device bfloat* y = Y + size_t(r) * (TOPK + 1) * D + d;");
    defer a.free(ca);
    const s6 = try std.mem.replaceOwned(u8, a, ca, "+ YS[size_t(r) * D + d];", "+ y[size_t(TOPK) * D];");
    errdefer a.free(s6);
    const s7 = try a.dupe(u8, rc);
    return .{ s0, s1, s2, s3, s4, s5, s6, s7 };
}

fn pipeFrom(device: mtl.Device, text: []const u8, sym: [:0]const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, sym, false);
}

fn depth(_: c.Shape) usize {
    return 0;
}

fn spec(v: Variant) c.Spec {
    return .{
        .name = entryName(v),
        .accum = .none,
        .depth = depth,
        .out = .bf16,
        .replaces = "moe_2_and_wide_q8",
        .nan = .propagate,
        .tiles = .{ .{ .size = 16 }, .{ .size = 64, .step = 1, .min = 1 }, .{ .size = 512, .step = 64, .min = 64 } },
        .served = if (v.kind == .batch) &batch_served else &pick_served,
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    gu4: mtl.Pipeline,
    dn4: mtl.Pipeline,
    gu2: mtl.Pipeline,
    dn2: mtl.Pipeline,
    gu_pick: mtl.Pipeline,
    dn_pick: mtl.Pipeline,
    comb0: mtl.Pipeline,
    base_gu: mtl.Pipeline,
    base_dn: mtl.Pipeline,
    base_comb: mtl.Pipeline,
    route_sel: mtl.Pipeline,
    wide: tf.glm.wide.Pipes,
    shape: c.Shape = undefined,
    refs: [27]rig_.Ref = undefined,
    acts_ref: rig_.Ref = undefined,
    ys_ref: rig_.Ref = undefined,
    base_y: rig_.Ref = undefined,
    out_bytes: std.ArrayList(u8) = .empty,

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.out_bytes.deinit(p.a);
        inline for (.{ p.gu4, p.dn4, p.gu2, p.dn2, p.gu_pick, p.dn_pick, p.comb0, p.base_gu, p.base_dn, p.base_comb, p.route_sel }) |pp| pp.deinit();
        p.wide.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.v), .ctx = p, .vt = &vt, .control = if (p.v.control) .shapes else null };
    }

    const vt = g.hooks(Probe);

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.rig.outputBytes();
        const want = try p.replaced();
        if (!std.mem.eql(u8, got, want)) return "sh8 fused MoE bytes differ from predecessor";
        return null;
    }

    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        p.rig.reset();
        if (s.n != N or s.k != D or s.rows == 0 or s.rows > 8) return false;
        p.shape = s;
        const n_exp: usize = E;
        const mspecs = [_][3]usize{
            .{ n_exp * N, D, 4 }, .{ n_exp * N, D, 4 }, .{ 2 * N, D, 8 }, .{ n_exp * D, N, 4 }, .{ D, N, 8 },
        };
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const xb = try p.a.alloc(u16, s.rows * D);
        defer p.a.free(xb);
        for (xb) |*v| v.* = c.toBf16((rnd.float(f32) - 0.5) * 2.0);
        if (f == .nonfinite) {
            xb[0] = 0x7fc0;
            if (s.rows > 1) xb[D + 5] = 0x7f80;
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(xb));
        for (mspecs, 0..) |ms, mi| {
            const wb = try p.a.alloc(u8, ms[0] * ms[1] * ms[2] / 8);
            defer p.a.free(wb);
            rnd.bytes(wb);
            const sb = try p.a.alloc(u16, ms[0] * (ms[1] / 64));
            defer p.a.free(sb);
            const bb = try p.a.alloc(u16, ms[0] * (ms[1] / 64));
            defer p.a.free(bb);
            const div: f32 = if (ms[2] == 8) 512.0 else 32.0;
            for (sb, bb) |*sv, *bv| {
                sv.* = c.toBf16((rnd.float(f32) + 0.5) / div);
                bv.* = c.toBf16(-(rnd.float(f32) + 0.5) * 0.1);
            }
            p.refs[1 + mi * 3] = try p.rig.input(wb);
            p.refs[2 + mi * 3] = try p.rig.input(std.mem.sliceAsBytes(sb));
            p.refs[3 + mi * 3] = try p.rig.input(std.mem.sliceAsBytes(bb));
        }
        const ids = try p.a.alloc(i32, MAX_EXPERTS);
        defer p.a.free(ids);
        const mem = try p.a.alloc(i32, MAX_EXPERTS * MAXR);
        defer p.a.free(mem);
        const pk = try p.a.alloc(i32, MAXR * TOPK);
        defer p.a.free(pk);
        const lg = try p.a.alloc(f32, s.rows * MAX_EXPERTS);
        defer p.a.free(lg);
        const bi = try p.a.alloc(f32, MAX_EXPERTS);
        defer p.a.free(bi);
        const wt = try p.a.alloc(f32, MAXR * TOPK);
        defer p.a.free(wt);
        @memset(ids, 0);
        @memset(mem, -1);
        @memset(lg, 0);
        @memset(bi, -std.math.inf(f32));
        for (0..E) |ei| bi[ei] = rnd.float(f32) * 0.1;
        for (0..s.rows) |r| {
            var k: usize = 0;
            while (k < TOPK) {
                const ev: i32 = @intCast(rnd.uintLessThan(usize, E));
                if (std.mem.indexOfScalar(i32, pk[r * TOPK ..][0..k], ev) == null) {
                    pk[r * TOPK + k] = ev;
                    wt[r * TOPK + k] = 0.05 + rnd.float(f32) * 0.2;
                    k += 1;
                }
            }
            for (0..E) |ei| lg[r * MAX_EXPERTS + ei] = rnd.float(f32) * 2.0 - 1.0;
        }
        var n_u: usize = 0;
        for (0..E) |ei| {
            var m: usize = 0;
            for (pk[0 .. s.rows * TOPK], 0..) |pv, pi| if (pv == @as(i32, @intCast(ei))) {
                mem[n_u * MAXR + m] = @intCast(pi);
                m += 1;
            };
            if (m > 0) {
                ids[n_u] = @intCast(ei);
                n_u += 1;
            }
        }
        const uc: [1]i32 = .{@intCast(n_u)};
        const rl_sz = TOPK * MAXR * 4;
        if (p.v.kind == .pick) {
            p.refs[16] = try p.rig.scratch(MAX_EXPERTS * 4);
            p.refs[17] = try p.rig.scratch(MAX_EXPERTS * MAXR * 4);
            p.refs[18] = try p.rig.scratch(64);
            p.refs[19] = try p.rig.scratch(MAXR * (TOPK + 1) * N * 2);
            p.refs[20] = try p.rig.scratch(MAXR * (TOPK + 1) * D * 2);
            p.refs[21] = try p.rig.scratch(MAXR * TOPK * 4);
        } else {
            p.refs[16] = try p.rig.input(std.mem.sliceAsBytes(ids));
            p.refs[17] = try p.rig.input(std.mem.sliceAsBytes(mem));
            p.refs[18] = try p.rig.input(std.mem.asBytes(&uc));
            p.refs[19] = try p.rig.scratch(MAXR * (TOPK + 1) * N * 2);
            p.refs[20] = try p.rig.scratch(MAXR * (TOPK + 1) * D * 2);
            p.refs[21] = try p.rig.input(std.mem.sliceAsBytes(wt));
        }
        p.refs[22] = try p.rig.output(s.rows * D * 2);
        p.refs[23] = try p.rig.input(std.mem.sliceAsBytes(lg));
        p.refs[24] = try p.rig.input(std.mem.sliceAsBytes(bi));
        p.refs[25] = try p.rig.scratch(MAXR * TOPK * 4);
        p.refs[26] = try p.rig.scratch(rl_sz * 2 + 128);
        p.acts_ref = try p.rig.scratch(s.rows * N * 2);
        p.ys_ref = try p.rig.scratch(s.rows * D * 2);
        p.base_y = try p.rig.scratch(s.rows * D * 2);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows: u32 = @intCast(p.shape.rows);
        const zs = rows * TOPK + 1;
        const gu_two = rows <= 4;
        const gu_rps: u32 = if (gu_two) 2 else 4;
        const dn_rps: u32 = if (rows == 1) 2 else 4;
        const is_pick = p.v.kind == .pick;
        if (is_pick) {
            for (0..rows) |row| {
                e.setPipeline(p.gu_pick);
                e.setBuffer(p.refs[0].buf, p.refs[0].off + row * D * 2, 0);
                e.setBytes(std.mem.asBytes(&[2]i32{ 1, D }), 1);
                for (1..10, 2..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
                e.setBuffer(p.refs[23].buf, p.refs[23].off + row * MAX_EXPERTS * 4, 11);
                e.setBuffer(p.refs[24].buf, p.refs[24].off, 12);
                e.setBuffer(p.refs[25].buf, p.refs[25].off + row * TOPK * 4, 13);
                e.setValue(swiglu_limit, 14);
                e.setBuffer(p.refs[19].buf, p.refs[19].off + row * (TOPK + 1) * N * 2, 15);
                e.setBuffer(p.refs[21].buf, p.refs[21].off + row * TOPK * 4, 16);
                e.setValue(routed_scale, 17);
                e.dispatchThreads(mtl.Size.of(32, N / 2, TOPK + 1), mtl.Size.of(32, 1, 1));
                e.setPipeline(p.dn_pick);
                e.setBuffer(p.refs[19].buf, p.refs[19].off + row * (TOPK + 1) * N * 2, 0);
                e.setBytes(std.mem.asBytes(&[3]i32{ 1, TOPK + 1, N }), 1);
                for (10..16, 2..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
                e.setBuffer(p.refs[25].buf, p.refs[25].off + row * TOPK * 4, 8);
                e.setBuffer(p.refs[25].buf, p.refs[25].off + row * TOPK * 4, 9);
                e.setBuffer(p.refs[25].buf, p.refs[25].off + row * TOPK * 4, 10);
                e.setBuffer(p.refs[20].buf, p.refs[20].off + row * (TOPK + 1) * D * 2, 11);
                e.dispatchThreads(mtl.Size.of(32, D / 2, TOPK + 1), mtl.Size.of(32, 1, 1));
            }
        } else {
            e.setPipeline(if (gu_two) p.gu2 else p.gu4);
            e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
            e.setBytes(std.mem.asBytes(&[2]i32{ @intCast(rows), D }), 1);
            for (1..10, 2..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
            for (16..19, 11..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
            e.setValue(swiglu_limit, 14);
            e.setBuffer(p.refs[19].buf, p.refs[19].off, 15);
            e.dispatchThreads(mtl.Size.of(32 * rows, N / gu_rps, zs), mtl.Size.of(32 * rows, 1, 1));
            e.setPipeline(if (rows == 1) p.dn2 else p.dn4);
            e.setBuffer(p.refs[19].buf, p.refs[19].off, 0);
            e.setBytes(std.mem.asBytes(&[3]i32{ @intCast(rows), TOPK + 1, N }), 1);
            for (10..16, 2..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
            for (16..19, 8..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
            e.setBuffer(p.refs[20].buf, p.refs[20].off, 11);
            e.dispatchThreads(mtl.Size.of(32 * rows, D / dn_rps, zs), mtl.Size.of(32 * rows, 1, 1));
        }
        e.setPipeline(p.comb0);
        e.setBuffer(p.refs[20].buf, p.refs[20].off, 0);
        e.setBuffer(p.refs[20].buf, p.refs[20].off, 1);
        e.setBuffer(p.refs[21].buf, p.refs[21].off, 2);
        e.setBytes(std.mem.asBytes(&[2]i32{ @intCast(rows), TOPK }), 3);
        e.setBuffer(p.refs[22].buf, p.refs[22].off, 4);
        e.dispatchThreads(mtl.Size.of(rows * D, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        const rows: u32 = @intCast(p.shape.rows);
        const rl_sz = TOPK * MAXR * 4;
        const pass = p.rig.begin();
        const e = pass.e;
        const sh_gu: tf.glm.weights.Affine = .{ .w = .{ .buf = p.refs[7].buf, .off = p.refs[7].off }, .s = .{ .buf = p.refs[8].buf, .off = p.refs[8].off }, .b = .{ .buf = p.refs[9].buf, .off = p.refs[9].off }, .n = 2 * N, .k = D, .bits = 8, .group = 64 };
        const sh_dn: tf.glm.weights.Affine = .{ .w = .{ .buf = p.refs[13].buf, .off = p.refs[13].off }, .s = .{ .buf = p.refs[14].buf, .off = p.refs[14].off }, .b = .{ .buf = p.refs[15].buf, .off = p.refs[15].off }, .n = D, .k = N, .bits = 8, .group = 64 };
        const acts = p.acts_ref;
        const ys = p.ys_ref;
        if (p.v.kind == .pick) {
            const rl = p.refs[26];
            moe_route.select(e, p.route_sel, .{ .buf = p.refs[23].buf, .off = p.refs[23].off }, .{ .buf = p.refs[24].buf, .off = p.refs[24].off }, routed_scale, rows, .{ 0, MAX_EXPERTS }, .{ .pick = .{ .buf = p.refs[25].buf, .off = p.refs[25].off }, .wts = .{ .buf = p.refs[21].buf, .off = p.refs[21].off }, .ids = .{ .buf = p.refs[16].buf, .off = p.refs[16].off }, .members = .{ .buf = p.refs[17].buf, .off = p.refs[17].off }, .count = .{ .buf = p.refs[18].buf, .off = p.refs[18].off }, .mine = .{ .buf = rl.buf, .off = rl.off }, .theirs = .{ .buf = rl.buf, .off = rl.off + rl_sz }, .counts = .{ .buf = rl.buf, .off = rl.off + 2 * rl_sz }, .word = .{ .buf = rl.buf, .off = rl.off + 2 * rl_sz + 64 } });
        }
        tf.glm.wide.glu(&p.wide, e, sh_gu, N, .{ .buf = p.refs[0].buf, .off = p.refs[0].off }, .{ .buf = acts.buf, .off = acts.off }, rows, swiglu_limit);
        tf.glm.wide.dense(&p.wide, e, sh_dn, .{ .in = .{ .buf = acts.buf, .off = acts.off }, .ldx = N, .out = .{ .buf = ys.buf, .off = ys.off }, .ldo = D, .rows = rows });
        e.setPipeline(p.base_gu);
        e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
        e.setBytes(std.mem.asBytes(&[2]i32{ @intCast(rows), D }), 1);
        for (1..10, 2..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
        for (16..19, 11..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
        e.setValue(swiglu_limit, 14);
        e.setBuffer(p.refs[19].buf, p.refs[19].off, 15);
        e.dispatchThreads(mtl.Size.of(32 * rows, N / 4, rows * TOPK), mtl.Size.of(32 * rows, 1, 1));
        e.setPipeline(p.base_dn);
        e.setBuffer(p.refs[19].buf, p.refs[19].off, 0);
        e.setBytes(std.mem.asBytes(&[3]i32{ @intCast(rows), TOPK, N }), 1);
        for (10..16, 2..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
        for (16..19, 8..) |j, slot| e.setBuffer(p.refs[j].buf, p.refs[j].off, slot);
        e.setBuffer(p.refs[20].buf, p.refs[20].off, 11);
        e.dispatchThreads(mtl.Size.of(32 * rows, D / 4, rows * TOPK), mtl.Size.of(32 * rows, 1, 1));
        e.setPipeline(p.base_comb);
        e.setBuffer(ys.buf, ys.off, 0);
        e.setBuffer(p.refs[20].buf, p.refs[20].off, 1);
        e.setBuffer(p.refs[21].buf, p.refs[21].off, 2);
        e.setBytes(std.mem.asBytes(&[2]i32{ @intCast(rows), TOPK }), 3);
        e.setBuffer(p.base_y.buf, p.base_y.off, 4);
        e.dispatchThreads(mtl.Size.of(rows * D, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
        return p.rig.bytes(p.base_y, p.shape.rows * D * 2);
    }

    pub fn count(p: *Probe) usize {
        return p.shape.rows * D;
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Probe) = .empty,

    const list = [_]Variant{ .{ .kind = .batch }, .{ .kind = .pick }, .{ .kind = .batch, .control = true }, .{ .kind = .pick, .control = true } };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ks: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        var srcs: ?[8][]u8 = null;
        defer if (srcs) |ss| for (ss) |txt| a.free(txt);
        for (list) |v| {
            if (only) |text| if (std.mem.indexOf(u8, entryName(v), text) == null) continue;
            if (srcs == null) srcs = try buildSources(a);
            const ss = srcs.?;
            const bad_gu = if (v.control and v.kind == .batch)
                try std.mem.replaceOwned(u8, a, ss[2], "int(threadgroup_position_in_grid.z) == int(X_shape[0]) * TOPK", "int(threadgroup_position_in_grid.z) == int(X_shape[0]) * TOPK - 1")
            else if (v.control and v.kind == .pick)
                try std.mem.replaceOwned(u8, a, ss[4], "tf_route_pick_at(RLOGITS, RBIAS, u, lane,", "tf_route_pick_at(RLOGITS, RBIAS, (u + 1) % TOPK, lane,")
            else
                try a.dupe(u8, ss[if (v.kind == .batch) 2 else 4]);
            defer a.free(bad_gu);
            const p = try a.create(Probe);
            p.* = .{
                .a = a,
                .rig = .{ .a = a, .device = device, .queue = queue },
                .v = v,
                .gu4 = try pipeFrom(device, ss[0], "glm_moe_gateup_0_sh8"),
                .dn4 = try pipeFrom(device, ss[1], "glm_moe_down_0_sh8"),
                .gu2 = try pipeFrom(device, if (v.kind == .batch) bad_gu else ss[2], "glm_moe_gateup_0_sh8_one"),
                .dn2 = try pipeFrom(device, ss[3], "glm_moe_down_0_sh8_one"),
                .gu_pick = try pipeFrom(device, if (v.kind == .pick) bad_gu else ss[4], "glm_moe_gateup_0_sh8_pick"),
                .dn_pick = try pipeFrom(device, ss[5], "glm_moe_down_0_sh8_pick"),
                .comb0 = try pipeFrom(device, ss[6], kernels.generatedFunction("moe_combine")),
                .base_gu = try pipeFrom(device, kernels.generatedSource("moe_gateup_2"), kernels.generatedFunction("moe_gateup_2")),
                .base_dn = try pipeFrom(device, kernels.generatedSource("moe_down_2"), kernels.generatedFunction("moe_down_2")),
                .base_comb = try pipeFrom(device, kernels.generatedSource("moe_combine"), kernels.generatedFunction("moe_combine")),
                .route_sel = try pipeFrom(device, ss[7], moe_route.names[1]),
                .wide = try tf.glm.wide.load(a, device),
            };
            try s.probes.append(a, p);
            try ks.append(a, p.kernel());
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
