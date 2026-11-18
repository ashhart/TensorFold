//! A one-row MoE layer's experts replaying the route select's picks (moe.zig's `both`, one row) against main's select.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const experts = @import("metal_glm_experts.zig");
const picks = @import("metal_glm_route_select.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const route = tf.moe_route;
const shape = tf.glm.kernels.route_shape;

const E = shape.experts;
const POOL = 16; // experts with weights: every other expert's bias is -inf, so the picks stay in the pool
const D = 4096;
const N = 2048;
const TOPK = 8;
const MAXR = 16;
const swiglu_limit: f32 = 10.0;
const scale: f32 = 2.5; // config.zig's routed_scale

/// kernels.zig's pickSources over the one-row gathers: groups take pick u by the select's routine, no member lists.
pub fn texts(a: std.mem.Allocator) ![2][]u8 {
    const t = try experts.texts(a);
    defer for (t) |x| a.free(x);
    const member = "  if (!shared && u >= UCOUNT[0]) return;\n  const int pick = shared ? (m < R ? m * TOPK : -1) : UMEM[u * MAXR + m];";
    const by_pick = "  const int pick = shared ? (m < R ? m * TOPK : -1) : u;";
    const edits = [_][2][]const u8{
        .{ member, by_pick },
        .{ "  const size_t e = shared ? 0 : size_t(UIDS[u]);", "  const size_t e = shared ? 0 : size_t(tf_route_pick_at(RLOGITS, RBIAS, u, lane, u == TOPK - 1 && threadgroup_position_in_grid.y == 0, RPICK, RWTS, RSCALE));" },
        .{ "  const device int32_t* UIDS [[buffer(11)]],\n  const device int32_t* UMEM [[buffer(12)]],\n  const device int32_t* UCOUNT [[buffer(13)]],", "  const device float* RLOGITS [[buffer(11)]],\n  const device float* RBIAS [[buffer(12)]],\n  device int32_t* RPICK [[buffer(13)]]," },
        .{ "  device bfloat16_t* ACT [[buffer(15)]],", "  device bfloat16_t* ACT [[buffer(15)]],\n  device float* RWTS [[buffer(16)]],\n  constant float& RSCALE [[buffer(17)]]," },
    };
    var gu = try a.dupe(u8, t[2]);
    defer a.free(gu);
    for (edits) |x| {
        if (std.mem.count(u8, gu, x[0]) != 1) return error.KernelCompile;
        const next = try std.mem.replaceOwned(u8, a, gu, x[0], x[1]);
        a.free(gu);
        gu = next;
    }
    const core = try route.source(a, shape);
    defer a.free(core);
    if (std.mem.count(u8, t[3], member) != 1) return error.KernelCompile;
    return .{ try std.mem.concat(a, u8, &.{ core, gu }), try std.mem.replaceOwned(u8, a, t[3], member, by_pick) };
}

/// Rows: one-row MoE layers launched one after another, each on its own input and logits row; n, k: N and D.
const served = [_]c.Shape{ .{ .rows = 6, .n = N, .k = D }, .{ .rows = 1, .n = N, .k = D }, .{ .rows = 16, .n = N, .k = D } };

const Slot = enum(u8) { x, gw, gs, gb, uw, us, ub, sw, ss, sb, dw, ds, db, hw, hs, hb, logits, bias, act, ye, pick, wts, lists, none };

pub const Case = struct {
    control: bool,
    new: [2]mtl.Pipeline,
    old: [3]mtl.Pipeline, // main's select, then the one-row gate/up and down

    pub fn init(s: *bits.Suite, control: bool) !Case {
        const loader = try s.production();
        const t = try texts(s.a);
        defer for (t) |x| s.a.free(x);
        for (t) |x| _ = try loader.require(x);
        const one = try experts.texts(s.a);
        defer for (one) |x| s.a.free(x);
        const gu = if (control) try bits.edited(s.a, t[0], "", "tf_route_pick_at(RLOGITS, RBIAS, u, lane,", "tf_route_pick_at(RLOGITS, RBIAS, (u + 1) % TOPK, lane,") else try s.a.dupe(u8, t[0]);
        defer s.a.free(gu);
        var x: Case = .{ .control = control, .new = undefined, .old = undefined };
        x.new[0] = try bits.pipeline(s.device, gu, "glm_moe_gateup_0");
        errdefer x.new[0].deinit();
        x.new[1] = try bits.pipeline(s.device, t[1], "glm_moe_down_0");
        errdefer x.new[1].deinit();
        const frozen = try picks.text(s.a, .glm, picks.main_route);
        defer s.a.free(frozen);
        x.old[0] = try bits.pipeline(s.device, frozen, route.names[1]);
        errdefer x.old[0].deinit();
        x.old[1] = try bits.pipeline(s.device, one[2], "glm_moe_gateup_0");
        errdefer x.old[1].deinit();
        x.old[2] = try bits.pipeline(s.device, one[3], "glm_moe_down_0");
        return x;
    }
    pub fn deinit(x: *Case) void {
        for (x.new) |p| p.deinit();
        for (x.old) |p| p.deinit();
    }
    pub fn name(control: bool) []const u8 {
        return if (control) "control: metal glm launch experts pick, each routed group taking the next pick's expert" else "metal glm launch experts pick one row, the gate/up groups replaying the route select";
    }
    pub fn spec(x: *const Case) c.Spec {
        return bits.spec(name(x.control), "main's tf_route_select, then the one-row gate/up and down gathers", "main's bytes in every pick's activations and outputs, the shared slot's, the picks and the weights", &served, "the select's weights and the generated kernels' sums in their orders; only who picks moves");
    }
    pub fn layout(_: *const Case, s: c.Shape) ?bits.Layout {
        if (s.rows == 0 or s.rows > 16 or s.n != N or s.k != D) return null;
        const w: tf.glm.affine.Width = .{ .bits = 4, .group = 64 };
        var l: bits.Layout = .{};
        _ = l.add(.input, s.rows * D * 2, .bf16);
        for ([_][2]usize{ .{ POOL * N, D }, .{ POOL * N, D }, .{ 2 * N, D }, .{ POOL * D, N }, .{ D, N } }) |m| {
            _ = l.add(.input, tf.glm.affine.wBytes(m[0], m[1], w), .int);
            _ = l.add(.input, tf.glm.affine.sBytes(m[0], m[1], w), .bf16);
            _ = l.add(.input, tf.glm.affine.sBytes(m[0], m[1], w), .bf16);
        }
        _ = l.add(.input, s.rows * E * 4, .f32);
        _ = l.add(.input, E * 4, .f32);
        _ = l.add(.output, s.rows * (TOPK + 1) * N * 2, .bf16);
        _ = l.add(.output, s.rows * (TOPK + 1) * D * 2, .bf16);
        _ = l.add(.output, s.rows * TOPK * 4, .int);
        _ = l.add(.output, s.rows * TOPK * 4, .f32);
        _ = l.add(.scratch, 4 * TOPK * MAXR * MAXR * 4, .int);
        _ = l.add(.scratch, 256, .int);
        return l;
    }
    /// Pool logits random, in a few levels (ties) or equal by row; gate 1 and every third shape add NaN, +-Inf, -0.
    pub fn fill(_: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const x = std.mem.bytesAsSlice(u16, data[@backingInt(Slot.x)]);
        for (x) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 2);
        for (0..5) |m| {
            const at = @backingInt(Slot.gw) + 3 * m;
            rnd.bytes(data[at]);
            for (std.mem.bytesAsSlice(u16, data[at + 1])) |*v| v.* = bits.bf(rnd.float(f32) * 0.02 + 0.001);
            for (std.mem.bytesAsSlice(u16, data[at + 2])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 0.1);
        }
        const lg = std.mem.bytesAsSlice(f32, data[@backingInt(Slot.logits)]);
        for (lg, 0..) |*v, i| v.* = switch ((seed + i / E) % 3) {
            0 => (rnd.float(f32) - 0.5) * 8,
            1 => @round((rnd.float(f32) - 0.5) * 6) / 2,
            else => 0.25,
        };
        for (std.mem.bytesAsSlice(f32, data[@backingInt(Slot.bias)]), 0..) |*v, i| v.* = if (i >= POOL) -std.math.inf(f32) else if (i % 5 == 0) -0.0 else (rnd.float(f32) - 0.5) * 0.2;
        if (f == .stress or (f == .random and seed % 3 != 2)) return;
        for (0..s.rows) |r| {
            lg[r * E + (r % POOL)] = if (r % 3 == 0) std.math.nan(f32) else if (r % 3 == 1) std.math.inf(f32) else -std.math.inf(f32);
            lg[r * E + ((r + 7) % POOL)] = -0.0;
            if (r % 4 == 0) x[r * D + 9] = if (r % 8 == 0) 0x7fc0 else 0xff80;
        }
    }
    /// Each one-row layer: moe.zig's one-row `both` with the picks replayed, or main's select then the gathers.
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const at = struct {
            fn of(v: rig_.Ref, extra: usize) rig_.Ref {
                return .{ .buf = v.buf, .off = v.off + extra };
            }
        }.of;
        const bind = struct {
            fn on(enc: mtl.ComputeEncoder, v: rig_.Ref, j: usize) void {
                enc.setBuffer(v.buf, v.off, j);
            }
        }.on;
        const ix = struct {
            fn of(slot: Slot) usize {
                return @backingInt(slot);
            }
        }.of;
        const T = TOPK * MAXR * 4;
        const lists = r[ix(.lists)];
        for (0..s.rows) |row| {
            const lg = at(r[ix(.logits)], row * E * 4);
            const pk = at(r[ix(.pick)], row * TOPK * 4);
            const wt = at(r[ix(.wts)], row * TOPK * 4);
            const act = at(r[ix(.act)], row * (TOPK + 1) * N * 2);
            if (!new) route.select(e, x.old[0], lg, r[ix(.bias)], scale, 1, .{ 0, E }, route.Outputs(rig_.Ref){ .pick = pk, .wts = wt, .ids = lists, .members = at(lists, T), .count = at(lists, T + T * MAXR), .mine = at(lists, 2 * T + T * MAXR), .theirs = at(lists, 3 * T + T * MAXR), .counts = at(lists, 4 * T + T * MAXR), .word = at(lists, 4 * T + T * MAXR + 64) });
            e.setPipeline(if (new) x.new[0] else x.old[1]);
            bind(e, at(r[ix(.x)], row * D * 2), 0);
            e.setBytes(std.mem.asBytes(&[2]i32{ 1, D }), 1);
            for (ix(.gw)..ix(.dw), 2..) |j, k| bind(e, r[j], k);
            if (new) {
                for ([_]rig_.Ref{ lg, r[ix(.bias)], pk }, 11..) |v, k| bind(e, v, k);
            } else for ([_]rig_.Ref{ lists, at(lists, T), at(lists, T + T * MAXR) }, 11..) |v, k| bind(e, v, k);
            e.setValue(swiglu_limit, 14);
            bind(e, act, 15);
            if (new) bind(e, wt, 16);
            if (new) e.setValue(scale, 17);
            e.dispatchThreads(mtl.Size.of(32, N / 2, TOPK + 1), mtl.Size.of(32, 1, 1));
            e.setPipeline(if (new) x.new[1] else x.old[2]);
            bind(e, act, 0);
            e.setBytes(std.mem.asBytes(&[3]i32{ 1, TOPK + 1, N }), 1);
            for (ix(.dw)..ix(.logits), 2..) |j, k| bind(e, r[j], k);
            if (new) {
                for ([_]rig_.Ref{ pk, r[ix(.none)], r[ix(.none)] }, 8..) |v, k| bind(e, v, k);
            } else for ([_]rig_.Ref{ lists, at(lists, T), at(lists, T + T * MAXR) }, 8..) |v, k| bind(e, v, k);
            bind(e, at(r[ix(.ye)], row * (TOPK + 1) * D * 2), 11);
            e.dispatchThreads(mtl.Size.of(32, D / 2, TOPK + 1), mtl.Size.of(32, 1, 1));
        }
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for ([_]bool{ false, true }) |control| {
        if (s.wanted(only, Case.name(control))) try s.add(Case, try Case.init(s, control));
    }
}

/// Gate 5's registrations `glm launch experts pick <function>`: the loader's pick gathers.
pub const prefix = "glm launch experts pick ";
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, label: []const u8) !bits.Texts {
    const t = try texts(a);
    const pick = t[@intFromBool(std.mem.endsWith(u8, label, "glm_moe_down_0"))];
    return .{ .here = pick, .pre_m5 = pick };
}
