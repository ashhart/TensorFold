//! GLM's decode experts as moe.zig's `both` launches them against four outputs a simdgroup, on the same route tables.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const kernels = tf.glm.kernels;

const E = 10; // experts in the pool: a row's 8 picks overlap the next row's
const D = 4096;
const N = 2048;
const TOPK = 8;
const MAXR = 16; // route_select's member rows an expert
const swiglu_limit: f32 = 10.0; // config.zig's
const pick_slot = "const int u = PART == 1 ? MAXU : int(threadgroup_position_in_grid.z);";

/// kernels.zig's routed-and-shared gathers: four outputs a simdgroup, and the `_one` pipelines' two.
pub fn texts(a: std.mem.Allocator) ![4][]u8 {
    const gu = comptime kernels.generatedSource("moe_gateup_2") ++ kernels.instance("moe_gateup_2", "4096, 2048, 4, 8, 16, 128, 0, 4, 16, 8", "glm_moe_gateup_0");
    const dn = comptime kernels.generatedSource("moe_down_2") ++ kernels.instance("moe_down_2", "2048, 4096, 4, 8, 16, 128, 0, 4, 16, 8", "glm_moe_down_0");
    const gu4 = try std.mem.replaceOwned(u8, a, gu, pick_slot, "const int u = PART == 1 ? MAXU : (int(threadgroup_position_in_grid.z) == int(X_shape[0]) * TOPK ? MAXU : int(threadgroup_position_in_grid.z));");
    const dn4 = try std.mem.replaceOwned(u8, a, dn, pick_slot, "const int u = PART == 1 ? MAXU : (int(threadgroup_position_in_grid.z) == int(ACT_shape[0]) * TOPK ? MAXU : int(threadgroup_position_in_grid.z));");
    const gu2 = try std.mem.replaceOwned(u8, a, gu4, "<4096, 2048, 4, 8, 16, 128, 0, 4, 16, 8>", "<4096, 2048, 2, 8, 16, 128, 0, 4, 16, 8>");
    const dn2 = try std.mem.replaceOwned(u8, a, dn4, "<2048, 4096, 4, 8, 16, 128, 0, 4, 16, 8>", "<2048, 4096, 2, 8, 16, 128, 0, 4, 16, 8>");
    return .{ gu4, dn4, gu2, dn2 };
}

/// Rows: the window's rows; n: experts' intermediate width; k: the hidden width.
const served = [_]c.Shape{ .{ .rows = 4, .n = N, .k = D }, .{ .rows = 1, .n = N, .k = D }, .{ .rows = 2, .n = N, .k = D }, .{ .rows = 3, .n = N, .k = D }, .{ .rows = 5, .n = N, .k = D }, .{ .rows = 8, .n = N, .k = D } };

const Slot = enum(u8) { x, gw, gs, gb, uw, us, ub, sw, ss, sb, dw, ds, db, hw, hs, hb, ids, members, count, act, y };

pub const Case = struct {
    control: bool,
    new: [2][2]mtl.Pipeline, // [gate/up, down][two outputs a simdgroup, four]
    old: [2]mtl.Pipeline,

    pub fn init(s: *bits.Suite, control: bool) !Case {
        const loader = try s.production();
        const t = try texts(s.a);
        defer for (t) |x| s.a.free(x);
        for (t) |x| _ = try loader.require(x);
        const shared = "(int(threadgroup_position_in_grid.z) == int(X_shape[0]) * TOPK ? MAXU";
        const gu2 = if (control) try bits.edited(s.a, t[2], "", shared, "(int(threadgroup_position_in_grid.z) == int(X_shape[0]) * TOPK - 1 ? MAXU") else try s.a.dupe(u8, t[2]);
        defer s.a.free(gu2);
        var x: Case = .{ .control = control, .new = undefined, .old = undefined };
        x.new[0][0] = try bits.pipeline(s.device, gu2, "glm_moe_gateup_0");
        errdefer x.new[0][0].deinit();
        x.new[1][0] = try bits.pipeline(s.device, t[3], "glm_moe_down_0");
        errdefer x.new[1][0].deinit();
        x.old[0] = try bits.pipeline(s.device, t[0], "glm_moe_gateup_0");
        errdefer x.old[0].deinit();
        x.old[1] = try bits.pipeline(s.device, t[1], "glm_moe_down_0");
        x.new[0][1] = x.old[0];
        x.new[1][1] = x.old[1];
        return x;
    }
    pub fn deinit(x: *Case) void {
        x.new[0][0].deinit();
        x.new[1][0].deinit();
        for (x.old) |p| p.deinit();
    }
    pub fn name(control: bool) []const u8 {
        return if (control) "control: metal glm launch experts, two gate/up outputs a simdgroup taking the shared slot one early" else "metal glm launch experts gate/up and down at moe.zig's outputs a simdgroup";
    }
    pub fn spec(x: *const Case) c.Spec {
        return bits.spec(name(x.control), "glm_moe_gateup_0 and glm_moe_down_0 at four outputs a simdgroup", "the four-output kernels' bytes in every pick's activations and outputs and the shared slot's", &served, "the generated kernels' sums; only the outputs a simdgroup change");
    }
    pub fn layout(_: *const Case, s: c.Shape) ?bits.Layout {
        if (s.rows == 0 or s.rows > 8 or s.n != N or s.k != D) return null;
        const w: tf.glm.affine.Width = .{ .bits = 4, .group = 64 };
        var l: bits.Layout = .{};
        _ = l.add(.input, s.rows * D * 2, .bf16);
        for ([_][2]usize{ .{ E * N, D }, .{ E * N, D }, .{ 2 * N, D }, .{ E * D, N }, .{ D, N } }) |m| {
            _ = l.add(.input, tf.glm.affine.wBytes(m[0], m[1], w), .int);
            _ = l.add(.input, tf.glm.affine.sBytes(m[0], m[1], w), .bf16);
            _ = l.add(.input, tf.glm.affine.sBytes(m[0], m[1], w), .bf16);
        }
        _ = l.add(.input, 128 * 4, .int);
        _ = l.add(.input, 128 * MAXR * 4, .int);
        _ = l.add(.input, 16, .int);
        _ = l.add(.output, s.rows * (TOPK + 1) * N * 2, .bf16);
        _ = l.add(.output, s.rows * (TOPK + 1) * D * 2, .bf16);
        return l;
    }
    /// tf-glm-moe-rows' draws and route: each row 8 distinct experts of the pool, grouped as route_select groups them.
    pub fn fill(_: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        for (std.mem.bytesAsSlice(u16, data[@backingInt(Slot.x)])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 2);
        for (0..5) |m| {
            const at = @backingInt(Slot.gw) + 3 * m;
            rnd.bytes(data[at]);
            for (std.mem.bytesAsSlice(u16, data[at + 1])) |*v| v.* = bits.bf(rnd.float(f32) * 0.02 + 0.001);
            for (std.mem.bytesAsSlice(u16, data[at + 2])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 0.1);
        }
        var picks: [MAXR * TOPK]i32 = undefined;
        for (0..s.rows) |row| {
            var k: usize = 0;
            while (k < TOPK) {
                const e: i32 = @intCast(rnd.uintLessThan(usize, E));
                if (std.mem.indexOfScalar(i32, picks[row * TOPK .. row * TOPK + k], e) == null) {
                    picks[row * TOPK + k] = e;
                    k += 1;
                }
            }
        }
        const ids = std.mem.bytesAsSlice(i32, data[@backingInt(Slot.ids)]);
        const members = std.mem.bytesAsSlice(i32, data[@backingInt(Slot.members)]);
        @memset(ids, 0);
        @memset(members, -1);
        var n: usize = 0;
        for (0..E) |e| {
            var m: usize = 0;
            for (picks[0 .. s.rows * TOPK], 0..) |p, pi| if (p == @as(i32, @intCast(e))) {
                members[n * MAXR + m] = @intCast(pi);
                m += 1;
            };
            if (m > 0) {
                ids[n] = @intCast(e);
                n += 1;
            }
        }
        std.mem.bytesAsSlice(i32, data[@backingInt(Slot.count)])[0] = @intCast(n);
        if (f == .stress or (f == .random and seed % 3 != 2)) return;
        const x = std.mem.bytesAsSlice(u16, data[@backingInt(Slot.x)]);
        for ([_][2]usize{ .{ 0, 5 }, .{ 1, 700 }, .{ 2, 64 }, .{ 3, 9 }, .{ 3, 4000 } }, [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80, 0xff80 }) |p, v| if (p[0] < s.rows) {
            x[p[0] * D + p[1]] = v;
        };
    }
    /// moe.zig's `both`: gate/up at two outputs a simdgroup to 4 rows, down at two at one row; the predecessor at four.
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const rows = s.rows;
        const zs = rows * TOPK + 1;
        const gu_rps: usize = if (new and rows <= 4) 2 else 4;
        const dn_rps: usize = if (new and rows == 1) 2 else 4;
        const bind = struct {
            fn at(enc: mtl.ComputeEncoder, v: rig_.Ref, j: usize) void {
                enc.setBuffer(v.buf, v.off, j);
            }
        }.at;
        e.setPipeline(if (new) x.new[0][if (gu_rps == 2) 0 else 1] else x.old[0]);
        bind(e, r[@backingInt(Slot.x)], 0);
        e.setBytes(std.mem.asBytes(&[2]i32{ @intCast(rows), D }), 1);
        for (@backingInt(Slot.gw)..@backingInt(Slot.dw), 2..) |j, k| bind(e, r[j], k);
        for ([_]Slot{ .ids, .members, .count }, 11..) |j, k| bind(e, r[@backingInt(j)], k);
        e.setValue(swiglu_limit, 14);
        bind(e, r[@backingInt(Slot.act)], 15);
        e.dispatchThreads(mtl.Size.of(32 * rows, N / gu_rps, zs), mtl.Size.of(32 * rows, 1, 1));
        e.setPipeline(if (new) x.new[1][if (dn_rps == 2) 0 else 1] else x.old[1]);
        bind(e, r[@backingInt(Slot.act)], 0);
        e.setBytes(std.mem.asBytes(&[3]i32{ @intCast(rows), TOPK + 1, N }), 1);
        for (@backingInt(Slot.dw)..@backingInt(Slot.ids), 2..) |j, k| bind(e, r[j], k);
        for ([_]Slot{ .ids, .members, .count, .y }, 8..) |j, k| bind(e, r[@backingInt(j)], k);
        e.dispatchThreads(mtl.Size.of(32 * rows, D / dn_rps, zs), mtl.Size.of(32 * rows, 1, 1));
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for ([_]bool{ false, true }) |control| {
        if (s.wanted(only, Case.name(control))) try s.add(Case, try Case.init(s, control));
    }
}

/// Gate 5's registrations `glm launch experts <form> <function>`: the loader's gather at that form.
pub const prefix = "glm launch experts ";
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, label: []const u8) !bits.Texts {
    const t = try texts(a);
    const one = std.mem.indexOf(u8, label, " two ") != null;
    const down = std.mem.endsWith(u8, label, "glm_moe_down_0");
    const pick = t[@as(usize, if (one) 2 else 0) + @intFromBool(down)];
    return .{ .here = pick, .pre_m5 = pick };
}
