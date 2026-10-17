//! The GPU gather gate: gathered dots and GLU equal independent sorted-row arithmetic, including mixed formats.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const c = @import("glm_rows_check.zig");
const wide = tf.glm.wide;
const gather = wide.gather;
const cases = @import("glm_gather_cases.zig");
const E = cases.experts;
const N = 64;
const K = 640;
const D = 12;
const pairs = cases.pairs;
const repeat = 3;
const F = struct { bits: u8, group: u16 };

fn reference(e: mtl.ComputeEncoder, g: *const c.Gpu, gate: c.Mat, up: c.Mat, ids: []const i32, order: []const i32, gu: mtl.Buffer, act: mtl.Buffer) void {
    for (order, 0..) |o, i| {
        const expert: usize = @intCast(ids[@intCast(o)]);
        const x = c.ref(gate.x()).at(@as(usize, @intCast(o)) / repeat * K * 2);
        for ([2]c.Mat{ gate, up }, 0..) |m, j| wide.dense(&g.p, e, m.part(expert, E), .{ .in = x, .ldx = K, .out = c.ref(gu).at((i * 2 + j) * N * 2), .ldo = 2 * N, .rows = 1 });
    }
    c.swiglu(e, g.swiglu, c.ref(gu), c.ref(act), pairs, N, 0.75);
}
fn downReference(e: mtl.ComputeEncoder, g: *const c.Gpu, down: c.Mat, ids: []const i32, order: []const i32, act: mtl.Buffer, y: mtl.Buffer) void {
    for (order, 0..) |o, i| {
        const expert: usize = @intCast(ids[@intCast(o)]);
        wide.dense(&g.p, e, down.part(expert, E), .{ .in = c.ref(act).at(i * N * 2), .ldx = N, .out = c.ref(y).at(@as(usize, @intCast(o)) * D * 2), .ldo = D, .rows = 1 });
    }
}
fn encode(e: mtl.ComputeEncoder, g: *const c.Gpu, q: tf.glm.affine.Affine, up: ?tf.glm.affine.Affine, io: gather.Io, ldo: u32) void {
    gather.encode(e, g.p.gather[@intFromBool(up == null)], q, up, io, pairs, E, if (up != null) repeat else 1, ldo, 0.75);
}
fn one(g: *const c.Gpu, gf: F, uf: F, df: F, kind: cases.Kind) !void {
    errdefer std.debug.print("gather failure: {s}, gate {d}/{d}, up {d}/{d}, down {d}/{d}\n", .{ @tagName(kind), gf.bits, gf.group, uf.bits, uf.group, df.bits, df.group });
    const gate = try c.Mat.init(g.device, E * N, K, gf.bits, gf.group, 188);
    defer gate.deinit();
    const up = try c.Mat.init(g.device, E * N, K, uf.bits, uf.group, 911);
    defer up.deinit();
    const down = try c.Mat.init(g.device, E * D, N, df.bits, df.group, 710);
    defer down.deinit();
    const ids = cases.ids(kind);
    var offsets: [E]i32 = undefined;
    var order: [pairs]i32 = undefined;
    var at: usize = 0;
    for (0..E) |expert| {
        offsets[expert] = @intCast(at);
        for (ids, 0..) |id, i| if (id == expert) {
            order[at] = @intCast(i);
            at += 1;
        };
    }
    const ob = try c.buf(g.device, (E + pairs) * 4);
    defer ob.deinit();
    @memcpy(ob.slice(i32, E), &offsets);
    @memcpy(ob.slice(i32, E + pairs)[E..], &order);
    const gu = try c.buf(g.device, pairs * 2 * N * 2);
    defer gu.deinit();
    const act = try c.buf(g.device, pairs * N * 2);
    defer act.deinit();
    const got = try c.buf(g.device, pairs * N * 2);
    defer got.deinit();
    const y = try c.buf(g.device, 2 * pairs * D * 2);
    defer y.deinit();
    try g.run(reference, .{ g, gate, up, &ids, &order, gu, act });
    var gq = gate.q;
    gq.n = N;
    var uq = up.q;
    uq.n = N;
    const io: gather.Io = .{ .x = c.ref(gate.x()), .out = c.ref(got), .offsets = c.ref(ob), .order = c.ref(ob).at(E * 4) };
    try g.run(encode, .{ g, gq, @as(?tf.glm.affine.Affine, uq), io, N });
    try c.eql(got.slice(u16, pairs * N), act.slice(u16, pairs * N));
    try g.run(downReference, .{ g, down, &ids, &order, act, y });
    var dq = down.q;
    dq.n = D;
    try g.run(encode, .{ g, dq, @as(?tf.glm.affine.Affine, null), gather.Io{ .x = c.ref(act), .out = c.ref(y).at(pairs * D * 2), .offsets = io.offsets, .order = io.order }, D });
    const both = y.slice(u16, 2 * pairs * D);
    try c.eql(both[0 .. pairs * D], both[pairs * D ..]);
}

pub fn check(g: *const c.Gpu) !void {
    const bits = tf.glm.affine.row_bits;
    for (cases.kinds) |kind| {
        for (bits) |b| for ([_]u16{ 32, 64 }) |group| {
            const f: F = .{ .bits = b, .group = group };
            try one(g, f, f, f, kind);
        };
        for (bits, 0..) |gb, gi| for (bits, 0..) |ub, ui| for ([_]u16{ 32, 64 }, 0..) |gg, ggi| for ([_]u16{ 32, 64 }, 0..) |ug, ugi| {
            try one(g, .{ .bits = gb, .group = gg }, .{ .bits = ub, .group = ug }, .{ .bits = bits[(gi + ui + 1) % bits.len], .group = if ((ggi + ugi) % 2 == 0) 32 else 64 }, kind);
        };
        std.debug.print("gather GLU and scatter == sorted arithmetic: {s}, 12 uniform and 144 mixed format cases\n", .{@tagName(kind)});
    }
    std.debug.print("gather equality: {d} distributions, {d} format/routing cases, 35 experts, 37 pairs\n", .{ cases.kinds.len, cases.kinds.len * 156 });
}
