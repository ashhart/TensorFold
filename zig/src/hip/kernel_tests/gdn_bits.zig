//! The DeltaNet scan's exact bits on fixed inputs: each prompt route, the decode plan and its replay.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const launches = @import("../launches.zig");

const Shape = struct { length: usize, key_heads: usize, value_heads: usize, dk: usize, dv: usize };

/// Inputs from integer draws only: unit-norm q and k rows, gates in [0.1, 1), beta in [0, 1), a state of 0.25 scale.
const Inputs = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    gate: []f32,
    beta: []f32,
    state: []f32,

    fn make(sh: Shape, seed: u64) !Inputs {
        var rng: rig.Rng = .{ .state = seed };
        const qk = sh.length * sh.key_heads * sh.dk;
        const gb = sh.length * sh.value_heads;
        const in: Inputs = .{
            .q = try gpa.alloc(f32, qk),
            .k = try gpa.alloc(f32, qk),
            .v = try gpa.alloc(f32, gb * sh.dv),
            .gate = try gpa.alloc(f32, gb),
            .beta = try gpa.alloc(f32, gb),
            .state = try gpa.alloc(f32, sh.value_heads * sh.dv * sh.dk),
        };
        for ([_][]f32{ in.q, in.k }) |a| {
            for (0..a.len / sh.dk) |row| {
                const r = a[row * sh.dk ..][0..sh.dk];
                var ss: f32 = 0;
                for (r) |*x| {
                    x.* = rng.unit();
                    ss += x.* * x.*;
                }
                const inv = 1 / @sqrt(ss);
                for (r) |*x| x.* *= inv;
            }
        }
        for (in.v) |*x| x.* = rng.unit();
        for (in.state) |*x| x.* = 0.25 * rng.unit();
        for (in.gate) |*x| x.* = 0.55 + 0.45 * rng.unit();
        for (in.beta) |*x| x.* = 0.5 + 0.5 * rng.unit();
        return in;
    }

    fn free(in: Inputs) void {
        for ([_][]f32{ in.q, in.k, in.v, in.gate, in.beta, in.state }) |a| gpa.free(a);
    }
};

/// FNV-1a over the bytes of each part in turn.
fn digest(parts: []const []const f32) u64 {
    var h = std.hash.Fnv1a_64.init();
    for (parts) |p| h.update(std.mem.sliceAsBytes(p));
    return h.final();
}

/// Every value finite and some output nonzero, so a digest never pins a launch that did nothing.
fn sane(y: []const f32, state: []const f32) !void {
    var any = false;
    for (y) |x| {
        try rig.expect(std.math.isFinite(x), "a non-finite output {d}", .{x});
        any = any or x != 0;
    }
    for (state) |x| try rig.expect(std.math.isFinite(x), "a non-finite state {d}", .{x});
    try rig.expect(any, "every output is zero", .{});
}

/// One prompt call (tf_gated_delta's route: half-head kernel, tile or wave): the digest of y, state and any snapshots.
fn prompt(t: *Rig, sh: Shape, snapshots: bool) !u64 {
    const in = try Inputs.make(sh, 0x6a09e667f3bcc908 +% sh.length *% 131 +% sh.dk *% 7 +% sh.dv);
    defer in.free();
    var q = try t.upload(in.q);
    defer q.free();
    var k = try t.upload(in.k);
    defer k.free();
    var v = try t.upload(in.v);
    defer v.free();
    var gate = try t.upload(in.gate);
    defer gate.free();
    var beta = try t.upload(in.beta);
    defer beta.free();
    var state = try t.upload(in.state);
    defer state.free();
    var y = try t.alloc(in.v.len * 4);
    defer y.free();
    try y.fill8(0);
    var states = try t.alloc(if (snapshots) sh.length * in.state.len * 4 else 4);
    defer states.free();
    try states.fill8(0);
    try t.ops(&t.on).gatedDelta(rig.at(q), rig.at(k), rig.at(v), rig.at(gate), rig.at(beta), rig.at(state), rig.at(y), sh.length, sh.key_heads, sh.value_heads, sh.dk, sh.dv, if (snapshots) rig.at(states) else null);
    try t.stream.synchronize();
    const gy = try rig.download(f32, y);
    defer gpa.free(gy);
    const gs = try rig.download(f32, state);
    defer gpa.free(gs);
    const gn = try rig.download(f32, states);
    defer gpa.free(gn);
    try sane(gy, gs);
    return digest(&.{ gy, gs, gn });
}

/// Six one-row rounds, a four-row window beside one row, a replay of two window rows: the digest of y and both states.
fn plan(t: *Rig, dk: usize) !u64 {
    const hk = 2;
    const hv = 6;
    const rows = 11;
    const rounds = 6;
    const sh: Shape = .{ .length = rows, .key_heads = hk, .value_heads = hv, .dk = dk, .dv = dk };
    const in = try Inputs.make(sh, 0xbb67ae8584caa73b +% dk);
    defer in.free();
    const other = try gpa.alloc(f32, in.state.len);
    defer gpa.free(other);
    for (other, 0..) |*x, i| x.* = in.state[in.state.len - 1 - i];

    // the last round's five rows packed as its snapshot: k, v, gate, beta
    const packed_rows = rows - rounds;
    var snap: std.ArrayList(f32) = .empty;
    defer snap.deinit(gpa);
    try snap.appendSlice(gpa, in.k[rounds * hk * dk ..][0 .. packed_rows * hk * dk]);
    try snap.appendSlice(gpa, in.v[rounds * hv * dk ..][0 .. packed_rows * hv * dk]);
    try snap.appendSlice(gpa, in.gate[rounds * hv ..][0 .. packed_rows * hv]);
    try snap.appendSlice(gpa, in.beta[rounds * hv ..][0 .. packed_rows * hv]);

    var q = try t.upload(in.q);
    defer q.free();
    var k = try t.upload(in.k);
    defer k.free();
    var v = try t.upload(in.v);
    defer v.free();
    var gate = try t.upload(in.gate);
    defer gate.free();
    var beta = try t.upload(in.beta);
    defer beta.free();
    var y = try t.alloc(in.v.len * 4);
    defer y.free();
    try y.fill8(0);
    var state0 = try t.upload(in.state);
    defer state0.free();
    var state1 = try t.upload(other);
    defer state1.free();
    var snapped = try t.upload(snap.items);
    defer snapped.free();
    var desc0 = try t.upload(&[4]u64{ 0, 0, 0, rig.at(state0) });
    defer desc0.free();
    var desc1 = try t.upload(&[4]u64{ 0, 0, 0, rig.at(state1) });
    defer desc1.free();
    var descs = try t.upload(&[2]u64{ rig.at(desc0), rig.at(desc1) });
    defer descs.free();
    var snaps = try t.upload(&[2]u64{ 0, rig.at(snapped) });
    defer snaps.free();
    // firsts of the six rounds, the window round's firsts and counts, a count of one, the replay's firsts and keeps
    var ints = try t.upload(&[_]i32{ 0, 1, 2, 3, 4, 5, rounds, rows - 1, 4, 1, 1, 0, 4, 2, 0 });
    defer ints.free();
    const int = rig.at(ints);

    const l = &t.on;
    const s = t.stream.handle;
    for (0..rounds) |i| {
        const p: launches.PlanRef = .{ .args = .{ .first = int + 4 * i, .count = int + 40, .desc = rig.at(descs) }, .rows = 1, .slots = 1 };
        try l.planGatedDelta(rig.at(q), rig.at(k), rig.at(v), rig.at(gate), rig.at(beta), rig.at(y), p, 0, hk, hv, dk, dk, s);
    }
    const round: launches.PlanRef = .{ .args = .{ .first = int + 24, .count = int + 32, .desc = rig.at(descs) }, .rows = packed_rows, .slots = 2 };
    try l.planGatedDelta(rig.at(q), rig.at(k), rig.at(v), rig.at(gate), rig.at(beta), rig.at(y), round, 0, hk, hv, dk, dk, s);
    const replay: launches.PlanRef = .{ .args = .{ .first = int + 44, .count = int + 32, .desc = rig.at(descs), .snaps = rig.at(snaps) }, .rows = packed_rows, .slots = 2 };
    try l.planGdnReplay(int + 52, replay, hk, hv, dk, dk, 1, s);
    try t.stream.synchronize();

    const gy = try rig.download(f32, y);
    defer gpa.free(gy);
    const g0 = try rig.download(f32, state0);
    defer gpa.free(g0);
    const g1 = try rig.download(f32, state1);
    defer gpa.free(g1);
    try sane(gy, g0);
    try sane(gy, g1);
    return digest(&.{ gy, g0, g1 });
}

test "the DeltaNet prompt routes keep the tile's and the wave's exact bits on fixed inputs" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    const Case = struct { shape: Shape, snapshots: bool = false, want: u64 };
    const cases = [_]Case{
        .{ .shape = .{ .length = 53, .key_heads = 2, .value_heads = 6, .dk = 128, .dv = 128 }, .want = 0x4eacafc73a369773 },
        .{ .shape = .{ .length = 16, .key_heads = 2, .value_heads = 6, .dk = 128, .dv = 128 }, .want = 0x473e0e8ffb309fcf },
        .{ .shape = .{ .length = 21, .key_heads = 2, .value_heads = 6, .dk = 128, .dv = 128 }, .snapshots = true, .want = 0x1d1567e82800c78a },
        .{ .shape = .{ .length = 37, .key_heads = 2, .value_heads = 6, .dk = 16, .dv = 16 }, .want = 0x47edc1580c7b0a52 },
        .{ .shape = .{ .length = 29, .key_heads = 2, .value_heads = 6, .dk = 128, .dv = 12 }, .want = 0x1cb94122ee1c760a },
        .{ .shape = .{ .length = 19, .key_heads = 2, .value_heads = 6, .dk = 16, .dv = 4 }, .want = 0xf0049ba65c446999 },
        // 64 rows or more at dk = dv = 128 take tf_gdn_prompt_128, held to the tile's bits (27B and 35B-A3B heads)
        .{ .shape = .{ .length = 64, .key_heads = 16, .value_heads = 48, .dk = 128, .dv = 128 }, .want = 0x27aebe37a9ff41d1 },
        .{ .shape = .{ .length = 83, .key_heads = 16, .value_heads = 32, .dk = 128, .dv = 128 }, .want = 0x24b5519dab7aaa08 },
    };
    var ok = true;
    for (cases, 0..) |c, i| {
        const got = try prompt(t, c.shape, c.snapshots);
        if (got != c.want) {
            std.debug.print("prompt case {d}: digest 0x{x:0>16}, want 0x{x:0>16}\n", .{ i, got, c.want });
            ok = false;
        }
    }
    try std.testing.expect(ok);
}

test "the DeltaNet decode plan and its replay keep their exact bits on fixed inputs" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    const cases = [_]struct { dk: usize, want: u64 }{ .{ .dk = 128, .want = 0xcece797e46731074 }, .{ .dk = 16, .want = 0xfa8203a4628d94ff } };
    var ok = true;
    for (cases) |c| {
        const got = try plan(t, c.dk);
        if (got != c.want) {
            std.debug.print("plan dk {d}: digest 0x{x:0>16}, want 0x{x:0>16}\n", .{ c.dk, got, c.want });
            ok = false;
        }
    }
    try std.testing.expect(ok);
}
