//! The DeltaNet recurrence, token-serial and chunked, against a float64 recurrence on random inputs.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;

const d = 128;
const Shape = struct { length: usize, key_heads: usize, value_heads: usize };

const Inputs = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    gate: []f32,
    beta: []f32,
    state: []f32,

    /// L2-normalized q and k, gates exp(-A softplus(x)) in (0, 1], beta = sigmoid, a state of 0.1 scale.
    fn make(sh: Shape, seed: u64) !Inputs {
        var rng: rig.Rng = .{ .state = seed };
        const qk = sh.length * sh.key_heads * d;
        const vy = sh.length * sh.value_heads * d;
        const gb = sh.length * sh.value_heads;
        const in: Inputs = .{
            .q = try gpa.alloc(f32, qk),
            .k = try gpa.alloc(f32, qk),
            .v = try gpa.alloc(f32, vy),
            .gate = try gpa.alloc(f32, gb),
            .beta = try gpa.alloc(f32, gb),
            .state = try gpa.alloc(f32, sh.value_heads * d * d),
        };
        for ([_][]f32{ in.q, in.k }) |a| {
            for (0..a.len / d) |row| {
                var ss: f64 = 0;
                for (a[row * d ..][0..d]) |*x| {
                    const z = rng.normal();
                    x.* = @floatCast(z);
                    ss += z * z;
                }
                const inv: f32 = @floatCast(1 / @sqrt(ss));
                for (a[row * d ..][0..d]) |*x| x.* *= inv;
            }
        }
        for (in.v) |*x| x.* = @floatCast(rng.normal());
        for (in.state) |*x| x.* = @floatCast(0.1 * rng.normal());
        for (in.gate, in.beta, 0..) |*g, *b, i| {
            const h = i % sh.value_heads;
            const a_log = 0.5 + 7.5 * @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(sh.value_heads));
            const x = rng.normal() - 2;
            const sp = if (x > 20) x else @log(1 + @exp(x));
            g.* = @floatCast(@exp(-a_log * sp));
            b.* = @floatCast(1 / (1 + @exp(-rng.normal())));
        }
        return in;
    }

    fn free(in: Inputs) void {
        for ([_][]f32{ in.q, in.k, in.v, in.gate, in.beta, in.state }) |a| gpa.free(a);
    }
};

/// The recurrence in float64: y (L, Hv, d) and the final state.
fn reference(sh: Shape, in: Inputs, y: []f64, state: []f64) !void {
    const s = try gpa.alloc(f64, d * d);
    defer gpa.free(s);
    const group = sh.value_heads / sh.key_heads;
    for (0..sh.value_heads) |h| {
        for (s, in.state[h * d * d ..][0 .. d * d]) |*a, b| a.* = b;
        const kh = h / group;
        for (0..sh.length) |t| {
            const gb = t * sh.value_heads + h;
            const decay: f64 = in.gate[gb];
            const beta: f64 = in.beta[gb];
            const k = in.k[(t * sh.key_heads + kh) * d ..][0..d];
            const q = in.q[(t * sh.key_heads + kh) * d ..][0..d];
            const v = in.v[gb * d ..][0..d];
            for (0..d) |r| {
                const row = s[r * d ..][0..d];
                var kv: f64 = 0;
                for (row, k) |*a, kk| {
                    a.* *= decay;
                    kv += a.* * kk;
                }
                const delta = (v[r] - kv) * beta;
                var out: f64 = 0;
                for (row, k, q) |*a, kk, qq| {
                    a.* += kk * delta;
                    out += a.* * qq;
                }
                y[gb * d + r] = out;
            }
        }
        @memcpy(state[h * d * d ..][0 .. d * d], s);
    }
}

/// The largest error over the reference's largest value.
fn relative(got: []const f32, want: []const f64) f64 {
    var scale: f64 = 0;
    var max: f64 = 0;
    for (got, want) |g, w| {
        max = @max(max, @abs(@as(f64, g) - w));
        scale = @max(scale, @abs(w));
    }
    return max / scale;
}

test "the serial and chunked DeltaNet stay within 5e-3 of the float64 recurrence at four lengths" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    for ([_]Shape{
        .{ .length = 64, .key_heads = 2, .value_heads = 4 },
        .{ .length = 200, .key_heads = 2, .value_heads = 4 },
        .{ .length = 1000, .key_heads = 2, .value_heads = 4 },
        .{ .length = 4170, .key_heads = 2, .value_heads = 4 },
    }) |sh| {
        const in = try Inputs.make(sh, 0x9e3779b97f4a7c15 + sh.length);
        defer in.free();
        const ry = try gpa.alloc(f64, sh.length * sh.value_heads * d);
        defer gpa.free(ry);
        const rs = try gpa.alloc(f64, sh.value_heads * d * d);
        defer gpa.free(rs);
        try reference(sh, in, ry, rs);
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
        var y = try t.alloc(ry.len * 4);
        defer y.free();
        var state = try t.upload(in.state);
        defer state.free();
        // `off` runs the token-serial kernel, `on` the chunked one for these lengths
        for ([_]*const @import("../launches.zig").Launcher{ &t.off, &t.on }) |l| {
            try state.upload(0, std.mem.sliceAsBytes(in.state));
            try y.fill8(0);
            t.arena.reset();
            try t.ops(l).gatedDelta(rig.at(q), rig.at(k), rig.at(v), rig.at(gate), rig.at(beta), rig.at(state), rig.at(y), sh.length, sh.key_heads, sh.value_heads, d, d, null);
            try t.stream.synchronize();
            const gy = try rig.download(f32, y);
            defer gpa.free(gy);
            const gs = try rig.download(f32, state);
            defer gpa.free(gs);
            try std.testing.expect(relative(gy, ry) < 5e-3);
            try std.testing.expect(relative(gs, rs) < 5e-3);
        }
    }
}
