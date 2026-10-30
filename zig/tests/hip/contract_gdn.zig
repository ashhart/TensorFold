//! The DeltaNet recurrence under the contract on HIP: its float64 rows, and the prompt kernel at half a head a block.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// Value heads a key head: Flash Next's 48 value heads over 16 key heads.
pub const group = 3;

/// A gate 4 replay's accumulator and order.
pub const Sim = struct { c.Accum, g.Order };

/// Gate 4's inputs: the exact scans' (both sums), or the matrix-core form's (k = 0, beta = 0: the readout alone).
pub const Mode = enum { exact, wy };

/// The longest chain a state element passes a row: decay, k product, the 32-lane sum, delta's two, the update's two.
pub fn depth(s: c.Shape) usize {
    return 15 * s.rows + 10;
}

/// One stream's float64 inputs: t tokens, h value heads, dk = dv = d; every value an fp32 one.
pub const Problem = struct {
    t: usize,
    h: usize,
    d: usize,
    q: []f64, // [t][h / group][d]
    k: []f64, // [t][h / group][d]
    v: []f64, // [t][h][d]
    gate: []f64, // [t][h]
    beta: []f64, // [t][h]
    state: []f64, // [h][d value rows][d]

    pub fn kh(p: *const Problem) usize {
        return p.h / group;
    }

    pub fn make(a: std.mem.Allocator, t: usize, h: usize, d: usize, fill: g.Fill, seed: u64, mode: Mode) !Problem {
        const n = h / group;
        var p: Problem = .{
            .t = t,
            .h = h,
            .d = d,
            .q = try a.alloc(f64, t * n * d),
            .k = try a.alloc(f64, t * n * d),
            .v = try a.alloc(f64, t * h * d),
            .gate = try a.alloc(f64, t * h),
            .beta = try a.alloc(f64, t * h),
            .state = try a.alloc(f64, h * d * d),
        };
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            p.stress(rng, mode);
            return p;
        }
        p.random(rng);
        if (fill == .nonfinite and t > 2 and h > 2) p.plant();
        return p;
    }

    /// L2-normalized q and k rows, normal v, a state of 0.1 scale, gates exp(-A softplus(x)), beta = sigmoid.
    fn random(p: *Problem, rng: std.Random) void {
        const d = p.d;
        for ([_][]f64{ p.q, p.k }) |x| for (0..x.len / d) |row| {
            const at = x[row * d ..][0..d];
            var ss: f64 = 0;
            for (at) |*e| {
                e.* = rng.floatNorm(f64);
                ss += e.* * e.*;
            }
            for (at) |*e| e.* = fp32(e.* / @sqrt(ss));
        };
        for (p.v) |*e| e.* = fp32(rng.floatNorm(f64));
        for (p.state) |*e| e.* = fp32(0.1 * rng.floatNorm(f64));
        const heads: f64 = @floatFromInt(p.h);
        for (p.gate, p.beta, 0..) |*gv, *bv, i| {
            const scale = 0.5 + 7.5 * @as(f64, @floatFromInt(i % p.h)) / heads;
            gv.* = fp32(@exp(-scale * std.math.log1p(@exp(rng.floatNorm(f64) - 2))));
            bv.* = fp32(1 / (1 + @exp(-rng.floatNorm(f64))));
        }
    }

    /// NaN in q (token 0, key head 0) and a gate (token 2, head 0), +Inf in v (token 1, head 1), -Inf in a state
    fn plant(p: *Problem) void {
        p.q[0] = std.math.nan(f64);
        p.gate[2 * p.h] = std.math.nan(f64);
        p.v[(p.h + 1) * p.d] = std.math.inf(f64);
        p.state[(2 * p.d + 3) * p.d] = -std.math.inf(f64);
    }

    /// Gate 4: decay 1; the state's halves cancel to e_0 against mirrored k and q, so every sum is a small integer.
    fn stress(p: *Problem, rng: std.Random, mode: Mode) void {
        const d = p.d;
        @memset(p.gate, 1);
        @memset(p.beta, if (mode == .wy) 0 else 1);
        for (0..p.state.len / d) |row| mirror(rng, p.state[row * d ..][0..d], -1);
        for (0..p.q.len / d) |row| {
            mirror(rng, p.q[row * d ..][0..d], 1);
            mirror(rng, p.k[row * d ..][0..d], 1);
            if (mode == .wy) @memset(p.k[row * d ..][0..d], 0);
        }
        // v is k's element 0, the remembered sum: delta is 0 and the state stays as it was
        for (0..p.t) |t| for (0..p.h) |h| {
            @memset(p.v[(t * p.h + h) * d ..][0..d], p.k[(t * p.kh() + h / group) * d]);
        };
    }

    /// y [t][h][d] and the state after `upto` tokens in float64 (sums in `sim`'s accumulator and order when given).
    pub fn walk(p: *const Problem, a: std.mem.Allocator, upto: usize, y: []f64, ymag: []f64, st: []f64, smag: []f64, sim: ?Sim) !void {
        const d = p.d;
        const terms = try a.alloc(f64, d);
        defer a.free(terms);
        @memcpy(st, p.state);
        for (smag, p.state) |*m, v| m.* = @abs(v);
        for (0..p.h) |h| for (0..d) |r| {
            const s = st[(h * d + r) * d ..][0..d];
            const m = smag[(h * d + r) * d ..][0..d];
            for (0..upto) |t| {
                const at = t * p.h + h;
                const decay = p.gate[at];
                const beta = p.beta[at];
                const key = p.k[(t * p.kh() + h / group) * d ..][0..d];
                const query = p.q[(t * p.kh() + h / group) * d ..][0..d];
                const value = p.v[at * d + r];
                var mrem: f64 = 0;
                for (s, m, key, terms) |*sv, *mv, kv, *o| {
                    sv.* *= decay;
                    mv.* *= decay;
                    o.* = sv.* * kv;
                    mrem += mv.* * @abs(kv);
                }
                const delta = (value - sum(terms, sim)) * beta;
                const mdelta = (@abs(value) + mrem) * @abs(beta);
                var my: f64 = 0;
                for (s, m, key, query, terms) |*sv, *mv, kv, qv, *o| {
                    sv.* += kv * delta;
                    mv.* += @abs(kv) * mdelta;
                    o.* = sv.* * qv;
                    my += mv.* * @abs(qv);
                }
                y[at * d + r] = sum(terms, sim);
                ymag[at * d + r] = my;
            }
        };
    }
};

fn fp32(x: f64) f64 {
    return @as(f32, @floatCast(x));
}

/// Integers in [1, 128] in the first half; the second half `sign` times the first, plus 1 at element 0 when negated.
fn mirror(rng: std.Random, x: []f64, sign: f64) void {
    const half = x.len / 2;
    for (x[0..half], x[half..], 0..) |*lo, *hi, i| {
        lo.* = @floatFromInt(1 + rng.uintLessThan(u32, 128));
        hi.* = sign * lo.* + @as(f64, if (sign < 0 and i == 0) 1 else 0);
    }
}

fn sum(t: []const f64, sim: ?Sim) f64 {
    if (sim) |how| return rows.sumIn(how[0], how[1], t);
    var x: f64 = 0;
    for (t) |v| x += v;
    return x;
}

/// q, k, v, gate and beta as fp32 inputs of `r`.
pub fn upload(r: *rig_.Rig, a: std.mem.Allocator, p: *const Problem) ![5]u64 {
    var at: [5]u64 = undefined;
    for (&at, [_][]const f64{ p.q, p.k, p.v, p.gate, p.beta }) |*x, v| x.* = try r.input(try kit.f32Bytes(a, v));
    return at;
}

/// The plain window scan (gated_delta_tile, no snapshots) over `length` tokens: the kernels below must give its bits.
pub fn plain(l: *const hip.Launcher, r: *rig_.Rig, in: [5]u64, state: u64, y: u64, p: *const Problem, length: usize) !void {
    var a: hip.launch.Args = .{};
    for ([_]u64{ in[0], in[1], in[2], in[3], in[4], state, y }) |x| try a.add(x);
    for ([_]usize{ length, p.kh(), p.h, p.d }) |x| try a.add(@as(c_int, @intCast(x)));
    try a.add(@as(u64, 0)); // no snapshots
    try a.add(@as(u64, 0)); // no start row
    const f = l.gated_delta_tile[if (p.d == 128) 0 else 1];
    const grid: hip.launch.Dim3 = .{ .x = @intCast(p.d / 8), .y = @intCast(p.h) };
    try hip.launch.launch(f, .{ .grid = grid, .block = .{ .x = 32, .y = 8 } }, r.s, &a);
}

/// A launch's two outputs, each registered on its own: y, and the state after the tokens.
pub const Part = enum { y, state };

/// Flash Next's 48 value heads at the shortest prompt that takes the kernel and a longer one; two key heads, 97 tokens.
const prompt_served = [_]c.Shape{
    .{ .rows = 64, .n = 48, .k = 128 },
    .{ .rows = 300, .n = 48, .k = 128 },
    .{ .rows = 97, .n = 6, .k = 128 },
};

pub fn promptSpec(part: Part) c.Spec {
    return .{
        .name = if (part == .y) "hip DeltaNet tf_gdn_prompt_128 (y)" else "hip DeltaNet tf_gdn_prompt_128 (the state)",
        .accum = .f32,
        .depth = depth,
        .out = .f32,
        .replaces = "gated_delta_tile<128, 8, 16>",
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = &prompt_served,
    };
}

/// tf_gdn_prompt_128: shape.rows tokens over shape.n value heads, dk = dv = 128, the state updated in place.
pub const Prompt = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    part: Part,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    p: Problem = undefined,
    in: [5]u64 = @splat(0),
    y: u64 = 0,
    st: u64 = 0,
    st_bytes: []u8 = &.{},
    base: [2]u64 = @splat(0), // the plain scan's y and state

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, part: Part) Prompt {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .part = part, .arena = .init(a) };
    }

    pub fn deinit(self: *Prompt) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Prompt) g.Kernel {
        return .{ .spec = promptSpec(self.part), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Prompt);

    pub fn prepare(self: *Prompt, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != 128 or shape.rows == 0 or shape.n == 0 or shape.n % group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        self.p = try Problem.make(a, shape.rows, shape.n, 128, fill, seed, .exact);
        self.in = try upload(&self.rig, a, &self.p);
        self.st_bytes = try kit.f32Bytes(a, self.p.state);
        const yb = shape.rows * shape.n * 128 * 4;
        const sb = self.st_bytes.len;
        // the other part's output is scratch: the gates read one part's bytes
        self.y = if (self.part == .y) try self.rig.output(yb) else try self.rig.scratch(yb);
        self.st = if (self.part == .state) try self.rig.output(sb) else try self.rig.scratch(sb);
        self.base = .{ try self.rig.scratch(yb), try self.rig.scratch(sb) };
        return true;
    }

    /// The state back to the prepared one, then one launch: half a value head a block, 4 waves.
    pub fn run(self: *Prompt) !void {
        try self.rig.put(self.st, self.st_bytes);
        var a: hip.launch.Args = .{};
        for ([_]u64{ self.in[0], self.in[1], self.in[2], self.in[3], self.in[4], self.st, self.y }) |x| try a.add(x);
        for ([_]usize{ self.shape.rows, self.p.kh(), self.shape.n }) |x| try a.add(@as(c_int, @intCast(x)));
        const grid: hip.launch.Dim3 = .{ .x = 2, .y = @intCast(self.shape.n) };
        try hip.launch.launch(self.l.gdn_prompt, .{ .grid = grid, .block = .{ .x = 32, .y = 4 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    /// The plain scan from the prepared state over the same tokens.
    pub fn replaced(self: *Prompt) ![]const u8 {
        try self.rig.put(self.base[1], self.st_bytes);
        try plain(self.l, &self.rig, self.in, self.base[1], self.base[0], &self.p, self.shape.rows);
        if (self.part == .y) return self.rig.bytes(self.base[0], self.count() * 4);
        return self.rig.bytes(self.base[1], self.st_bytes.len);
    }

    pub fn count(self: *Prompt) usize {
        return if (self.part == .y) self.shape.rows * self.shape.n * 128 else self.p.state.len;
    }

    pub fn reference(self: *Prompt, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null);
    }

    pub fn simulate(self: *Prompt, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(mag);
        return self.outputs(out, mag, .{ acc, order });
    }

    fn outputs(self: *Prompt, want: []f64, mag: []f64, sim: ?Sim) !void {
        return partOf(self.rig.a, &self.p, self.shape.rows, self.part, want, mag, sim);
    }
};

/// One part's float64 outputs and magnitudes after `upto` tokens, the other part's in scratch.
pub fn partOf(a: std.mem.Allocator, p: *const Problem, upto: usize, part: Part, want: []f64, mag: []f64, sim: ?Sim) !void {
    const ny = p.t * p.h * p.d;
    const ns = p.state.len;
    const buf = try a.alloc(f64, 2 * (ny + ns));
    defer a.free(buf);
    const y = buf[0..ny];
    const ymag = buf[ny..][0..ny];
    const st = buf[2 * ny ..][0..ns];
    const smag = buf[2 * ny + ns ..][0..ns];
    try p.walk(a, upto, y, ymag, st, smag, sim);
    const n = upto * p.h * p.d;
    @memcpy(want, if (part == .y) y[0..n] else st);
    @memcpy(mag, if (part == .y) ymag[0..n] else smag);
}
