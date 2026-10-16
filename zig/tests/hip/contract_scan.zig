//! Nemotron's Mamba-2 state recurrence under the contract on HIP: y and the state after the rows, in float64.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const mamba = @import("contract_mamba.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const dh = 64;
const ds = 128;

/// Each row's state update (an exp, two products and an fma) on the state's chain, then 32 fmas, four lanes and an fma.
fn depth(s: c.Shape) usize {
    return 6 * s.rows + 40;
}

/// A gate 4 replay's accumulator and order.
const Sim = struct { c.Accum, g.Order };

/// The launch's two outputs, each registered on its own: y, and the state after the rows.
pub const Part = enum { y, state };

const served = [_]c.Shape{
    .{ .rows = 4, .n = 32, .k = ds },
    .{ .rows = 1, .n = 64, .k = ds },
    .{ .rows = 2048, .n = 16, .k = ds },
};

pub fn spec(part: Part, k: kit.Kind) c.Spec {
    var s = c.Spec{ .name = "", .accum = .f32, .depth = depth, .out = .f32, .served = &served };
    s.tiles = .{ .{ .size = 16 }, .{}, .{} };
    if (part == .state) {
        s.name = "hip mamba tf_mamba_scan (the state)";
        return s;
    }
    s.name = if (k == .bf16) "hip mamba tf_mamba_scan (y, bf16)" else "hip mamba tf_mamba_scan (y, fp16)";
    s.out = kit.elem(k);
    // the gate's silu and the product with it
    s.ops = 8;
    // the gate and x D + s C each rounded to the kind before their product
    s.mids = 2;
    return s;
}

/// shape.rows rows over shape.n heads: y [rows][xd] of the kind and the state [heads][dh][128] fp32 after them.
pub const Scan = struct {
    rig: rig_.Rig,
    part: Part,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    d: mamba.Dims = undefined,
    z: []f64 = &.{}, // [rows][xd]
    xc: []f64 = &.{}, // [rows][cd]
    dt: []f64 = &.{}, // [rows][heads]
    st: []f64 = &.{}, // [heads][dh][128]
    a_log: []f64 = &.{},
    dsk: []f64 = &.{},
    in: [6]u64 = @splat(0), // P, xc, dt, a_log, d, the state each run starts from
    y: u64 = 0,
    st_out: u64 = 0,
    st_bytes: []u8 = &.{},
    gate: []f64 = &.{}, // gate 4: the gate's value, as the kind holds it

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, part: Part) Scan {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .part = part, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Scan) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Scan) g.Kernel {
        return .{ .spec = spec(self.part, self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Scan);

    fn signed(rng: std.Random, lo: f64, hi: f64) f64 {
        const v = lo + rng.float(f64) * (hi - lo);
        return if (rng.boolean()) v else -v;
    }

    pub fn prepare(self: *Scan, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        const d = mamba.Dims{ .heads = shape.n };
        self.d = d;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows;
        const k = self.k;
        self.z = try a.alloc(f64, n * d.xd());
        self.xc = try a.alloc(f64, n * d.cd());
        self.dt = try a.alloc(f64, n * d.heads);
        self.st = try a.alloc(f64, d.heads * dh * ds);
        self.a_log = try kit.floats(a, rng, d.heads, -1, 2.5);
        self.dsk = try kit.floats(a, rng, d.heads, 0.5, 1.5);
        for (self.z) |*v| v.* = kit.round(k, signed(rng, 0.5, 4));
        for (self.xc, 0..) |*v, i| {
            v.* = kit.round(k, if (i % d.cd() < d.xd()) signed(rng, 0.25, 2) else rng.float(f64) * 2 - 1);
        }
        for (self.dt) |*v| v.* = @as(f32, @floatCast(0.001 + rng.float(f64) * 0.099));
        for (self.st) |*v| v.* = @as(f32, @floatCast(rng.float(f64) * 0.4 - 0.2));
        if (fill == .stress) self.stress(rng);
        if (fill == .nonfinite) self.plant();
        self.gate = try a.alloc(f64, self.z.len);
        for (self.gate, self.z) |*o, v| o.* = kit.round(k, kit.silu(v));
        const p = try a.alloc(u8, n * d.proj() * kit.size(k));
        @memset(p, 0);
        for (0..n) |row| for (0..d.xd()) |i| kit.put(k, p, row * d.proj() + i, self.z[row * d.xd() + i]);
        const xb = try a.alloc(u8, self.xc.len * kit.size(k));
        for (self.xc, 0..) |v, i| kit.put(k, xb, i, v);
        self.st_bytes = try kit.f32Bytes(a, self.st);
        const ins = [_][]const u8{
            p,
            xb,
            try kit.f32Bytes(a, self.dt),
            try kit.f32Bytes(a, self.a_log),
            try kit.f32Bytes(a, self.dsk),
            self.st_bytes,
        };
        for (ins, &self.in) |b, *at| at.* = try self.rig.input(b);
        // the other part's output is scratch: the gates read one part's bytes
        const y_bytes = n * d.xd() * kit.size(k);
        self.y = if (self.part == .y) try self.rig.output(y_bytes) else try self.rig.scratch(y_bytes);
        const st_bytes = self.st_bytes.len;
        self.st_out = if (self.part == .state) try self.rig.output(st_bytes) else try self.rig.scratch(st_bytes);
        return true;
    }

    /// Gate 4: A = -0, dt = 1: the state sums x B in integers, rows 2 and 3 undoing rows 0 and 1 but for one x.
    fn stress(self: *Scan, rng: std.Random) void {
        const d = self.d;
        @memset(self.a_log, -std.math.inf(f64));
        @memset(self.dt, 1);
        @memset(self.st, 0);
        @memset(self.dsk, 0);
        @memset(self.z, 16);
        for (0..self.shape.rows) |row| {
            const r = self.xc[row * d.cd() ..][0..d.cd()];
            for (r[0..d.xd()], 0..) |*x, i| x.* = switch (row % 4) {
                0, 1 => @floatFromInt(1 + rng.uintLessThan(u32, 64)),
                2 => -self.xc[(row - 2) * d.cd() + i],
                else => 1 - self.xc[(row - 2) * d.cd() + i],
            };
            if (row > 0) {
                @memcpy(r[d.xd()..], self.xc[d.xd()..d.cd()]); // every row's B and C are row 0's
                continue;
            }
            // C's second half cancels its first but for state 0, whose B is 1; B repeats in the second half
            for (0..d.groups()) |grp| {
                const b = r[d.xd() + grp * ds ..][0..ds];
                const cc = r[d.xd() + (d.groups() + grp) * ds ..][0..ds];
                for (0..ds / 2) |i| {
                    b[i] = if (i == 0) 1 else @floatFromInt(1 + rng.uintLessThan(u32, 64));
                    b[i + ds / 2] = b[i];
                    cc[i] = @floatFromInt(1 + rng.uintLessThan(u32, 16));
                    cc[i + ds / 2] = -cc[i] + @as(f64, if (i == 0) 1 else 0);
                }
            }
        }
    }

    /// NaN in row 0's x (head 0), +Inf in row 1's gate (head 1) and row 2's dt (head 2), -Inf in head 3's state.
    fn plant(self: *Scan) void {
        const d = self.d;
        self.xc[0] = std.math.nan(f64);
        if (self.shape.rows > 1) self.z[d.xd() + dh] = std.math.inf(f64);
        if (self.shape.rows > 2) self.dt[2 * d.heads + 2] = std.math.inf(f64);
        self.st[3 * dh * ds] = -std.math.inf(f64);
    }

    /// The state back to the prepared one, then one launch: a run never reads the last run's state.
    pub fn run(self: *Scan) !void {
        try self.rig.put(self.st_out, self.st_bytes);
        const d = self.d;
        const C = ?*const anyopaque;
        const CF = ?[*]const f32;
        const dims = [_]usize{ self.shape.rows, d.proj(), d.xd(), d.cd(), d.heads, dh, d.groups() };
        var v: [7]c_int = undefined;
        for (&v, dims) |*o, x| o.* = @intCast(x);
        try self.l.tf_mamba_scan(
            kit.ptr(C, self.in[0]),
            kit.ptr(C, self.in[1]),
            @backingInt(self.k),
            kit.ptr(CF, self.in[2]),
            kit.ptr(?[*]f32, self.st_out),
            kit.ptr(CF, self.in[3]),
            kit.ptr(CF, self.in[4]),
            kit.ptr(?*anyopaque, self.y),
            v[0],
            v[1],
            v[2],
            v[3],
            v[4],
            v[5],
            v[6],
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Scan) usize {
        return if (self.part == .y) self.z.len else self.st.len;
    }

    /// y and the state in float64 (in `sim`'s accumulator and order when given), with each one's magnitudes.
    fn walk(self: *Scan, y: []f64, ymag: []f64, st: []f64, smag: []f64, sim: ?Sim) !void {
        const d = self.d;
        const a = self.rig.a;
        const terms = try a.alloc(f64, ds + 1);
        defer a.free(terms);
        const hist = try a.alloc(f64, self.shape.rows + 1);
        defer a.free(hist);
        for (0..d.heads) |h| {
            const grp = h / (d.heads / d.groups());
            const decay = -@exp(self.a_log[h]);
            for (0..dh) |vd| {
                const at = (h * dh + vd) * ds;
                const s = st[at..][0..ds];
                const m = smag[at..][0..ds];
                @memcpy(s, self.st[at..][0..ds]);
                for (m, s) |*o, v| o.* = @abs(v);
                for (0..self.shape.rows) |row| {
                    const r = self.xc[row * d.cd() ..][0..d.cd()];
                    const x = r[h * dh + vd];
                    const dtv = self.dt[row * d.heads + h];
                    const da = @exp(decay * dtv);
                    var mag: f64 = 0;
                    for (s, m, 0..) |*sv, *mv, i| {
                        const add = x * dtv * r[d.xd() + grp * ds + i];
                        sv.* = sv.* * da + add;
                        mv.* = mv.* * da + @abs(add);
                        if (sim) |how| sv.* = self.replay(how, h, vd, i, row, hist);
                        terms[i] = sv.* * r[d.xd() + (d.groups() + grp) * ds + i];
                        mag += mv.* * @abs(r[d.xd() + (d.groups() + grp) * ds + i]);
                    }
                    terms[ds] = x * self.dsk[h];
                    const sum = if (sim) |how| rows.sumIn(how[0], how[1], terms) else blk: {
                        var t: f64 = 0;
                        for (terms) |v| t += v;
                        break :blk t;
                    };
                    const i = row * d.xd() + h * dh + vd;
                    const gate = if (sim != null) self.gate[i] else kit.silu(self.z[i]);
                    y[i] = gate * sum;
                    ymag[i] = @abs(gate) * (mag + @abs(terms[ds]));
                }
            }
        }
    }

    /// Gate 4's state after `row`: its x dt B terms from row 0 (A = -0: no decay) in the accumulator and order.
    fn replay(self: *Scan, how: Sim, h: usize, vd: usize, i: usize, row: usize, hist: []f64) f64 {
        const d = self.d;
        const grp = h / (d.heads / d.groups());
        for (0..row + 1) |t| {
            const r = self.xc[t * d.cd() ..][0..d.cd()];
            hist[t] = r[h * dh + vd] * self.dt[t * d.heads + h] * r[d.xd() + grp * ds + i];
        }
        return rows.sumIn(how[0], how[1], hist[0 .. row + 1]);
    }

    pub fn reference(self: *Scan, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null);
    }

    pub fn simulate(self: *Scan, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.a;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        return self.outputs(out, mag, .{ acc, order });
    }

    /// This part's outputs and magnitudes, the other part's in scratch.
    fn outputs(self: *Scan, want: []f64, mag: []f64, sim: ?Sim) !void {
        const a = self.rig.a;
        const n = self.z.len + self.st.len;
        const buf = try a.alloc(f64, 2 * n);
        defer a.free(buf);
        const y = buf[0..self.z.len];
        const ymag = buf[n..][0..self.z.len];
        const st = buf[self.z.len..n];
        const smag = buf[n + self.z.len ..];
        try self.walk(y, ymag, st, smag, sim);
        @memcpy(want, if (self.part == .y) y else st);
        @memcpy(mag, if (self.part == .y) ymag else smag);
    }
};
