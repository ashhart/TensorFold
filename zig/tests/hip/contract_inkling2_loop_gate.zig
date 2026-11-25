//! Inkling 2 Mini's HIP message gates under the contract: every output's bits equal their named baselines'.
const std = @import("std");
const core = @import("core");
const K = @import("inkling2_loop_hip");
const Rig = @import("contract_rig.zig").Rig;
const Profile = @import("contract_entry.zig").Profile;
const ref = @import("contract_inkling2_loop_ref.zig");
const rg = @import("contract_inkling2_loop_ref_gate.zig");
const bf = @import("bf16_round");

const c = core.contract.spec;
const g = core.contract.gates;
const H = K.hidden;
const R = K.rank;

fn depth(_: c.Shape) usize {
    return ref.depthW32(H) + ref.depthW32(2 * H) + R;
}

fn sh(rows: usize, gates: usize) c.Shape {
    return .{ .rows = rows, .n = gates, .k = 2 * H };
}

// rows a call, gates a call (q, k and v together): decode at 1, 4 and 8 rows, prompt rows up to 2,048
const full = [_]c.Shape{ sh(1, 1), sh(1, 3), sh(4, 3), sh(8, 3), sh(16, 1), sh(64, 3), sh(2048, 1) };
const smoke = [_]c.Shape{ sh(1, 1), sh(1, 3), sh(8, 3), sh(16, 1) };

const nan_rule = "each output is NaN where the host model's is, and every other output has its bits";

fn label(which: K.Gates) []const u8 {
    return switch (which) {
        .served => "inkling2_loop hip tf_ink2_loop_gate_down + _up (G(h, m) for 1 or 3 gates, the second and third in place)",
        .tanhf => "control: inkling2_loop hip gates with the device's own tanhf for Zig's",
        .fma => "control: inkling2_loop hip gates with (1 - F) * h + G * m fused into one fma",
        .serial_rho => "control: inkling2_loop hip gates with rho's squares in one serial unfused chain",
        .butterfly_desc => "control: inkling2_loop hip gates with the W32 butterfly at 16, 8, 4, 2, 1",
        .unfused_sq => "control: inkling2_loop hip gates with rho's squares rounded before each lane's add",
    };
}

pub const Gate = struct {
    rig: Rig,
    k: *const K.Kernels,
    which: K.Gates = .served,
    profile: Profile = .full,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    // each gate's h, m [rows][H] and weights, gate after gate
    h: []f32 = &.{},
    m: []f32 = &.{},
    down: []u16 = &.{},
    forget: []u16 = &.{},
    gain: []u16 = &.{},
    want: []f32 = &.{},
    calls: [K.max_gates]K.Gate = undefined,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Gate) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Gate) void {
        const a = self.rig.a;
        for ([_][]f32{ self.h, self.m, self.want }) |x| a.free(x);
        for ([_][]u16{ self.down, self.forget, self.gain }) |x| a.free(x);
        self.* = .{ .rig = self.rig, .k = self.k, .which = self.which, .profile = self.profile };
    }

    pub fn kernel(self: *Gate) g.Kernel {
        var s: c.Spec = .{ .name = label(self.which), .accum = .f32, .depth = depth, .out = .f32, .served = if (self.profile == .full) &full else &smoke, .nan = .{ .stated = nan_rule } };
        s.tiles = .{ .{ .size = K.up_rows }, .{}, .{} }; // rows at 1, 3, 4, 5: an up block's row group and its edges
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.which == .served) null else .shapes };
    }

    const vt = g.hooks(Gate);

    pub fn count(self: *const Gate) usize {
        return self.shape.n * self.shape.rows * H;
    }

    fn call(self: *const Gate, z: usize) rg.GateCall {
        const rows = self.shape.rows;
        return .{ .h = self.h[z * rows * H ..][0 .. rows * H], .m = self.m[z * rows * H ..][0 .. rows * H], .down = self.down[z * R * 2 * H ..][0 .. R * 2 * H], .forget = self.forget[z * H * R ..][0 .. H * R], .gain = self.gain[z * H * R ..][0 .. H * R], .rows = rows };
    }

    pub fn prepare(self: *Gate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > K.max_rows or shape.n == 0 or shape.n > K.max_gates or shape.k != 2 * H) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.a;
        const n = shape.n;
        self.h = try a.alloc(f32, n * shape.rows * H);
        self.m = try a.alloc(f32, n * shape.rows * H);
        self.down = try a.alloc(u16, n * R * 2 * H);
        self.forget = try a.alloc(u16, n * H * R);
        self.gain = try a.alloc(u16, n * H * R);
        var prng = std.Random.DefaultPrng.init(seed);
        try make(self, prng.random());
        // gate 0 writes its own out; gates 1 and 2 write over their h, as the engine's gates do
        for (0..n) |z| {
            const cl = self.call(z);
            var x: K.Gate = undefined;
            x.out = try self.rig.output(4 * cl.h.len);
            x.h = if (z == 0) try self.rig.input(std.mem.sliceAsBytes(cl.h)) else x.out;
            x.m = try self.rig.input(std.mem.sliceAsBytes(cl.m));
            x.down = try self.rig.input(std.mem.sliceAsBytes(cl.down));
            x.forget = try self.rig.input(std.mem.sliceAsBytes(cl.forget));
            x.gain = try self.rig.input(std.mem.sliceAsBytes(cl.gain));
            x.zeta = try self.rig.scratch(2 * shape.rows * R);
            self.calls[z] = x;
        }
        return true;
    }

    pub fn run(self: *Gate) !void {
        for (self.calls[1..self.shape.n], 1..) |x, z| try self.rig.put(x.out, std.mem.sliceAsBytes(self.call(z).h));
        try K.gatesAs(self.k, self.rig.s, self.which, self.calls[0..self.shape.n], @intCast(self.shape.rows));
        try self.rig.s.synchronize();
    }

    fn expected(self: *Gate) ![]const f32 {
        if (self.want.len == self.count()) return self.want;
        self.want = try self.rig.a.alloc(f32, self.count());
        const per = self.shape.rows * H;
        for (0..self.shape.n) |z| try rg.gate(self.rig.a, self.call(z), self.want[z * per ..][0..per]);
        return self.want;
    }

    pub fn reference(self: *Gate, out: []f64, mag: []f64) !void {
        for (out, mag, try self.expected()) |*o, *m, w| {
            o.* = w;
            m.* = @abs(w);
        }
    }

    /// Gate 4: every gate and row with each sum in `acc` and `order` (the stress rows make fp32 exact in any order).
    pub fn simulate(self: *Gate, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.a;
        const x = try a.alloc(f64, 2 * H + 4 * R * H);
        defer a.free(x);
        const w = x[2 * H ..];
        for (0..self.shape.n) |z| {
            const cl = self.call(z);
            for (w[0 .. 2 * R * H], cl.down) |*v, b| v.* = bf.value(b);
            for (w[2 * R * H .. 3 * R * H], cl.forget) |*v, b| v.* = bf.value(b);
            for (w[3 * R * H ..], cl.gain) |*v, b| v.* = bf.value(b);
            const gw: rg.GateWeights = .{ .down = w[0 .. 2 * R * H], .forget = w[2 * R * H .. 3 * R * H], .gain = w[3 * R * H ..] };
            for (0..cl.rows) |t| {
                for (x[0..H], cl.h[t * H ..][0..H]) |*v, f| v.* = f;
                for (x[H .. 2 * H], cl.m[t * H ..][0..H]) |*v, f| v.* = f;
                rg.simGate(acc, order, x[0..H], x[H .. 2 * H], gw, out[(z * cl.rows + t) * H ..][0..H]);
            }
        }
    }

    /// Every output's bits against the host model's (NaN as a class); at gate 1, NaN and finite outputs both occur.
    pub fn check(self: *Gate, at: g.Gate) !?[]const u8 {
        const want = try self.expected();
        const got = try self.rig.outputBytes();
        var nans: usize = 0;
        for (want, 0..) |w, i| {
            const y: f32 = @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little));
            if (std.math.isNan(w)) nans += 1;
            if (!ref.same32(y, w)) {
                const rows = self.shape.rows;
                const fmt = "at {d} rows x {d} gates, gate {d} row {d} column {d} is 0x{x:0>8} where the baseline gives 0x{x:0>8}";
                const args = .{ rows, self.shape.n, i / (rows * H), i / H % rows, i % H, @as(u32, @bitCast(y)), @as(u32, @bitCast(w)) };
                return std.fmt.bufPrint(&self.msg, fmt, args) catch "an output differs";
            }
        }
        if (at == .nonfinite and (nans == 0 or nans == want.len)) return "the planted values give no mix of NaN and finite outputs: no proof";
        return null;
    }
};

/// Random h, m and weights (some -0, some large), gate 0's order cells; gate 1's NaN and infinities; gate 4's stress.
fn make(gt: *Gate, r: std.Random) !void {
    if (gt.fill == .stress) {
        for (0..gt.shape.n) |z| {
            const rows = gt.shape.rows;
            rg.gateStress(r, gt.h[z * rows * H ..][0 .. rows * H], gt.m[z * rows * H ..][0 .. rows * H], gt.down[z * R * 2 * H ..][0 .. R * 2 * H], gt.forget[z * H * R ..][0 .. H * R], gt.gain[z * H * R ..][0 .. H * R]);
        }
        return;
    }
    for (gt.h, gt.m, 0..) |*x, *y, i| {
        x.* = r.floatNorm(f32) * (if (i % 89 == 0) @as(f32, 60) else 4);
        y.* = r.floatNorm(f32) * 0.5;
        if (i % 1009 == 5) x.* = -0.0;
    }
    for ([_][]u16{ gt.down, gt.forget, gt.gain }, [_]f32{ 0.02, 0.15, 0.15 }) |w, scale| for (w) |*v| {
        v.* = bf.bits(r.floatNorm(f32) * scale);
    };
    // gate 0's cell weights, and in its last row (rows >= 2) the cell row
    rg.gateCellWeights(gt.down[0 .. R * 2 * H], gt.forget[0 .. H * R], gt.gain[0 .. H * R]);
    const rows = gt.shape.rows;
    if (rows >= 2) try rg.gateCellRow(gt.h[(rows - 1) * H ..][0..H], gt.m[(rows - 1) * H ..][0..H]);
    if (gt.fill == .nonfinite) {
        gt.h[1 * H + 5] = std.math.nan(f32); // gate 0, row 1: every output NaN
        gt.m[3 * H + 8] = std.math.inf(f32); // row 3: an infinite message
        gt.forget[11 * R + 3] = bf.bits(std.math.inf(f32)); // column 11's forget dot is infinite: F is +-1
        gt.gain[7 * R + 9] = bf.bits(std.math.nan(f32)); // column 7's gain is NaN in every row
    }
}
