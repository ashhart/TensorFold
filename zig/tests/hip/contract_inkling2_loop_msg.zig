//! Inkling 2 Mini's HIP cross-pass messages under the contract: every output's bits equal their named baselines'.
const std = @import("std");
const core = @import("core");
const K = @import("inkling2_loop_hip");
const Rig = @import("contract_rig.zig").Rig;
const Profile = @import("contract_entry.zig").Profile;
const ref = @import("contract_inkling2_loop_ref.zig");
const bf = @import("bf16_round");

const c = core.contract.spec;
const g = core.contract.gates;
const H = K.hidden;

fn depth(_: c.Shape) usize {
    return 2 * ref.depthW32(H);
}

fn sh(rows: usize, count: usize) c.Shape {
    return .{ .rows = rows, .n = count, .k = H };
}

// rows a call, core layers a call: decode (1, 4, 8 rows) for one layer or all 12, prompt rows up to 2,048
const full = [_]c.Shape{ sh(1, 1), sh(1, 12), sh(4, 12), sh(8, 12), sh(16, 1), sh(64, 12), sh(2048, 1) };
const smoke = [_]c.Shape{ sh(1, 1), sh(1, 12), sh(8, 12), sh(16, 1) };

const nan_rule = "each output is NaN where the host model's is, and every other output has its bits";

fn label(which: K.Messages) []const u8 {
    return switch (which) {
        .served => "inkling2_loop hip tf_ink2_loop_messages (B_12 and P_18 against 4 roles' queries, 1 or 12 core layers)",
        .rsqrt => "control: inkling2_loop hip messages with rsqrt for the baselines' 1 / sqrt",
        .fma => "control: inkling2_loop hip messages with the second share fused into an fma",
        .signed_zero => "control: inkling2_loop hip messages summing m0 + m1 without the +0 start",
        .serial_rho => "control: inkling2_loop hip messages with rho's squares in one serial unfused chain",
        .butterfly_desc => "control: inkling2_loop hip messages with the W32 butterfly at 16, 8, 4, 2, 1",
    };
}

pub const Msg = struct {
    rig: Rig,
    k: *const K.Kernels,
    which: K.Messages = .served,
    profile: Profile = .full,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    first: usize = 0,
    bank0: []u16 = &.{},
    bank1: []u16 = &.{},
    qm: []f32 = &.{},
    kv: []f32 = &.{},
    want: []f32 = &.{},
    dev: struct { bank0: u64 = 0, bank1: u64 = 0, qm: u64 = 0, kv: u64 = 0, out: u64 = 0 } = .{},
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Msg) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Msg) void {
        const a = self.rig.a;
        a.free(self.bank0);
        a.free(self.bank1);
        a.free(self.qm);
        a.free(self.kv);
        a.free(self.want);
        self.* = .{ .rig = self.rig, .k = self.k, .which = self.which, .profile = self.profile };
    }

    pub fn kernel(self: *Msg) g.Kernel {
        const s: c.Spec = .{ .name = label(self.which), .accum = .f32, .depth = depth, .out = .f32, .served = if (self.profile == .full) &full else &smoke, .nan = .{ .stated = nan_rule } };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.which == .served) null else .shapes };
    }

    const vt = g.hooks(Msg);

    pub fn count(self: *const Msg) usize {
        return self.shape.n * K.roles * self.shape.rows * H;
    }

    fn call(self: *const Msg) ref.MessageCall {
        return .{ .bank0 = self.bank0, .bank1 = self.bank1, .qm = self.qm, .kv = self.kv, .first = self.first, .count = self.shape.n, .rows = self.shape.rows };
    }

    pub fn prepare(self: *Msg, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > K.max_rows or shape.n == 0 or shape.n > K.core_layers or shape.k != H) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        self.first = K.core_layers - shape.n; // one layer: the last core layer's queries, rows 22 and 23
        const a = self.rig.a;
        self.bank0 = try a.alloc(u16, shape.rows * H);
        self.bank1 = try a.alloc(u16, shape.rows * H);
        self.qm = try a.alloc(f32, 24 * H);
        self.kv = try a.alloc(f32, 24 * H);
        var prng = std.Random.DefaultPrng.init(seed);
        make(self, prng.random());
        const d = &self.dev;
        d.bank0 = try self.rig.input(std.mem.sliceAsBytes(self.bank0));
        d.bank1 = try self.rig.input(std.mem.sliceAsBytes(self.bank1));
        d.qm = try self.rig.input(std.mem.sliceAsBytes(self.qm));
        d.kv = try self.rig.input(std.mem.sliceAsBytes(self.kv));
        d.out = try self.rig.output(4 * self.count());
        return true;
    }

    pub fn run(self: *Msg) !void {
        const d = self.dev;
        const at = 2 * self.first * H * 4;
        try K.messagesAs(self.k, self.rig.s, self.which, .{ .bank0 = d.bank0, .bank1 = d.bank1, .qm = d.qm + at, .kv = d.kv + at, .out = d.out, .rows = @intCast(self.shape.rows), .count = @intCast(self.shape.n) });
        try self.rig.s.synchronize();
    }

    fn expected(self: *Msg) ![]const f32 {
        if (self.want.len == self.count()) return self.want;
        self.want = try self.rig.a.alloc(f32, self.count());
        try ref.messages(self.call(), self.want);
        return self.want;
    }

    pub fn reference(self: *Msg, out: []f64, mag: []f64) !void {
        for (out, mag, try self.expected()) |*o, *m, w| {
            o.* = w;
            m.* = @abs(w);
        }
    }

    /// Gate 4: every message with each sum in `acc` and `order` (the stress rows make fp32 exact in any order).
    pub fn simulate(self: *Msg, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.a;
        const x = try a.alloc(f64, 2 * H);
        defer a.free(x);
        const roles = [_]usize{ 0, 1, 2, 3 };
        for (0..self.shape.rows) |t| {
            for (x[0..H], self.bank0[t * H ..][0..H]) |*v, h| v.* = bf.value(h);
            for (x[H..], self.bank1[t * H ..][0..H]) |*v, h| v.* = bf.value(h);
            for (0..self.shape.n) |j| for (roles) |r| {
                const row = 2 * (self.first + j) + @intFromBool(r >= 2);
                const q = (if (r == 0 or r == 3) self.qm else self.kv)[row * H ..][0..H];
                ref.simMessage(acc, order, x[0..H], x[H..], q, out[((j * 4 + r) * self.shape.rows + t) * H ..][0..H]);
            };
        }
    }

    /// Every output's bits against the host model's (NaN as a class); at gate 1, NaN and finite outputs both occur.
    pub fn check(self: *Msg, at: g.Gate) !?[]const u8 {
        const want = try self.expected();
        const got = try self.rig.outputBytes();
        var nans: usize = 0;
        for (want, 0..) |w, i| {
            const y: f32 = @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little));
            if (std.math.isNan(w)) nans += 1;
            if (!ref.same32(y, w)) {
                const rows = self.shape.rows;
                const fmt = "at {d} rows x {d} layers, message {d} row {d} column {d} is 0x{x:0>8} where the baseline gives 0x{x:0>8}";
                const args = .{ rows, self.shape.n, i / (rows * H), i / H % rows, i % H, @as(u32, @bitCast(y)), @as(u32, @bitCast(w)) };
                return std.fmt.bufPrint(&self.msg, fmt, args) catch "an output differs";
            }
        }
        if (at == .nonfinite and (nans == 0 or nans == want.len)) return "the planted values give no mix of NaN and finite outputs: no proof";
        return null;
    }
};

/// Random banks (some -0, some large) and queries, the order cells; gate 1's NaN and infinities; gate 4's stress.
fn make(m: *Msg, r: std.Random) void {
    const rows = m.shape.rows;
    switch (m.fill) {
        .random, .nonfinite => {
            for (m.bank0, m.bank1, 0..) |*x, *y, i| {
                const big: f32 = if (i % 97 == 0) 40 else 3;
                x.* = bf.bits(r.floatNorm(f32) * big);
                y.* = bf.bits(r.floatNorm(f32) * 3);
            }
            for (m.qm, m.kv) |*x, *y| {
                x.* = r.floatNorm(f32) * 2;
                y.* = r.floatNorm(f32) * 2;
            }
            // both banks -0 at two columns a row: the shares' +0 start gives +0 there
            for (0..rows) |t| for ([_]usize{ (7 * t + 11) % H, (13 * t + 1000) % H }) |col| {
                m.bank0[t * H + col] = 0x8000;
                m.bank1[t * H + col] = 0x8000;
            };
            cells(m);
            if (m.fill == .nonfinite) {
                m.bank0[1 * H + 5] = bf.bits(std.math.nan(f32)); // row 1: every message NaN
                m.bank1[2 * H + 7] = bf.bits(std.math.inf(f32)); // row 2: an infinite bank
                m.bank0[3 * H + 9] = bf.bits(-std.math.inf(f32));
                const q = 2 * (m.first + m.shape.n - 1) * H;
                m.kv[q + 3] = std.math.inf(f32); // the last layer's k: an infinite query
            }
        },
        .stress => stress(m, r),
    }
}

/// The order cells: the last layer's q a dot cell, the last row's banks +-1,024 (b is +-1), the row before it rho's.
fn cells(m: *Msg) void {
    const rows = m.shape.rows;
    ref.dotCell(m.qm[2 * (m.first + m.shape.n - 1) * H ..][0..H]);
    if (rows >= 2) {
        @memset(m.bank0[(rows - 1) * H ..][0..H], bf.bits(1024));
        @memset(m.bank1[(rows - 1) * H ..][0..H], bf.bits(-1024));
    }
    if (rows >= 3) ref.rhoCellBank(m.bank0[(rows - 2) * H ..][0..H]);
}

/// Banks of +-1 (squares sum to 2,304 in fp32, 2,048 in fp16), queries nonzero at 3 columns: every fp32 sum is exact.
fn stress(m: *Msg, r: std.Random) void {
    for (m.bank0, m.bank1) |*x, *y| {
        x.* = bf.bits(if (r.boolean()) 1 else -1);
        y.* = bf.bits(if (r.boolean()) 1 else -1);
    }
    @memset(m.qm, 0);
    @memset(m.kv, 0);
    for (0..24) |row| for ([_][]f32{ m.qm, m.kv }) |q| {
        const vals = [3]f32{ 2, -1, 1 };
        for (vals, 0..) |v, i| q[row * H + (row * 131 + i * 977) % H] = v;
    };
}
