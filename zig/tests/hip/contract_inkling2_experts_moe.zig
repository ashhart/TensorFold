//! Inkling 2 Mini's HIP down projections under the contract: every output's bits equal loop_model.zig's.
const std = @import("std");
const core = @import("core");
const K = @import("inkling2_experts_hip");
const Rig = @import("contract_rig.zig").Rig;
const Profile = @import("contract_entry.zig").Profile;
const ref = @import("contract_inkling2_experts_ref.zig");
const bf = @import("bf16_round");

const c = core.contract.spec;
const g = core.contract.gates;
const topk = K.topk;

pub const Kind = enum { nv_down, mx_down };

/// A control's planted fault, through the device's inputs only: the kernels run as they are.
pub const Plant = enum { none, late_scale };

fn format(k: Kind) ref.Format {
    return if (k == .nv_down) .nvfp4 else .mxfp8;
}

fn depth(s: c.Shape) usize {
    return s.k;
}

fn sh(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

const down_full = [_]c.Shape{ sh(1, K.hidden, K.width), sh(4, K.hidden, K.width), sh(16, K.hidden, K.width), sh(64, K.hidden, K.width) };
const down_smoke = [_]c.Shape{ sh(1, K.hidden, K.width), sh(4, K.hidden, K.width) };
// every (code, scale) pair: 32 one-hot routes of 2,304 rows hold 73,728 of them
const decode_shapes = [_]c.Shape{sh(4, K.hidden, K.width)};

const nan_rule = "a NaN output is 0x7fc0; an id past the experts gives a NaN row; every other output has loop_model's bits";

fn label(k: Kind, plant: Plant, decode: bool) []const u8 {
    if (decode) return if (k == .mx_down) "inkling2_experts hip tf_ink2_mxfp8_down at every (code, scale) pair, scale 255 as 2^128 (one-hot rows)" else "inkling2_experts hip tf_ink2_nvfp4_down at every (code, scale) pair, both nibbles, eight amax (one-hot rows)";
    return switch (plant) {
        .none => switch (k) {
            .nv_down => "inkling2_experts hip tf_ink2_nvfp4_down (w2 1,152 -> 2,304, ids 0-7)",
            .mx_down => "inkling2_experts hip tf_ink2_mxfp8_down (w2 1,152 -> 2,304, ids 120-127)",
        },
        .late_scale => "control: inkling2_experts hip nvfp4 down reading each row's scales one group late",
    };
}

pub fn spec(k: Kind, plant: Plant, decode: bool, profile: Profile) c.Spec {
    const full = profile == .full;
    var s: c.Spec = .{ .name = label(k, plant, decode), .accum = .f32, .depth = depth, .out = .bf16, .served = &.{}, .nan = .{ .stated = nan_rule } };
    s.served = if (decode) &decode_shapes else if (full) &down_full else &down_smoke;
    // edges: outputs a route at 1, 63, 64 and 65 (64 a block); reductions of 32 and 64 inputs
    if (!decode) s.tiles = .{ .{}, .{ .size = K.block_outputs }, .{ .size = 32, .step = 32, .min = 32 } };
    return s;
}

pub const Moe = struct {
    rig: Rig,
    k: *const K.Kernels,
    kind: Kind,
    plant: Plant = .none,
    decode: bool = false,
    profile: Profile = .full,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    first: usize = 0, // the id of the matrix's expert 0: MXFP8 cells hold Mini's last experts (120-127)
    x: []u16 = &.{},
    codes: []u8 = &.{},
    scales: []u8 = &.{},
    amax: []f32 = &.{},
    ids: []i32 = &.{},
    want: []u16 = &.{},
    dev: Dev = .{},
    msg: [200]u8 = undefined,

    const Dev = struct { x: u64 = 0, codes: u64 = 0, scales: u64 = 0, amax: u64 = 0, den: u64 = 0, ids: u64 = 0, y: u64 = 0 };
    pub const den: f32 = 2695.75;

    pub fn deinit(self: *Moe) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Moe) void {
        const a = self.rig.a;
        a.free(self.x);
        a.free(self.codes);
        a.free(self.scales);
        a.free(self.amax);
        a.free(self.ids);
        a.free(self.want);
        self.* = .{ .rig = self.rig, .k = self.k, .kind = self.kind, .plant = self.plant, .decode = self.decode, .profile = self.profile };
    }

    pub fn kernel(self: *Moe) g.Kernel {
        return .{ .spec = spec(self.kind, self.plant, self.decode, self.profile), .ctx = self, .vt = &vt, .control = if (self.plant == .none) null else .shapes };
    }

    const vt = g.hooks(Moe);

    fn routes(self: *const Moe) usize {
        return self.shape.rows * topk;
    }

    pub fn count(self: *const Moe) usize {
        return self.routes() * self.shape.n;
    }

    /// The matrix the kernel reads: w2, n rows an expert.
    pub fn mat(self: *const Moe) ref.Mat {
        const f = format(self.kind);
        return .{ .format = f, .rows = self.shape.n, .k = self.shape.k, .codes = self.codes, .scales = self.scales, .amax = self.amax, .den = den, .first = self.first };
    }

    pub fn prepare(self: *Moe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.rows * topk > K.max_rows) return false;
        if (shape.k == 0 or shape.k > K.max_k or shape.k % 32 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const fills = @import("contract_inkling2_experts_fill.zig");
        try fills.make(self, prng.random());
        try self.upload();
        return true;
    }

    /// The device's buffers (a control's carry its fault); codes and scales start `first` experts early.
    fn upload(self: *Moe) !void {
        const a = self.rig.a;
        const d = &self.dev;
        d.ids = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        d.y = try self.rig.output(self.count() * 2);
        const x = try a.dupe(u16, self.x);
        defer a.free(x);
        d.x = try self.rig.input(std.mem.sliceAsBytes(x));
        const m = self.mat();
        const codes = try a.dupe(u8, self.codes);
        defer a.free(codes);
        const scales = try a.dupe(u8, self.scales);
        defer a.free(scales);
        const rb = m.rowBytes();
        const rs = m.rowScales();
        for (0..m.count()) |e| for (0..m.rows) |r| {
            const at = e * m.rows + r;
            if (self.plant == .late_scale) for (0..rs) |j| {
                scales[at * rs + j] = self.scales[at * rs + (j + 1) % rs];
            };
        };
        const expert_codes = m.rows * rb;
        const expert_scales = m.rows * rs;
        d.codes = try self.rig.input(codes) - self.first * expert_codes;
        d.scales = try self.rig.input(scales) - self.first * expert_scales;
        if (m.format == .nvfp4) {
            d.amax = try self.rig.input(std.mem.sliceAsBytes(self.amax)) - self.first * 4;
            d.den = try self.rig.input(std.mem.asBytes(&den));
        }
    }

    fn weights(self: *const Moe) K.Weights {
        const d = self.dev;
        return .{ .format = if (format(self.kind) == .nvfp4) .nvfp4 else .mxfp8, .codes = d.codes, .scales = d.scales, .amax = d.amax, .den = d.den };
    }

    pub fn run(self: *Moe) !void {
        const d = self.dev;
        const call: K.Call = .{ .rows = @intCast(self.shape.rows), .n = @intCast(self.shape.n), .k = @intCast(self.shape.k) };
        try K.down(self.k, self.rig.s, self.weights(), d.x, d.ids, d.y, call);
        try self.rig.s.synchronize();
    }

    /// loop_model's outputs on the true inputs, once a prepare.
    fn expected(self: *Moe) ![]const u16 {
        if (self.want.len == self.count()) return self.want;
        const a = self.rig.a;
        self.want = try a.alloc(u16, self.count());
        try ref.project(a, .{ .mat = self.mat(), .x = self.x, .ids = self.ids, .n = self.shape.n }, self.want);
        return self.want;
    }

    pub fn reference(self: *Moe, out: []f64, mag: []f64) !void {
        for (out, mag, try self.expected()) |*o, *m, h| {
            o.* = if (h == ref.nan_bits) std.math.nan(f64) else bf.value(h);
            m.* = @abs(o.*);
        }
    }

    /// Gate 4 on the stress rows: every row and token is the same, so one replay serves every output.
    pub fn simulate(self: *Moe, acc: c.Accum, order: g.Order, out: []f64) !void {
        std.debug.assert(self.fill == .stress);
        @memset(out, @import("contract_inkling2_experts_fill.zig").simulate(self, acc, order));
    }

    /// Every output's bits against loop_model's; at gate 1, NaN and finite outputs both occur.
    pub fn check(self: *Moe, at: g.Gate) !?[]const u8 {
        const want = try self.expected();
        const out = try self.rig.outputBytes();
        var nans: usize = 0;
        for (want, 0..) |w, i| {
            const got = std.mem.readInt(u16, out[2 * i ..][0..2], .little);
            if (w == ref.nan_bits) nans += 1;
            if (got != w) {
                const fmt = "at {d}x{d}x{d} output {d} (route {d}, column {d}) is 0x{x:0>4} where loop_model gives 0x{x:0>4}";
                const sh_ = self.shape;
                return std.fmt.bufPrint(&self.msg, fmt, .{ sh_.rows, sh_.n, sh_.k, i, i / sh_.n, i % sh_.n, got, w }) catch "an output differs";
            }
        }
        if (at == .nonfinite and (nans == 0 or nans == want.len)) return "the planted values give no mix of NaN and finite outputs: no proof";
        return null;
    }
};
