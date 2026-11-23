//! tf_nemo_attn64 under the contract: the speaker stack's full attention at head dim 64, head by head.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const nemo_audio = @import("nemo_audio");

const c = core.contract.spec;
const g = core.contract.gates;

const D = 64;

/// A score's 64 products, then the P x V chain over every key in order.
fn keyChain(s: c.Shape) usize {
    return s.k + 64;
}

fn scale() f32 {
    return 1.0 / @sqrt(@as(f32, D));
}

/// Shape: heads (rows), head width (n), rows L (k): a short clip's, the first window's and a full window's calls.
pub fn spec(name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = keyChain,
        .ops = 3,
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 64 } },
        .served = &.{ .{ .rows = 8, .n = D, .k = 828 }, .{ .rows = 8, .n = D, .k = 550 }, .{ .rows = 8, .n = D, .k = 74 } },
        .sums = .{ .why = "P stays fp32 through the P x V sums: gates 1 and 3 hold each output to attention_ref's float64 bound" },
    };
}

pub const Full = struct {
    rig: rig_.Rig,
    k: nemo_audio.kernels.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    r: ?attn.Ref = null,
    shape: c.Shape = undefined,
    /// [L, 3 heads 128]: q heads, then k, then v (tf_moss_qk's packed rows)
    qkv: []u16 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Full) void {
        self.rig.gpu.gpa.free(self.qkv);
        if (self.r) |*r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Full) g.Kernel {
        return .{ .spec = spec(self.name), .ctx = self, .vt = &vt, .control = self.control };
    }

    const vt = g.hooks(Full);

    fn stride(self: *const Full) usize {
        return 3 * self.shape.rows * D;
    }

    fn at(self: *const Full, row: usize, which: usize, head: usize) []u16 {
        return self.qkv[row * self.stride() + (which * self.shape.rows + head) * D ..][0..D];
    }

    pub fn prepare(self: *Full, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.n != D or shape.rows == 0 or shape.k == 0 or shape.k > nemo_audio.kernels.attention_rows) return false;
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        const L = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, D, L);
        a.free(self.qkv);
        self.qkv = &.{};
        self.qkv = try a.alloc(u16, L * self.stride());
        for (self.qkv) |*e| e.* = attn.normal(rng);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        self.dev = try self.rig.input(std.mem.sliceAsBytes(self.qkv));
        self.out = try self.rig.output(L * shape.rows * D * 2);
        return true;
    }

    /// Head 0: NaN in row 0's query, +inf and -inf in values, both in one column; head 1: NaN in one key.
    fn plant(self: *Full) void {
        const L = self.shape.k;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        self.at(0, 0, 0)[3] = nan;
        self.at(L - 1, 2, 0)[7] = inf;
        self.at(L / 2, 2, 0)[9] = ninf;
        if (L >= 3) {
            self.at(1, 2, 0)[11] = inf;
            self.at(2, 2, 0)[11] = ninf;
        }
        if (self.shape.rows > 1) self.at(L / 3, 1, 1)[5] = nan;
    }

    /// Gate 4: every key one vector (each p is 1); each value column mirrored over the keys.
    fn stress(self: *Full, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const L = self.shape.k;
        const key = try a.alloc(u16, D);
        defer a.free(key);
        for (key) |*e| e.* = attn.normal(rng);
        const col = try a.alloc(f32, L);
        defer a.free(col);
        for (0..self.shape.rows) |h| {
            for (0..L) |j| @memcpy(self.at(j, 1, h), key);
            for (0..D) |e| {
                attn.mirrored(rng, col);
                for (col, 0..) |y, j| self.at(j, 2, h)[e] = c.toBf16(y);
            }
        }
    }

    pub fn run(self: *Full) !void {
        try self.k.attention(self.rig.s, self.dev, self.out, self.shape.k, self.shape.rows, scale());
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Full) usize {
        return self.shape.k * self.shape.rows * D;
    }

    /// Output head i (row i / heads, head i % heads) over every row's key and value.
    pub fn load(self: *Full, i: usize, hd: *attn.Head) !void {
        const row = i / self.shape.rows;
        const head = i % self.shape.rows;
        for (hd.q, self.at(row, 0, head)) |*y, x| y.* = c.fromBf16(x);
        for (0..self.shape.k) |j| for (0..D) |e| {
            hd.k[j * D + e] = c.fromBf16(self.at(j, 1, head)[e]);
            hd.v[j * D + e] = c.fromBf16(self.at(j, 2, head)[e]);
        };
        hd.n = self.shape.k;
    }

    fn heads(self: *Full) usize {
        return self.shape.k * self.shape.rows;
    }

    pub fn check(self: *Full, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), scale(), try self.rig.outputBytes(), gate);
    }

    pub fn reference(self: *Full, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), scale(), want, mag);
    }

    pub fn simulate(self: *Full, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), scale(), acc, order, out);
    }
};
