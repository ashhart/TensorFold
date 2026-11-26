//! prefill_attention.cu's vision_kernel<D, 8> under the contract: a tower's full attention, head by head.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;

/// The P x V chain of a row: every key, a 64-key tile at a time.
fn keyChain(s: c.Shape) usize {
    return s.k + 64;
}

/// The test type at head dim `D`, registered as `name` at the shapes `served`: heads (rows), D (n), rows L (k).
pub fn Full(comptime D: usize, comptime name: []const u8, comptime served: []const c.Shape) type {
    return struct {
        const Self = @This();

        fn scale() f32 {
            return 1.0 / @sqrt(@as(f32, D));
        }

        pub fn spec() c.Spec {
            return .{
                .name = name,
                .accum = .f32,
                .depth = keyChain,
                .ops = 1,
                .out = .bf16,
                .tiles = .{ .{}, .{}, .{ .size = 128 } },
                .served = served,
                .sums = .{ .why = attn.why },
            };
        }

        rig: rig_.Rig,
        att: *const cuda.vit.Attention,
        r: ?attn.Ref = null,
        shape: c.Shape = undefined,
        /// [L, 3 heads D]: q heads, then k, then v
        qkv: []u16 = &.{},
        dev: u64 = 0,
        out: u64 = 0,

        pub fn deinit(self: *Self) void {
            self.rig.gpu.gpa.free(self.qkv);
            if (self.r) |*r| r.deinit();
            self.rig.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = spec(), .ctx = self, .vt = &vt };
        }

        const vt = g.hooks(Self);

        fn stride(self: *const Self) usize {
            return 3 * self.shape.rows * D;
        }

        fn at(self: *const Self, row: usize, which: usize, head: usize) []u16 {
            return self.qkv[row * self.stride() + (which * self.shape.rows + head) * D ..][0..D];
        }

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            self.rig.reset();
            if (shape.n != D or shape.rows == 0 or shape.k == 0) return false;
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
                .random => self.plantAdversarial(),
                .nonfinite => self.plant(),
                .stress => try self.stress(rng),
            }
            self.dev = try self.rig.input(std.mem.sliceAsBytes(self.qkv));
            self.out = try self.rig.output(L * shape.rows * D * 2);
            return true;
        }

        /// Head 0: NaN in row 0's query, +inf and -inf in values, both in one column; head 1: NaN in one key.
        fn plant(self: *Self) void {
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

        /// Maxima tie across a 64-key tile then change; short shapes keep the sequence and special bf16 bits in Q/K/V.
        fn plantAdversarial(self: *Self) void {
            const q = self.at(0, 0, 0);
            @memset(q, 0);
            q[0] = c.toBf16(1);
            q[1] = 0x8000;
            q[2] = 0x0001;
            q[3] = 0x8001;
            q[4] = 0x007f;
            q[5] = 0x807f;
            for (0..self.shape.k) |row| @memset(self.at(row, 1, 0), 0);
            const tile = @min(64, (self.shape.k - 1) / 2);
            if (tile == 0) return;
            const spots = [3]usize{ tile - 1, tile, @min(2 * tile, self.shape.k - 1) };
            for (spots, 0..) |row, i| {
                const key = self.at(row, 1, 0);
                key[0] = c.toBf16(if (i < 2) 16 else 32);
                key[1] = if (i == 0) 0x8000 else 0;
                key[2] = if (i == 0) 0x0001 else 0x8001;
                key[3] = if (i == 0) 0x807f else 0x007f;
                const value = self.at(row, 2, 0);
                value[0] = c.toBf16(1 + 2 * @as(f32, @floatFromInt(i)));
                value[1] = if (i == 0) 0x8000 else 0;
                value[2] = if (i == 0) 0x0001 else 0x8001;
                value[3] = if (i == 0) 0x8001 else 0x007f;
                value[4] = if (i == 0) 0x007f else 0x807f;
            }
        }

        /// Gate 4: every key one vector (each p is 1); each value column mirrored over the keys.
        fn stress(self: *Self, rng: std.Random) !void {
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

        pub fn run(self: *Self) !void {
            try self.att.full(self.rig.s, self.dev, self.out, self.shape.k, self.shape.rows, D, scale());
            try self.rig.s.synchronize();
        }

        pub fn count(self: *Self) usize {
            return self.shape.k * self.shape.rows * D;
        }

        /// Output head i (row i / heads, head i % heads) over every row's key and value.
        pub fn load(self: *Self, i: usize, hd: *attn.Head) !void {
            const row = i / self.shape.rows;
            const head = i % self.shape.rows;
            for (hd.q, self.at(row, 0, head)) |*y, x| y.* = c.fromBf16(x);
            for (0..self.shape.k) |j| for (0..D) |e| {
                hd.k[j * D + e] = c.fromBf16(self.at(j, 1, head)[e]);
                hd.v[j * D + e] = c.fromBf16(self.at(j, 2, head)[e]);
            };
            hd.n = self.shape.k;
        }

        fn heads(self: *Self) usize {
            return self.shape.k * self.shape.rows;
        }

        pub fn check(self: *Self, gate: g.Gate) !?[]const u8 {
            return self.r.?.check(self, self.heads(), scale(), try self.rig.outputBytes(), gate);
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            return self.r.?.reference(self, self.heads(), scale(), want, mag);
        }

        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            return self.r.?.simulate(self, self.heads(), scale(), acc, order, out);
        }
    };
}
