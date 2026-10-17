//! The checkpoint math's NVFP4 experts at decode-sized calls under the contract on CUDA: ck_experts.cu's _few pair.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Ck = cuda.ck.Ck;
const grouped = cuda.grouped;

pub const Kind = enum { up, down };

const experts = 4;
const slots = 4; // pairs a token row
const rule = "as the prompt kernel it replaces gives them: NaN and infinite alphas, a NaN block scale";

/// A 64-input mma step a link, after the products inside it.
fn stepDepth(s: c.Shape) usize {
    return s.k / 64 + 64;
}

/// Super's routed experts (latent 1,024, a share 1,344 wide) and its shared expert (hidden 4,096, a share 2,688 wide).
pub fn spec(kind: Kind) c.Spec {
    const tiles = [3]c.Tile{ .{ .size = 16 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 256, .step = 64, .min = 64 } };
    return switch (kind) {
        .up => .{ .name = "cuda ck experts up tf_ck4_up_relu2_few", .accum = .f32, .depth = stepDepth, .ops = 2, .mids = 1, .out = .bf16, .replaces = "tf_ck4_up_relu2", .nan = .{ .stated = rule }, .tiles = tiles, .served = &.{ .{ .rows = 1, .n = 1344, .k = 1024 }, .{ .rows = 32, .n = 1344, .k = 1024 }, .{ .rows = 1, .n = 2688, .k = 4096 }, .{ .rows = 32, .n = 2688, .k = 4096 } } },
        .down => .{ .name = "cuda ck experts down tf_ck4_down_f32_few", .accum = .f32, .depth = stepDepth, .ops = 1, .out = .f32, .replaces = "tf_ck4_down_f32", .nan = .{ .stated = rule }, .tiles = tiles, .served = &.{ .{ .rows = 1, .n = 1024, .k = 1344 }, .{ .rows = 32, .n = 1024, .k = 1344 }, .{ .rows = 1, .n = 4096, .k = 2688 }, .{ .rows = 32, .n = 4096, .k = 2688 } } },
    };
}

/// shape.rows token rows, `slots` pairs each over `experts` experts; up reads a token's row, down a pair's.
pub const Experts = struct {
    rig: rig_.Rig,
    kind: Kind,
    ck: *const Ck,
    router: grouped.Router,
    shape: c.Shape = undefined,
    x: []u8 = &.{}, // NVFP4 rows: codes, then block scales
    w: []u8 = &.{}, // [experts][n][k/2] codes
    ws: []u8 = &.{}, // [experts][n][k/16] block scales
    alpha: [experts]f32 = undefined,
    picks: []i32 = &.{},
    xd: u64 = 0,
    wd: [2]u64 = undefined,
    ad: u64 = 0,
    plan: grouped.Plan = undefined,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Experts) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Experts) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.w);
        a.free(self.ws);
        a.free(self.picks);
        self.x = &.{};
        self.w = &.{};
        self.ws = &.{};
        self.picks = &.{};
    }

    pub fn kernel(self: *Experts) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Experts);

    fn pairs(self: *const Experts) usize {
        return self.shape.rows * slots;
    }

    /// Activation rows: a token's for up, a pair's for down.
    fn xRows(self: *const Experts) usize {
        return if (self.kind == .up) self.shape.rows else self.pairs();
    }

    fn row(self: *const Experts, pair: usize) usize {
        return if (self.kind == .up) pair / slots else pair;
    }

    fn items(self: *const Experts) usize {
        return grouped.maxItems(self.pairs(), experts, 16);
    }

    /// Random codes and scales; non-finite: NaN and infinite alphas and a NaN block scale; stress: mirrored halves.
    pub fn prepare(self: *Experts, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > cuda.ck.few_tokens or shape.n % 64 != 0 or shape.n == 0 or shape.k % 64 != 0 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u8, cuda.ck.rowsBytes(self.xRows(), k));
        self.w = try a.alloc(u8, experts * n * k / 2);
        self.ws = try a.alloc(u8, experts * n * k / 16);
        self.picks = try a.alloc(i32, self.pairs());
        const xs = self.x[self.xRows() * k / 2 ..];
        rng.bytes(self.x[0 .. self.xRows() * k / 2]);
        rng.bytes(self.w);
        for (xs) |*s| s.* = 0x20 + rng.uintLessThan(u8, 0x28);
        for (self.ws) |*s| s.* = 0x20 + rng.uintLessThan(u8, 0x28);
        for (&self.alpha) |*v| v.* = 0.005 + rng.float(f32) * 0.03;
        for (self.picks, 0..) |*p, i| p.* = @intCast(if (i < experts) i else rng.uintLessThan(usize, experts));
        switch (fill) {
            .random => {},
            .nonfinite => {
                self.alpha[1] = std.math.nan(f32);
                self.alpha[2] = std.math.inf(f32);
                self.alpha[3] = -std.math.inf(f32);
                self.ws[0] = 0x7f;
            },
            .stress => self.stress(rng),
        }
        self.xd = try self.rig.input(self.x);
        const packed_bytes = Ck.packedBytes(experts, n, k);
        const zeros = try a.alloc(u8, packed_bytes[0]);
        defer a.free(zeros);
        @memset(zeros, 0);
        self.wd = .{ try self.rig.input(zeros), try self.rig.input(zeros[0..packed_bytes[1]]) };
        const src = [2]u64{ try self.rig.scratch(self.w.len), try self.rig.scratch(self.ws.len) };
        for (src, [2][]const u8{ self.w, self.ws }) |p, h| try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = p, .len = h.len }).upload(0, h);
        try self.ck.pack(self.rig.s, src[0], src[1], experts, n, k, 0, k, self.wd[0], self.wd[1]);
        self.ad = try self.rig.input(std.mem.sliceAsBytes(&self.alpha));
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        const routed = [3]usize{ self.pairs() * 4, self.items() * 12, 8 };
        self.plan = .{ .members = try self.rig.input(zeros[0..routed[0]]), .items = try self.rig.input(zeros[0..routed[1]]), .counts = try self.rig.input(zeros[0..routed[2]]), .rank = try self.rig.scratch(4), .hist = try self.rig.scratch(4) };
        try self.router.route(self.rig.s, ids, self.pairs(), experts, 16, self.plan);
        try self.rig.s.synchronize();
        const bytes = self.pairs() * n * c.size(spec(self.kind).out);
        self.out = try self.rig.output(bytes);
        self.base = try self.rig.scratch(bytes);
        return true;
    }

    /// Gate 4: block scales 16 (x) and 8 (w), values 4 and 6 mirrored across halves (8 of x's a step lower), alphas 1.
    fn stress(self: *Experts, rng: std.Random) void {
        const k = self.shape.k;
        const xs = self.x[self.xRows() * k / 2 ..];
        @memset(xs, 0x58);
        @memset(self.ws, 0x50);
        @memset(&self.alpha, 1);
        for (0..self.xRows()) |r| for (0..k / 2) |i| {
            const v: u8 = 6 + rng.uintLessThan(u8, 2);
            put(self.x[r * k / 2 ..], i, v);
            put(self.x[r * k / 2 ..], k / 2 + i, if (i < 8) v - 1 else v);
        };
        for (0..experts * self.shape.n) |j| for (0..k / 2) |i| {
            const v: u8 = 6 + rng.uintLessThan(u8, 2);
            put(self.w[j * k / 2 ..], i, v);
            put(self.w[j * k / 2 ..], k / 2 + i, v | 8);
        };
    }

    fn put(codes: []u8, i: usize, v: u8) void {
        const b = &codes[i / 2];
        b.* = if (i % 2 == 0) (b.* & 0xf0) | v else (b.* & 0x0f) | v << 4;
    }

    fn value(codes: []const u8, scales: []const u8, r: usize, k: usize, i: usize) f64 {
        const b = codes[r * k / 2 + i / 2];
        const nib = if (i % 2 == 0) b & 0xf else b >> 4;
        return core.quant.nvfp4.e2m1(nib) * core.quant.fp8.e4m3(scales[r * k / 16 + i / 16]);
    }

    /// Pair p's products with column j of its expert, in input order.
    fn terms(self: *const Experts, p: usize, j: usize, t: []f64) void {
        const k = self.shape.k;
        const xs = self.x[self.xRows() * k / 2 ..];
        const wr = @as(usize, @intCast(self.picks[p])) * self.shape.n + j;
        for (t, 0..) |*o, i| o.* = value(self.x, xs, self.row(p), k, i) * value(self.w, self.ws, wr, k, i);
    }

    /// The epilogue on a pair's sum: alpha times it, then relu^2 for up.
    fn epilogue(self: *const Experts, p: usize, sum: f64) f64 {
        const v = sum * self.alpha[@intCast(self.picks[p])];
        return if (self.kind == .up) @max(v, 0) * @max(v, 0) else v;
    }

    pub fn run(self: *Experts) !void {
        try self.launch(1, self.out);
        try self.rig.s.synchronize();
    }

    /// Form 0 (prompt chunks, 64 columns a warp) or 1 (decode-sized, 32) of this kind's kernel into `out`.
    fn launch(self: *Experts, form: usize, out: u64) !void {
        const up = self.kind == .up;
        const f = if (up) self.ck.up_fns[form] else self.ck.down_fns[form];
        const x = self.shape;
        try Ck.run(self.rig.s, f, self.ck.resident[form][@intFromBool(!up)], Ck.warp_columns[form], self.xd, self.xRows(), if (up) slots else 0, self.wd[0], self.wd[1], self.ad, x.k, x.n, self.plan, out, self.items());
    }

    pub fn replaced(self: *Experts) ![]const u8 {
        try self.launch(0, self.base);
        return self.rig.bytes(self.base, self.count() * c.size(spec(self.kind).out));
    }

    /// Gate 1: the replaced kernel's bytes, with some output non-finite to prove the planted values reach it.
    pub fn check(self: *Experts, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from {s}'s", .{ at, got.len, spec(self.kind).replaces.? });
        const out = spec(self.kind).out;
        for (0..self.count()) |i| if (c.classOf(c.read(out, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    pub fn count(self: *Experts) usize {
        return self.pairs() * self.shape.n;
    }

    pub fn reference(self: *Experts, want: []f64, mag: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.pairs()) |p| for (0..self.shape.n) |j| {
            self.terms(p, j, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            const o = p * self.shape.n + j;
            const al = @abs(self.alpha[@intCast(self.picks[p])]);
            want[o] = self.epilogue(p, sum);
            mag[o] = if (self.kind == .up) 2 * @sqrt(want[o]) * al * abs else al * abs;
        };
    }

    pub fn simulate(self: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.pairs()) |p| for (0..self.shape.n) |j| {
            self.terms(p, j, t);
            out[p * self.shape.n + j] = self.epilogue(p, contract.rows.sumIn(acc, order, t));
        };
    }
};
