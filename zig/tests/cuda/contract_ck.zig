//! The checkpoint math's NVFP4 experts under the contract on CUDA: ck_experts.cu's _few pair and its prompt tiles.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Ck = cuda.ck.Ck;
const grouped = cuda.grouped;

/// The decode-sized pair, then the prompt kernels: up to NVFP4 rows, fp32 down, and down summing two width shares.
pub const Kind = enum { up, down, prompt_up, prompt_down, prompt_sum };

const experts = 4; // prompt_sum: 2 experts in 2 shares, share s of expert e expert 2 s + e
const slots = 4; // pairs a token row
const shares = 2;
const rule = "as the prompt kernel it replaces gives them: NaN and infinite alphas, a NaN block scale";
const prompt_rule = "as the kernels it replaces give them: NaN and infinite alphas, a NaN block scale";

/// A 64-input mma step a link, after the products inside it.
fn stepDepth(s: c.Shape) usize {
    return s.k / 64 + 64;
}

/// Each share's mma chain, then the add of the two.
fn shareDepth(s: c.Shape) usize {
    return stepDepth(s) + 1;
}

/// Super's routed experts (latent 1,024, a share 1,344 wide) and its shared expert (hidden 4,096, a share 2,688 wide).
pub fn spec(kind: Kind) c.Spec {
    const tiles = [3]c.Tile{ .{ .size = 16 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 256, .step = 64, .min = 64 } };
    // a prompt block: a plan item of 128 pairs (32 token rows) by 128 columns, four 64-input steps in flight
    const prompt_tiles = [3]c.Tile{ .{ .size = 128 / slots }, .{ .size = 128, .step = 64, .min = 64 }, .{ .size = 256, .step = 64, .min = 64 } };
    const ups = &[_]c.Shape{ .{ .rows = 129, .n = 1344, .k = 1024 }, .{ .rows = 300, .n = 1344, .k = 1024 }, .{ .rows = 129, .n = 2688, .k = 4096 } };
    const downs = &[_]c.Shape{ .{ .rows = 129, .n = 1024, .k = 1344 }, .{ .rows = 300, .n = 1024, .k = 1344 }, .{ .rows = 129, .n = 4096, .k = 2688 } };
    return switch (kind) {
        .up => .{ .name = "cuda ck experts up tf_ck4_up_relu2_few", .accum = .f32, .depth = stepDepth, .ops = 2, .mids = 1, .out = .bf16, .replaces = "tf_ck4_up_relu2", .nan = .{ .stated = rule }, .tiles = tiles, .served = &.{ .{ .rows = 1, .n = 1344, .k = 1024 }, .{ .rows = 32, .n = 1344, .k = 1024 }, .{ .rows = 1, .n = 2688, .k = 4096 }, .{ .rows = 32, .n = 2688, .k = 4096 } } },
        .down => .{ .name = "cuda ck experts down tf_ck4_down_f32_few", .accum = .f32, .depth = stepDepth, .ops = 1, .out = .f32, .replaces = "tf_ck4_down_f32", .nan = .{ .stated = rule }, .tiles = tiles, .served = &.{ .{ .rows = 1, .n = 1024, .k = 1344 }, .{ .rows = 32, .n = 1024, .k = 1344 }, .{ .rows = 1, .n = 4096, .k = 2688 }, .{ .rows = 32, .n = 4096, .k = 2688 } } },
        .prompt_up => .{ .name = "cuda ck experts prompt up tf_ck4_up_nvfp4_prompt (NVFP4 rows out)", .accum = .f32, .depth = stepDepth, .out = .raw, .replaces = "tf_ck4_up_relu2 then tf_ck_quant4", .nan = .{ .stated = prompt_rule }, .tiles = prompt_tiles, .served = ups },
        .prompt_down => .{ .name = "cuda ck experts prompt down tf_ck4_down_f32_prompt", .accum = .f32, .depth = stepDepth, .ops = 1, .out = .f32, .replaces = "tf_ck4_down_f32", .nan = .{ .stated = prompt_rule }, .tiles = prompt_tiles, .served = downs },
        .prompt_sum => .{ .name = "cuda ck experts prompt down tf_ck4_down_sum_prompt (two shares summed, bf16)", .accum = .f32, .depth = shareDepth, .ops = 2, .out = .bf16, .replaces = "tf_ck4_down_f32 then tf_super_combine's share sum", .nan = .{ .stated = prompt_rule }, .tiles = prompt_tiles, .served = downs },
    };
}

/// shape.rows token rows, `slots` pairs each over `experts` experts; up reads a token's row, down a pair's.
pub const Experts = struct {
    rig: rig_.Rig,
    kind: Kind,
    ck: *const Ck,
    router: grouped.Router,
    latent: *const nemotron.latent.Fns,
    shape: c.Shape = undefined,
    x: []u8 = &.{}, // NVFP4 rows: codes, then block scales
    w: []u8 = &.{}, // [experts][n][k/2] codes
    ws: []u8 = &.{}, // [experts][n][k/16] block scales
    alpha: [experts]f32 = undefined,
    act: f32 = 1, // prompt_up: down's input scale, which quantizes its rows
    picks: []i32 = &.{},
    xd: u64 = 0,
    wd: [2]u64 = undefined,
    ad: u64 = 0,
    plan: grouped.Plan = undefined, // 16 pairs an item: the decode-sized and the replaced kernels'
    wide: grouped.Plan = undefined, // cuda.ck.prompt_tile pairs an item: the prompt kernels'
    ones: u64 = 0, // prompt_sum: tf_super_combine's weights, one pick a row
    mid: u64 = 0, // the replaced kernels' rows before quant4 or the share sum
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

    fn up(self: *const Experts) bool {
        return self.kind == .up or self.kind == .prompt_up;
    }

    fn pairs(self: *const Experts) usize {
        return self.shape.rows * slots;
    }

    /// Activation rows: a token's for up, a pair's for down.
    fn xRows(self: *const Experts) usize {
        return if (self.up()) self.shape.rows else self.pairs();
    }

    fn row(self: *const Experts, pair: usize) usize {
        return if (self.up()) pair / slots else pair;
    }

    /// Output rows: a pair's, or for prompt_sum a share pair's.
    fn outRows(self: *const Experts) usize {
        return if (self.kind == .prompt_sum) self.pairs() / shares else self.pairs();
    }

    fn items(self: *const Experts, tile: usize) usize {
        return grouped.maxItems(self.pairs(), experts, tile);
    }

    /// A plan's buffers for `tile` pairs an item, routed from the picks.
    fn route(self: *Experts, ids: u64, tile: usize, zeros: []const u8) !grouped.Plan {
        const blocks = (self.pairs() + 1023) / 1024;
        const p: grouped.Plan = .{ .members = try self.rig.input(zeros[0 .. self.pairs() * 4]), .items = try self.rig.input(zeros[0 .. self.items(tile) * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = try self.rig.scratch(self.pairs() * 4), .hist = try self.rig.scratch(blocks * experts * 4) };
        try self.router.route(self.rig.s, ids, self.pairs(), experts, tile, p);
        return p;
    }

    /// Random codes and scales; non-finite: NaN and infinite alphas and a NaN block scale; stress: mirrored halves.
    pub fn prepare(self: *Experts, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const few = self.kind == .up or self.kind == .down;
        if (shape.rows == 0 or (few and shape.rows > cuda.ck.few_tokens) or shape.n % 64 != 0 or shape.n == 0 or shape.k % 64 != 0 or shape.k == 0) return false;
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
        if (self.kind == .prompt_sum) {
            // a share pair's expert, every expert in the first ones; share s at expert s * experts / shares + e
            for (0..self.pairs() / shares) |q| {
                const e = if (q < experts / shares) q else rng.uintLessThan(usize, experts / shares);
                for (0..shares) |s| self.picks[q * shares + s] = @intCast(s * experts / shares + e);
            }
        } else {
            for (self.picks, 0..) |*p, i| p.* = @intCast(if (i < experts) i else rng.uintLessThan(usize, experts));
        }
        self.act = 0.375;
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
        const zeros = try a.alloc(u8, @max(packed_bytes[0], self.pairs() * 4, self.items(16) * 12));
        defer a.free(zeros);
        @memset(zeros, 0);
        self.wd = .{ try self.rig.input(zeros[0..packed_bytes[0]]), try self.rig.input(zeros[0..packed_bytes[1]]) };
        const src = [2]u64{ try self.rig.scratch(self.w.len), try self.rig.scratch(self.ws.len) };
        for (src, [2][]const u8{ self.w, self.ws }) |p, h| try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = p, .len = h.len }).upload(0, h);
        try self.ck.pack(self.rig.s, src[0], src[1], experts, n, k, 0, k, self.wd[0], self.wd[1]);
        self.ad = try self.rig.input(std.mem.sliceAsBytes(&self.alpha));
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        self.plan = try self.route(ids, 16, zeros);
        if (!few) self.wide = try self.route(ids, cuda.ck.prompt_tile, zeros);
        const ones = try a.alloc(f32, self.outRows());
        defer a.free(ones);
        @memset(ones, 1);
        if (self.kind == .prompt_sum) self.ones = try self.rig.input(std.mem.sliceAsBytes(ones));
        try self.rig.s.synchronize();
        self.out = try self.rig.output(self.outBytes());
        self.base = try self.rig.scratch(self.outBytes());
        self.mid = try self.rig.scratch(self.pairs() * n * 4);
        return true;
    }

    fn outBytes(self: *const Experts) usize {
        return if (self.kind == .prompt_up) cuda.ck.rowsBytes(self.pairs(), self.shape.n) else self.outRows() * self.shape.n * c.size(spec(self.kind).out);
    }

    /// Gate 4: block scales 16 (x) and 8 (w), values 4 and 6 mirrored across halves (8 of x's a step lower), alphas 1.
    fn stress(self: *Experts, rng: std.Random) void {
        const k = self.shape.k;
        const xs = self.x[self.xRows() * k / 2 ..];
        @memset(xs, 0x58);
        @memset(self.ws, 0x50);
        @memset(&self.alpha, 1);
        self.act = 65536; // relu^2 of the stress sums stays inside down's e4m3 block scales
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
        return if (self.up()) @max(v, 0) * @max(v, 0) else v;
    }

    pub fn run(self: *Experts) !void {
        const l = self.layer();
        const wide = self.items(cuda.ck.prompt_tile);
        switch (self.kind) {
            .up, .down => try self.launch(1, self.out),
            .prompt_up => try self.ck.promptUp(self.rig.s, l, self.xd, self.xRows(), slots, self.wide, self.pairs(), self.out, wide),
            .prompt_down => try self.ck.promptDown(self.rig.s, l, self.xd, self.pairs(), self.wide, self.out, wide),
            .prompt_sum => try self.ck.promptSum(self.rig.s, l, self.xd, self.pairs(), self.wide, self.out, wide, shares),
        }
        try self.rig.s.synchronize();
    }

    /// The layer's view of the packed experts: up n outputs from k inputs, down n outputs from k.
    fn layer(self: *const Experts) cuda.ck.Layer {
        const u = self.up();
        return .{ .up = self.wd[0], .up_s = self.wd[1], .down = self.wd[0], .down_s = self.wd[1], .up_alpha = self.ad, .down_alpha = self.ad, .up_act = 1, .down_act = self.act, .experts = experts, .width = if (u) self.shape.n else self.shape.k, .dims = if (u) self.shape.k else self.shape.n };
    }

    /// Form 0 (calls under cuda.ck.prompt_tokens, 64 columns a warp) or 1 (decode-sized, 32) of this kind's kernel.
    fn launch(self: *Experts, form: usize, out: u64) !void {
        const u = self.up();
        const f = if (u) self.ck.up_fns[form] else self.ck.down_fns[form];
        const x = self.shape;
        try Ck.run(self.rig.s, f, self.ck.resident[form][@intFromBool(!u)], Ck.warp_columns[form], self.xd, self.xRows(), if (u) slots else 0, self.wd[0], self.wd[1], self.ad, x.k, x.n, self.plan, out, self.items(16));
    }

    pub fn replaced(self: *Experts) ![]const u8 {
        const n = self.shape.n;
        switch (self.kind) {
            .up, .down, .prompt_down => try self.launch(0, self.base),
            .prompt_up => {
                try self.launch(0, self.mid);
                try self.ck.quant4(self.rig.s, self.mid, n, self.pairs(), n, self.act, self.base);
            },
            .prompt_sum => {
                try self.launch(0, self.mid);
                var a: cuda.Args = .{};
                a.add(self.mid);
                for ([_]usize{ n, shares * n }) |v| a.add(@as(i64, @intCast(v)));
                a.add(@as(c_int, shares));
                a.add(self.ones);
                a.add(@as(c_int, 1));
                a.add(self.base);
                a.add(@as(c_int, @intCast(n)));
                try cuda.launch.launch(self.latent.combine, .{ .grid = .{ .x = @intCast(self.outRows()), .y = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &a);
            },
        }
        return self.rig.bytes(self.base, self.outBytes());
    }

    /// Gate 1: the replaced kernels' bytes, and some output (prompt_up: up's bf16 rows) non-finite: the planted values.
    pub fn check(self: *Experts, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from {s}'s", .{ at, got.len, spec(self.kind).replaces.? });
        const out = if (self.kind == .prompt_up) c.Elem.bf16 else spec(self.kind).out;
        const seen = if (self.kind == .prompt_up) try self.rig.bytes(self.mid, self.pairs() * self.shape.n * 2) else got;
        for (0..seen.len / c.size(out)) |i| if (c.classOf(c.read(out, seen, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    /// Outputs: values, or prompt_up's NVFP4 bytes.
    pub fn count(self: *Experts) usize {
        return if (self.kind == .prompt_up) self.outBytes() else self.outRows() * self.shape.n;
    }

    pub fn reference(self: *Experts, want: []f64, mag: []f64) !void {
        if (self.kind == .prompt_up) {
            @memset(mag, 0);
            return self.sums(.none, .sequential, want);
        }
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        @memset(want, 0);
        @memset(mag, 0);
        for (0..self.pairs()) |p| for (0..self.shape.n) |j| {
            self.terms(p, j, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            const o = p / (self.pairs() / self.outRows()) * self.shape.n + j;
            const al = @abs(self.alpha[@intCast(self.picks[p])]);
            const v = self.epilogue(p, sum);
            want[o] += v;
            mag[o] += if (self.up()) 2 * @sqrt(v) * al * abs else al * abs;
        };
    }

    pub fn simulate(self: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.sums(acc, order, out);
    }

    /// Every output with each pair's sum in `acc` (none: float64); prompt_up's as the bytes quant4 makes of its rows.
    fn sums(self: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const n = self.shape.n;
        const t = try a.alloc(f64, self.shape.k);
        defer a.free(t);
        const rows = try a.alloc(u16, if (self.kind == .prompt_up) self.pairs() * n else 0);
        defer a.free(rows);
        if (self.kind != .prompt_up) @memset(out, 0);
        for (0..self.pairs()) |p| for (0..n) |j| {
            self.terms(p, j, t);
            const sum = contract.rows.sumIn(acc, order, t);
            if (self.kind != .prompt_up) {
                out[p / (self.pairs() / self.outRows()) * n + j] += self.epilogue(p, sum);
                continue;
            }
            // tf_ck4_up_relu2's epilogue: alpha times the sum to bf16, relu, its square to bf16
            const u = @max(c.fromBf16(c.toBf16(@floatCast(sum * self.alpha[@intCast(self.picks[p])]))), 0);
            rows[p * n + j] = c.toBf16(u * u);
        };
        if (self.kind != .prompt_up) return;
        const src = try self.rig.scratch(rows.len * 2);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = src, .len = rows.len * 2 }).upload(0, std.mem.sliceAsBytes(rows));
        const dst = try self.rig.scratch(self.outBytes());
        try self.ck.quant4(self.rig.s, src, n, self.pairs(), n, self.act, dst);
        for (out, try self.rig.bytes(dst, self.outBytes())) |*o, b| o.* = @floatFromInt(b);
    }
};
