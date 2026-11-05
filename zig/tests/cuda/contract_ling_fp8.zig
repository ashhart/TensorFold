//! Ling-3.1-flash's FP8 gate-up instances in Ling's clamp order (Layer.after) under the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");
const more = @import("contract_ling_more.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const fp8 = core.quant.fp8;
const fx8 = cuda.experts.Fp8Experts;
const experts = more.experts;
const act = more.act;
const bf = more.bf;

pub const Fp8Entry = struct { name: []const u8, prompt: bool, limit: f32 = 0 };

pub const fp8_entries = [_]Fp8Entry{
    .{ .name = "cuda ling fp8_experts decode gate-up in Ling's clamp order (fp8_expert_kernel<2,4,4>, no limit)", .prompt = false },
    .{ .name = "cuda ling fp8_experts decode gate-up in Ling's clamp order (fp8_expert_kernel<2,4,4>, limit 10)", .prompt = false, .limit = 10 },
    .{ .name = "cuda ling fp8_experts prompt gate-up in Ling's clamp order (fp8_expert_prompt_kernel<2,4,2,2>, no limit)", .prompt = true },
    .{ .name = "cuda ling fp8_experts prompt gate-up in Ling's clamp order (fp8_expert_prompt_kernel<2,4,2,2>, limit 10)", .prompt = true, .limit = 10 },
};

/// A 128-block: eight k16 MMAs of up to 17-term chains, then one fma a block of the scales.
fn fp8Depth(s: c.Shape) usize {
    return 8 * 17 + s.k / 128;
}

const fp8_decode = [_]c.Shape{ .{ .rows = 1, .n = 1280, .k = 4096 }, .{ .rows = 8, .n = 640, .k = 4096 } };
const fp8_prompt = [_]c.Shape{ .{ .rows = 3, .n = 1280, .k = 4096 }, .{ .rows = 70, .n = 640, .k = 4096 } };
const slots = 2;

fn fp8Spec(e: Fp8Entry) c.Spec {
    return .{ .name = e.name, .accum = .f32, .depth = fp8Depth, .mids = 3, .out = .bf16, .served = if (e.prompt) &fp8_prompt else &fp8_decode };
}

/// Fp8Experts.run's gate-up with Layer.after: token rows x, e4m3 gate and up [E][n][k], fp32 scales a 128 x 128 block.
pub const Fp8 = struct {
    rig: rig_.Rig,
    ex: *const fx8,
    entry: Fp8Entry,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    w: []u8 = &.{}, // [2][E][n][k] codes: gate then up
    sc: []f32 = &.{}, // [E][2][n/128][k/128]
    picks: []u8 = &.{},
    extra: []f64 = &.{},
    layer: cuda.experts.Layer = undefined,
    plan: cuda.grouped.Plan = undefined,
    items: usize = 0,
    xd: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Fp8) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Fp8) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.w);
        a.free(self.sc);
        a.free(self.picks);
        a.free(self.extra);
        self.x = &.{};
        self.w = &.{};
        self.sc = &.{};
        self.picks = &.{};
        self.extra = &.{};
    }

    pub fn kernel(self: *Fp8) g.Kernel {
        return .{ .spec = fp8Spec(self.entry), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Fp8);

    fn code(v: i32) u8 {
        var b: u8 = 0;
        while (fp8.e4m3(b) != @as(f64, @floatFromInt(v))) b += 1;
        return b;
    }

    /// Random: x in [-1, 1), |w| < 4, scales in [2^-6, 2^-5); stress: g 1024 (silu exact), u 1, halves a step apart.
    pub fn prepare(self: *Fp8, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const n = shape.n;
        const k = shape.k;
        if (shape.rows == 0 or n % 128 != 0 or k % 128 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const pairs = shape.rows * slots;
        self.x = try a.alloc(u16, shape.rows * k);
        self.w = try a.alloc(u8, 2 * experts * n * k);
        self.sc = try a.alloc(f32, experts * 2 * (n / 128) * (k / 128));
        self.picks = try a.alloc(u8, pairs);
        for (0..shape.rows) |r| {
            self.picks[r * slots] = if (self.entry.prompt) 0 else rng.uintLessThan(u8, experts / 2);
            self.picks[r * slots + 1] = experts / 2 + rng.uintLessThan(u8, experts / 2);
        }
        if (fill == .stress) {
            var tab: [33]u8 = undefined;
            for (&tab, 0..) |*t, i| t.* = code(@as(i32, @intCast(i)) - 16);
            const h = k / 2;
            for (0..shape.rows) |r| for (0..h) |i| {
                const v: f32 = if (i == 5) 1 else if (i == 7) 64 else @floatFromInt(1 + rng.uintLessThan(u32, 127));
                self.x[r * k + i] = c.toBf16(v);
                self.x[r * k + h + (i + 1) % h] = c.toBf16(v);
            };
            @memset(self.w[0 .. experts * n * k], tab[16]);
            for (0..experts * n) |row| {
                self.w[row * k + 7] = tab[32];
                const up = self.w[(experts * n + row) * k ..][0..k];
                for (0..h) |i| {
                    const v: i32 = 1 + @as(i32, @intCast(rng.uintLessThan(u32, 16)));
                    up[i] = tab[@intCast(16 + v)];
                    up[h + (i + 1) % h] = tab[@intCast(16 - v + @as(i32, @intFromBool(i == 5)))];
                }
            }
            @memset(self.sc, 1);
        } else {
            for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
            for (self.w) |*b| b.* = (@as(u8, rng.uintLessThan(u8, 2)) << 7) | (rng.uintLessThan(u8, 9) << 3) | rng.uintLessThan(u8, 8);
            for (self.sc) |*s| s.* = (1 + rng.float(f32)) * 0x1p-6;
        }
        if (fill == .nonfinite) {
            self.x[5] = c.toBf16(std.math.nan(f32));
            if (shape.rows > 1) self.x[k + 9] = c.toBf16(std.math.inf(f32));
            if (shape.rows > 2) {
                self.x[2 * k + 3] = c.toBf16(std.math.inf(f32));
                self.x[2 * k + 200] = c.toBf16(-std.math.inf(f32));
            }
        }
        const one = n * k;
        const pu = try a.alloc(u32, experts * one / 2);
        defer a.free(pu);
        for (0..experts) |e| try fx8.packGateUp(a, pu[e * one / 2 ..][0 .. one / 2], self.w[e * one ..][0..one], self.w[(experts + e) * one ..][0..one], n, k);
        const plan = try more.Plan.make(a, self.picks, experts, if (self.entry.prompt) 64 else 16);
        defer plan.deinit(a);
        self.items = plan.n;
        const cnt: i32 = @intCast(plan.n);
        self.plan = .{ .items = try self.rig.input(std.mem.sliceAsBytes(plan.items[0 .. 3 * plan.n])), .counts = try self.rig.input(std.mem.asBytes(&cnt)), .members = try self.rig.input(std.mem.sliceAsBytes(plan.members)), .rank = 0, .hist = 0 };
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        const up = try self.rig.input(std.mem.sliceAsBytes(pu));
        const scd = try self.rig.input(std.mem.sliceAsBytes(self.sc));
        self.layer = .{ .format = .fp8g, .up = up, .down = 0, .up_scale = scd, .width = n, .dims = k, .experts = experts, .limit = self.entry.limit, .after = true };
        self.out = try self.rig.output(pairs * n * 2);
        return true;
    }

    pub fn run(self: *Fp8) !void {
        try self.ex.run(self.rig.s, .up, .{ .x = self.xd, .stride = self.shape.k, .slots = slots }, self.layer, self.plan, self.entry.prompt, self.out, self.items, -1);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Fp8) usize {
        return self.shape.rows * slots * self.shape.n;
    }

    /// Output o's products for matrix m (0 gate, 1 up), each times its block's scale.
    fn terms(self: *const Fp8, m: usize, o: usize, t: []f64) void {
        const n = self.shape.n;
        const k = self.shape.k;
        const p = o / n;
        const col = o % n;
        const e: usize = self.picks[p];
        const w = self.w[((m * experts + e) * n + col) * k ..][0..k];
        const sc = self.sc[((e * 2 + m) * (n / 128) + col / 128) * (k / 128) ..][0 .. k / 128];
        for (t, 0..) |*v, i| v.* = bf(self.x[(p / slots) * k + i]) * fp8.e4m3(w[i]) * sc[i / 128];
    }

    pub fn reference(self: *Fp8, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.extra);
        self.extra = try a.alloc(f64, want.len);
        const t = try a.alloc(f64, self.shape.k);
        defer a.free(t);
        const gam = c.gamma(@floatFromInt(fp8Depth(self.shape)), 0x1p-24);
        for (want, mag, self.extra, 0..) |*w, *m, *x, o| {
            var v: [2]f64 = undefined;
            var e: [2]f64 = undefined;
            for (0..2) |mi| {
                self.terms(mi, o, t);
                var abs: f64 = 0;
                for (t) |y| abs += @abs(y);
                v[mi] = pages.sumIn(.none, .sequential, t);
                e[mi] = gam * abs;
            }
            w.* = act(v[0], v[1], self.entry.limit);
            m.* = 0;
            x.* = if (self.fill == .stress) 0 else more.actSpread(v[0], v[1], e[0], e[1], 0x1p-8);
        }
    }

    pub fn bounds(self: *Fp8, _: []const f64, _: []const f64, allow: []f64) !void {
        more.widen(allow, self.extra);
    }

    pub fn simulate(self: *Fp8, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (out, 0..) |*v, o| {
            var x: [2]f64 = undefined;
            for (0..2) |mi| {
                self.terms(mi, o, t);
                x[mi] = pages.sumIn(acc, order, t);
            }
            v.* = act(x[0], x[1], self.entry.limit);
        }
    }
};
