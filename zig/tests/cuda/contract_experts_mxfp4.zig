//! MXFP4 experts (experts_mxfp4.cu): SiTU's up and the downs on ERNIE 5.5's routed shapes, both forms; four controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const grouped = cuda.grouped;
const ex_ = cuda.experts;
const Mx = ex_.Mxfp4Experts;
const aref = core.act_ref;

pub const Kind = enum { up, down_f32, down_bf16 };

/// A planted fault: a down a group short or each scale the next's; the up's matrices swapped, or SiTU unclamped.
pub const Fault = enum { none, short, scale_shift, swapped, no_clamp };

pub const Entry = struct { name: []const u8, kind: Kind, prompt: bool, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda mxfp4 experts down tf_mxfp4_experts_down_f32", .kind = .down_f32, .prompt = false },
    .{ .name = "cuda mxfp4 experts down tf_mxfp4_experts_down_bf16", .kind = .down_bf16, .prompt = false },
    .{ .name = "cuda mxfp4 experts prompt down tf_mxfp4_experts_prompt_down_f32", .kind = .down_f32, .prompt = true },
    .{ .name = "cuda mxfp4 experts prompt down tf_mxfp4_experts_prompt_down_bf16", .kind = .down_bf16, .prompt = true },
    .{ .name = "control: mxfp4 experts fp32 down launched one 32-input group short", .kind = .down_f32, .prompt = false, .fault = .short },
    .{ .name = "control: mxfp4 experts fp32 down reading each group's scale from the next", .kind = .down_f32, .prompt = false, .fault = .scale_shift },
    .{ .name = "cuda mxfp4 experts up tf_mxfp4_experts_up_situ (gate and up under SiTU at beta 4, linear beta 25)", .kind = .up, .prompt = false },
    .{ .name = "cuda mxfp4 experts prompt up tf_mxfp4_experts_prompt_up_situ", .kind = .up, .prompt = true },
    .{ .name = "control: mxfp4 experts up reading the up matrix as the gate and the gate as the up", .kind = .up, .prompt = false, .fault = .swapped },
    .{ .name = "control: mxfp4 experts up with SiTU's up unclamped (linear beta 1e30)", .kind = .up, .prompt = false, .fault = .no_clamp },
};

const experts = 4;
const slots = 10; // pairs a token row: ERNIE 5.5's ten routed picks
/// ERNIE 5.5's SiTU: activation_situ_beta and activation_situ_linear_beta.
const beta = 4;
const lin = 25;
/// SiTU after the sums: expf and two tanhf (2 ulp each, 4 units of 2^-24), three divisions, an add and four products.
const situ_ops = 20;
/// The prompt up's pass: 16 pairs a step at RT 2 (experts_mxfp4.cu), half the 64-pair items it walks.
const up_prompt_pass = 32;
const up_why = "fp16's strided sums come out exact here, so bf16's miss is the one shown";

/// An mma k16 step a link of each row's chain, plus one step's error over its own products (attention_range's 36).
fn depth(s: c.Shape) usize {
    return s.k / 16 + core.attention_range.mma_step;
}

/// ERNIE 5.5's routed ups and downs, 2,048 by 2,048; decode windows of 1 to 32 rows, prompt items of 4 and 7.
const downs = [2][]const c.Shape{ &.{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 32, .n = 2048, .k = 2048 } }, &.{ .{ .rows = 40, .n = 2048, .k = 2048 }, .{ .rows = 72, .n = 2048, .k = 2048 } } };

pub fn spec(e: Entry) c.Spec {
    const served = downs[@intFromBool(e.prompt)];
    const pass: usize = if (!e.prompt) 16 else if (e.kind == .up) up_prompt_pass else ex_.promptTile(.mxfp4);
    const tiles = [3]c.Tile{ .{ .size = @max(1, pass / slots) }, .{ .size = 64, .step = 32, .min = 32 }, .{ .size = 64, .step = 32, .min = 32 } };
    if (e.kind == .up) return .{ .name = e.name, .accum = .f32, .depth = depth, .ops = situ_ops, .out = .bf16, .tiles = tiles, .served = served, .sums = .{ .f16 = 0, .why = up_why } };
    return .{ .name = e.name, .accum = .f32, .depth = depth, .out = if (e.kind == .down_f32) .f32 else .bf16, .tiles = tiles, .served = served };
}

/// shape.rows token rows, `slots` pairs each over `experts` experts; a down reads its pair's row, an up its token's.
pub const Experts = struct {
    rig: rig_.Rig,
    entry: Entry,
    kx: *const Mx,
    router: grouped.Router,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{}, // [experts][m][n][k]: the codes decoded on the host, the values the kernel must use
    codes: []u8 = &.{}, // [experts][m][n][k/2], low nibble first (m 2 for the up: gate, then up)
    scales: []u8 = &.{}, // [experts][m][n][k/32]
    extra: []f64 = &.{}, // the up's: how far each output may move with its gate's and up's bf16 roundings
    picks: []i32 = &.{},
    xd: u64 = 0,
    cd: u64 = 0,
    sd: u64 = 0,
    plan: grouped.Plan = undefined,
    out: u64 = 0,

    pub fn deinit(self: *Experts) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Experts) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.w);
        a.free(self.codes);
        a.free(self.scales);
        a.free(self.picks);
        a.free(self.extra);
        self.extra = &.{};
        self.x = &.{};
        self.w = &.{};
        self.codes = &.{};
        self.scales = &.{};
        self.picks = &.{};
    }

    pub fn kernel(self: *Experts) g.Kernel {
        const control: ?g.Gate = switch (self.entry.fault) {
            .none => null,
            .short, .scale_shift, .swapped, .no_clamp => .shapes,
        };
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Experts);

    fn pairs(self: *const Experts) usize {
        return self.shape.rows * slots;
    }

    fn up(self: *const Experts) bool {
        return self.entry.kind == .up;
    }

    /// Matrices an expert: the up's gate and up, or a down.
    fn mats(self: *const Experts) usize {
        return if (self.up()) 2 else 1;
    }

    fn tile(self: *const Experts) usize {
        return if (self.entry.prompt) ex_.promptTile(.mxfp4) else 16;
    }

    fn items(self: *const Experts) usize {
        return grouped.maxItems(self.pairs(), experts, self.tile());
    }

    /// Weight i of its row: code `q` times 2^(s - 127) in bf16, exact for the scales Mxfp4Experts takes, -0 kept.
    fn decoded(q: u8, s: u8) u16 {
        const mag = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
        const b: u32 = @bitCast(mag[q & 7] * @as(f32, @bitCast(@as(u32, s) << 23)));
        return @intCast((b >> 16) | (if (q & 8 != 0) @as(u32, 0x8000) else 0));
    }

    fn setCode(self: *Experts, i: usize, q: u8) void {
        const byte = &self.codes[i / 2];
        byte.* = if (i % 2 == 0) (byte.* & 0xF0) | q else (byte.* & 0x0F) | (q << 4);
    }

    /// Random rows, codes and scales; non-finite: NaN, +Inf and -Inf in the first rows; stress: cancelling sums.
    pub fn prepare(self: *Experts, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = if (self.up()) shape.rows else self.pairs();
        self.x = try a.alloc(u16, rows * k);
        self.w = try a.alloc(u16, experts * self.mats() * n * k);
        self.codes = try a.alloc(u8, Mx.codeBytes(experts * self.mats(), n, k));
        self.scales = try a.alloc(u8, Mx.scaleBytes(experts * self.mats(), n, k));
        self.picks = try a.alloc(i32, self.pairs());
        for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.codes) |*v| v.* = rng.int(u8);
        // Scales 2^-5 to 2^-2: E2M1's 6 times these stays under 1.5.
        for (self.scales) |*v| v.* = 122 + rng.uintLessThan(u8, 4);
        for (self.picks, 0..) |*p, i| p.* = @intCast(if (i < experts) i else rng.uintLessThan(usize, experts));
        switch (fill) {
            .random => {},
            .nonfinite => {
                // a row holding NaN, one holding +Inf, one holding both infinities
                const planted = [_]struct { usize, usize, f32 }{ .{ 0, k / 3, std.math.nan(f32) }, .{ 1, k / 2, std.math.inf(f32) }, .{ 2, 1, std.math.inf(f32) }, .{ 2, k - 1, -std.math.inf(f32) } };
                for (planted) |p| if (p[0] < rows) {
                    self.x[p[0] * k + p[1]] = c.toBf16(p[2]);
                };
            },
            .stress => if (self.up()) self.stressUp(rng) else self.stress(rng),
        }
        try Mx.checkScales(self.scales);
        for (self.w, 0..) |*v, i| v.* = decoded(if (i % 2 == 0) self.codes[i / 2] & 15 else self.codes[i / 2] >> 4, self.scales[i / 32]);
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        if (self.entry.fault == .swapped) {
            const cs = try a.dupe(u8, self.codes);
            defer a.free(cs);
            const ss = try a.dupe(u8, self.scales);
            defer a.free(ss);
            swapHalves(cs);
            swapHalves(ss);
            self.cd = try self.rig.input(cs);
            self.sd = try self.rig.input(ss);
        } else {
            self.cd = try self.rig.input(self.codes);
            self.sd = try self.rig.input(self.scales);
        }
        const zeros = try a.alloc(u8, @max(self.pairs() * 4, self.items() * 12));
        defer a.free(zeros);
        @memset(zeros, 0);
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        const hist = (self.pairs() + 1023) / 1024;
        self.plan = .{ .members = try self.rig.input(zeros[0 .. self.pairs() * 4]), .items = try self.rig.input(zeros[0 .. self.items() * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = try self.rig.scratch(self.pairs() * 4), .hist = try self.rig.scratch(hist * experts * 4) };
        try self.router.route(self.rig.s, ids, self.pairs(), experts, self.tile(), self.plan);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(self.pairs() * n * c.size(spec(self.entry).out));
        return true;
    }

    /// The up's gate 4: a code a column at 2^-7; rows of 1/16 (the second 3/16) between 2^13 and -2^13: g, u its codes.
    fn stressUp(self: *Experts, rng: std.Random) void {
        const k = self.shape.k;
        @memset(self.scales, 120);
        for (0..experts * 2 * self.shape.n) |j| {
            const q: u8 = if (rng.boolean()) 2 else 4; // E2M1 1.0, 2.0
            for (0..k) |i| self.setCode(j * k + i, q);
        }
        for (0..self.shape.rows) |r| {
            const row = self.x[r * k ..][0..k];
            @memset(row, c.toBf16(1.0 / 16.0));
            row[0] = c.toBf16(8192);
            row[1] = c.toBf16(3.0 / 16.0);
            row[k - 1] = c.toBf16(-8192);
        }
    }

    /// Gate 4: codes 1.0 and 2.0 at scale 2^0 and mirrored integer rows, every partial sum exact under 2^20.
    fn stress(self: *Experts, rng: std.Random) void {
        const k = self.shape.k;
        const h = k / 2;
        @memset(self.scales, 127);
        for (0..experts * self.shape.n) |j| for (0..h) |i| {
            const q: u8 = if (rng.boolean()) 2 else 4; // E2M1 1.0, 2.0
            self.setCode(j * k + i, q);
            self.setCode(j * k + h + i, q);
        };
        for (0..self.pairs()) |r| for (0..h) |i| {
            const v: f32 = @floatFromInt(4 * (1 + 2 * rng.uintLessThan(u32, 64)));
            self.x[r * k + i] = c.toBf16(v);
            self.x[r * k + h + i] = c.toBf16(if (i < 8) 4 - v else -v);
        };
    }

    pub fn run(self: *Experts) !void {
        const n = self.shape.n;
        const k = self.shape.k;
        if (self.up()) {
            const in: ex_.Rows = .{ .x = self.xd, .stride = k, .slots = slots };
            const lb: f32 = if (self.entry.fault == .no_clamp) 1e30 else lin;
            const l: ex_.Layer = .{ .format = .mxfp4, .up = self.cd, .up_scale = self.sd, .down = 0, .width = n, .dims = k, .experts = experts, .situ = .{ beta, lb } };
            try self.kx.run(self.rig.s, self.entry.prompt, .up, in, l, self.plan, self.out, self.items(), -1);
            return self.rig.s.synchronize();
        }
        const in: ex_.Rows = .{ .x = self.xd, .stride = k };
        const epi: ex_.Epi = if (self.entry.kind == .down_f32) .down_f32 else .down_bf16;
        const l: ex_.Layer = .{ .format = .mxfp4, .up = 0, .down = self.cd, .down_scale = self.sd, .width = k, .dims = n, .experts = experts };
        switch (self.entry.fault) {
            .none => try self.kx.run(self.rig.s, self.entry.prompt, epi, in, l, self.plan, self.out, self.items(), -1),
            .short => try self.launch(k / 32 - 1, self.sd),
            .scale_shift => try self.launch(k / 32, self.sd + 1),
            .swapped, .no_clamp => unreachable,
        }
        try self.rig.s.synchronize();
    }

    /// The fp32 decode down with its own input groups and scales: experts_mxfp4.cu's arguments.
    fn launch(self: *Experts, kg: usize, scales: u64) !void {
        var a: cuda.Args = .{};
        a.add(self.xd);
        a.add(@as(i32, @intCast(self.shape.k)));
        a.add(@as(i32, 0));
        a.add(self.cd);
        a.add(scales);
        a.add(@as(i32, @intCast(kg)));
        a.add(@as(i32, @intCast(self.shape.n / Mx.cols)));
        for ([_]u64{ self.plan.items, self.plan.counts, self.plan.members, self.out }) |v| a.add(v);
        a.add(@as(i32, @intCast(self.shape.n)));
        a.add(@as(i32, -1));
        const grid: u32 = @intCast((self.items() * self.shape.n / Mx.cols + Mx.warps - 1) / Mx.warps);
        try cuda.launch.launch(self.kx.fns[0][1], .{ .grid = .{ .x = grid, .y = 1, .z = 1 }, .block = .{ .x = Mx.warps * 32 } }, self.rig.s, &a);
    }

    pub fn count(self: *Experts) usize {
        return self.pairs() * self.shape.n;
    }

    /// Pair p's products with column j of its expert's matrix m (the up's gate 0, up 1), in input order.
    fn terms(self: *const Experts, p: usize, j: usize, m: usize, t: []f64) void {
        const k = self.shape.k;
        const wr = ((@as(usize, @intCast(self.picks[p])) * self.mats() + m) * self.shape.n + j) * k;
        const xr = if (self.up()) p / slots else p;
        for (t, 0..) |*o, i| o.* = c.fromBf16(self.x[xr * k + i]) * c.fromBf16(self.w[wr + i]);
    }

    pub fn reference(self: *Experts, want: []f64, mag: []f64) !void {
        if (self.up()) return self.referenceUp(want, mag);
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.pairs()) |p| for (0..self.shape.n) |j| {
            self.terms(p, j, 0, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            want[p * self.shape.n + j] = sum;
            mag[p * self.shape.n + j] = abs;
        };
    }

    /// SiTU of the bf16 sums; mag 0, extra each sum's slope times how far its bf16 can move within the sum's error.
    fn referenceUp(self: *Experts, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const k = self.shape.k;
        const t = try a.alloc(f64, 2 * k);
        defer a.free(t);
        a.free(self.extra);
        self.extra = try a.alloc(f64, want.len);
        @memset(mag, 0);
        const gamma = c.gamma(@floatFromInt(depth(self.shape)), 0x1p-24);
        for (0..self.pairs()) |p| for (0..self.shape.n) |j| {
            var gu: [2]f64 = undefined;
            var moved: [2]f64 = undefined;
            for (0..2) |m| {
                const tm = t[m * k ..][0..k];
                self.terms(p, j, m, tm);
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (tm) |v| {
                    sum += v;
                    abs += @abs(v);
                }
                gu[m] = bf(sum);
                const e = gamma * abs;
                moved[m] = @max(@abs(bf(sum - e) - gu[m]), @abs(bf(sum + e) - gu[m]));
            }
            const at = p * self.shape.n + j;
            want[at] = aref.situ(gu[0], gu[1], beta, lin);
            var slope: [2]f64 = @splat(0);
            for ([_]f64{ -1, 0, 1 }) |d| {
                const sg = aref.situSlopes(gu[0] + d * moved[0], gu[1], beta, lin);
                const su = aref.situSlopes(gu[0], gu[1] + d * moved[1], beta, lin);
                slope = .{ @max(slope[0], @abs(sg[0])), @max(slope[1], @abs(su[1])) };
            }
            const x = slope[0] * moved[0] + slope[1] * moved[1];
            self.extra[at] = if (std.math.isFinite(x)) x else 0;
        };
    }

    pub fn bounds(self: *Experts, _: []const f64, _: []const f64, allow: []f64) !void {
        if (!self.up()) return;
        for (allow, self.extra) |*l, x| l.* += x * (1 + 0x1p-7);
    }

    pub fn simulate(self: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.pairs()) |p| for (0..self.shape.n) |j| {
            if (self.up()) {
                var gu: [2]f64 = undefined;
                for (&gu, 0..) |*v, m| {
                    self.terms(p, j, m, t);
                    v.* = bf(contract.rows.sumIn(acc, order, t));
                }
                out[p * self.shape.n + j] = aref.situ(gu[0], gu[1], beta, lin);
                continue;
            }
            self.terms(p, j, 0, t);
            out[p * self.shape.n + j] = contract.rows.sumIn(acc, order, t);
        };
    }
};

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// Each expert's two halves (the up's gate and up matrices) exchanged, for the swapped control.
fn swapHalves(bytes: []u8) void {
    const half = bytes.len / experts / 2;
    for (0..experts) |e| {
        const at = bytes[e * 2 * half ..][0 .. 2 * half];
        for (at[0..half], at[half..]) |*x, *y| std.mem.swap(u8, x, y);
    }
}

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    kx: Mx,
    mod: cuda.Module,
    items: [entries.len]Experts = undefined,

    /// The set, its kernels appended to `list` (each control after the kernels it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        var kx = try Mx.load(gpu.d, sms);
        errdefer kx.unload();
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
        errdefer mod.unload();
        set.* = .{ .gpa = gpu.gpa, .kx = kx, .mod = mod };
        const router = try grouped.Router.resolve(set.mod);
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .kx = &set.kx, .router = router };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.mod.unload();
        set.kx.unload();
        set.gpa.destroy(set);
    }
};

/// The MXFP4 experts' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6d786634, &t);
    std.debug.print("SUMMARY mxfp4 experts: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "mxfp4-experts contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}
