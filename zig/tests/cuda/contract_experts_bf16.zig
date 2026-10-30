//! bf16 experts (experts_bf16.cu) under the contract on Transcribe's shapes, decode and prompt forms; two controls.
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
const Bf16 = ex_.Bf16Experts;

pub const Kind = enum { up, down_f32, down_bf16 };

/// A planted fault: the fp32 down launched one 32-input group short, or contract_controls.cu's bf16-summing down.
pub const Fault = enum { none, short, bf16_sums };

pub const Entry = struct { name: []const u8, kind: Kind, prompt: bool, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda bf16 experts up tf_bf16_experts_up (relu^2)", .kind = .up, .prompt = false },
    .{ .name = "cuda bf16 experts down tf_bf16_experts_down_f32", .kind = .down_f32, .prompt = false },
    .{ .name = "cuda bf16 experts down tf_bf16_experts_down_bf16", .kind = .down_bf16, .prompt = false },
    .{ .name = "cuda bf16 experts prompt up tf_bf16_experts_prompt_up (relu^2)", .kind = .up, .prompt = true },
    .{ .name = "cuda bf16 experts prompt down tf_bf16_experts_prompt_down_f32", .kind = .down_f32, .prompt = true },
    .{ .name = "cuda bf16 experts prompt down tf_bf16_experts_prompt_down_bf16", .kind = .down_bf16, .prompt = true },
    .{ .name = "control: bf16 experts fp32 down launched one 32-input group short", .kind = .down_f32, .prompt = false, .fault = .short },
    .{ .name = "control: bf16 experts fp32 down summing in bf16", .kind = .down_f32, .prompt = false, .fault = .bf16_sums },
};

const experts = 4;
const slots = 8; // pairs a token row: Transcribe's 6 routed picks and its shared expert's two halves

/// An mma k16 step a link of each row's chain, plus one step's error over its own products (attention_range's 36).
fn depth(s: c.Shape) usize {
    return s.k / 16 + core.attention_range.mma_step;
}

/// Transcribe's routed experts, up 2,688 to 1,856 and back; decode windows of 1 to 32 rows, prompt items of 2 and 9.
const ups = [2][]const c.Shape{ &.{ .{ .rows = 1, .n = 1856, .k = 2688 }, .{ .rows = 32, .n = 1856, .k = 2688 } }, &.{ .{ .rows = 40, .n = 1856, .k = 2688 }, .{ .rows = 72, .n = 1856, .k = 2688 } } };
const downs = [2][]const c.Shape{ &.{ .{ .rows = 1, .n = 2688, .k = 1856 }, .{ .rows = 32, .n = 2688, .k = 1856 } }, &.{ .{ .rows = 40, .n = 2688, .k = 1856 }, .{ .rows = 72, .n = 2688, .k = 1856 } } };

pub fn spec(e: Entry) c.Spec {
    const served = (if (e.kind == .up) ups else downs)[@intFromBool(e.prompt)];
    const tiles = [3]c.Tile{ .{ .size = (if (e.prompt) ex_.promptTile(.bf16) else 16) / slots }, .{ .size = 64, .step = 32, .min = 32 }, .{ .size = 64, .step = 32, .min = 32 } };
    return switch (e.kind) {
        // bf16(acc)'s rounding enters the square twice
        .up => .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 1, .mids = 2, .out = .bf16, .tiles = tiles, .served = served, .sums = .{ .f16 = 30, .bf16 = 30, .why = "a sum that collapses below 0 gives relu^2 0, which misses by want itself, and the bf16 roundings already allow 3 u of it: at most 1 / (3 u) = 85x is provable" } },
        .down_f32 => .{ .name = e.name, .accum = .f32, .depth = depth, .out = .f32, .tiles = tiles, .served = served },
        .down_bf16 => .{ .name = e.name, .accum = .f32, .depth = depth, .out = .bf16, .tiles = tiles, .served = served },
    };
}

/// shape.rows token rows, `slots` pairs each over `experts` experts; up reads a token's row, down a pair's.
pub const Experts = struct {
    rig: rig_.Rig,
    entry: Entry,
    kx: *const Bf16,
    router: grouped.Router,
    ctl: ?cuda.Function = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{}, // [experts][n][k], as nn.Linear stores them
    picks: []i32 = &.{},
    xd: u64 = 0,
    wd: u64 = 0,
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
        a.free(self.picks);
        self.x = &.{};
        self.w = &.{};
        self.picks = &.{};
    }

    pub fn kernel(self: *Experts) g.Kernel {
        const control: ?g.Gate = switch (self.entry.fault) {
            .none => null,
            .short => .shapes,
            .bf16_sums => .sums,
        };
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Experts);

    fn up(self: *const Experts) bool {
        return self.entry.kind == .up;
    }

    fn pairs(self: *const Experts) usize {
        return self.shape.rows * slots;
    }

    fn xRows(self: *const Experts) usize {
        return if (self.up()) self.shape.rows else self.pairs();
    }

    fn tile(self: *const Experts) usize {
        return if (self.entry.prompt) ex_.promptTile(.bf16) else 16;
    }

    fn items(self: *const Experts) usize {
        return grouped.maxItems(self.pairs(), experts, self.tile());
    }

    /// Random bf16 rows and weights; non-finite: NaN, +Inf and -Inf in the first rows; stress: mirrored integers.
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
        self.x = try a.alloc(u16, self.xRows() * k);
        self.w = try a.alloc(u16, experts * n * k);
        self.picks = try a.alloc(i32, self.pairs());
        for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.w) |*v| v.* = c.toBf16((rng.float(f32) - 0.5) * 0.25);
        for (self.picks, 0..) |*p, i| p.* = @intCast(if (i < experts) i else rng.uintLessThan(usize, experts));
        switch (fill) {
            .random => {},
            .nonfinite => {
                // a row holding NaN, one holding +Inf, one holding both infinities
                const planted = [_]struct { usize, usize, f32 }{ .{ 0, k / 3, std.math.nan(f32) }, .{ 1, k / 2, std.math.inf(f32) }, .{ 2, 1, std.math.inf(f32) }, .{ 2, k - 1, -std.math.inf(f32) } };
                for (planted) |p| if (p[0] < self.xRows()) {
                    self.x[p[0] * k + p[1]] = c.toBf16(p[2]);
                };
            },
            .stress => self.stress(rng),
        }
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        const src = try self.rig.scratch(self.w.len * 2);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = src, .len = self.w.len * 2 }).upload(0, std.mem.sliceAsBytes(self.w));
        const blocks = Bf16.bytes(experts, n, k);
        const zeros = try a.alloc(u8, @max(blocks, self.pairs() * 4, self.items() * 12));
        defer a.free(zeros);
        @memset(zeros, 0);
        self.wd = try self.rig.input(zeros[0..blocks]);
        try self.kx.pack(self.rig.s, src, .{ .stride = n * k, .ld = k, .in_out = false }, self.wd, experts, n, k);
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        const hist = (self.pairs() + 1023) / 1024;
        self.plan = .{ .members = try self.rig.input(zeros[0 .. self.pairs() * 4]), .items = try self.rig.input(zeros[0 .. self.items() * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = try self.rig.scratch(self.pairs() * 4), .hist = try self.rig.scratch(hist * experts * 4) };
        try self.router.route(self.rig.s, ids, self.pairs(), experts, self.tile(), self.plan);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(self.pairs() * n * c.size(spec(self.entry).out));
        return true;
    }

    /// Gate 4: mirrored integer weights and inputs, every partial sum an exact integer under 2^20, the row's sum small.
    fn stress(self: *Experts, rng: std.Random) void {
        const k = self.shape.k;
        const h = k / 2;
        for (0..experts * self.shape.n) |j| for (0..h) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 2));
            self.w[j * k + i] = c.toBf16(v);
            self.w[j * k + h + i] = c.toBf16(v);
        };
        for (0..self.xRows()) |r| for (0..h) |i| {
            const v: f32 = @floatFromInt(4 * (1 + 2 * rng.uintLessThan(u32, 64)));
            self.x[r * k + i] = c.toBf16(v);
            self.x[r * k + h + i] = c.toBf16(if (i < 8) 4 - v else -v);
        };
    }

    pub fn run(self: *Experts) !void {
        const n = self.shape.n;
        const k = self.shape.k;
        const in: ex_.Rows = .{ .x = self.xd, .stride = k, .slots = if (self.up()) slots else 0 };
        switch (self.entry.fault) {
            .none => {
                const epi: ex_.Epi = switch (self.entry.kind) {
                    .up => .up,
                    .down_f32 => .down_f32,
                    .down_bf16 => .down_bf16,
                };
                const l: ex_.Layer = .{ .format = .bf16, .up = self.wd, .down = self.wd, .width = if (self.up()) n else k, .dims = if (self.up()) k else n, .experts = experts, .relu2 = true };
                try self.kx.run(self.rig.s, self.entry.prompt, epi, in, l, self.plan, self.out, self.items(), -1);
            },
            .short => try self.launch(self.kx.fns[0][1], k / 32 - 1, .{ .x = @intCast((self.items() * n / Bf16.cols + Bf16.warps - 1) / Bf16.warps) }, Bf16.warps * 32),
            .bf16_sums => try self.launch(self.ctl.?, k / 32, .{ .x = @intCast((n + 127) / 128), .y = @intCast(self.items()) }, 128),
        }
        try self.rig.s.synchronize();
    }

    /// A down kernel's launch with its own input groups: experts_bf16.cu's arguments, less `slots` for the control.
    fn launch(self: *Experts, f: cuda.Function, kg: usize, grid: struct { x: u32, y: u32 = 1 }, threads: u32) !void {
        var a: cuda.Args = .{};
        a.add(self.xd);
        a.add(@as(i32, @intCast(self.shape.k)));
        if (self.entry.fault == .short) a.add(@as(i32, 0));
        a.add(self.wd);
        a.add(@as(i32, @intCast(kg)));
        a.add(@as(i32, @intCast(self.shape.n / Bf16.cols)));
        for ([_]u64{ self.plan.items, self.plan.counts, self.plan.members, self.out }) |v| a.add(v);
        a.add(@as(i32, @intCast(self.shape.n)));
        if (self.entry.fault == .short) a.add(@as(i32, -1));
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid.x, .y = grid.y, .z = 1 }, .block = .{ .x = threads } }, self.rig.s, &a);
    }

    pub fn count(self: *Experts) usize {
        return self.pairs() * self.shape.n;
    }

    /// Pair p's products with column j of its expert, in input order.
    fn terms(self: *const Experts, p: usize, j: usize, t: []f64) void {
        const k = self.shape.k;
        const xr = (if (self.up()) p / slots else p) * k;
        const wr = (@as(usize, @intCast(self.picks[p])) * self.shape.n + j) * k;
        for (t, 0..) |*o, i| o.* = c.fromBf16(self.x[xr + i]) * c.fromBf16(self.w[wr + i]);
    }

    /// relu^2 for up, as float64 keeps a NaN; down's sum as it is.
    fn epilogue(self: *const Experts, sum: f64) f64 {
        if (!self.up()) return sum;
        const u = if (std.math.isNan(sum)) sum else @max(sum, 0);
        return u * u;
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
            want[o] = self.epilogue(sum);
            // a sum's error reaches relu^2 through its derivative 2 u
            mag[o] = if (self.up()) 2 * @sqrt(want[o]) * abs else abs;
        };
    }

    pub fn simulate(self: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.pairs()) |p| for (0..self.shape.n) |j| {
            self.terms(p, j, t);
            out[p * self.shape.n + j] = self.epilogue(contract.rows.sumIn(acc, order, t));
        };
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    kx: Bf16,
    mod: cuda.Module,
    ctl: cuda.Module,
    items: [entries.len]Experts = undefined,

    const controls = struct {
        pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
    };

    /// The set, its kernels appended to `list` (each control after the kernels it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        var kx = try Bf16.load(gpu.d, sms);
        errdefer kx.unload();
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
        errdefer mod.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .kx = kx, .mod = mod, .ctl = ctl };
        const router = try grouped.Router.resolve(set.mod);
        const sums = try set.ctl.function("tf_ctl_bf16_experts_bf16_sums");
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .kx = &set.kx, .router = router, .ctl = if (e.fault == .bf16_sums) sums else null };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.ctl.unload();
        set.mod.unload();
        set.kx.unload();
        set.gpa.destroy(set);
    }
};

/// The bf16 experts' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x62663136, &t);
    std.debug.print("SUMMARY bf16 experts: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "experts-bf16-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}
