//! fn_int4.cu at the checkpoint's shapes: its registrations, five controls, the epilogue's model, a skipped slot.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ir = @import("int4_ref.zig");
const scalar = @import("int4_controls.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const grouped = cuda.grouped;
const int4 = cuda.int4;

/// The form a registration runs.
pub const Kind = enum { gu, down_f32, down_bf16, head, head_f32, prompt_down, prompt_down_bf16, prompt_gu };

/// A planted fault: gate 4's control kernel, or a head packed with its nibbles unshuffled (gate 3 must catch it).
pub const Fault = enum { none, bf16_group_sums, unshuffled_nibbles };

/// A registration: its form, a planted `fault`, or the controls image's scalar gate/up at `gu` (`epilogue`: its model).
pub const Entry = struct { name: []const u8, kind: Kind, tile: usize = int4.plan_tile, fault: Fault = .none, gu: ?scalar.Fault = null };

/// The checkpoint's shapes: hidden 2,560, experts of width 640, a head of 248,320 rows (its config.json).
pub const dims = 2560;
pub const width = 640;
const vocab = 248320;
/// Experts the tables hold: every expert the plan's picks name. The plan itself routes over the checkpoint's 512.
const used = 8;
/// Pairs a token row: five routed picks and the shared expert, whose slot is skipped.
const slots = 6;
pub const shared = used - 1;
/// The inputs whose mirror carries the nudge: small, so the nudge 4 is exact in bf16 at their size.
const nudged = 8;
/// Gate 4's bf16 proof of a bf16 SwiGLU output: a mirrored row's bf16 sum is 0 in the halves order, missing by want.
const gate_up_why = "a bf16 sum of a mirrored row is 0 in the halves order, so it misses by want itself, which the epilogue's three bf16 roundings and the store already allow 4.0393 u of: at most 1 / (4.0393 u) = 63.4x is provable";

const up_rows = [4]c.Shape{ .{ .rows = 1, .n = width, .k = dims }, .{ .rows = 3, .n = width, .k = dims }, .{ .rows = 16, .n = width, .k = dims }, .{ .rows = 64, .n = width, .k = dims } };
const prompt_rows = [2]c.Shape{ .{ .rows = 64, .n = width, .k = dims }, .{ .rows = 200, .n = width, .k = dims } };
const down_rows = [4]c.Shape{ .{ .rows = 1, .n = dims, .k = width }, .{ .rows = 3, .n = dims, .k = width }, .{ .rows = 16, .n = dims, .k = width }, .{ .rows = 64, .n = dims, .k = width } };
const prompt_down_rows = [2]c.Shape{ .{ .rows = 64, .n = dims, .k = width }, .{ .rows = 200, .n = dims, .k = width } };
const head_rows = [4]c.Shape{ .{ .rows = 1, .n = 2560, .k = dims }, .{ .rows = 16, .n = 2560, .k = dims }, .{ .rows = 32, .n = 2560, .k = dims }, .{ .rows = 1, .n = vocab, .k = dims } };

/// An mma k16 step and one fma a group, plus one step's error over its own products.
fn depth(s: c.Shape) usize {
    return s.k / 16 + s.k / 128 + core.attention_range.mma_step;
}

fn tiles(e: Entry) [3]c.Tile {
    return switch (e.kind) {
        .head, .head_f32 => .{ .{}, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 128, .step = 128, .min = 128 } },
        .prompt_down, .prompt_down_bf16, .prompt_gu => .{ .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 128, .step = 128, .min = 128 } },
        else => .{ .{ .size = 16, .step = 1, .min = 1 }, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 128, .step = 128, .min = 128 } },
    };
}

fn served(e: Entry) []const c.Shape {
    return switch (e.kind) {
        .head, .head_f32 => &head_rows,
        .down_f32, .down_bf16 => if (e.tile == int4.prompt_tile) &prompt_rows else &down_rows,
        .prompt_down, .prompt_down_bf16 => &prompt_down_rows,
        else => if (e.tile == int4.prompt_tile) &prompt_rows else &up_rows,
    };
}

/// The prompt forms must give the plain form's bytes; the others are checked against float64 at their own bound.
fn replaces(e: Entry) ?[]const u8 {
    if (e.fault != .none) return null;
    return switch (e.kind) {
        .prompt_down => "the routed down of 16-pair items",
        .prompt_down_bf16 => "the routed down of 16-pair items to bf16",
        .prompt_gu => "the routed gate/up of 16-pair items",
        else => null,
    };
}

pub fn spec(e: Entry) c.Spec {
    const base: c.Spec = .{ .name = e.name, .accum = .f32, .depth = depth, .out = .f32, .tiles = tiles(e), .served = served(e), .replaces = replaces(e) };
    var s = base;
    switch (e.kind) {
        // exp's 2 ulps, 1 + e, the quotient and the product; gate, up and silu rounded to bf16 before the product
        .gu, .prompt_gu => {
            s.out = .bf16;
            s.ops = 5;
            s.mids = 3;
            s.sums = .{ .bf16 = 60, .why = gate_up_why };
        },
        .down_bf16, .head, .prompt_down_bf16 => s.out = .bf16,
        .down_f32, .prompt_down, .head_f32 => {},
    }
    return s;
}

pub const entries = [_]Entry{
    .{ .name = "cuda int4 routed gate/up SwiGLU (int4_kernel, 16-pair items)", .kind = .gu },
    .{ .name = "cuda int4 routed gate/up SwiGLU (int4_kernel, 64-pair items)", .kind = .gu, .tile = int4.prompt_tile },
    .{ .name = "cuda int4 routed down to fp32 (int4_kernel)", .kind = .down_f32 },
    .{ .name = "cuda int4 routed down to bf16 (int4_kernel)", .kind = .down_bf16 },
    .{ .name = "cuda int4 lm head to bf16 (int4_kernel, dense items)", .kind = .head },
    .{ .name = "cuda int4 lm head to fp32 (int4_kernel, dense items)", .kind = .head_f32 },
    .{ .name = "cuda int4 routed down to fp32 (int4_prompt_kernel, 64-pair items)", .kind = .prompt_down },
    .{ .name = "cuda int4 routed down to bf16 (int4_prompt_kernel, 64-pair items)", .kind = .prompt_down_bf16 },
    .{ .name = "cuda int4 routed gate/up SwiGLU (int4_prompt_kernel, 64-pair items)", .kind = .prompt_gu },
    .{ .name = "cuda int4 routed gate/up SwiGLU (the epilogue's scalar model, int4_controls)", .kind = .gu, .gu = .epilogue },
    .{ .name = "control: int4 routed down with each group's sum rounded to bf16", .kind = .down_f32, .fault = .bf16_group_sums },
    .{ .name = "control: int4 lm head with each word's nibbles unshuffled", .kind = .head, .fault = .unshuffled_nibbles },
    .{ .name = "control: int4 routed gate/up with silu of the up sum times the gate sum (gate and up swapped)", .kind = .gu, .gu = .swapped },
    .{ .name = "control: int4 routed gate/up with gate, up and silu cut toward zero to bf16", .kind = .gu, .gu = .cut },
    .{ .name = "control: int4 routed gate/up with each group's sums rounded to bf16", .kind = .gu, .gu = .bf16_sums },
};

/// swiglu as float64 keeps it: silu of the gate sum times the up sum.
fn swiglu(gv: f64, uv: f64) f64 {
    return gv / (1 + @exp(-gv)) * uv;
}

/// One call's buffers: the checkpoint matrices the plan's picks name, the rows in, the plan and the output.
pub const Fixture = struct {
    rig: rig_.Rig,
    entry: Entry,
    kx: *const int4.Kernels,
    router: grouped.Router,
    ctl: ?cuda.Function = null,
    gu_ctl: ?cuda.Function = null,
    shape: c.Shape = undefined,
    pairs: usize = 0,
    mats: usize = 1,
    qw: []u8 = &.{},
    sc: []u8 = &.{},
    src: []int4.Source = &.{},
    x: []u16 = &.{},
    ids: []i32 = &.{},
    ex: int4.Experts = .{},
    mat: int4.Mat = .{},
    plan: grouped.Plan = undefined,
    plan16: grouped.Plan = undefined,
    /// the expert id whose slot is left untouched (the shared expert), or -1
    skip: c_int = -1,
    xd: u64 = 0,
    tab: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    /// what a gate asked the inputs to hold: gate 4's stress rows make the sums exact in fp32
    fill: g.Fill = .random,
    /// every output's derived allowance, from its own terms (reference), which `bounds` hands the harness
    allow: []f64 = &.{},
    msg: [192]u8 = undefined,

    pub fn deinit(self: *Fixture) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Fixture) void {
        const a = self.rig.gpu.gpa;
        a.free(self.qw);
        a.free(self.sc);
        a.free(self.src);
        a.free(self.x);
        a.free(self.ids);
        a.free(self.allow);
        self.qw = &.{};
        self.sc = &.{};
        self.src = &.{};
        self.x = &.{};
        self.ids = &.{};
        self.allow = &.{};
    }

    pub fn kernel(self: *Fixture) g.Kernel {
        const control: ?g.Gate = if (scalar.gate(self.entry.gu)) |at| at else switch (self.entry.fault) {
            .none => null,
            .unshuffled_nibbles => .shapes,
            .bf16_group_sums => .sums,
        };
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Fixture);

    fn up(self: *const Fixture) bool {
        return self.entry.kind == .gu or self.entry.kind == .prompt_gu;
    }

    fn prompt(self: *const Fixture) bool {
        return self.entry.kind == .prompt_down or self.entry.kind == .prompt_down_bf16 or self.entry.kind == .prompt_gu;
    }

    /// The head's dense rows: one matrix, a row a row, either store width.
    fn denseForm(self: *const Fixture) bool {
        return self.entry.kind == .head or self.entry.kind == .head_f32;
    }

    fn elem(self: *const Fixture) usize {
        return c.size(spec(self.entry).out);
    }

    /// One checkpoint matrix of the shapes a call reads: rows in, the matrix out, its codes and scales.
    fn matrices(self: *Fixture, a: std.mem.Allocator, n: usize, k: usize, seed: u64) !void {
        self.mats = if (self.up()) 2 else 1;
        const per_qw = k / 8 * n * 4;
        const per_sc = k / 128 * n * 2;
        self.qw = try a.alloc(u8, used * self.mats * per_qw);
        self.sc = try a.alloc(u8, used * self.mats * per_sc);
        self.src = try a.alloc(int4.Source, used * self.mats);
        var prng = std.Random.DefaultPrng.init(seed | 0x1_0000);
        const rng = prng.random();
        for (0..used * self.mats) |m| {
            const q = self.qw[m * per_qw ..][0..per_qw];
            const s = self.sc[m * per_sc ..][0..per_sc];
            rng.bytes(q);
            for (0..s.len / 2) |i| {
                // the format's own scales: 0.25 to 0.5, well inside fp16 and away from zero
                const bits: u16 = @intCast(0x3400 + rng.uintLessThan(u16, 0x0400));
                std.mem.writeInt(u16, s[2 * i ..][0..2], bits, .little);
            }
            self.src[m] = .{ .qweight = q, .scales = s, .n_full = n, .k_full = k };
        }
    }

    /// Gate 4's inputs: integers whose terms cancel in pairs, so a low-precision sum loses the whole result.
    fn stress(self: *Fixture, rng: std.Random) void {
        const k = self.shape.k;
        const h = k / 2;
        for (0..self.x_rows()) |r| for (0..h) |i| {
            const small = i < nudged;
            const v: f32 = @floatFromInt(if (small) 4 * (1 + rng.uintLessThan(u32, 4)) else 8 * (1 + 2 * rng.uintLessThan(u32, 128)));
            self.x[r * k + i] = c.toBf16(v);
            self.x[r * k + h + i] = c.toBf16(if (small) 4 - v else -v);
        };
        // the same code at both halves of an input pair, so a column's sums cancel to a small integer
        for (self.src) |s| {
            const q = @constCast(s.qweight);
            for (0..self.shape.n) |col| for (0..k / 2 / 8) |w| {
                const a = (w * self.shape.n + col) * 4;
                const b = ((w + h / 8) * self.shape.n + col) * 4;
                var word: u32 = 0;
                for (0..8) |j| word |= @as(u32, 9 + rng.uintLessThan(u32, 3)) << @intCast(4 * j);
                std.mem.writeInt(u32, q[a..][0..4], word, .little);
                std.mem.writeInt(u32, q[b..][0..4], word, .little);
            };
            const sc = @constCast(s.scales);
            for (0..sc.len / 2) |i| std.mem.writeInt(u16, sc[2 * i ..][0..2], 0x3C00, .little);
        }
    }

    fn x_rows(self: *const Fixture) usize {
        return if (self.denseForm()) self.shape.rows else self.pairs;
    }

    /// Random or planted rows, the checkpoint matrices of the shape, the plan at the entry's tile and the buffers.
    pub fn prepare(self: *Fixture, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        if (shape.n % 8 != 0 or shape.k % int4.checkpoint_group != 0) return false;
        if (self.prompt() and shape.n % 128 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.pairs = shape.rows * slots;
        try self.matrices(a, shape.n, shape.k, seed);
        // the rows in: gate/up read a token row a pair, down reads its own row a pair, the head a row
        const rows = if (self.denseForm()) shape.rows else self.pairs;
        self.x = try a.alloc(u16, rows * shape.k);
        for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        self.ids = try a.alloc(i32, @max(1, self.pairs));
        for (0..shape.rows) |r| for (0..slots) |s| {
            self.ids[r * slots + s] = @intCast(if (s == slots - 1) shared else (r * 3 + s) % (used - 1));
        };
        switch (fill) {
            .random => {},
            .nonfinite => {
                const planted = [_]struct { usize, usize, f32 }{ .{ 0, shape.k / 3, std.math.nan(f32) }, .{ 1, shape.k / 2, std.math.inf(f32) }, .{ 2, 1, std.math.inf(f32) }, .{ 2, shape.k - 1, -std.math.inf(f32) } };
                for (planted) |p| if (p[0] < rows) {
                    self.x[p[0] * shape.k + p[1]] = c.toBf16(p[2]);
                };
            },
            .stress => self.stress(rng),
        }
        // the packed matrices the kernel reads: every picked expert's, at one or two matrices an expert
        const words = try a.alloc(u32, used * self.mats * int4.wordsOf(shape.n, shape.k));
        defer a.free(words);
        const scales = try a.alloc(u16, used * self.mats * int4.scalesOf(shape.n, shape.k));
        defer a.free(scales);
        const sl: int4.Slice = .{ .n0 = 0, .n = shape.n, .k0 = 0, .k = shape.k, .gs = int4.checkpoint_group };
        for (0..used * self.mats) |m| {
            const w = words[m * int4.wordsOf(shape.n, shape.k) ..][0..int4.wordsOf(shape.n, shape.k)];
            const s = scales[m * int4.scalesOf(shape.n, shape.k) ..][0..int4.scalesOf(shape.n, shape.k)];
            if (self.entry.fault == .unshuffled_nibbles) {
                try packRaw(self.src[m], sl, w);
            } else try int4.packWords(self.src[m], sl, w);
            try int4.packScales(self.src[m], sl, s);
        }
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.tab = try self.rig.input(std.mem.sliceAsBytes(words));
        const tab_s = try self.rig.input(std.mem.sliceAsBytes(scales));
        self.mat = .{ .w = self.tab, .s = tab_s, .n = @intCast(shape.n), .k = @intCast(shape.k) };
        self.ex = .{
            .up = self.tab,
            .up_s = tab_s,
            .down = self.tab,
            .down_s = tab_s,
            .width = @intCast(if (self.up()) shape.n else shape.k),
            .dims = @intCast(if (self.up()) shape.k else shape.n),
        };
        const items64 = grouped.maxItems(self.pairs, 512, int4.prompt_tile);
        const items16 = grouped.maxItems(self.pairs, 512, int4.plan_tile);
        const zeros = try a.alloc(u8, @max(@max(self.pairs * 4, items64 * 12), items16 * 12 + 8));
        defer a.free(zeros);
        @memset(zeros, 0);
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        self.plan16 = .{ .members = try self.rig.input(zeros[0 .. self.pairs * 4]), .items = try self.rig.input(zeros[0 .. items16 * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = try self.rig.scratch(@max(4, self.pairs * 4)), .hist = try self.rig.scratch(((self.pairs + 1023) / 1024) * 512 * 4) };
        self.plan = self.plan16;
        try self.router.route(self.rig.s, ids, self.pairs, 512, int4.plan_tile, self.plan16);
        if (items64 != items16) {
            self.plan = .{ .members = self.plan16.members, .items = try self.rig.input(zeros[0 .. items64 * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = self.plan16.rank, .hist = self.plan16.hist };
            try self.router.route(self.rig.s, ids, self.pairs, 512, int4.prompt_tile, self.plan);
        }
        try self.rig.s.synchronize();
        self.allow = try a.alloc(f64, self.count());
        self.out = try self.rig.output(self.count() * self.elem());
        self.base = try self.rig.scratch(self.count() * self.elem());
        return true;
    }

    fn launch(self: *Fixture) !void {
        const sh = self.shape;
        switch (self.entry.kind) {
            .gu => try int4.gateUp(self.kx, self.rig.s, self.xd, sh.k, self.ex, self.plan, slots, 512, self.out, sh.rows, self.skip, self.entry.tile),
            .down_f32, .down_bf16 => try int4.down(self.kx, self.rig.s, self.xd, sh.k, self.ex, self.plan, slots, 512, self.out, self.entry.kind == .down_f32, sh.rows, self.skip, self.entry.tile),
            .prompt_down => try int4.downPrompt(self.kx, self.rig.s, self.xd, sh.k, self.ex, self.plan, slots, 512, self.out, true, sh.rows, self.skip),
            .prompt_down_bf16 => try int4.downPrompt(self.kx, self.rig.s, self.xd, sh.k, self.ex, self.plan, slots, 512, self.out, false, sh.rows, self.skip),
            .prompt_gu => try int4.gateUpPrompt(self.kx, self.rig.s, self.xd, sh.k, self.ex, self.plan, slots, 512, self.out, sh.rows, self.skip),
            .head => try int4.dense(self.kx, self.rig.s, self.xd, sh.k, self.mat, self.out, false, sh.rows),
            .head_f32 => try int4.dense(self.kx, self.rig.s, self.xd, sh.k, self.mat, self.out, true, sh.rows),
        }
    }

    pub fn run(self: *Fixture) !void {
        switch (self.entry.fault) {
            .bf16_group_sums => try self.controlSums(),
            .none, .unshuffled_nibbles => if (self.entry.gu) |f| try scalar.launch(self.rig.s, self.gu_ctl.?, self.xd, self.shape, slots, self.ex, self.plan, self.out, self.pairs, f) else try self.launch(),
        }
        try self.rig.s.synchronize();
    }

    /// The control's own launch: the down form with each group's sum rounded to bf16 before its scale.
    fn controlSums(self: *Fixture) !void {
        const sh = self.shape;
        var a: cuda.Args = .{};
        a.add(self.xd);
        a.add(@as(i32, @intCast(sh.k)));
        a.add(@as(i32, 0));
        a.add(self.tab);
        a.add(self.ex.down_s);
        a.add(@as(i32, @intCast(sh.k)));
        a.add(@as(i32, @intCast(sh.n)));
        a.add(self.plan.items);
        a.add(self.plan.counts);
        a.add(self.plan.members);
        a.add(@as(i32, @intCast(self.pairs)));
        a.add(self.out);
        a.add(@as(i32, @intCast(sh.n)));
        a.add(@as(f32, 0));
        a.add(@as(c_int, -1));
        try cuda.launch.launch(self.ctl.?, .{ .grid = .{ .x = @intCast((sh.n + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, self.rig.s, &a);
    }

    pub fn count(self: *Fixture) usize {
        return if (self.denseForm()) self.shape.rows * self.shape.n else self.pairs * self.shape.n;
    }

    /// The rows a pair's products read: its token row for gate/up, its own row for down.
    fn xrow(self: *const Fixture, p: usize) usize {
        return if (self.up()) p / slots else p;
    }

    /// Pair p's expert, and the matrix of its gate and up (or its down).
    fn srcOf(self: *const Fixture, p: usize, m: usize) int4.Source {
        const e: usize = @intCast(self.ids[p] & (used - 1));
        return self.src[e * self.mats + m];
    }

    /// Pair p's terms against one column, in input order.
    fn terms(self: *const Fixture, p: usize, j: usize, m: usize, t: []f64) void {
        const k = self.shape.k;
        const src = self.srcOf(p, m);
        const xr = self.xrow(p) * k;
        for (t, 0..) |*o, i| o.* = int4.weight64(src, j, i) * c.fromBf16(self.x[xr + i]);
    }

    /// The head's rows are its own, its one matrix the first.
    fn headTerms(self: *const Fixture, r: usize, j: usize, t: []f64) void {
        const k = self.shape.k;
        for (t, 0..) |*o, i| o.* = int4.weight64(self.src[0], j, i) * c.fromBf16(self.x[r * k + i]);
    }

    pub fn reference(self: *Fixture, want: []f64, mag: []f64) !void {
        const gpa = self.rig.gpu.gpa;
        const t = try gpa.alloc(f64, self.shape.k);
        defer gpa.free(t);
        const u = try gpa.alloc(f64, self.shape.k);
        defer gpa.free(u);
        const sp = spec(self.entry);
        const d = depth(self.shape);
        // gate 4's stress inputs are integers whose partial sums are exact in fp32 in any order
        const ex = self.fill == .stress;
        if (self.denseForm()) {
            for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
                self.headTerms(r, j, t);
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (t) |v| {
                    sum += v;
                    abs += @abs(v);
                }
                const o = r * self.shape.n + j;
                want[o] = sum;
                mag[o] = abs;
                self.allow[o] = ir.rows(d, abs, sum, sp.out, ex);
            };
            return;
        }
        for (0..self.pairs) |p| for (0..self.shape.n) |j| {
            const o = p * self.shape.n + j;
            self.terms(p, j, 0, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            if (!self.up()) {
                want[o] = sum;
                mag[o] = abs;
                self.allow[o] = ir.rows(d, abs, sum, sp.out, ex);
                continue;
            }
            self.terms(p, j, 1, u);
            var usum: f64 = 0;
            var uabs: f64 = 0;
            for (u) |v| {
                usum += v;
                uabs += @abs(v);
            }
            want[o] = swiglu(sum, usum);
            mag[o] = abs + uabs;
            self.allow[o] = ir.gateUp(d, sp.ops, sum, usum, abs, uabs, want[o], ex);
        };
    }

    /// Each output's derived allowance (from reference): the generic formula counts a SwiGLU's roundings at |want|.
    pub fn bounds(self: *Fixture, want: []const f64, _: []const f64, allow: []f64) !void {
        if (want.len != self.allow.len or allow.len != self.allow.len) return error.BadLength;
        @memcpy(allow, self.allow);
    }

    pub fn simulate(self: *Fixture, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        const u = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(u);
        if (self.denseForm()) {
            for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
                self.headTerms(r, j, t);
                out[r * self.shape.n + j] = contract.rows.sumIn(acc, order, t);
            };
            return;
        }
        for (0..self.pairs) |p| for (0..self.shape.n) |j| {
            self.terms(p, j, 0, t);
            const sum = contract.rows.sumIn(acc, order, t);
            if (!self.up()) {
                out[p * self.shape.n + j] = sum;
                continue;
            }
            self.terms(p, j, 1, u);
            out[p * self.shape.n + j] = swiglu(sum, contract.rows.sumIn(acc, order, u));
        };
    }

    /// The plain form's bytes on the same inputs: the prompt kernels must equal them.
    pub fn replaced(self: *Fixture) ![]const u8 {
        const sh = self.shape;
        switch (self.entry.kind) {
            .prompt_down => try int4.down(self.kx, self.rig.s, self.xd, sh.k, self.ex, self.plan16, slots, 512, self.base, true, sh.rows, -1, int4.plan_tile),
            .prompt_down_bf16 => try int4.down(self.kx, self.rig.s, self.xd, sh.k, self.ex, self.plan16, slots, 512, self.base, false, sh.rows, -1, int4.plan_tile),
            .prompt_gu => try int4.gateUp(self.kx, self.rig.s, self.xd, sh.k, self.ex, self.plan16, slots, 512, self.base, sh.rows, -1, int4.plan_tile),
            else => return error.NoBaseline,
        }
        return self.rig.bytes(self.base, self.count() * self.elem());
    }
};

/// The kernel's word order skipped: the GPTQ word as stored, which reads nibble i as input 2i (gate 3 must catch it).
fn packRaw(src: int4.Source, sl: int4.Slice, out: []u32) !void {
    try int4.checkSlice(src, sl);
    if (out.len != int4.wordsOf(sl.n, sl.k)) return error.BadLength;
    var at: usize = 0;
    for (0..sl.n / 8) |nt| for (0..sl.k / sl.gs) |grp| for (0..32) |lane| {
        const col = sl.n0 + nt * 8 + lane / 4;
        const t = lane % 4;
        for (0..sl.gs / 32) |b| {
            const kp = (sl.k0 + grp * sl.gs + b * 32) / 8 + t;
            out[at] = std.mem.readInt(u32, src.qweight[(kp * src.n_full + col) * 4 ..][0..4], .little);
            at += 1;
        }
    };
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none or (e.gu != null and e.gu.? != .epilogue), std.mem.startsWith(u8, e.name, "control: "));
}
