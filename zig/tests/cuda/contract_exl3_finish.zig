//! tf_exl3_plan_finish under the contract: modes 0 and 1 and the downs bit-equal to main's, SwiGLU-OAI against float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const iv = @import("contract_interval.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = iv.Iv;
const Sums = iv.Sums;

const experts = 4;
const had_scale: f64 = @as(f32, 0.08838834764831845); // exl3_decode.cuh's HAD_SCALE, as fp32 holds it
const oai_alpha: f32 = 1.702; // MiniMax-M3.1-Flash's
const oai_limit: f32 = 7.0;

/// What one entry runs: plan_finish's epilogue (0 fp32 down, 2 gate-up, 3 bf16 down), its activation and clamp.
pub const Mode = struct {
    epi: i32,
    act: i32 = 0,
    limit: f32 = 0,
    name: []const u8,
};

pub const modes = [_]Mode{
    .{ .epi = 2, .act = 0, .limit = oai_limit, .name = "exl3_experts tf_exl3_plan_finish gate-up, bf16 SwiGLU clamped (act_mode 0)" },
    .{ .epi = 2, .act = 1, .name = "exl3_experts tf_exl3_plan_finish gate-up, fp32 SwiGLU (act_mode 1)" },
    .{ .epi = 0, .name = "exl3_experts tf_exl3_plan_finish fp32 down" },
    .{ .epi = 3, .name = "exl3_experts tf_exl3_plan_finish bf16 down" },
    .{ .epi = 2, .act = 2, .limit = oai_limit, .name = "exl3_experts tf_exl3_plan_finish gate-up, SwiGLU-OAI (act_mode 2)" },
};

/// The served kernel, or a control: the SwiGLU-OAI without the up's + 1, or with alpha 1 (exl3_plan.cuh's faults).
pub const Fault = enum { none, no_plus_one, alpha_one };

const clamp_nan = "torch.clamp's bounds: a NaN gate or up stays NaN, +Inf reads as the limit and a -Inf up as -limit";

/// Pairs x outputs x splits: MiniMax-M3.1-Flash at TP=2 (1,536 / 6,144 a rank), the Qwen3.5 EXL3 pack (1,024 / 3,072).
fn spec(m: Mode, fault: Fault) c.Spec {
    const up = &[_]c.Shape{ .{ .rows = 16, .n = 1536, .k = 4 }, .{ .rows = 64, .n = 1536, .k = 4 }, .{ .rows = 1, .n = 1536, .k = 4 }, .{ .rows = 64, .n = 1024, .k = 4 } };
    const down = &[_]c.Shape{ .{ .rows = 16, .n = 6144, .k = 1 }, .{ .rows = 64, .n = 6144, .k = 1 }, .{ .rows = 64, .n = 3072, .k = 1 } };
    var s: c.Spec = .{ .name = m.name, .accum = .f32, .depth = depth, .out = if (m.epi == 0) .f32 else .bf16, .tiles = .{ .{}, .{ .size = 128, .step = 128, .min = 128 }, .{} }, .served = if (m.epi == 2) up else down };
    if (m.act != 2) s.replaces = "main's tf_exl3_plan_finish (contract_controls tf_ctl_exl3_plan_finish_old)";
    if (m.limit > 0) s.nan = .{ .stated = clamp_nan };
    switch (fault) {
        .none => {},
        .no_plus_one => s.name = "control: tf_exl3_plan_finish SwiGLU-OAI without the up's + 1",
        .alpha_one => s.name = "control: tf_exl3_plan_finish SwiGLU-OAI with alpha 1",
    }
    return s;
}

/// The splits' sums in order (k - 1 adds), then the Hadamard transform's 7 butterfly levels.
fn depth(s: c.Shape) usize {
    return s.k + 7;
}

/// One plan_finish launch: `rows` pairs over three experts' items, each pair's splits of every output column.
pub const Finish = struct {
    rig: rig_.Rig,
    mode: Mode,
    fault: Fault = .none,
    f: cuda.Function,
    old: ?cuda.Function = null, // main's kernel, for the entries that must equal it
    shape: c.Shape = undefined,
    exact: bool = false,
    z: []f32 = &.{}, // [(mat * sk + split) * rows + pair] * n + column
    sv: [2][]u16 = .{ &.{}, &.{} }, // fp16 [experts][n]: gate (or down) and up
    expert: []u8 = &.{}, // each pair's expert
    stage: cuda.exl3_grouped.Stage = undefined,
    base: u64 = 0,
    cache: []Iv = &.{},
    fresh: bool = false,

    pub fn deinit(self: *Finish) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Finish) void {
        const a = self.rig.gpu.gpa;
        a.free(self.z);
        for (&self.sv) |*v| a.free(v.*);
        a.free(self.expert);
        a.free(self.cache);
        self.z = &.{};
        self.sv = .{ &.{}, &.{} };
        self.expert = &.{};
        self.cache = &.{};
    }

    pub fn kernel(self: *Finish) g.Kernel {
        return .{ .spec = spec(self.mode, self.fault), .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
    }

    const vt = g.hooks(Finish); // gate 3 asks for replaced only of the entries whose spec names main's kernel

    fn mats(self: *const Finish) usize {
        return if (self.mode.epi == 2) 2 else 1;
    }

    /// Splits in [-2, 2), or integers in [-2048, 2048] (stress: fp32 exact); NaN, +Inf and -Inf in three pairs.
    pub fn prepare(self: *Finish, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 128 != 0 or !(shape.k == 1 or shape.k == 2 or shape.k == 4)) return false;
        self.free();
        self.rig.reset();
        self.fresh = false;
        self.shape = shape;
        self.exact = fill == .stress;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const P = shape.rows;
        const n = shape.n;
        self.z = try a.alloc(f32, self.mats() * shape.k * P * n);
        for (self.z) |*v| v.* = if (self.exact) @floatFromInt(rng.intRangeAtMost(i32, -2048, 2048)) else rng.float(f32) * 4 - 2;
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, i| if (i < P) {
                self.z[i * n + 3 + 128 * (i % (n / 128))] = v; // split 0 of the gate (or down)
                if (self.mats() == 2) self.z[(shape.k * P + i) * n + 5] = v; // and of the up
            };
        }
        const level: f32 = if (self.exact) 0x1p-12 else 0.5;
        for (&self.sv, 0..) |*sv, mi| {
            sv.* = try a.alloc(u16, experts * n);
            for (sv.*) |*h| h.* = @bitCast(@as(f16, @floatCast(level * (0.5 + rng.float(f32)) * @as(f32, if (mi == 1) 0.8 else 1))));
        }
        // three items: experts 1, 3 and 0, uneven, their pairs shuffled
        self.expert = try a.alloc(u8, P);
        const members = try a.alloc(i32, P);
        defer a.free(members);
        for (members, 0..) |*m, i| m.* = @intCast(i);
        rng.shuffle(i32, members);
        const cut = [_]usize{ 0, P / 2, P / 2 + P / 3, P };
        const ids = [_]i32{ 1, 3, 0 };
        var items: [9]i32 = undefined;
        for (0..3) |it| {
            items[3 * it] = ids[it];
            items[3 * it + 1] = @intCast(cut[it]);
            items[3 * it + 2] = @intCast(cut[it + 1] - cut[it]);
            for (members[cut[it]..cut[it + 1]]) |p| self.expert[@intCast(p)] = @intCast(ids[it]);
        }
        const counts = [1]i32{3};
        const out_bytes = P * n * @as(usize, if (self.mode.epi == 0) 4 else 2);
        self.stage = .{ .items = try self.rig.input(std.mem.sliceAsBytes(&items)), .counts = try self.rig.input(std.mem.sliceAsBytes(&counts)), .members = try self.rig.input(std.mem.sliceAsBytes(members)), .x = 0, .su0 = 0, .su1 = 0, .sv0 = try self.rig.input(std.mem.sliceAsBytes(self.sv[0])), .sv1 = try self.rig.input(std.mem.sliceAsBytes(self.sv[1])), .x0 = 0, .x1 = 0, .z = try self.rig.input(std.mem.sliceAsBytes(self.z)), .out = try self.rig.output(out_bytes), .k = 0, .n = @intCast(n), .p = @intCast(P), .sk = @intCast(shape.k), .slots = 0, .stride = 0, .skip = -1, .epi = self.mode.epi, .act_mode = self.mode.act, .limit = self.mode.limit, .alpha = if (self.mode.act == 2) oai_alpha else 0 };
        if (self.old != null) self.base = try self.rig.scratch(out_bytes);
        return true;
    }

    fn launch(self: *Finish, f: cuda.Function, out: u64) !void {
        var st = self.stage;
        st.out = out;
        var args: cuda.Args = .{};
        args.add(st);
        try cuda.launch.launch(f, .{ .grid = .{ .x = 3, .y = @intCast(self.shape.n / 128) }, .block = .{ .x = 32 } }, self.rig.s, &args);
    }

    pub fn run(self: *Finish) !void {
        try self.launch(self.f, self.stage.out);
        try self.rig.s.synchronize();
    }

    /// Main's kernel on the same inputs (PlanStage's fields but alpha, which it does not read).
    pub fn replaced(self: *Finish) ![]const u8 {
        try self.launch(self.old.?, self.base);
        return self.rig.bytes(self.base, self.count() * @as(usize, if (self.mode.epi == 0) 4 else 2));
    }

    pub fn count(self: *Finish) usize {
        return self.shape.rows * self.shape.n;
    }

    fn scale(self: *const Finish, mat: usize, p: usize, col: usize) f64 {
        return @as(f16, @bitCast(self.sv[mat][@as(usize, self.expert[p]) * self.shape.n + col]));
    }

    /// A matrix's output before its scale: a 128-block's split sums, Hadamard transformed, one sum of 128 x sk terms.
    fn rotated(self: *const Finish, sums: Sums, mat: usize, p: usize, col: usize, terms: []Iv, scratch: []f64) Iv {
        const sk = self.shape.k;
        const P = self.shape.rows;
        const n = self.shape.n;
        const block = col / 128 * 128;
        const i = col % 128;
        for (0..sk) |s| for (0..128) |j| {
            const v: f64 = self.z[((mat * sk + s) * P + p) * n + block + j];
            terms[s * 128 + j] = Iv.of(if (@popCount(i & j) % 2 == 1) -v else v);
        };
        return sums.sum(terms[0 .. sk * 128], depth(self.shape), scratch);
    }

    /// Every output at the kernel's rounding points, as main's kernel and exl3_plan.cuh compute it.
    fn solve(self: *const Finish, a: std.mem.Allocator, sums: Sums, out: []Iv) !void {
        const terms = try a.alloc(Iv, self.shape.k * 128);
        defer a.free(terms);
        const scratch = try a.alloc(f64, self.shape.k * 128);
        defer a.free(scratch);
        const m = self.mode;
        const lim: f64 = m.limit;
        for (0..self.shape.rows) |p| for (0..self.shape.n) |col| {
            var v = self.rotated(sums, 0, p, col, terms, scratch).mul(Iv.of(had_scale)).mul(Iv.of(self.scale(0, p, col)));
            if (m.epi == 2) {
                var up = self.rotated(sums, 1, p, col, terms, scratch).mul(Iv.of(had_scale)).mul(Iv.of(self.scale(1, p, col)));
                if (m.act == 2) {
                    const gc = v.bf().atMost(lim);
                    const uc = up.bf().atMost(lim).atLeast(-lim);
                    const s = gc.mul(Iv.of(oai_alpha)).bf().sigmoid().bf();
                    v = uc.add(Iv.of(1)).bf().mul(gc.mul(s).bf());
                } else {
                    if (m.act == 0) {
                        v = v.bf();
                        up = up.bf();
                    }
                    if (m.limit > 0) {
                        v = v.atMost(lim);
                        up = up.atMost(lim).atLeast(-lim);
                    }
                    v = v.div(Iv.of(1).add(v.neg().exp()));
                    if (m.act == 0) v = v.bf();
                    v = v.mul(up);
                }
            }
            out[p * self.shape.n + col] = if (m.epi == 0) v else v.bf();
        };
    }

    fn solved(self: *Finish) ![]const Iv {
        if (self.fresh) return self.cache;
        const a = self.rig.gpu.gpa;
        a.free(self.cache);
        self.cache = &.{};
        self.cache = try a.alloc(Iv, self.count());
        try self.solve(a, .{ .exact = self.exact }, self.cache);
        self.fresh = true;
        return self.cache;
    }

    pub fn reference(self: *Finish, want: []f64, mag: []f64) !void {
        for (want, mag, try self.solved()) |*w, *m, v| {
            w.* = v.m;
            m.* = 0;
        }
    }

    pub fn bounds(self: *Finish, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, try self.solved()) |*l, w, v| l.* = if (std.math.isFinite(w)) @max(0, @max(w - v.lo, v.hi - w)) + 0x1p-140 else std.math.floatMax(f64);
    }

    pub fn intervals(self: *Finish, low: []f64, high: []f64) !void {
        for (low, high, try self.solved()) |*l, *h, v| {
            l.* = v.lo;
            h.* = v.hi;
        }
    }

    pub fn simulate(self: *Finish, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const tmp = try a.alloc(Iv, out.len);
        defer a.free(tmp);
        try self.solve(a, .{ .sim = .{ .acc = acc, .order = order }, .exact = true }, tmp);
        for (out, tmp) |*o, v| o.* = v.m;
    }
};

/// The five modes on the served kernel, then the two SwiGLU-OAI controls.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    entries: [modes.len + 2]Finish,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
        errdefer self.mod.unload();
        const served = try self.mod.function("tf_exl3_plan_finish");
        const old = try controls.function("tf_ctl_exl3_plan_finish_old");
        for (&self.entries, 0..) |*x, i| {
            const fault: Fault = if (i < modes.len) .none else if (i == modes.len) .no_plus_one else .alpha_one;
            const m = if (i < modes.len) modes[i] else modes[modes.len - 1];
            const f = switch (fault) {
                .none => served,
                .no_plus_one => try controls.function("tf_ctl_exl3_plan_finish_no_plus_one"),
                .alpha_one => try controls.function("tf_ctl_exl3_plan_finish_alpha_one"),
            };
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .mode = m, .fault = fault, .f = f, .old = if (m.act != 2) old else null };
        }
        errdefer for (&self.entries) |*x| x.deinit();
        for (&self.entries) |*x| try kernels.append(a, x.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (&self.entries) |*x| x.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};
