//! Inkling 2 Mini's stored experts under the contract: NVFP4 and MXFP8 on Mini's shapes, soft-capped SwiGLU, controls.
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
const mx = cuda.experts_mx;
const dec = core.expert_decode;

pub const Format = enum { nvfp4, mxfp8 };
pub const Kind = enum { up, down_f32, down_bf16 };

pub const Entry = struct { name: []const u8, format: Format, kind: Kind, prompt: bool, short: bool = false };

/// Every new entry point, the reused NVFP4 downs at Mini's shapes, and a one-group-short control per new entry.
pub const entries = [_]Entry{
    .{ .name = "cuda stored experts nvfp4 tf_nvfp4_softcap (soft-capped SwiGLU)", .format = .nvfp4, .kind = .up, .prompt = false },
    .{ .name = "cuda stored experts nvfp4 tf_nvfp4_prompt_softcap", .format = .nvfp4, .kind = .up, .prompt = true },
    .{ .name = "cuda stored experts nvfp4 decode down f32 (Mini's factor)", .format = .nvfp4, .kind = .down_f32, .prompt = false },
    .{ .name = "cuda stored experts nvfp4 prompt down bf16 (Mini's factor)", .format = .nvfp4, .kind = .down_bf16, .prompt = true },
    .{ .name = "cuda stored experts mxfp8 tf_mxfp8_softcap", .format = .mxfp8, .kind = .up, .prompt = false },
    .{ .name = "cuda stored experts mxfp8 tf_mxfp8_down_f32", .format = .mxfp8, .kind = .down_f32, .prompt = false },
    .{ .name = "cuda stored experts mxfp8 tf_mxfp8_down_bf16", .format = .mxfp8, .kind = .down_bf16, .prompt = false },
    .{ .name = "cuda stored experts mxfp8 tf_mxfp8_prompt_softcap", .format = .mxfp8, .kind = .up, .prompt = true },
    .{ .name = "cuda stored experts mxfp8 tf_mxfp8_prompt_down", .format = .mxfp8, .kind = .down_bf16, .prompt = true },
    .{ .name = "control: tf_nvfp4_softcap one 32-input group short", .format = .nvfp4, .kind = .up, .prompt = false, .short = true },
    .{ .name = "control: tf_nvfp4_prompt_softcap one 32-input group short", .format = .nvfp4, .kind = .up, .prompt = true, .short = true },
    .{ .name = "control: tf_mxfp8_softcap one 32-input group short", .format = .mxfp8, .kind = .up, .prompt = false, .short = true },
    .{ .name = "control: tf_mxfp8_down_f32 one 32-input group short", .format = .mxfp8, .kind = .down_f32, .prompt = false, .short = true },
    .{ .name = "control: tf_mxfp8_down_bf16 one 32-input group short", .format = .mxfp8, .kind = .down_bf16, .prompt = false, .short = true },
    .{ .name = "control: tf_mxfp8_prompt_softcap one 32-input group short", .format = .mxfp8, .kind = .up, .prompt = true, .short = true },
    .{ .name = "control: tf_mxfp8_prompt_down one 32-input group short", .format = .mxfp8, .kind = .down_bf16, .prompt = true, .short = true },
};

const experts = 4;
const slots = 8; // Mini's top-8
pub const caps = [2]f32{ 4.0, 25.0 }; // Mini's mlp_gate_softcap, mlp_up_softcap

/// An mma k16 step a link of each row's chain, its own error (mma_step), and NVFP4's block-scale fma and factor.
fn depth(s: c.Shape) usize {
    return s.k / 16 + core.attention_range.mma_step + 2;
}

/// Mini's experts: 2,304 to 1,152 (gate and up each) and back; decode 1 and 4 tokens, prompt items of 9 and 16 tokens.
const ups = [2][]const c.Shape{ &.{ .{ .rows = 1, .n = 1152, .k = 2304 }, .{ .rows = 4, .n = 1152, .k = 2304 } }, &.{ .{ .rows = 9, .n = 1152, .k = 2304 }, .{ .rows = 16, .n = 1152, .k = 2304 } } };
const downs = [2][]const c.Shape{ &.{ .{ .rows = 1, .n = 2304, .k = 1152 }, .{ .rows = 4, .n = 2304, .k = 1152 } }, &.{ .{ .rows = 9, .n = 2304, .k = 1152 }, .{ .rows = 16, .n = 2304, .k = 1152 } } };

pub fn spec(e: Entry) c.Spec {
    const up = e.kind == .up;
    const served = (if (up) ups else downs)[@intFromBool(e.prompt)];
    // MXFP8's prompt CTAs take 2 (gate-up) or 4 (down) column blocks of 32
    const n_tile: c.Tile = if (e.format == .mxfp8 and e.prompt) (if (up) .{ .size = 128, .step = 64, .min = 64 } else .{ .size = 256, .step = 128, .min = 128 }) else .{ .size = 64, .step = 32, .min = 32 };
    const tiles = [3]c.Tile{ .{ .size = (if (e.prompt) ex_.promptTile(.nvfp4) else 16) / slots }, n_tile, .{ .size = 64, .step = 32, .min = 32 } };
    return switch (e.kind) {
        // fp32 tanhf, expf and the divisions after the sums: a few ulps each
        .up => .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 16, .out = .bf16, .tiles = tiles, .served = served, .sums = .{ .f16 = 30, .bf16 = 30, .why = "the caps saturate: a replay's sum far off can land on the same plateau, so only 30x is provable" } },
        .down_f32 => .{ .name = e.name, .accum = .f32, .depth = depth, .out = .f32, .tiles = tiles, .served = served },
        .down_bf16 => .{ .name = e.name, .accum = .f32, .depth = depth, .out = .bf16, .tiles = tiles, .served = served },
    };
}

/// gc tanh(g / gc) sigmoid(g) uc tanh(u / uc) in float64, and its partial derivatives in g and u.
pub fn softcap(gs: f64, us: f64) [3]f64 {
    const gc: f64 = caps[0];
    const uc: f64 = caps[1];
    const tg = std.math.tanh(gs / gc);
    const e = @exp(-@abs(gs));
    const sg = if (gs >= 0) 1 / (1 + e) else e / (1 + e);
    const tu = std.math.tanh(us / uc);
    const a = gc * tg * sg;
    const b = uc * tu;
    const da = (1 - tg * tg) * sg + gc * tg * sg * (1 - sg);
    return .{ a * b, da * b, a * (1 - tu * tu) };
}

/// shape.rows token rows, `slots` pairs each over `experts` experts; up reads a token's row, down a pair's.
pub const Experts = struct {
    rig: rig_.Rig,
    entry: Entry,
    k4: *const ex_.Nvfp4Experts,
    k8: *const mx.Mxfp8Experts,
    router: grouped.Router,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []f32 = &.{}, // [M][experts][n][k] stored weights decoded, less NVFP4's factor (exact in f32)
    factor: [2][experts]f64 = undefined, // NVFP4's per-expert factor by matrix (1 for MXFP8)
    picks: []i32 = &.{},
    xd: u64 = 0,
    wd: u64 = 0,
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
        a.free(self.picks);
        self.x = &.{};
        self.w = &.{};
        self.picks = &.{};
    }

    pub fn kernel(self: *Experts) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.short) .shapes else null };
    }

    const vt = g.hooks(Experts);

    fn up(self: *const Experts) bool {
        return self.entry.kind == .up;
    }

    fn parts(self: *const Experts) usize {
        return if (self.up()) 2 else 1;
    }

    fn pairs(self: *const Experts) usize {
        return self.shape.rows * slots;
    }

    fn xRows(self: *const Experts) usize {
        return if (self.up()) self.shape.rows else self.pairs();
    }

    fn tile(self: *const Experts) usize {
        return if (self.entry.prompt) ex_.promptTile(.nvfp4) else 16;
    }

    fn items(self: *const Experts) usize {
        return grouped.maxItems(self.pairs(), experts, self.tile());
    }

    fn layer(self: *const Experts) ex_.Layer {
        const n = self.shape.n;
        const k = self.shape.k;
        return .{ .format = if (self.entry.format == .nvfp4) .nvfp4 else .mxfp8, .up = self.wd, .down = self.wd, .up_scale = self.sd, .down_scale = self.sd, .width = if (self.up()) n else k, .dims = if (self.up()) k else n, .experts = experts, .caps = caps };
    }

    /// Random stored codes and scales; non-finite: NaN, +Inf and -Inf in the first rows; stress: mirrored integers.
    pub fn prepare(self: *Experts, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        if (self.entry.format == .mxfp8 and self.entry.prompt and shape.n % (32 * mx.Mxfp8Experts.prompt_cw[@intFromBool(!self.up())]) != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u16, self.xRows() * k);
        self.w = try a.alloc(f32, self.parts() * experts * n * k);
        self.picks = try a.alloc(i32, self.pairs());
        for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.picks, 0..) |*p, i| p.* = @intCast(if (i < experts) i else rng.uintLessThan(usize, experts));
        const stress = fill == .stress;
        const nv = self.entry.format == .nvfp4;
        const cbytes = if (nv) n * k / 2 else n * k;
        const sbytes = if (nv) n * k / 16 else n * k / 32;
        const codes = try a.alloc(u8, self.parts() * experts * cbytes);
        defer a.free(codes);
        const scales = try a.alloc(u8, self.parts() * experts * sbytes);
        defer a.free(scales);
        for (codes, 0..) |*b, i| b.* = if (stress) (if (nv) @as(u8, if (i % 3 == 0) 0x42 else 0x24) else @as(u8, if (i % 3 == 0) 0x40 else 0x38)) else rng.int(u8);
        if (!nv and !stress) for (codes) |*b| if (b.* & 0x7F == 0x7F) {
            b.* ^= 1; // no NaN codes
        };
        // E4M3 scales 2^-3 .. 2^3, E8M0 2^-11 .. 2^-7: a few units a sum; stress 1, MXFP8 down 8 (fp16 lanes overflow)
        for (scales) |*b| b.* = if (stress) (if (nv) 0x38 else if (self.up()) @as(u8, 127) else 130) else if (nv) 0x20 + rng.uintLessThan(u8, 0x30) else 116 + rng.uintLessThan(u8, 5);
        for (0..2) |m| for (0..experts) |e| {
            const amax: f32 = if (stress) 2688 else 64 + rng.float(f32) * 256;
            self.factor[m][e] = if (nv) mx.nvfp4Factor(amax, 2688) else 1;
        };
        if (stress) self.mirror(codes, cbytes);
        for (0..self.parts() * experts) |me| for (0..n) |row| for (0..k) |i| {
            const cs = codes[me * cbytes ..][0..cbytes];
            const ss = scales[me * sbytes ..][0..sbytes];
            self.w[(me * n + row) * k + i] = @floatCast(if (nv) dec.nvfp4(cs, ss, k, row, i, 1) else dec.mxfp8(cs, ss, k, row, i));
        };
        if (fill == .nonfinite) {
            // a row holding NaN, one holding +Inf, one holding both infinities
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, k / 3, std.math.nan(f32) }, .{ 1, k / 2, std.math.inf(f32) }, .{ 2, 1, std.math.inf(f32) }, .{ 2, k - 1, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < self.xRows()) {
                self.x[p[0] * k + p[1]] = c.toBf16(p[2]);
            };
        }
        if (stress) self.stressRows(rng);
        try self.upload(codes, scales, cbytes, sbytes);
        return true;
    }

    /// The packed blocks, the factors, the plan and the output.
    fn upload(self: *Experts, codes: []const u8, scales: []const u8, cbytes: usize, sbytes: usize) !void {
        const a = self.rig.gpu.gpa;
        const n = self.shape.n;
        const k = self.shape.k;
        const nv = self.entry.format == .nvfp4;
        const per = (if (nv) ex_.Nvfp4Experts.bytes(1, n, k, self.parts()) else mx.Mxfp8Experts.bytes(1, n, k, self.parts())) / 4;
        const blocks = try a.alloc(u32, experts * per);
        defer a.free(blocks);
        for (0..experts) |e| {
            const out = blocks[e * per ..][0..per];
            var cs: [2][]const u8 = undefined;
            var ss: [2][]const u8 = undefined;
            for (0..self.parts()) |m| {
                cs[m] = codes[(m * experts + e) * cbytes ..][0..cbytes];
                ss[m] = scales[(m * experts + e) * sbytes ..][0..sbytes];
            }
            if (nv and self.up()) ex_.Nvfp4Experts.packGateUp(out, .{ cs[0], ss[0] }, .{ cs[1], ss[1] }, n, k);
            if (nv and !self.up()) ex_.Nvfp4Experts.packOne(out, cs[0], ss[0], n, k);
            if (!nv and self.up()) mx.Mxfp8Experts.packGateUp(out, .{ cs[0], ss[0] }, .{ cs[1], ss[1] }, n, k);
            if (!nv and !self.up()) mx.Mxfp8Experts.packOne(out, cs[0], ss[0], n, k);
        }
        self.wd = try self.rig.input(std.mem.sliceAsBytes(blocks));
        var f: [experts * 2]f32 = undefined;
        for (0..experts) |e| for (0..self.parts()) |m| {
            f[e * self.parts() + m] = @floatCast(self.factor[m][e]);
        };
        self.sd = try self.rig.input(std.mem.sliceAsBytes(f[0 .. experts * self.parts()]));
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        const zeros = try a.alloc(u8, @max(self.pairs() * 4, self.items() * 12, 8));
        defer a.free(zeros);
        @memset(zeros, 0);
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        const hist = (self.pairs() + 1023) / 1024;
        self.plan = .{ .members = try self.rig.input(zeros[0 .. self.pairs() * 4]), .items = try self.rig.input(zeros[0 .. self.items() * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = try self.rig.scratch(self.pairs() * 4), .hist = try self.rig.scratch(hist * experts * 4) };
        try self.router.route(self.rig.s, ids, self.pairs(), experts, self.tile(), self.plan);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(self.pairs() * n * c.size(spec(self.entry).out));
    }

    /// Gate 4's weights: input k - 1 - i repeats input i (weights 1 and 2; MXFP8 down 8, 16). so twins sit in
    /// different 32-input lanes and no replay order cancels them early.
    fn mirror(self: *Experts, codes: []u8, cbytes: usize) void {
        const k = self.shape.k;
        const nv = self.entry.format == .nvfp4;
        const row = if (nv) k / 2 else k;
        for (0..self.parts() * experts * (cbytes / row)) |r| {
            const b = codes[r * row ..][0..row];
            for (0..row / 2) |j| b[row - 1 - j] = if (nv) (b[j] >> 4) | (b[j] << 4) else b[j];
        }
    }

    /// Gate 4's rows: input k - 1 - i is minus input i, input k - 1 is 4 - x_0: each sum is 4 w_0, partials under 2^23.
    fn stressRows(self: *Experts, rng: std.Random) void {
        const k = self.shape.k;
        for (0..self.xRows()) |r| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(4 * (1 + 2 * rng.uintLessThan(u32, 64)));
            self.x[r * k + i] = c.toBf16(v);
            self.x[r * k + k - 1 - i] = c.toBf16(if (i == 0) 4 - v else -v); // a multiple of 8: exact in bf16
        };
    }

    pub fn run(self: *Experts) !void {
        const l = self.layer();
        const in: ex_.Rows = .{ .x = self.xd, .stride = self.shape.k, .slots = if (self.up()) slots else 0 };
        if (self.entry.short) return self.short(l, in);
        const epi: ex_.Epi = switch (self.entry.kind) {
            .up => .up,
            .down_f32 => .down_f32,
            .down_bf16 => .down_bf16,
        };
        const gx: ex_.Grouped = .{ .nvfp4 = self.k4, .mxfp8 = self.k8 };
        if (self.entry.prompt) {
            try gx.prompt(self.rig.s, epi, in, l, self.plan, self.out, self.items(), self.pairs(), 64, -1);
        } else try gx.decode(self.rig.s, epi, in, l, self.plan, self.out, self.items(), self.pairs(), 16, -1);
        try self.rig.s.synchronize();
    }

    /// The control: the entry's own launch with one k32 group fewer than the layer has.
    fn short(self: *Experts, l: ex_.Layer, in: ex_.Rows) !void {
        const up_ = self.up();
        const n = self.shape.n;
        const nb = n / 32;
        const kg = self.shape.k / 32 - 1;
        const nv = self.entry.format == .nvfp4;
        var a = mx.argsOf(in, up_, self.wd, if (nv) self.sd else null, kg, nb, self.plan, self.out, n, l.caps, -1);
        var cfg: cuda.Config = .{ .grid = .{ .x = @intCast((self.items() * nb + 3) / 4) }, .block = .{ .x = 128 } };
        const f = if (nv) self.k4.softcap.fns[@intFromBool(self.entry.prompt)] else self.k8.fns[
            try mx.Mxfp8Experts.entry(self.entry.prompt, switch (self.entry.kind) {
                .up => .up,
                .down_f32 => .down_f32,
                .down_bf16 => .down_bf16,
            })
        ];
        if (self.entry.prompt and nv) {
            cfg.grid.x = @intCast(self.items() * ((nb + 3) / 4));
            cfg.shared = ex_.Nvfp4Experts.promptSmem(2, mx.Softcap.prompt_wn);
        } else if (self.entry.prompt) cfg.grid.x = @intCast(self.items() * (nb / mx.Mxfp8Experts.prompt_cw[@intFromBool(!up_)]));
        try cuda.launch.launch(f, cfg, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Experts) usize {
        return self.pairs() * self.shape.n;
    }

    /// Pair p's products with column j of matrix m of its expert, in input order, and that matrix's factor.
    fn terms(self: *const Experts, m: usize, p: usize, j: usize, t: []f64) f64 {
        const k = self.shape.k;
        const xr = (if (self.up()) p / slots else p) * k;
        const e: usize = @intCast(self.picks[p]);
        const wr = ((m * experts + e) * self.shape.n + j) * k;
        for (t, 0..) |*o, i| o.* = c.fromBf16(self.x[xr + i]) * @as(f64, self.w[wr + i]);
        return self.factor[m][e];
    }

    pub fn reference(self: *Experts, want: []f64, mag: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.pairs()) |p| for (0..self.shape.n) |j| {
            var sum: [2]f64 = .{ 0, 0 };
            var abs: [2]f64 = .{ 0, 0 };
            for (0..self.parts()) |m| {
                const f = self.terms(m, p, j, t);
                for (t) |v| {
                    sum[m] += v;
                    abs[m] += @abs(v);
                }
                sum[m] *= f;
                abs[m] *= f;
            }
            const o = p * self.shape.n + j;
            if (self.up()) {
                const r = softcap(sum[0], sum[1]);
                want[o] = r[0];
                mag[o] = @abs(r[1]) * abs[0] + @abs(r[2]) * abs[1]; // the sums' errors through the slopes
            } else {
                want[o] = sum[0];
                mag[o] = abs[0];
            }
        };
    }

    pub fn simulate(self: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.pairs()) |p| for (0..self.shape.n) |j| {
            var sum: [2]f64 = .{ 0, 0 };
            for (0..self.parts()) |m| {
                const f = self.terms(m, p, j, t);
                sum[m] = contract.rows.sumIn(acc, order, t) * f;
            }
            out[p * self.shape.n + j] = if (self.up()) softcap(sum[0], sum[1])[0] else sum[0];
        };
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k4: ex_.Nvfp4Experts,
    k8: mx.Mxfp8Experts,
    mod: cuda.Module,
    items: [entries.len]Experts = undefined,

    /// The set, its kernels appended to `list` (each control after the kernels it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        var k4 = try ex_.Nvfp4Experts.load(gpu.d, sms);
        errdefer k4.unload();
        var k8 = try mx.Mxfp8Experts.load(gpu.d, sms);
        errdefer k8.unload();
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
        errdefer mod.unload();
        set.* = .{ .gpa = gpu.gpa, .k4 = k4, .k8 = k8, .mod = mod };
        const router = try grouped.Router.resolve(set.mod);
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .k4 = &set.k4, .k8 = &set.k8, .router = router };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.mod.unload();
        set.k8.unload();
        set.k4.unload();
        set.gpa.destroy(set);
    }
};

/// The stored experts' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6d696e69, &t);
    std.debug.print("SUMMARY stored experts: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "stored-experts contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.short, std.mem.startsWith(u8, e.name, "control: "));
}

test "the soft-capped SwiGLU's slopes are its float64 difference quotients" {
    for ([_][2]f64{ .{ 0.3, 1.5 }, .{ -6, 30 }, .{ 9, -2 } }) |gu| {
        const r = softcap(gu[0], gu[1]);
        const h = 1e-6;
        try std.testing.expectApproxEqRel((softcap(gu[0] + h, gu[1])[0] - softcap(gu[0] - h, gu[1])[0]) / (2 * h), r[1], 1e-6);
        try std.testing.expectApproxEqRel((softcap(gu[0], gu[1] + h)[0] - softcap(gu[0], gu[1] - h)[0]) / (2 * h), r[2], 1e-6);
    }
}
