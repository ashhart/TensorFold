//! dots.mocr's vision glue against float64: tf_dots_vision_rope (2-D rotary in place), tf_dots_patch_bias, controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const bfr = core.bf16_round;

const hd = 128;
const half = hd / 2;
const quarter = half / 2;
const heads = 12; // the tower's heads, q, k and v alike
const width = 3 * heads * hd; // a row of QKV: q | k | v
const turned = 2 * heads; // q and k heads rotate, v heads stay
const hidden = 1536; // the patch embedding's outputs
const top = 3393; // the largest patch row or column placed
const theta: f32 = 10000;
/// Rows 0 to 3 put 0, 1, 239 and 3,393 at both h and w; rows 4 to 7 pair each with itself.
const special = [_][2]i32{ .{ top, 0 }, .{ 0, top }, .{ 1, 239 }, .{ 239, 1 }, .{ top, top }, .{ 0, 0 }, .{ 1, 1 }, .{ 239, 239 } };
/// CUDA's cosf and sinf (no fast math) are within 2 fp32 ulps: the reference allows 4 either side.
const trig_ulps = 4;

/// A registration: its name, symbol, and whether it is the planted control (from the contract_controls image).
pub const Form = struct { name: []const u8, sym: [:0]const u8, control: bool = false };

pub const rope_forms = [_]Form{
    .{ .name = "cuda dots_mocr tf_dots_vision_rope (2-D rotary in place, fp32 cos and sin, 12 heads of 128)", .sym = "tf_dots_vision_rope" },
    .{ .name = "control: dots.mocr's vision rotary with cos and sin rounded to bf16 first", .sym = "tf_ctl_dots_vision_rope", .control = true },
};

pub const bias_forms = [_]Form{
    .{ .name = "cuda dots_mocr tf_dots_patch_bias (bf16 y + bias rounded again, in place)", .sym = "tf_dots_patch_bias" },
    .{ .name = "control: dots.mocr's patch bias from the fp32 accumulator, rounded once", .sym = "tf_ctl_dots_patch_bias", .control = true },
};

/// Shape: rows x heads or outputs a row (n) x the terms each sum adds (k).
const rope_served = [_]c.Shape{ .{ .rows = 4, .n = heads, .k = 2 }, .{ .rows = 196, .n = heads, .k = 2 }, .{ .rows = 4096, .n = heads, .k = 2 }, .{ .rows = 19520, .n = heads, .k = 2 } };
const bias_served = [_]c.Shape{ .{ .rows = 4, .n = hidden, .k = 2 }, .{ .rows = 196, .n = hidden, .k = 2 }, .{ .rows = 4096, .n = hidden, .k = 2 }, .{ .rows = 19520, .n = hidden, .k = 2 } };

const rope_why = "a bf16 value times an fp32 cos or sin is not exact in fp32, so gate 4 proves fp16's miss alone; gate 3 checks HF's points";
const bias_why = "one add of two bf16 values: a bf16 sum rounds where the output does, so gate 4 proves fp16's miss alone";

fn ropeDepth(_: c.Shape) usize {
    return 2; // two fp32 products, then their sum
}

fn biasDepth(_: c.Shape) usize {
    return 1; // y + b: one fp32 add
}

/// Gates 1 and 3 check every output at HF's rounding points; gate 4 at (0, 0) sees the rows unrotated.
pub fn ropeSpec(f: Form) c.Spec {
    return .{ .name = f.name, .accum = .f32, .depth = ropeDepth, .out = .bf16, .in_place = true, .served = &rope_served, .sums = .{ .bf16 = 0, .why = rope_why } };
}

/// Gates 1 and 3 check every output exactly: bf16(bf16 y + b), HIP's order.
pub fn biasSpec(f: Form) c.Spec {
    return .{ .name = f.name, .accum = .f32, .depth = biasDepth, .out = .bf16, .in_place = true, .served = &bias_served, .sums = .{ .bf16 = 0, .why = bias_why } };
}

/// x rounded to fp32 (float64 then fp32 equals one fp32 rounding of + and x) and then to bf16, as the kernels store.
fn b16(x: f64) f64 {
    return bfr.value(bfr.bits(@floatCast(x)));
}

const Trig = enum { cos, sin };

/// The fp32 values a device cosf or sinf of `angle` can return: exact at 0, else within trig_ulps of float64's.
fn trig(angle: f32, f: Trig) [2]f32 {
    if (angle == 0) return @splat(if (f == .cos) 1 else 0);
    const v = if (f == .cos) @cos(@as(f64, angle)) else @sin(@as(f64, angle));
    const ulp = std.math.ldexp(@as(f64, 1), @max(std.math.frexp(v).exponent - 24, -149));
    return .{ @floatCast(v - trig_ulps * ulp), @floatCast(v + trig_ulps * ulp) };
}

/// One output of a q or k head at cos cs and sin sn: HF's fp32 products, their fp32 sum, then bf16.
fn rotate(a: f64, b: f64, upper: bool, cs: f32, sn: f32) f64 {
    const p: f32 = @floatCast((if (upper) b else a) * cs); // a bf16 times an fp32 value: exact in float64
    const q: f32 = @floatCast((if (upper) a else b) * sn);
    const sum: f32 = @floatCast(if (upper) @as(f64, p) + q else @as(f64, p) - q);
    return bfr.value(bfr.bits(sum));
}

/// Rows' (h, w): the special pairs, then runs of random grids in dots' 2 x 2 merge order; stress rows at (0, 0).
fn place(hw: []i32, fill: g.Fill, rng: std.Random) void {
    if (fill == .stress) return @memset(hw, 0);
    const r = hw.len / 2;
    var row: usize = 0;
    while (row < r and row < special.len) : (row += 1) {
        hw[2 * row] = special[row][0];
        hw[2 * row + 1] = special[row][1];
    }
    while (row < r) {
        const gh = 2 * (1 + rng.uintAtMost(usize, top / 2)); // an even grid up to 3,394 patches a side
        const gw = 2 * (1 + rng.uintAtMost(usize, top / 2));
        const blocks = gh / 2 * (gw / 2);
        var blk = rng.uintLessThan(usize, blocks); // a random start inside the grid
        while (row < r and blk < blocks) : (blk += 1) for (0..4) |q| if (row < r) {
            hw[2 * row] = @intCast(blk / (gw / 2) * 2 + q / 2);
            hw[2 * row + 1] = @intCast(blk % (gw / 2) * 2 + q % 2);
            row += 1;
        };
    }
}

/// The sums' accumulator and order (gate 4's simulation), or null: fp32, as the kernels add.
const Sim = ?struct { acc: c.Accum, order: rows_.Order };

pub const Rope = struct {
    rig: rig_.Rig,
    f: Form,
    fn_: cuda.Function,
    arena: std.heap.ArenaAllocator,
    rows: usize = 0,
    qkv: []u16 = &.{}, // the input rows' bf16 bits
    hw: []i32 = &.{}, // [rows][h, w]
    cs: [][2]f32 = &.{}, // a row's fp32 cos interval for each of the 64 frequencies
    sn: [][2]f32 = &.{},
    dev: struct { src: u64 = 0, qkv: u64 = 0, hw: u64 = 0, inv: u64 = 0 } = .{},
    msg: [240]u8 = undefined,

    pub fn deinit(self: *Rope) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Rope) g.Kernel {
        return .{ .spec = ropeSpec(self.f), .ctx = self, .vt = &vt, .control = if (self.f.control) .shapes else null };
    }

    const vt = g.hooks(Rope);

    pub fn prepare(self: *Rope, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        if (shape.rows == 0 or shape.n != heads or shape.k != 2) return false;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        self.rows = r;
        const inv = try a.alloc(f32, quarter);
        for (inv, 0..) |*v, i| v.* = 1 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / half); // dim 64
        self.hw = try a.alloc(i32, 2 * r);
        place(self.hw, fill, rng);
        self.qkv = try a.alloc(u16, r * width);
        for (self.qkv, 0..) |*v, i| v.* = bfr.bits(rng.floatNorm(f32) * (if (i % 37 == 0) @as(f32, 64) else 2));
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => ropeStress(self.qkv, rng),
        }
        self.cs = try a.alloc([2]f32, r * half);
        self.sn = try a.alloc([2]f32, r * half);
        for (0..r) |row| for (0..half) |j| {
            const angle = @as(f32, @floatFromInt(self.hw[2 * row + j / quarter])) * inv[j % quarter]; // __fmul_rn
            self.cs[row * half + j] = trig(angle, .cos);
            self.sn[row * half + j] = trig(angle, .sin);
        };
        self.dev.src = try self.rig.input(std.mem.sliceAsBytes(self.qkv));
        self.dev.hw = try self.rig.input(std.mem.sliceAsBytes(self.hw));
        self.dev.inv = try self.rig.input(std.mem.sliceAsBytes(inv));
        self.dev.qkv = try self.rig.output(r * width * 2);
        return true;
    }

    /// NaN in a q head (row 0) and a v head (row 2); +Inf in row 1's k head; both infinities in row 3's q head.
    fn plant(self: *Rope) void {
        const put = [_]struct { usize, f32 }{
            .{ 3, std.math.nan(f32) },
            .{ width + heads * hd + 70, std.math.inf(f32) },
            .{ 2 * width + turned * hd + 7, std.math.nan(f32) },
            .{ 3 * width + 5 * hd + 20, std.math.inf(f32) },
            .{ 3 * width + 5 * hd + 90, -std.math.inf(f32) },
            .{ 4 * width + (turned + 6) * hd + 100, -std.math.inf(f32) },
        };
        for (put) |p| if (p[0] < self.qkv.len) {
            self.qkv[p[0]] = bfr.bits(p[1]);
        };
    }

    pub fn run(self: *Rope) !void {
        const bytes = self.rows * width * 2; // in place: the rows go back before each independent run
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.dev.qkv, .len = bytes }).copyFrom(0, self.dev.src, bytes, self.rig.s.handle);
        var a: cuda.Args = .{};
        inline for (.{ "qkv", "hw", "inv" }) |name| a.add(@field(self.dev, name));
        a.add(@as(c_int, @intCast(self.rows)));
        try cuda.launch.launch(self.fn_, .{ .grid = .{ .x = @intCast(self.rows), .y = turned, .z = 1 }, .block = .{ .x = half } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Rope) usize {
        return self.rows * width;
    }

    /// Output i's value at a cos and sin corner (ci, si) of its intervals; a v head's is its input.
    fn at(self: *const Rope, i: usize, ci: usize, si: usize) f64 {
        const row = i / width;
        const col = i % width;
        if (col >= turned * hd) return bfr.value(self.qkv[i]);
        const base = row * width + col / hd * hd;
        const j = col % hd % half;
        const x = row * half + j;
        return rotate(bfr.value(self.qkv[base + j]), bfr.value(self.qkv[base + j + half]), col % hd >= half, self.cs[x][ci], self.sn[x][si]);
    }

    /// Output i's least and greatest value over its four corners (each rounding is monotone), and corner (0, 0).
    fn span(self: *const Rope, i: usize) [3]f64 {
        const first = self.at(i, 0, 0);
        var lo = first;
        var hi = first;
        for (1..4) |k| {
            const v = self.at(i, k / 2, k % 2);
            if (c.classOf(v) != .finite or c.classOf(first) != .finite) return .{ first, first, first };
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        return .{ lo, hi, first };
    }

    pub fn reference(self: *Rope, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            w.* = self.at(i, 0, 0);
            m.* = @abs(bfr.value(self.qkv[i])) + @abs(bfr.value(self.qkv[i ^ half]));
        }
    }

    /// The q and k outputs' two products and their sum in `acc`, at corner (0, 0); v heads pass.
    pub fn simulate(self: *Rope, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*o, i| {
            const col = i % width;
            if (col >= turned * hd) {
                o.* = bfr.value(self.qkv[i]);
                continue;
            }
            const base = i / width * width + col / hd * hd;
            const j = col % hd % half;
            const x = i / width * half + j;
            const a: f64 = bfr.value(self.qkv[base + j]);
            const b: f64 = bfr.value(self.qkv[base + j + half]);
            const cs: f64 = self.cs[x][0];
            const sn: f64 = self.sn[x][0];
            const terms = if (col % hd >= half) [2]f64{ b * cs, a * sn } else [2]f64{ a * cs, -(b * sn) };
            o.* = b16(rows_.sumIn(acc, order, &terms));
        }
    }

    /// Every output inside HF's values at the cos and sin the device can return; NaN and infinities by class.
    pub fn check(self: *Rope, gate: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        var planted: usize = 0;
        for (0..self.count()) |i| {
            const got = c.read(.bf16, out, i);
            const s = self.span(i);
            const finite = c.classOf(s[2]) == .finite;
            if (!finite) planted += 1;
            if (if (finite) got >= s[0] and got <= s[1] else c.classOf(got) == c.classOf(s[2])) continue;
            const row = i / width;
            const col = i % width;
            return try std.fmt.bufPrint(&self.msg, "row {d} head {d} dim {d} at (h {d}, w {d}): {e} where HF gives [{e}, {e}]", .{ row, col / hd, col % hd, self.hw[2 * row], self.hw[2 * row + 1], got, s[0], s[1] });
        }
        if (gate == .nonfinite and (planted == 0 or planted == self.count())) return "the planted values leave no class to prove";
        return null;
    }
};

/// Gate 4 at (0, 0): q, k and v values s (2^20 + d 2^13), exact in bf16 and fp32, past fp16's range.
fn ropeStress(qkv: []u16, rng: std.Random) void {
    for (qkv) |*v| {
        const d: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 127));
        v.* = bfr.bits((if (rng.boolean()) @as(f32, 1) else -1) * (0x1p20 + d * 0x1p13));
    }
}

/// Patch sums and bias: columns 0 mod 4 walk every bf16 pattern, 1 mod 4 tie on a power-of-two bias, the rest random.
fn biasFill(acc: []f32, b: []u16, n: usize, fill: g.Fill, rng: std.Random) void {
    const sign = [2]f32{ 1, -1 };
    if (fill == .stress) {
        for (b, 0..) |*v, col| v.* = bfr.bits(-sign[col % 2] * 0x1p20);
        for (acc, 0..) |*v, i| v.* = sign[i % n % 2] * (0x1p20 + @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 127))) * 0x1p13);
        return;
    }
    for (b, 0..) |*v, col| v.* = bfr.bits(if (col % 4 == 1) sign[rng.uintLessThan(usize, 2)] * std.math.ldexp(@as(f32, 1), -rng.intRangeAtMost(i32, 0, 12)) else rng.floatNorm(f32) * 0.5);
    for (acc, 0..) |*v, i| {
        const row = i / n;
        const col = i % n;
        v.* = switch (col % 4) {
            0 => bfr.value(@truncate((row * ((n + 3) / 4) + col / 4) *% 40503)), // odd: every pattern once in 65,536
            1 => sign[rng.uintLessThan(usize, 2)] * (1 + @as(f32, @floatFromInt(rng.uintLessThan(u32, 128))) / 128) * @abs(bfr.value(b[col])) * 256,
            else => rng.floatNorm(f32) * (if (i % 37 == 0) @as(f32, 64) else 2),
        };
    }
}

pub const Bias = struct {
    rig: rig_.Rig,
    f: Form,
    fn_: cuda.Function,
    arena: std.heap.ArenaAllocator,
    rows: usize = 0,
    n: usize = 0,
    acc: []f32 = &.{},
    y: []u16 = &.{}, // acc rounded to bf16: what the patch GEMM stores
    b: []u16 = &.{},
    dev: struct { y: u64 = 0, b: u64 = 0, o: u64 = 0 } = .{},
    msg: [240]u8 = undefined,

    pub fn deinit(self: *Bias) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Bias) g.Kernel {
        return .{ .spec = biasSpec(self.f), .ctx = self, .vt = &vt, .control = if (self.f.control) .shapes else null };
    }

    const vt = g.hooks(Bias);

    pub fn prepare(self: *Bias, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        if (shape.rows == 0 or shape.n == 0 or shape.k != 2) return false;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        self.rows = shape.rows;
        self.n = shape.n;
        self.acc = try a.alloc(f32, shape.rows * shape.n);
        self.b = try a.alloc(u16, shape.n);
        biasFill(self.acc, self.b, shape.n, fill, prng.random());
        if (fill == .nonfinite) self.plant();
        self.y = try a.alloc(u16, self.acc.len);
        for (self.y, self.acc) |*y, v| y.* = bfr.bits(v);
        self.dev.y = try self.rig.input(if (self.f.control) std.mem.sliceAsBytes(self.acc) else std.mem.sliceAsBytes(self.y));
        self.dev.b = try self.rig.input(std.mem.sliceAsBytes(self.b));
        self.dev.o = try self.rig.output(self.y.len * 2);
        return true;
    }

    /// NaN (row 0), +Inf (row 1), -Inf (row 2) and both infinities (row 3) in the sums; -Inf in a column's bias.
    fn plant(self: *Bias) void {
        const n = self.n;
        const put = [_]struct { usize, f32 }{
            .{ 3, std.math.nan(f32) },          .{ n + 70, std.math.inf(f32) },      .{ 2 * n + 6, -std.math.inf(f32) },
            .{ 3 * n + 22, std.math.inf(f32) }, .{ 3 * n + 91, -std.math.inf(f32) },
        };
        for (put) |p| if (p[0] < self.acc.len) {
            self.acc[p[0]] = p[1];
        };
        self.b[9 % n] = bfr.bits(-std.math.inf(f32));
    }

    pub fn run(self: *Bias) !void {
        var a: cuda.Args = .{};
        if (self.f.control) {
            a.add(self.dev.y);
        } else {
            // in place, as the tower calls it: the GEMM's bf16 rows go back before each independent run
            const bytes = self.y.len * 2;
            try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.dev.o, .len = bytes }).copyFrom(0, self.dev.y, bytes, self.rig.s.handle);
            a.add(self.dev.o);
        }
        a.add(self.dev.b);
        a.add(self.dev.o);
        a.add(@as(c_int, @intCast(self.rows)));
        a.add(@as(c_int, @intCast(self.n)));
        const grid: cuda.launch.Dim3 = .{ .x = @intCast(self.rows), .y = @intCast((self.n + 255) / 256), .z = 1 };
        try cuda.launch.launch(self.fn_, .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Bias) usize {
        return self.y.len;
    }

    fn want(self: *const Bias, i: usize) f64 {
        return b16(@as(f64, bfr.value(self.y[i])) + bfr.value(self.b[i % self.n]));
    }

    pub fn reference(self: *Bias, want_: []f64, mag: []f64) !void {
        for (want_, mag, 0..) |*w, *m, i| {
            w.* = self.want(i);
            m.* = @abs(bfr.value(self.y[i])) + @abs(bfr.value(self.b[i % self.n]));
        }
    }

    pub fn simulate(self: *Bias, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*o, i| o.* = b16(rows_.sumIn(acc, order, &[2]f64{ bfr.value(self.y[i]), bfr.value(self.b[i % self.n]) }));
    }

    /// Every output equals bf16(bf16 y + b) exactly; NaN and infinities by class.
    pub fn check(self: *Bias, gate: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        var planted: usize = 0;
        for (0..self.count()) |i| {
            const got = c.read(.bf16, out, i);
            const w = self.want(i);
            if (c.classOf(w) != .finite) planted += 1;
            if (c.classOf(got) == c.classOf(w) and (c.classOf(w) != .finite or got == w)) continue;
            return try std.fmt.bufPrint(&self.msg, "row {d} column {d}: {e} where bf16(bf16 y + b) gives {e} (y {e}, b {e})", .{ i / self.n, i % self.n, got, w, bfr.value(self.y[i]), bfr.value(self.b[i % self.n]) });
        }
        if (gate == .nonfinite and (planted == 0 or planted == self.count())) return "the planted values leave no class to prove";
        return null;
    }
};

/// The kernels and their controls, registered onto `kernels`; the controls from the contract_controls image `ctl`.
pub const Set = struct {
    mod: ?cuda.Module = null,
    ropes: [rope_forms.len]Rope = undefined,
    biases: [bias_forms.len]Bias = undefined,
    nr: usize = 0,
    nb: usize = 0,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.dots_mocr);
        for (rope_forms, &self.ropes) |f, *x| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .fn_ = try (if (f.control) ctl else self.mod.?).function(f.sym), .arena = .init(gpu.gpa) };
            self.nr += 1;
        }
        for (bias_forms, &self.biases) |f, *x| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .fn_ = try (if (f.control) ctl else self.mod.?).function(f.sym), .arena = .init(gpu.gpa) };
            self.nb += 1;
        }
        for (self.ropes[0..self.nr]) |*x| try kernels.append(gpu.gpa, x.kernel());
        for (self.biases[0..self.nb]) |*x| try kernels.append(gpu.gpa, x.kernel());
    }

    pub fn deinit(self: *Set) void {
        for (self.biases[0..self.nb]) |*x| x.deinit();
        for (self.ropes[0..self.nr]) |*x| x.deinit();
        if (self.mod) |*m| m.unload();
    }
};

test "bf16 cos and sin leave HF's fp32 window often, so the rotary control cannot pass gate 3" {
    var prng = std.Random.DefaultPrng.init(0x76697331);
    const rng = prng.random();
    var outside: usize = 0;
    for (0..4096) |_| {
        const a = b16(rng.floatNorm(f64) * 2);
        const b = b16(rng.floatNorm(f64) * 2);
        const j = rng.uintLessThan(usize, quarter);
        const inv = 1 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * j)) / half);
        const angle = @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, top))) * inv;
        const cs = trig(angle, .cos);
        const sn = trig(angle, .sin);
        const bc: f32 = @floatCast(b16(@cos(@as(f64, angle))));
        const bs: f32 = @floatCast(b16(@sin(@as(f64, angle))));
        for ([_]bool{ false, true }) |upper| {
            const ctl = rotate(a, b, upper, bc, bs);
            var lo = std.math.inf(f64);
            var hi = -std.math.inf(f64);
            for (0..4) |k| {
                const v = rotate(a, b, upper, cs[k / 2], sn[k % 2]);
                lo = @min(lo, v);
                hi = @max(hi, v);
            }
            outside += @intFromBool(ctl < lo or ctl > hi);
        }
    }
    try std.testing.expect(outside > 8192 / 10);
}

test "one rounding of the patch sums differs from HIP's second rounding, so the bias control cannot pass gate 3" {
    var prng = std.Random.DefaultPrng.init(0x70617463);
    const rng = prng.random();
    var differ: usize = 0;
    for (0..8192) |_| {
        const acc = rng.floatNorm(f32) * 2;
        const b = b16(rng.floatNorm(f64) * 0.5);
        differ += @intFromBool(b16(acc + b) != b16(b16(acc) + b));
    }
    try std.testing.expect(differ > 8192 / 20);
}

test "the trig window: exact at angle 0, a few fp32 ulps either side elsewhere" {
    try std.testing.expectEqual([2]f32{ 1, 1 }, trig(0, .cos));
    try std.testing.expectEqual([2]f32{ 0, 0 }, trig(0, .sin));
    const w = trig(3393, .cos);
    const v: f32 = @floatCast(@cos(@as(f64, 3393)));
    try std.testing.expect(w[0] < v and v < w[1]);
    const ulp = std.math.ldexp(@as(f32, 1), std.math.frexp(v).exponent - 24);
    try std.testing.expect(w[1] - w[0] <= (2 * trig_ulps + 2) * ulp and w[1] - w[0] >= (2 * trig_ulps - 2) * ulp);
}

test "positions: rows 0 to 3 put 0, 1, 239 and 3,393 at h and w; random grids in range and 2 x 2 merge order" {
    var prng = std.Random.DefaultPrng.init(7);
    var hw: [2 * 4096]i32 = undefined;
    place(&hw, .random, prng.random());
    for ([_]i32{ 0, 1, 239, top }) |p| for (0..2) |axis| {
        var seen = false;
        for (0..4) |row| seen = seen or hw[2 * row + axis] == p;
        try std.testing.expect(seen);
    };
    for (hw) |p| try std.testing.expect(p >= 0 and p <= top);
    var row: usize = special.len;
    while (row + 4 <= 4096) : (row += 4) {
        const h = hw[2 * row];
        const w = hw[2 * row + 1];
        try std.testing.expect(@mod(h, 2) == 0 and @mod(w, 2) == 0);
        try std.testing.expectEqualSlices(i32, &.{ h, w, h, w + 1, h + 1, w, h + 1, w + 1 }, hw[2 * row ..][0..8]);
    }
}

test "patch fill: every bf16 pattern once, ties halfway between two bf16 values, stress sums exact only in fp32" {
    var prng = std.Random.DefaultPrng.init(11);
    const n = hidden;
    const rows = 196;
    const acc = try std.testing.allocator.alloc(f32, rows * n);
    defer std.testing.allocator.free(acc);
    var b: [n]u16 = undefined;
    biasFill(acc, &b, n, .random, prng.random());
    const seen = try std.testing.allocator.alloc(bool, 65536);
    defer std.testing.allocator.free(seen);
    @memset(seen, false);
    var ties: usize = 0;
    for (acc, 0..) |v, i| switch (i % n % 4) {
        0 => seen[@as(u32, @bitCast(v)) >> 16] = true,
        1 => {
            const s = @as(f64, v) + bfr.value(b[i % n]);
            ties += @intFromBool(@abs(s - b16(s)) == std.math.ldexp(@as(f64, 1), std.math.frexp(s).exponent - 9));
        },
        else => {},
    };
    for (seen) |s| try std.testing.expect(s);
    try std.testing.expect(ties > rows * n / 4 / 2);
    biasFill(acc[0 .. 4 * n], &b, n, .stress, prng.random());
    const terms = [2]f64{ acc[5], bfr.value(b[5]) };
    try std.testing.expectEqual(terms[0] + terms[1], rows_.sumIn(.f32, .sequential, &terms));
    try std.testing.expect(@abs(terms[0] + terms[1]) < 0x1p20 and std.math.isNan(rows_.sumIn(.f16, .sequential, &terms)));
    try std.testing.expectEqual(terms[0] + terms[1], rows_.sumIn(.bf16, .sequential, &terms));
}

test "served shapes are the tower's: 12 heads of 128 per q, k and v, 1,536 patch outputs, rows to 19,520" {
    for (rope_served) |s| try std.testing.expect(s.n == heads and s.k == 2);
    for (bias_served) |s| try std.testing.expect(s.n == hidden and s.k == 2);
    try std.testing.expectEqual(@as(usize, 4608), width);
    try std.testing.expectEqual(@as(usize, 19520), rope_served[3].rows);
}
