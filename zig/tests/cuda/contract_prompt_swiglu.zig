//! prompt_lean's gate/up SwiGLU pair: byte-equal to two lean products then tf_q35_mlp_act, with planted controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const gpu_check = @import("check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const nvfp4 = cuda.nvfp4;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// The 27B's MLP at 2048 and 4096 rows, a partial row tile, one row, 17 rows of 17 groups, 300 rows of 9 groups.
const served = [_]c.Shape{
    .{ .rows = 2048, .n = 17408, .k = 5120 }, .{ .rows = 4096, .n = 17408, .k = 5120 },
    .{ .rows = 129, .n = 17408, .k = 5120 },  .{ .rows = 1, .n = 128, .k = 64 },
    .{ .rows = 17, .n = 256, .k = 1088 },     .{ .rows = 300, .n = 384, .k = 576 },
};
/// The device row count's shapes: each drops 37 rows, so a row tile ends inside the launch's last one.
const counted = [_]c.Shape{ .{ .rows = 2048, .n = 17408, .k = 5120 }, .{ .rows = 129, .n = 17408, .k = 5120 }, .{ .rows = 300, .n = 384, .k = 576 } };
/// Every bf16 bit pattern as a gate value (row r's gate is r's pattern), against 128 up values a row.
const sweep_shape = [_]c.Shape{.{ .rows = 65536, .n = 128, .k = 64 }};

const rule = "every output byte equals two prompt_lean products, then tf_q35_mlp_act, on the same rows and weights";

/// What a registration plants or covers: the served launch, a device row count, every gate value, or a control.
const Form = enum { served, device_rows, sweep, swapped, gate_ulp };

/// One FP4 weight's host fixture: e2m1 codes, e4m3 block scales, packed as the loader packs them.
const Weight = struct {
    codes: []u8,
    scales: []u8,

    fn init(a: std.mem.Allocator, n: usize, k: usize) !Weight {
        return .{ .codes = try a.alloc(u8, n * k / 2), .scales = try a.alloc(u8, n * k / 16) };
    }

    fn deinit(w: Weight, a: std.mem.Allocator) void {
        a.free(w.codes);
        a.free(w.scales);
    }

    /// Random codes and scales: magnitudes 1/8 ... 4, all nibbles; `stress` every nibble -6 at the largest scale.
    fn fill(w: Weight, rng: std.Random, fill_: g.Fill) void {
        for (w.codes) |*q| q.* = if (fill_ == .stress) 0xEE else rng.int(u8);
        for (w.scales) |*s| s.* = if (fill_ == .stress) 0x7E else 0x20 + rng.uintLessThan(u8, 0x29);
        if (fill_ == .nonfinite) w.scales[w.scales.len / 3] = 0x7F; // a NaN block scale
    }

    fn upload(w: Weight, a: std.mem.Allocator, rig: *rig_.Rig, scale: f32, n: usize, k: usize) !cuda.qmmf.Weight {
        const words = try a.alloc(u32, nvfp4.padded(n) / 64 * (k / nvfp4.group) * 512);
        defer a.free(words);
        nvfp4.packWords(words, w.codes, n, k);
        const tiles = try a.alloc(u8, nvfp4.padded(n) * (k / 16));
        defer a.free(tiles);
        nvfp4.packScales(tiles, w.scales, n, k);
        return nvfp4.weight(try rig.input(std.mem.sliceAsBytes(words)), try rig.input(tiles), scale, n, k);
    }
};

/// Sums near 1 ... 10 at K 5120 for these codes and scales, so silu's every branch is reached.
const gate_scale: f32 = 0.0123;
const up_scale: f32 = 0.0071;

pub const Product = struct {
    rig: rig_.Rig,
    prompt: *const nvfp4.Prompt,
    act: cuda.Function,
    form: Form = .served,
    gate: cuda.qmmf.Weight = undefined,
    up: cuda.qmmf.Weight = undefined,
    x: u64 = 0,
    rows_at: u64 = 0, // the device row count's table (P[2]), 0 for none
    rows: usize = 0,
    n: usize = 0,
    k: usize = 0,
    out: u64 = 0,
    base: [3]u64 = .{ 0, 0, 0 }, // the separate path's gate, up and act

    pub fn kernel(self: *Product) g.Kernel {
        const name = switch (self.form) {
            .served => "cuda prompt16 FP4 lean SwiGLU pair",
            .device_rows => "cuda prompt16 FP4 lean SwiGLU pair, device row count",
            .sweep => "cuda prompt16 FP4 lean SwiGLU pair, every bf16 gate value",
            .swapped => "control: lean SwiGLU pair with gate and up swapped",
            .gate_ulp => "control: lean SwiGLU pair with the gate's scale one fp32 ulp up",
        };
        const control = self.form == .swapped or self.form == .gate_ulp;
        return .{ .spec = .{
            .name = name,
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = if (control) null else "two prompt_lean products, then tf_q35_mlp_act",
            .nan = .{ .stated = rule },
            .served = switch (self.form) {
                .sweep => &sweep_shape,
                .device_rows => &counted,
                else => &served,
            },
        }, .ctx = self, .vt = &vt, .control = if (control) .shapes else null };
    }

    const vt = g.hooks(Product);

    pub fn deinit(self: *Product) void {
        self.rig.deinit();
    }

    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.rows == 0 or shape.k % 64 != 0 or shape.n % 128 != 0) return false;
        const a = self.rig.gpu.gpa;
        self.rows = shape.rows;
        self.n = shape.n;
        self.k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const x = try a.alloc(u16, shape.rows * shape.k);
        defer a.free(x);
        for (x, 0..) |*v, i| v.* = switch (fill) {
            .stress => c.toBf16(rng.float(f32) * 2.0e18 + 1.0e17),
            else => if (i % 29 == 0) 0 else c.toBf16(rng.float(f32) * 4 - 2),
        };
        if (fill == .nonfinite) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, r| {
            if (r < shape.rows) x[r * shape.k + (r * 7 + 3) % shape.k] = c.toBf16(v);
        };
        const wg = try Weight.init(a, shape.n, shape.k);
        defer wg.deinit(a);
        const wu = try Weight.init(a, shape.n, shape.k);
        defer wu.deinit(a);
        wg.fill(rng, fill);
        wu.fill(rng, fill);
        var gs = gate_scale;
        if (self.form == .sweep) { // gate = x[r, 0]: a 1.0 code at input 0, scale 1, else 0; up skips input 0
            for (x, 0..) |*v, i| if (i % shape.k == 0) {
                v.* = @intCast(i / shape.k);
            };
            @memset(wg.codes, 0);
            @memset(wg.scales, 0x38);
            for (0..shape.n) |j| {
                wg.codes[j * shape.k / 2] = 0x02;
                wu.codes[j * shape.k / 2] &= 0xF0;
            }
            gs = 1.0;
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        self.gate = try wg.upload(a, &self.rig, gs, shape.n, shape.k);
        self.up = try wu.upload(a, &self.rig, up_scale, shape.n, shape.k);
        self.rows_at = 0;
        if (self.form == .device_rows) {
            const table = [4]i32{ 0, 0, @intCast(shape.rows - 37), 0 }; // P[2]: fewer rows than the launch's
            self.rows_at = try self.rig.input(std.mem.sliceAsBytes(&table));
        }
        for (&self.base, [_]usize{ 2, 2, 2 }) |*b, size| b.* = try self.rig.scratch(shape.rows * shape.n * size);
        self.out = try self.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(self: *Product) !void {
        var gate = self.gate;
        var up = self.up;
        if (self.form == .swapped) std.mem.swap(cuda.qmmf.Weight, &gate, &up);
        if (self.form == .gate_ulp) gate.scale = std.math.nextAfter(f32, gate.scale, std.math.inf(f32));
        try self.prompt.swiglu(self.rig.s, self.x, self.k, self.rows, gate, up, self.out, self.rows_at);
        try self.rig.s.synchronize();
    }

    /// main's path on the same inputs: the gate and up products, then the act kernel into an output-poisoned buffer.
    pub fn replaced(self: *Product) ![]const u8 {
        try self.separate();
        return self.rig.bytes(self.base[2], self.rows * self.n * 2);
    }

    fn separate(self: *Product) !void {
        const s = self.rig.s;
        try self.prompt.matmulTile(s, self.x, self.k, self.rows, self.gate, self.base[0], .bf16, .lean);
        try self.prompt.matmulTile(s, self.x, self.k, self.rows, self.up, self.base[1], .bf16, .lean);
        const act: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.base[2], .len = self.rows * self.n * 2 };
        try act.fill8(0xff, s.handle); // as rig.output() poisons: rows past a device count stay unwritten in both
        const total: c_longlong = @intCast(self.rows * self.n);
        var a: cuda.Args = .{};
        a.add(self.base[0]);
        a.add(self.base[1]);
        a.add(self.base[2]);
        a.add(total);
        a.add(self.rows_at);
        a.add(@as(c_int, @intCast(self.n)));
        a.add(@as(u64, 0));
        const blocks: u32 = @intCast((self.rows * self.n + 1023) / 1024);
        try cuda.launch.launch(self.act, .{ .grid = .{ .x = blocks, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn check(self: *Product, _: g.Gate) !?[]const u8 {
        const got = try self.rig.gpu.gpa.dupe(u8, try self.rig.outputBytes());
        defer self.rig.gpu.gpa.free(got);
        const want = try self.replaced();
        return if (std.mem.eql(u8, got, want)) null else rule;
    }

    pub fn count(self: *Product) usize {
        return self.rows * self.n * 2;
    }

    pub fn reference(_: *Product, _: []f64, _: []f64) !void {}

    /// Mean ns of `reps` fused launches, or of the separate path (two products and the act kernel).
    fn time(self: *Product, fused: bool, reps: usize, io: std.Io) !i96 {
        try self.rig.s.synchronize();
        const start = gpu_check.now(io);
        for (0..reps) |_| {
            if (fused) try self.run() else try self.separate();
            try self.rig.s.synchronize();
        }
        return @divTrunc(gpu_check.now(io) - start, @as(i96, @intCast(reps)));
    }
};

/// The act kernel's module (qwen35) and the registrations over one prompt module.
pub const Registrations = struct {
    mod: cuda.Module,
    items: [5]Product,

    pub fn init(gpu: gpu_check.Gpu, s: cuda.Stream, prompt: *const nvfp4.Prompt) !Registrations {
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
        errdefer mod.unload();
        const act = try mod.function("tf_q35_mlp_act");
        var r: Registrations = .{ .mod = mod, .items = undefined };
        for (&r.items, std.enums.values(Form)) |*p, form| p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .prompt = prompt, .act = act, .form = form };
        return r;
    }

    pub fn deinit(self: *Registrations) void {
        for (&self.items) |*p| p.deinit();
        self.mod.unload();
    }

    pub fn append(self: *Registrations, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        for (&self.items) |*p| try kernels.append(a, p.kernel());
    }
};

/// The focused gate run (tf-cuda-test prompt-swiglu-contract); kernel-contract runs it after its registered gates.
pub fn run(gpu: gpu_check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prompt = try nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    var registrations = try Registrations.init(gpu, s, &prompt);
    defer registrations.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try registrations.append(gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x5719, &tally);
    std.debug.print("SUMMARY prompt_swiglu: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed,
    });
    try gpu_check.expect(tally.problems() == 0, "prompt_swiglu contract: {d} failures", .{tally.problems()});
}

/// Kernel-level A-B-A per shape: two lean products and the act kernel against the pair launch (not a served claim).
pub fn bench(gpu: gpu_check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prompt = try nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    var registrations = try Registrations.init(gpu, s, &prompt);
    defer registrations.deinit();
    const product = &registrations.items[0];
    const cases = [_]c.Shape{
        .{ .rows = 2048, .n = 17408, .k = 5120 },
        .{ .rows = 4096, .n = 17408, .k = 5120 },
        .{ .rows = 8192, .n = 17408, .k = 5120 },
    };
    for (cases, 0..) |shape, i| {
        _ = try product.prepare(shape, .random, 0x5719 + i);
        _ = try product.time(true, 2, gpu.io); // warm both paths once
        _ = try product.time(false, 2, gpu.io);
        const a1 = try product.time(false, 7, gpu.io);
        const b = try product.time(true, 7, gpu.io);
        const a2 = try product.time(false, 7, gpu.io);
        const ms = std.time.ns_per_ms;
        const base: f64 = @floatFromInt(@divTrunc(a1 + a2, 2));
        std.debug.print("BENCH prompt_swiglu rows={d}: A1={d:.3} ms B={d:.3} ms A2={d:.3} ms separate/pair={d:.4}x\n", .{
            shape.rows,                       @as(f64, @floatFromInt(a1)) / ms,  @as(f64, @floatFromInt(b)) / ms,
            @as(f64, @floatFromInt(a2)) / ms, base / @as(f64, @floatFromInt(b)),
        });
    }
}
