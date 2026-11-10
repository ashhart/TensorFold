//! Raw and prepared TMA products retain the native FP4 reference and the two BF16 SwiGLU boundaries.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const base = @import("contract_nvfp4_prompt.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Tma = std.meta.Child(@FieldType(cuda.nvfp4.native_prompt.Native, "tma"));
pub const Fault = enum { none, guard, missing_tail, dropped_nonfinite, narrow_sum, wrong_branch };
pub const Config = struct { prepared: bool, rows: usize, out: cuda.qmmf.Out = .bf16, fused: bool = false, fault: Fault = .none };
pub const entries = [_]Config{
    .{ .prepared = false, .rows = 128 },                                            .{ .prepared = false, .rows = 128, .out = .f32 },
    .{ .prepared = false, .rows = 256 },                                            .{ .prepared = false, .rows = 256, .out = .f32 },
    .{ .prepared = true, .rows = 128 },                                             .{ .prepared = true, .rows = 128, .out = .f32 },
    .{ .prepared = true, .rows = 256 },                                             .{ .prepared = true, .rows = 256, .out = .f32 },
    .{ .prepared = false, .rows = 128, .fused = true },                             .{ .prepared = false, .rows = 256, .fused = true },
    .{ .prepared = true, .rows = 128, .fused = true },                              .{ .prepared = true, .rows = 256, .fused = true },
    .{ .prepared = false, .rows = 128, .fault = .guard },                           .{ .prepared = true, .rows = 256, .out = .f32, .fault = .missing_tail },
    .{ .prepared = true, .rows = 128, .fused = true, .fault = .dropped_nonfinite }, .{ .prepared = true, .rows = 256, .out = .f32, .fault = .narrow_sum },
    .{ .prepared = false, .rows = 128, .fused = true, .fault = .narrow_sum },       .{ .prepared = true, .rows = 256, .fused = true, .fault = .wrong_branch },
};
const served = [_]c.Shape{
    .{ .rows = 129, .n = 129, .k = 256 },  .{ .rows = 255, .n = 65, .k = 512 },
    .{ .rows = 256, .n = 128, .k = 768 },  .{ .rows = 257, .n = 127, .k = 256 },
    .{ .rows = 4096, .n = 1, .k = 256 },   .{ .rows = 3, .n = 3, .k = 17408 },
    .{ .rows = 1, .n = 17408, .k = 5120 }, .{ .rows = 1, .n = 5120, .k = 17408 },
};

fn label(comptime prepared: bool, comptime rows: usize, config: Config) []const u8 {
    const prefix = "cuda nvfp4 TMA " ++ (if (prepared) "prepared " else "raw ") ++ std.fmt.comptimePrint("{d}", .{rows});
    return switch (config.fault) {
        .none => if (config.fused) prefix ++ " SwiGLU" else if (config.out == .f32) prefix ++ " f32" else prefix ++ " bf16",
        .guard => prefix ++ " CONTROL guard write",
        .missing_tail => prefix ++ " CONTROL omitted tail",
        .dropped_nonfinite => prefix ++ " CONTROL erased nonfinite",
        .narrow_sum => if (config.fused) prefix ++ " CONTROL narrow SwiGLU" else prefix ++ " CONTROL narrow product",
        .wrong_branch => prefix ++ " CONTROL duplicated gate branch",
    };
}

pub fn spec(config: Config) c.Spec {
    var out = base.spec(.{ .out = config.out });
    out.name = if (config.prepared)
        (if (config.rows == 128) label(true, 128, config) else label(true, 256, config))
    else
        (if (config.rows == 128) label(false, 128, config) else label(false, 256, config));
    out.tiles = .{ .{ .size = config.rows }, .{ .size = if (config.fused) 64 else 128 }, .{ .size = 256, .step = 256, .min = 256 } };
    out.served = &served;
    if (config.fused) {
        out.ops = 6;
        out.replaces = "two native FP4 BF16 projections and tf_q35_mlp_act";
    }
    return out;
}

pub const Cases = struct {
    items: [entries.len]Case = undefined,
    initialized: usize = 0,
    pub fn append(self: *Cases, list: *std.ArrayList(g.Kernel), gpu: @import("check.zig").Gpu, stream: cuda.Stream, sm: u32) !void {
        if (!cuda.nvfp4.native_prompt.supported(sm)) return;
        for (&self.items, entries) |*item, config| {
            item.* = .{ .base = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = .{ .out = config.out } }, .config = config };
            self.initialized += 1;
            try list.append(gpu.gpa, item.kernel());
        }
    }
    pub fn deinit(self: *Cases) void {
        for (self.items[0..self.initialized]) |*item| item.deinit();
    }
};

pub fn run(gpu: @import("check.zig").Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var cases: Cases = .{};
    defer cases.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try cases.append(&kernels, gpu, stream, try gpu.ctx.capability());
    if (kernels.items.len == 0) return error.UnsupportedNativeNvfp4Prefill;
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x4f544d41, &tally);
    const limits = try checkLimits(gpu);
    std.debug.print("SUMMARY NVFP4 TMA: {d} kernels pass, {d} fail, {d} controls caught, {d} missed, {d} limit problems\n", .{ tally.passed, tally.failed, tally.caught, tally.missed, limits });
    if (tally.problems() + limits != 0) return error.TestFailed;
}

fn sharedBytes(rows: usize) u32 {
    return @intCast(3 * (rows * 80 + 128 * 72) + 48);
}

fn checkLimits(gpu: @import("check.zig").Gpu) !usize {
    const registry = core.contract.registry;
    const sm = try gpu.ctx.capability();
    const index: usize = if (sm == 120) 3 else if (sm == 121) 4 else return error.UnsupportedNativeNvfp4Prefill;
    const release = try core.contract.cubin.releaseFor(gpu.gpa, cuda.kernels.nvfp4_tma, sm);
    var tma = try Tma.load(gpu.d);
    defer tma.unload();
    var problems: usize = 0;
    for (registry.all_cuda) |record| {
        if (!std.mem.eql(u8, record.launch.image, "nvfp4_tma")) continue;
        const symbol = try gpu.gpa.dupeSentinel(u8, record.launch.symbol, 0);
        defer gpu.gpa.free(symbol);
        const f = try tma.mod.function(symbol);
        const regs = try f.attribute(.num_regs);
        const shared = try f.attribute(.shared_size_bytes);
        const threads = try f.attribute(.max_threads_per_block);
        const local = try f.attribute(.local_size_bytes);
        const dynamic = try f.attribute(.max_dynamic_shared_size_bytes);
        const expected = registry.usageFor(record, release)[index];
        const ok = regs == expected.regs and shared == expected.static and threads >= record.launch.threads and dynamic >= record.launch.dynamic;
        problems += @intFromBool(!ok);
        std.debug.print("{s} NVFP4 TMA limits {s}: regs={d} static={d} local={d} dynamic={d} threads={d}\n", .{ if (ok) "PASS" else "FAIL", record.launch.symbol, regs, shared, local, record.launch.dynamic, threads });
    }
    return problems;
}

pub const Case = struct {
    base: base.Case,
    config: Config,
    rig: *Rig = undefined,
    tma: ?Tma = null,
    activation_module: ?cuda.Module = null,
    activation: ?cuda.Function = null,
    out: u64 = 0,
    up: cuda.qmmf.Weight = undefined,
    prepared: [2]u64 = .{ 0, 0 },
    up_out: u64 = 0,
    baseline: u64 = 0,
    host_output: []u8 = &.{},
    host_baseline: []u8 = &.{},

    fn clear(self: *Case) void {
        const gpu = self.base.rig.gpu;
        const stream = self.base.rig.s;
        stream.synchronize() catch {};
        if (self.tma) |*t| t.unload();
        if (self.activation_module) |*m| m.unload();
        self.tma = null;
        self.activation_module = null;
        self.activation = null;
        self.base.deinit();
        self.base = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = .{ .out = self.config.out } };
        self.rig = &self.base.rig;
    }
    pub fn deinit(self: *Case) void {
        self.clear();
    }
    fn release(ctx: *anyopaque) void {
        const self: *Case = @ptrCast(@alignCast(ctx));
        self.clear();
    }
    fn output(ctx: *anyopaque) ![]const u8 {
        const self: *Case = @ptrCast(@alignCast(ctx));
        @memcpy(self.host_output, try self.rig.bytes(self.out, self.host_output.len));
        return self.host_output;
    }
    const vt = blk: {
        var hooks = g.hooks(Case);
        hooks.output = output;
        hooks.release = release;
        break :blk hooks;
    };
    pub fn kernel(self: *Case) g.Kernel {
        self.rig = &self.base.rig;
        return .{ .spec = spec(self.config), .ctx = self, .vt = &vt, .control = switch (self.config.fault) {
            .none => null,
            .guard => .repeat,
            .missing_tail, .wrong_branch => .shapes,
            .dropped_nonfinite => .nonfinite,
            .narrow_sum => .sums,
        } };
    }
    pub fn count(self: *const Case) usize {
        return self.base.count();
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 256 != 0) return false;
        self.clear();
        self.base.config.fault = if (self.config.fault == .dropped_nonfinite) .dropped_nonfinite else .none;
        if (!try self.base.prepare(shape, fill, seed)) return false;
        errdefer self.clear();
        // The oracle stays the original native path even when the caller enables the new fast path.
        if (self.base.native.?.tma) |*t| t.unload();
        self.base.native.?.tma = null;
        self.tma = try Tma.load(self.rig.gpu.d);
        const a = self.base.arena.?.allocator();
        self.host_output = try a.alloc(u8, self.count() * c.size(spec(self.config).out));
        self.host_baseline = try a.alloc(u8, self.host_output.len);
        self.out = try self.rig.output(self.host_output.len);
        const w = self.base.weight;
        if (self.config.prepared) self.prepared[0] = try self.rig.scratch(@as(usize, w.npad) * w.k / 2);
        if (self.config.fused) {
            const codes = try a.dupe(u8, self.base.codes);
            for (codes) |*byte| byte.* ^= 0x88;
            const words = try a.alloc(u32, @as(usize, w.npad) * w.k / 8);
            cuda.nvfp4.packWords(words, codes, w.n, w.k);
            self.up = w;
            self.up.codes = try self.rig.input(std.mem.sliceAsBytes(words));
            self.up.scale *= 2;
            self.up_out = try self.rig.scratch(self.count() * 2);
            self.baseline = try self.rig.scratch(std.mem.alignForward(usize, self.count(), 4) * 2);
            if (self.config.prepared) self.prepared[1] = try self.rig.scratch(@as(usize, w.npad) * w.k / 2);
            self.activation_module = try cuda.Module.load(self.rig.gpu.d, cuda.kernels.qwen35);
            self.activation = try self.activation_module.?.function("tf_q35_mlp_act");
        }
        return true;
    }

    fn launch(self: *Case, rows: usize, weight: cuda.qmmf.Weight, up: cuda.qmmf.Weight) !void {
        const t = &self.tma.?;
        const x = self.base.native.?.scratch.ptr;
        const xs = x + self.base.shape.rows * self.base.shape.k / 2;
        const bm = self.config.rows;
        var args: cuda.launch.Args = .{};
        if (self.config.fused) {
            args.add(try t.map.pair(x, xs, .{ weight.codes, up.codes }, .{ weight.scales, up.scales }, rows, weight.npad, weight.k, bm));
        } else args.add(try t.map.single(x, xs, weight.codes, weight.scales, rows, weight.npad, weight.k, bm));
        args.add(self.out);
        for ([_]usize{ rows, weight.n, weight.k }) |v| args.add(@as(i32, @intCast(v)));
        args.add(@as(f32, weight.act * weight.scale));
        if (self.config.fused) args.add(@as(f32, up.act * up.scale));
        const tile = @intFromBool(bm == 256);
        const f = if (self.config.fused)
            (if (self.config.prepared) t.packed_pair[tile] else t.pair[tile])
        else if (self.config.prepared) t.packed_single[tile][@intFromBool(self.config.out == .f32)] else t.single[tile][@intFromBool(self.config.out == .f32)];
        const bn: usize = if (self.config.fused) 64 else 128;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((weight.n + bn - 1) / bn), .y = @intCast((rows + bm - 1) / bm) }, .block = .{ .x = 288 }, .shared = sharedBytes(bm) }, self.rig.s, &args);
    }

    pub fn run(self: *Case) !void {
        try self.base.run();
        var w = self.base.weight;
        var up = if (self.config.fused) self.up else w;
        if (self.config.fused) {
            try self.base.native.?.matmul(self.rig.s, self.base.dx, w.k + 64, self.base.shape.rows, up, self.up_out, .bf16);
            var args: cuda.launch.Args = .{};
            args.add(self.base.out);
            args.add(self.up_out);
            args.add(self.baseline);
            args.add(@as(i64, @intCast(self.count())));
            args.add(@as(u64, 0));
            args.add(@as(i32, @intCast(w.n)));
            args.add(@as(u64, 0));
            try cuda.launch.launch(self.activation.?, .{ .grid = .{ .x = @intCast((self.count() + 1023) / 1024) }, .block = .{ .x = 256 } }, self.rig.s, &args);
        }
        if (self.config.prepared) {
            try self.tma.?.repack(self.rig.s, w.codes, self.prepared[0], @as(usize, w.npad) * w.k / 2);
            w.codes = self.prepared[0];
            if (self.config.fused) {
                try self.tma.?.repack(self.rig.s, up.codes, self.prepared[1], @as(usize, up.npad) * up.k / 2);
                up.codes = self.prepared[1];
            }
        }
        if (self.config.fault == .wrong_branch) up = w;
        const rows = self.base.shape.rows - @as(usize, @intFromBool(self.config.fault == .missing_tail and self.base.shape.rows > 1));
        try self.launch(rows, w, up);
        try self.rig.s.synchronize();
        if (self.config.fault == .guard) {
            const target: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out - 1, .len = 1, .borrowed = true };
            try target.upload(0, &.{0});
        }
        if (self.config.fault == .narrow_sum) {
            const a = self.base.arena.?.allocator();
            const values = try a.alloc(f64, self.count());
            try self.simulate(.bf16, .sequential, values);
            const bytes = try a.alloc(u8, self.host_output.len);
            for (values, 0..) |value, i| if (self.config.out == .f32) {
                std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(value))), .little);
            } else std.mem.writeInt(u16, bytes[i * 2 ..][0..2], c.toBf16(@floatCast(value)), .little);
            const target: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out, .len = bytes.len, .borrowed = true };
            try target.upload(0, bytes);
        }
    }

    pub fn reference(self: *Case, want: []f64, magnitude: []f64) !void {
        try self.base.reference(want, magnitude);
        if (self.config.fused) for (want, magnitude) |*value, *mag| {
            value.* = fused(value.*);
            mag.* = @abs(value.*);
        };
    }
    pub fn simulate(self: *Case, accum: c.Accum, order: g.Order, values: []f64) !void {
        try self.base.simulate(accum, order, values);
        if (self.config.fused) for (values) |*value| {
            value.* = fused(value.*);
        };
    }
    pub fn replaced(self: *Case) ![]const u8 {
        const a = self.base.arena.?.allocator();
        const want = try a.alloc(f64, self.count());
        const magnitude = try a.alloc(f64, self.count());
        try self.base.reference(want, magnitude);
        for ([_]u64{ self.base.out, self.up_out }, [_]f64{ 1, -2 }) |ptr, scale| {
            const actual = try self.rig.bytes(ptr, self.count() * 2);
            for (want, magnitude, 0..) |value, mag, i| {
                const reference_value = value * scale;
                const got = c.read(.bf16, actual, i);
                const allowed = c.bound(base.spec(.{ .out = .bf16 }), self.base.shape, reference_value, mag * @abs(scale));
                if (!(got == reference_value or @abs(got - reference_value) <= allowed)) return error.NativeFp4OracleOutsideFp64Bound;
            }
        }
        @memcpy(self.host_baseline, try self.rig.bytes(self.baseline, self.host_baseline.len));
        return self.host_baseline;
    }
};

fn fused(value: f64) f64 {
    const gate: f64 = c.fromBf16(c.toBf16(@floatCast(value)));
    const up: f64 = c.fromBf16(c.toBf16(@floatCast(-2 * value)));
    return gate / (1 + @exp(-gate)) * up;
}

test "NVFP4 TMA variants cover raw prepared tiles outputs and fused paths" {
    var singles: [2][2][2]bool = std.mem.zeroes([2][2][2]bool);
    var pairs: [2][2]bool = std.mem.zeroes([2][2]bool);
    for (entries) |config| {
        if (config.fault != .none) continue;
        const prepared = @intFromBool(config.prepared);
        const tile = @intFromBool(config.rows == 256);
        if (config.fused) pairs[prepared][tile] = true else singles[prepared][tile][@intFromBool(config.out == .f32)] = true;
        try std.testing.expectEqual(@as(f64, 100), spec(config).sums.f16);
        try std.testing.expectEqual(@as(f64, 100), spec(config).sums.bf16);
    }
    for (singles) |tiles| for (tiles) |outputs| for (outputs) |covered| try std.testing.expect(covered);
    for (pairs) |tiles| for (tiles) |covered| try std.testing.expect(covered);
    try std.testing.expectEqual(@as(u32, 58416), sharedBytes(128));
    try std.testing.expectEqual(@as(u32, 89136), sharedBytes(256));
}

test "NVFP4 TMA fused witness retains each BF16 projection rounding" {
    const witness: f64 = 0.999;
    const rounded = fused(witness);
    const unrounded = witness / (1 + @exp(-witness)) * (-2 * witness);
    try std.testing.expect(@abs(rounded - unrounded) > 0.0001);
    try std.testing.expect(fused(16) < -511.99);
    try std.testing.expect(std.math.isNan(fused(std.math.nan(f64))));
}

test "NVFP4 TMA actual FP4 stress products expose half and bf16 accumulation in every order" {
    const products = try std.testing.allocator.alloc(f64, 17408);
    defer std.testing.allocator.free(products);
    for (products, 0..) |*value, i| value.* = if (i < products.len / 2) 768 else -768;
    products[0] = 64;
    products[products.len / 2] = -48;
    for ([_]bool{ false, true }) |is_fused| {
        const s = spec(.{ .prepared = true, .rows = 256, .fused = is_fused });
        const want = if (is_fused) fused(16) else 16;
        const allowance = c.boundExact(s, want);
        for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
            const sum = core.contract.rows.sumIn(.f32, order, products);
            try std.testing.expectEqual(@as(f64, 16), sum);
            for ([_]c.Accum{ .f16, .bf16 }) |accum| {
                const narrow = core.contract.rows.sumIn(accum, order, products);
                const value = if (is_fused) fused(narrow) else narrow;
                try std.testing.expect(!std.math.isFinite(value) or @abs(value - want) >= 100 * allowance);
            }
        }
    }
}
