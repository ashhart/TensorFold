//! Optional target-only Q4 FP8 prompts. Drafter, decode and verification math remain unchanged.
const std = @import("std");
const cuda = @import("cuda");
pub const plan = @import("prefill8_plan.zig");

pub const Kernels = struct {
    modules: [2]cuda.Module,
    prepare_fn: cuda.Function,
    matmul_fn: cuda.Function,
    matmul_w8_fn: cuda.Function,
    expand_w8_fn: cuda.Function,

    pub fn init(d: *const cuda.Driver) !Kernels {
        var first = try cuda.Module.load(d, cuda.kernels.qmm_prefill8);
        errdefer first.unload();
        var second = try cuda.Module.load(d, cuda.kernels.qwen35_prefill8);
        errdefer second.unload();
        const mm = try first.function("tf_qmm_prefill8");
        const mm_w8 = try first.function("tf_qmm_prefill8_w8");
        const expand_w8 = try first.function("tf_qmm_prefill8_expand_w8");
        const prep = try second.function("tf_q35_prefill8_prepare");
        try mm.allowDynamicShared(plan.matmul_shared);
        try mm_w8.allowDynamicShared(matmul_w8_shared);
        try prep.allowDynamicShared(plan.prepareShared(plan.max_columns));
        return .{ .modules = .{ first, second }, .prepare_fn = prep, .matmul_fn = mm, .matmul_w8_fn = mm_w8, .expand_w8_fn = expand_w8 };
    }

    pub fn deinit(k: *Kernels) void {
        for (&k.modules) |*m| m.unload();
    }

    pub fn prepare(k: *const Kernels, stream: cuda.Stream, x: u64, y: u64, w: u64, q: Rows, rows: usize, columns: usize, mode: Mode, eps: f32) !void {
        try plan.validate(rows, columns);
        var args: cuda.Args = .{};
        for ([_]u64{ x, y, w, q.values, q.sums, q.scales }) |v| args.add(v);
        args.add(@as(c_int, @intCast(columns)));
        args.add(@as(c_int, @backingInt(mode)));
        args.add(eps);
        try cuda.launch.launch(k.prepare_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 }, .shared = plan.prepareShared(columns) }, stream, &args);
    }

    pub fn matmul(k: *const Kernels, stream: cuda.Stream, x: Rows, w: cuda.qlinear.Affine4, out: u64, rows: usize) !void {
        try plan.validate(rows, w.k);
        const tiles = (rows + 127) / 128;
        const group = @max(1, @min(tiles, (12 << 20) / (128 * w.k)));
        const expanded = w.w8 != 0;
        var args: cuda.Args = .{};
        for ([_]u64{ x.values, x.sums, x.scales, if (expanded) w.w8 else w.w, w.s, w.b, out }) |v| args.add(v);
        for ([_]usize{ rows, w.n, w.k, w.npad, group }) |v| args.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(if (expanded) k.matmul_w8_fn else k.matmul_fn, .{ .grid = .{ .x = @intCast(tiles * ((w.n + 127) / 128)) }, .block = .{ .x = 128 }, .shared = if (expanded) matmul_w8_shared else plan.matmul_shared }, stream, &args);
    }

    pub fn expandW8(k: *const Kernels, stream: cuda.Stream, weight: cuda.qlinear.Affine4) !void {
        if (weight.w == 0 or weight.w8 == 0 or weight.k == 0 or weight.npad == 0) return error.InvalidFp8PrefillW8Shape;
        const elements = std.math.mul(usize, weight.npad, weight.k) catch return error.InvalidFp8PrefillW8Shape;
        if (elements % 8 != 0) return error.InvalidFp8PrefillW8Shape;
        const words = elements / 8;
        if (words == 0 or words > std.math.maxInt(c_int)) return error.InvalidFp8PrefillW8Shape;
        var args: cuda.Args = .{};
        args.add(weight.w);
        args.add(weight.w8);
        args.add(@as(c_int, @intCast(words)));
        try cuda.launch.launch(k.expand_w8_fn, .{ .grid = .{ .x = @intCast((words + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
    }
};

pub const matmul_w8_shared: u32 = 49_920;
pub const Mode = enum(c_int) { norm = 0, swiglu = 1, gated_norm = 2, attention_gate = 3, quantize = 4 };
pub const Rows = struct { values: u64 = 0, sums: u64 = 0, scales: u64 = 0 };

pub const State = struct {
    on: bool = false,
    kernels: ?Kernels = null,
    scratch: ?cuda.DeviceBuffer = null,
    weights8: ?cuda.DeviceBuffer = null,
    rows: Rows = .{},

    pub fn enable(s: *State, e: anytype, sm: u32) !void {
        const g = e.g;
        if (!plan.supported(sm, g.hidden, g.intermediate, g.layers, g.moe(), g.bits.mixed(g.layers), g.kinds.modelopt, g.exl3, g.pocket))
            return error.UnsupportedFp8Prefill;
        const columns = @max(g.intermediate, g.valueWidth(), g.attnWidth(), g.hidden);
        const sizes = try plan.sizes(@import("cuda_round.zig").max_rows, columns);
        s.scratch = try cuda.DeviceBuffer.alloc(e.d, try plan.bytes(true, @import("cuda_round.zig").max_rows, columns));
        errdefer {
            s.scratch.?.free();
            s.scratch = null;
        }
        var at: usize = 0;
        s.rows.values = s.scratch.?.ptr + at;
        at += std.mem.alignForward(usize, sizes[0], 256);
        s.rows.sums = s.scratch.?.ptr + at;
        at += std.mem.alignForward(usize, sizes[1], 256);
        s.rows.scales = s.scratch.?.ptr + at;
        s.kernels = try Kernels.init(e.d);
        errdefer {
            s.kernels.?.deinit();
            s.kernels = null;
        }
        try s.expandGateUp(e);
        s.on = true;
    }

    /// Keep exact e4m3 copies for dense MLP gate/up prompts while decode and verification retain the packed Q4 words.
    fn expandGateUp(s: *State, e: anytype) !void {
        var bytes: usize = 0;
        for (e.layers) |*layer| switch (layer.mlp) {
            .dense => |*mlp| inline for (.{ "gate", "up" }) |name| switch (@field(mlp, name)) {
                .a4 => |q| bytes = std.math.add(usize, bytes, expandedBytes(q.npad, q.k)) catch return error.InvalidFp8PrefillW8Shape,
                else => return error.UnsupportedFp8Prefill,
            },
            else => return error.UnsupportedFp8Prefill,
        };
        s.weights8 = try cuda.DeviceBuffer.alloc(e.d, bytes);
        errdefer {
            s.weights8.?.free();
            s.weights8 = null;
        }
        var at: usize = 0;
        const kernels = if (s.kernels) |*v| v else return error.Fp8PrefillNotInitialized;
        for (e.layers) |*layer| switch (layer.mlp) {
            .dense => |*mlp| inline for (.{ "gate", "up" }) |name| switch (@field(mlp, name)) {
                .a4 => |*q| {
                    q.w8 = s.weights8.?.ptr + at;
                    at += expandedBytes(q.npad, q.k);
                    try kernels.expandW8(e.stream, q.*);
                },
                else => unreachable,
            },
            else => unreachable,
        };
        std.debug.assert(at == bytes);
    }

    pub fn deinit(s: *State) void {
        if (s.kernels) |*k| k.deinit();
        s.kernels = null;
        if (s.weights8) |*b| b.free();
        s.weights8 = null;
        if (s.scratch) |*b| b.free();
        s.scratch = null;
        s.rows = .{};
        s.on = false;
    }

    pub fn prepare(s: *State, e: anytype, x: u64, y: u64, w: u64, rows: usize, columns: usize, mode: Mode) !void {
        const k = if (s.kernels) |*v| v else return error.Fp8PrefillNotInitialized;
        try k.prepare(e.stream, x, y, w, s.rows, rows, columns, mode, e.g.eps);
    }

    pub fn project(s: *State, e: anytype, l: anytype, out: u64, rows: usize) !void {
        const k = if (s.kernels) |*v| v else return error.Fp8PrefillNotInitialized;
        switch (l) {
            .a4 => |q| try k.matmul(e.stream, s.rows, q, out, rows),
            else => return error.UnsupportedFp8Prefill,
        }
    }
};

fn expandedBytes(npad: usize, k: usize) usize {
    return std.mem.alignForward(usize, npad * k, 256);
}

test "FP8 prefill: actual driver launch ABI and invalid-shape no-launch gate" {
    const Fake = struct {
        var calls: usize = 0;
        var grid: u32 = 0;
        var block: u32 = 0;
        var shared: u32 = 0;
        var function: usize = 0;
        var pointers: [7]u64 = @splat(0);
        var ints: [5]c_int = @splat(0);
        fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, smem: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.debug.assert(gy == 1 and gz == 1 and by == 1 and bz == 1);
            calls += 1;
            grid = gx;
            block = bx;
            shared = smem;
            function = @intFromPtr(f);
            const p = params.?;
            const np: usize = if (@intFromPtr(f) == 1) 6 else 7;
            const ni: usize = if (np == 6) 2 else 5;
            for (0..np) |i| pointers[i] = @as(*align(1) const u64, @ptrCast(p[i].?)).*;
            for (0..ni) |i| ints[i] = @as(*align(1) const c_int, @ptrCast(p[np + i].?)).*;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const kernels: Kernels = .{ .modules = undefined, .prepare_fn = .{ .d = &d, .handle = @ptrFromInt(1) }, .matmul_fn = .{ .d = &d, .handle = @ptrFromInt(2) }, .matmul_w8_fn = .{ .d = &d, .handle = @ptrFromInt(3) }, .expand_w8_fn = .{ .d = &d, .handle = @ptrFromInt(4) } };
    const rows: Rows = .{ .values = 11, .sums = 22, .scales = 33 };
    Fake.calls = 0;
    try kernels.prepare(stream, 1, 2, 3, rows, 17, 5120, .norm, 1e-6);
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3, 11, 22, 33 }, Fake.pointers[0..6]);
    try std.testing.expectEqualSlices(c_int, &.{ 5120, 0 }, Fake.ints[0..2]);
    try std.testing.expectEqual(@as(u32, 17), Fake.grid);
    try std.testing.expectEqual(@as(u32, 256), Fake.block);
    try std.testing.expectEqual(@as(u32, 21504), Fake.shared);
    try kernels.matmul(stream, rows, .{ .w = 44, .s = 55, .b = 66, .n = 129, .k = 17408, .npad = 256 }, 77, 128);
    try std.testing.expectEqualSlices(u64, &.{ 11, 22, 33, 44, 55, 66, 77 }, &Fake.pointers);
    try std.testing.expectEqualSlices(c_int, &.{ 128, 129, 17408, 256, 1 }, &Fake.ints);
    try std.testing.expectEqual(@as(u32, 2), Fake.grid);
    try std.testing.expectEqual(@as(u32, 128), Fake.block);
    try std.testing.expectEqual(@as(u32, 37632), Fake.shared);
    try std.testing.expectEqual(@as(usize, 2), Fake.function);
    try kernels.matmul(stream, rows, .{ .w = 44, .w8 = 88, .s = 55, .b = 66, .n = 129, .k = 17408, .npad = 256 }, 77, 128);
    try std.testing.expectEqual(@as(u64, 88), Fake.pointers[3]);
    try std.testing.expectEqual(@as(u32, matmul_w8_shared), Fake.shared);
    try std.testing.expectEqual(@as(usize, 3), Fake.function);
    try std.testing.expectError(error.InvalidFp8PrefillShape, kernels.prepare(stream, 1, 2, 3, rows, 4097, 5120, .norm, 1e-6));
    try std.testing.expectEqual(@as(usize, 3), Fake.calls);
}

test "FP8 prefill: W8 expansion launch covers every packed Q4 word" {
    const Fake = struct {
        var calls: usize = 0;
        var grid: u32 = 0;
        var block: u32 = 0;
        var input: u64 = 0;
        var output: u64 = 0;
        var words: c_int = 0;

        fn launch(_: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, smem: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.debug.assert(gy == 1 and gz == 1 and by == 1 and bz == 1);
            std.debug.assert(smem == 0);
            calls += 1;
            grid = gx;
            block = bx;
            const p = params.?;
            input = @as(*align(1) const u64, @ptrCast(p[0].?)).*;
            output = @as(*align(1) const u64, @ptrCast(p[1].?)).*;
            words = @as(*align(1) const c_int, @ptrCast(p[2].?)).*;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const kernels: Kernels = .{ .modules = undefined, .prepare_fn = undefined, .matmul_fn = undefined, .matmul_w8_fn = undefined, .expand_w8_fn = .{ .d = &d, .handle = @ptrFromInt(4) } };
    try kernels.expandW8(stream, .{ .w = 41, .w8 = 42, .s = 51, .b = 61, .n = 129, .k = 5120, .npad = 256 });
    try std.testing.expectEqual(@as(usize, 1), Fake.calls);
    try std.testing.expectEqual(@as(u64, 41), Fake.input);
    try std.testing.expectEqual(@as(u64, 42), Fake.output);
    try std.testing.expectEqual(@as(c_int, 256 * 5120 / 8), Fake.words);
    try std.testing.expectEqual(@as(u32, (256 * 5120 / 8 + 255) / 256), Fake.grid);
    try std.testing.expectEqual(@as(u32, 256), Fake.block);
    try std.testing.expectError(error.InvalidFp8PrefillW8Shape, kernels.expandW8(stream, .{ .w = 41, .s = 51, .b = 61, .n = 129, .k = 5120, .npad = 256 }));
}

test "FP8 prefill: gate/up W8 storage cost is explicit" {
    try std.testing.expectEqual(@as(usize, 89_128_960), expandedBytes(17_408, 5_120));
    try std.testing.expectEqual(@as(usize, 11_408_506_880), 64 * 2 * expandedBytes(17_408, 5_120));
}
