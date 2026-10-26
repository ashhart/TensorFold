//! Stage 1 Vulkan operators: cast, raw copy, gather and affine dequant, each validated before any dispatch.
const std = @import("std");
const memory = @import("memory.zig");
const device = @import("device.zig");
const commands = @import("commands.zig");
const dispatch = @import("dispatch.zig");
const params = @import("stage1_params");
const tensor = @import("tensor.zig");
const Device = device.Device;
const Commands = commands.Commands;
const Buffer = memory.Buffer;
const Error = dispatch.Error;

/// The compiled shader images Ops runs; every field is a real SPIR-V image of the named variant.
pub const Shaders = struct {
    qmm_bf16: []align(4) const u8,
    dequant_bf16: []align(4) const u8,
    gather_u32: []align(4) const u8,
    gather_u16: []align(4) const u8,
    cast_bf16_f32: []align(4) const u8,
    cast_f32_bf16: []align(4) const u8,
    copy_u32: []align(4) const u8,
    copy_u16: []align(4) const u8,
};

const dummy_bytes: u64 = 256;

pub fn cast32(v: u64) Error!u32 {
    return std.math.cast(u32, v) orelse error.TooLarge;
}

pub fn dispatchGroups(count: u32, local: u32) [3]u32 {
    return .{ @intCast((@as(u64, count) + local - 1) / local), 1, 1 };
}

/// gather_take geometry: one index array over a row-major table, window elements per row.
pub const GatherPlan = struct { window: u64, count: u32, ids: u32, rows: u32, stride: u32 };

/// The gather's derived geometry, or the named reason it cannot run; pure and host-tested.
pub fn gatherPlan(out_elements: u64, table_elements: u64, ids_elements: u64, rows: u64) Error!GatherPlan {
    if (ids_elements == 0 or rows == 0) return error.Invalid;
    if (out_elements != ids_elements *| (table_elements / rows)) return error.Invalid;
    const window = table_elements / rows;
    if (window * rows != table_elements or window == 0) return error.Invalid;
    return .{
        .window = window,
        .count = try cast32(out_elements),
        .ids = try cast32(ids_elements),
        .rows = try cast32(rows),
        .stride = try cast32(window),
    };
}

/// dequant geometry: rows of k elements, k/group packed words and scale groups per row.
pub const DequantPlan = struct { rows: u64, words: u64, groups: u32, count: u32, group: u32 };

/// The dequant's derived geometry, or the named reason it cannot run; pure and host-tested.
pub fn dequantPlan(out_elements: u64, w_elements: u64, scales_elements: u64, biases_elements: u64, k: u64, group: u64) Error!DequantPlan {
    if (k == 0 or group == 0 or k % group != 0 or group % 8 != 0) return error.Invalid;
    const words = k / 8;
    const groups = k / group;
    if (out_elements == 0 or out_elements % k != 0) return error.Invalid;
    const rows = out_elements / k;
    if (w_elements != rows * words or scales_elements != rows * groups or biases_elements != rows * groups) return error.Invalid;
    return .{
        .rows = rows,
        .words = words,
        .groups = try cast32(groups),
        .count = try cast32(rows * words),
        .group = try cast32(group),
    };
}

pub const Ops = struct {
    dev: *const Device,
    qmm_bf16: dispatch.Pipeline,
    dequant_bf16: dispatch.Pipeline,
    gather_u32: dispatch.Pipeline,
    gather_u16: dispatch.Pipeline,
    cast_bf16_f32: dispatch.Pipeline,
    cast_f32_bf16: dispatch.Pipeline,
    copy_u32: dispatch.Pipeline,
    copy_u16: dispatch.Pipeline,
    /// Read-only stand-in for declared bindings a kernel never touches (cast 1, copy 1 and 3).
    dummy: Buffer,

    /// Builds the eight pipelines; the device must have been opened with storage16 and int16 for the bf16 paths.
    pub fn init(dev: *const Device, shaders: Shaders) Error!Ops {
        if (!dev.features.storage16 or !dev.features.int16) return error.MissingFeature;
        var o: Ops = .{
            .dev = dev,
            .qmm_bf16 = undefined,
            .dequant_bf16 = undefined,
            .gather_u32 = undefined,
            .gather_u16 = undefined,
            .cast_bf16_f32 = undefined,
            .cast_f32_bf16 = undefined,
            .copy_u32 = undefined,
            .copy_u16 = undefined,
            .dummy = try Buffer.init(dev, .device, dummy_bytes),
        };
        errdefer o.dummy.deinit();
        o.qmm_bf16 = try dispatch.Pipeline.init(dev, shaders.qmm_bf16, .{ .bindings = 5, .push_bytes = params.push_bytes, .writes = 1 << 4 });
        errdefer o.qmm_bf16.deinit();
        o.dequant_bf16 = try dispatch.Pipeline.init(dev, shaders.dequant_bf16, .{ .bindings = 4, .push_bytes = params.push_bytes, .writes = 1 << 3 });
        errdefer o.dequant_bf16.deinit();
        o.gather_u32 = try dispatch.Pipeline.init(dev, shaders.gather_u32, .{ .bindings = 3, .push_bytes = params.push_bytes, .writes = 1 << 2 });
        errdefer o.gather_u32.deinit();
        o.gather_u16 = try dispatch.Pipeline.init(dev, shaders.gather_u16, .{ .bindings = 3, .push_bytes = params.push_bytes, .writes = 1 << 2 });
        errdefer o.gather_u16.deinit();
        o.cast_bf16_f32 = try dispatch.Pipeline.init(dev, shaders.cast_bf16_f32, .{ .bindings = 3, .push_bytes = params.cast_push_bytes, .writes = 1 << 2 });
        errdefer o.cast_bf16_f32.deinit();
        o.cast_f32_bf16 = try dispatch.Pipeline.init(dev, shaders.cast_f32_bf16, .{ .bindings = 3, .push_bytes = params.cast_push_bytes, .writes = 1 << 2 });
        errdefer o.cast_f32_bf16.deinit();
        o.copy_u32 = try dispatch.Pipeline.init(dev, shaders.copy_u32, .{ .bindings = 4, .push_bytes = params.push_bytes, .writes = 1 << 2 });
        errdefer o.copy_u32.deinit();
        o.copy_u16 = try dispatch.Pipeline.init(dev, shaders.copy_u16, .{ .bindings = 4, .push_bytes = params.push_bytes, .writes = 1 << 2 });
        return o;
    }

    /// Destroys the pipelines and the dummy; call only after the last submission using them returned success.
    pub fn deinit(o: *Ops) void {
        o.qmm_bf16.deinit();
        o.dequant_bf16.deinit();
        o.gather_u32.deinit();
        o.gather_u16.deinit();
        o.cast_bf16_f32.deinit();
        o.cast_f32_bf16.deinit();
        o.copy_u32.deinit();
        o.copy_u16.deinit();
        o.dummy.deinit();
    }

    pub fn dummyBinding(o: *const Ops) dispatch.Binding {
        return .{ .buffer = &o.dummy, .len = dummy_bytes };
    }

    /// bf16 -> f32 (RNE-free widening) or f32 -> bf16 (RNE with NaN quiet bit), one dispatch.
    pub fn cast(o: *Ops, cmds: *Commands, comptime to_bf16: bool, out: tensor.View, src: tensor.View, timeout_ns: u64) Error!void {
        const src_kind: tensor.Kind = if (to_bf16) .f32 else .bf16;
        const out_kind: tensor.Kind = if (to_bf16) .bf16 else .f32;
        if (src.kind != src_kind or out.kind != out_kind or src.elements != out.elements) return error.Invalid;
        const pipe = if (to_bf16) &o.cast_f32_bf16 else &o.cast_bf16_f32;
        try pipe.bind(&.{ src.bind(), o.dummyBinding(), out.bind() });
        var p: params.Params = .{ .count = try src.elemCount32() };
        try pipe.run(cmds, std.mem.asBytes(&p)[0..params.cast_push_bytes], dispatchGroups(p.count, pipe.local[0]), timeout_ns);
    }

    pub fn castF32ToBf16(o: *Ops, cmds: *Commands, out: tensor.View, src: tensor.View, timeout_ns: u64) Error!void {
        return o.cast(cmds, true, out, src, timeout_ns);
    }

    pub fn castBf16ToF32(o: *Ops, cmds: *Commands, out: tensor.View, src: tensor.View, timeout_ns: u64) Error!void {
        return o.cast(cmds, false, out, src, timeout_ns);
    }

    /// Contiguous same-kind raw copy through copy_general (dims 1, unit strides); strided copies are refused.
    pub fn copy(o: *Ops, cmds: *Commands, out: tensor.View, src: tensor.View, timeout_ns: u64) Error!void {
        if (out.kind != src.kind or out.elements != src.elements) return error.Invalid;
        const pipe = switch (out.kind) {
            .f32, .u32 => &o.copy_u32,
            .bf16, .u16 => &o.copy_u16,
        };
        try pipe.bind(&.{ src.bind(), o.dummyBinding(), out.bind(), o.dummyBinding() });
        const count = try src.elemCount32();
        var p: params.Params = .{
            .count = count,
            .dims = 1,
            .shape = .{ count, 0, 0, 0 },
            .in_strides = .{ 1, 0, 0, 0 },
            .out_strides = .{ 1, 0, 0, 0 },
        };
        try pipe.run(cmds, std.mem.asBytes(&p), dispatchGroups(count, pipe.local[0]), timeout_ns);
    }

    /// gather_take with u32 indices: out element t reads table[id * window + s]; an out-of-range id gathers zero.
    /// The caller owns index validation (see ops_q4.checkIds); the zero write is the shader's documented read.
    pub fn gather(o: *Ops, cmds: *Commands, out: tensor.View, table: tensor.View, ids: tensor.View, rows: u64, timeout_ns: u64) Error!void {
        if (ids.kind != .u32 or table.kind != out.kind) return error.Invalid;
        const pipe = switch (out.kind) {
            .u32 => &o.gather_u32,
            .bf16, .u16 => &o.gather_u16,
            .f32 => return error.Invalid,
        };
        const plan = try gatherPlan(out.elements, table.elements, ids.elements, rows);
        try pipe.bind(&.{ table.bind(), ids.bind(), out.bind() });
        var p: params.Params = .{
            .count = plan.count,
            .operation = 1, // raw u32 indices
            .reduce_size = plan.rows,
            .matrix_m = plan.stride,
            .dims = 1,
            .shape = .{ try cast32(plan.window), 0, 0, 0 },
            .in_strides = .{ 1, 0, 0, 0 },
            .out_strides = .{ plan.ids, 0, 0, 0 },
            .flags = 1, // one window axis after the gathered one
        };
        try pipe.run(cmds, std.mem.asBytes(&p), dispatchGroups(plan.count, pipe.local[0]), timeout_ns);
    }

    /// Affine dequant: out = q * scale + bias per packed unit, bf16 round-to-nearest-even output.
    pub fn dequant(o: *Ops, cmds: *Commands, out: tensor.View, w: tensor.View, scales: tensor.View, biases: tensor.View, k: u64, group: u64, timeout_ns: u64) Error!void {
        if (out.kind != .bf16 or w.kind != .u32 or scales.kind != .bf16 or biases.kind != .bf16) return error.Invalid;
        const plan = try dequantPlan(out.elements, w.elements, scales.elements, biases.elements, k, group);
        try o.dequant_bf16.bind(&.{ w.bind(), scales.bind(), biases.bind(), out.bind() });
        var p: params.Params = .{
            .count = plan.count,
            .operation = 4, // 4-bit codes
            .reduce_size = plan.group,
            .matrix_n = try cast32(plan.words),
            .matrix_k = plan.groups,
        };
        try o.dequant_bf16.run(cmds, std.mem.asBytes(&p), dispatchGroups(plan.count, o.dequant_bf16.local[0]), timeout_ns);
    }
};

const testing = std.testing;

test "gather plans derive the window and refuse torn shapes" {
    const plan = try gatherPlan(6 * 128, 512 * 128, 6, 512);
    try testing.expectEqual(@as(u64, 128), plan.window);
    try testing.expectEqual(@as(u32, 768), plan.count);
    try testing.expectEqual(@as(u32, 512), plan.rows);
    try testing.expectEqual(@as(u32, 128), plan.stride);
    try testing.expectError(error.Invalid, gatherPlan(0, 128, 1, 1));
    try testing.expectError(error.Invalid, gatherPlan(128, 128, 1, 0));
    try testing.expectError(error.Invalid, gatherPlan(768, 512, 6, 512)); // torn window
    try testing.expectError(error.Invalid, gatherPlan(768, 510 * 128, 6, 511));
    try testing.expectError(error.TooLarge, gatherPlan((1 << 32) * 4, (1 << 32) * 4, 1 << 32, 1 << 32));
}

test "dequant plans derive rows, words and groups and refuse torn geometry" {
    const plan = try dequantPlan(4 * 4096, 4 * 512, 4 * 64, 4 * 64, 4096, 64);
    try testing.expectEqual(@as(u64, 4), plan.rows);
    try testing.expectEqual(@as(u64, 512), plan.words);
    try testing.expectEqual(@as(u32, 64), plan.groups);
    try testing.expectEqual(@as(u32, 2048), plan.count);
    try testing.expectError(error.Invalid, dequantPlan(4096, 512, 64, 64, 4096, 65)); // k % group
    try testing.expectError(error.Invalid, dequantPlan(4096, 511, 64, 64, 4096, 64)); // torn words
    try testing.expectError(error.Invalid, dequantPlan(4096, 512, 63, 64, 4096, 64)); // torn scales
    try testing.expectError(error.Invalid, dequantPlan(4095, 511, 64, 64, 4096, 64)); // torn rows
    try testing.expectError(error.TooLarge, dequantPlan(8 << 35, 8 << 32, 8 << 29, 8 << 29, 1 << 35, 64));
}

test "dispatch groups round up without wrapping" {
    try testing.expectEqual([3]u32{ 1, 1, 1 }, dispatchGroups(1, 256));
    try testing.expectEqual([3]u32{ 1, 1, 1 }, dispatchGroups(256, 256));
    try testing.expectEqual([3]u32{ 2, 1, 1 }, dispatchGroups(257, 256));
    try testing.expectEqual([3]u32{ (1 << 32) / 256, 1, 1 }, dispatchGroups(std.math.maxInt(u32), 256));
}
