//! Rank-local intermediate-channel TP for NVFP4 routed experts; the shared expert retains its own projection format.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const ex = cuda.experts.Nvfp4Experts;
const slice = core.nvfp4_slice;
const threads = @import("cuda_modelopt_threads.zig");

pub const Cut = struct {
    from: usize,
    to: usize,

    pub fn init(total_width: usize, world: usize, rank: usize) !Cut {
        if (world == 0 or rank >= world or total_width == 0 or total_width % world != 0) return error.UnevenExpertSplit;
        const part = total_width / world;
        if (part % 32 != 0) return error.UnevenExpertSplit;
        return .{ .from = rank * part, .to = (rank + 1) * part };
    }

    pub fn width(self: Cut) usize {
        return self.to - self.from;
    }
};

pub const Owned = struct {
    up: cuda.DeviceBuffer,
    down: cuda.DeviceBuffer,
    up_scale: cuda.DeviceBuffer,
    down_scale: cuda.DeviceBuffer,
    width: usize,
    dims: usize,
    count: usize,

    pub fn free(self: *Owned) void {
        self.up.free();
        self.down.free();
        self.up_scale.free();
        self.down_scale.free();
        self.* = undefined;
    }

    pub fn view(self: *const Owned) cuda.experts.Layer {
        return .{ .format = .nvfp4, .up = self.up.ptr, .down = self.down.ptr, .up_scale = self.up_scale.ptr, .down_scale = self.down_scale.ptr, .width = self.width, .dims = self.dims, .experts = self.count };
    }
};

/// Own only the selected rectangle; the exclusive loader store releases every borrowed plane before return.
pub fn readPart(a: std.mem.Allocator, store: anytype, name: []const u8, cut: Cut, down: bool, dims: usize, full_width: usize) !slice.Linear {
    defer store.release();
    var buf: [512]u8 = undefined;
    const w = try store.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}));
    const n = if (down) dims else full_width;
    const k = if (down) full_width else dims;
    if (!w.is(.u8, &.{ n, k / 2 })) return error.UnexpectedTensor;
    var linear = try slice.read(store, a, name, .{ .from = if (down) 0 else cut.from, .to = if (down) dims else cut.to }, .{ .from = if (down) cut.from else 0, .to = if (down) cut.to else dims });
    errdefer linear.deinit(a);
    if (!std.math.isFinite(linear.global) or linear.global <= 0) return error.UnsupportedQuantization;
    return linear;
}

/// Pack one expert at a time with BF16 activations (W4A16); input_scale is validated as metadata.
pub fn load(a: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, prefix: []const u8, world: usize, rank: usize) !Owned {
    return loadShape(a, d, store, prefix, world, rank, wt.dims, wt.moe_w, wt.experts_n);
}

pub fn loadShape(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, world: usize, rank: usize, dims: usize, full_width: usize, count: usize) !Owned {
    const cut = try Cut.init(full_width, world, rank);
    if (dims == 0 or dims % 32 != 0 or count == 0) return error.UnexpectedTensor;
    // one GPU and the weight store: packed on threads (cuda_modelopt_threads.zig); other readers or 1 thread: this loop
    if (comptime @typeInfo(@TypeOf(store)).pointer.child == wt.Store) {
        const n = try threads.configured();
        if (world == 1 and n > 1) return threads.load(a, d, store, prefix, dims, cut.width(), count, n);
    }
    const width = cut.width();
    const up_bytes = try packedBytes(width, dims, 2);
    const down_bytes = try packedBytes(dims, width, 1);
    const up_total = std.math.mul(usize, count, up_bytes) catch return error.UnexpectedTensor;
    const down_total = std.math.mul(usize, count, down_bytes) catch return error.UnexpectedTensor;
    const globals_count = std.math.mul(usize, count, 2) catch return error.UnexpectedTensor;
    var up = try cuda.DeviceBuffer.alloc(d, up_total);
    errdefer up.free();
    var down = try cuda.DeviceBuffer.alloc(d, down_total);
    errdefer down.free();
    const up_globals = try a.alloc(f32, globals_count);
    defer a.free(up_globals);
    const down_globals = try a.alloc(f32, count);
    defer a.free(down_globals);
    const packed_up = try a.alloc(u32, up_bytes / 4);
    defer a.free(packed_up);
    const packed_down = try a.alloc(u32, down_bytes / 4);
    defer a.free(packed_down);
    var buf: [512]u8 = undefined;
    for (0..count) |i| {
        var gate = try readPart(a, store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.gate_proj", .{ prefix, i }), cut, false, dims, full_width);
        defer gate.deinit(a);
        var value = try readPart(a, store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.up_proj", .{ prefix, i }), cut, false, dims, full_width);
        defer value.deinit(a);
        ex.packGateUp(packed_up, .{ gate.codes, gate.scales }, .{ value.codes, value.scales }, width, dims);
        try up.upload(i * up_bytes, std.mem.sliceAsBytes(packed_up));
        up_globals[i * 2] = gate.global;
        up_globals[i * 2 + 1] = value.global;
        var output = try readPart(a, store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.down_proj", .{ prefix, i }), cut, true, dims, full_width);
        defer output.deinit(a);
        ex.packOne(packed_down, output.codes, output.scales, dims, width);
        try down.upload(i * down_bytes, std.mem.sliceAsBytes(packed_down));
        down_globals[i] = output.global;
    }
    var us = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_globals));
    errdefer us.free();
    const ds = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_globals));
    return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .width = width, .dims = dims, .count = count };
}

test "intermediate-channel cuts cover the source once and preserve 32-input kernel groups" {
    const a = try Cut.init(640, 2, 0);
    const b = try Cut.init(640, 2, 1);
    try std.testing.expectEqual(@as(usize, 0), a.from);
    try std.testing.expectEqual(@as(usize, 320), a.to);
    try std.testing.expectEqual(a.to, b.from);
    try std.testing.expectEqual(@as(usize, 640), b.to);
    try std.testing.expectEqual(a.width(), b.width());
    try std.testing.expectError(error.UnevenExpertSplit, Cut.init(640, 0, 0));
    try std.testing.expectError(error.UnevenExpertSplit, Cut.init(640, 2, 2));
    try std.testing.expectError(error.UnevenExpertSplit, Cut.init(640, 3, 0));
    try std.testing.expectError(error.UnevenExpertSplit, Cut.init(640, 32, 0));
}

pub fn packedBytes(n: usize, k: usize, parts: usize) !usize {
    var size = std.math.mul(usize, n / ex.cols, k / 32) catch return error.UnexpectedTensor;
    for ([_]usize{ parts, ex.words, 4 }) |factor| size = std.math.mul(usize, size, factor) catch return error.UnexpectedTensor;
    return size;
}

test "expert packed extents reject overflow before device allocation" {
    try std.testing.expectEqual(ex.bytes(1, 320, 2560, 2), try packedBytes(320, 2560, 2));
    try std.testing.expectError(error.UnexpectedTensor, packedBytes(std.math.maxInt(usize), std.math.maxInt(usize), 2));
}
