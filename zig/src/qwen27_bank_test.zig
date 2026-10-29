//! Actual pool construction uses a recording allocator, never a Metal device.
const std = @import("std");
const metal = @import("metal");
const gpu = @import("families/qwen27/gdn_pool_gpu.zig");
const abi = @import("families/qwen27/gdn_contract.zig");
const Config = @import("families/qwen27/config.zig").Config;

fn shape() !abi.Shape {
    var c: Config = undefined;
    c.k_heads = 1;
    c.v_heads = 1;
    c.dk = 128;
    c.dv = 128;
    c.conv_kernel = 2;
    c.layers = 2;
    c.eps = 1e-6;
    return abi.Shape.init(c);
}

test "qwen27 bank: construction does not allocate rollback storage for idle or lone use" {
    var allocations: metal.Allocations = .{};
    const device = metal.Device{ .allocations = &allocations };
    var pool = try gpu.Pool.init(device, try shape(), 2, 8, 16);
    defer pool.deinit();
    try std.testing.expect(allocations.bytes <= pool.budget.totalBytes() - pool.budget.state);
}

const Plan = @import("families/qwen27/gdn_plan.zig").Plan;
const Window = @import("families/qwen27/state.zig").Window;
const kinds = abi.WeightKinds{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
const a = std.testing.allocator;

fn makePlan(pool: *const gpu.Pool, inputs: []const @import("families/qwen27/gdn_plan.zig").Input) !Plan {
    var rows: usize = 0;
    for (inputs) |input| rows += input.window.parents.len;
    return Plan.init(a, try pool.budget.shape.params(rows, pool.budget.slots, kinds), inputs);
}

test "qwen27 bank: constructor allocation failures unwind every earlier buffer" {
    var count: usize = 0;
    {
        var allocations: metal.Allocations = .{};
        var pool = try gpu.Pool.init(.{ .allocations = &allocations }, try shape(), 2, 8, 16);
        count = allocations.calls;
        pool.deinit();
        try std.testing.expectEqual(@as(usize, 0), allocations.live);
    }
    for (0..count) |fail| {
        var allocations: metal.Allocations = .{ .fail = fail };
        try std.testing.expectError(error.NoBuffer, gpu.Pool.init(.{ .allocations = &allocations }, try shape(), 2, 8, 16));
        try std.testing.expectEqual(@as(usize, 0), allocations.live);
        try std.testing.expectEqual(@as(usize, 0), allocations.bytes);
    }
}

test "qwen27 bank: actual plans allocate only the one-row rollback extent" {
    var allocations: metal.Allocations = .{};
    const device = metal.Device{ .allocations = &allocations };
    var pool = try gpu.Pool.init(device, try shape(), 2, 8, 16);
    defer pool.deinit();
    var one = try Window.init(a, 3, &.{-1}, 2, 64);
    defer one.deinit();
    var wide = try Window.init(a, 7, &.{ -1, 0, 1, 2, 3, 4, 5, 6 }, 2, 64);
    defer wide.deinit();
    var lone = try makePlan(&pool, &.{.{ .window = one, .state_slot = 7, .next_slot = 7, .path = &.{0} }});
    defer lone.deinit();
    try pool.prepare(device, lone, false);
    try std.testing.expect(pool.before == null);
    var tree = try makePlan(&pool, &.{.{ .window = wide, .state_slot = 0, .next_slot = 0, .path = &.{ 0, 1 } }});
    defer tree.deinit();
    try pool.prepare(device, tree, false);
    try std.testing.expect(pool.before == null);
    var wide_pair = try makePlan(&pool, &.{
        .{ .window = wide, .state_slot = 0, .next_slot = 0, .path = &.{ 0, 1 } },
        .{ .window = wide, .state_slot = 7, .next_slot = 7, .path = &.{ 0, 1 } },
    });
    defer wide_pair.deinit();
    try pool.prepare(device, wide_pair, false);
    try std.testing.expect(pool.before == null);
    var mixed = try makePlan(&pool, &.{
        .{ .window = one, .state_slot = 1, .next_slot = 1, .path = &.{0} },
        .{ .window = wide, .state_slot = 7, .next_slot = 7, .path = &.{ 0, 1 } },
    });
    defer mixed.deinit();
    try pool.prepare(device, mixed, false);
    try std.testing.expectEqual(@as(usize, 2), pool.before_slots);
    try std.testing.expectEqual(pool.budget.state / 4, pool.before.?.length());
    try std.testing.expectEqual(pool.budget.state - pool.before.?.length(), pool.missingBeforeBytes());
    try pool.prepare(device, mixed, true);
    try std.testing.expect(pool.before == null);
    pool.tree_mode = .snapshot;
    try pool.prepare(device, mixed, false);
    try std.testing.expect(pool.before == null);
    try std.testing.expectEqual(@as(usize, 0), pool.missingBeforeBytes());
    pool.tree_mode = .auto;
    try std.testing.expectEqual(pool.budget.state, pool.missingBeforeBytes());
}

test "qwen27 bank: growth shrink reuse and failed allocation leave committed words intact" {
    var allocations: metal.Allocations = .{};
    const device = metal.Device{ .allocations = &allocations };
    var pool = try gpu.Pool.init(device, try shape(), 2, 8, 16);
    defer pool.deinit();
    const base_bytes = allocations.bytes;
    var one = try Window.init(a, 3, &.{-1}, 2, 64);
    defer one.deinit();
    var small = try makePlan(&pool, &.{
        .{ .window = one, .state_slot = 0, .next_slot = 0, .path = &.{0} },
        .{ .window = one, .state_slot = 1, .next_slot = 1, .path = &.{0} },
    });
    defer small.deinit();
    var high = try makePlan(&pool, &.{
        .{ .window = one, .state_slot = 0, .next_slot = 0, .path = &.{0} },
        .{ .window = one, .state_slot = 7, .next_slot = 7, .path = &.{0} },
    });
    defer high.deinit();
    const state = try pool.committed(1, 7, true);
    @memset(@constCast(state), 0x59);
    try pool.prepare(device, small, false);
    const calls = allocations.calls;
    const id = pool.before.?.id;
    try pool.prepare(device, small, false);
    try std.testing.expectEqual(calls, allocations.calls);
    try std.testing.expectEqual(id, pool.before.?.id);
    allocations.fail = allocations.calls;
    try std.testing.expectError(error.NoBuffer, pool.prepare(device, high, false));
    try std.testing.expect(pool.before == null);
    try std.testing.expectEqual(@as(usize, 0), pool.before_slots);
    allocations.fail = std.math.maxInt(usize);
    try pool.prepare(device, high, false);
    try std.testing.expectEqual(pool.budget.state, pool.before.?.length());
    try std.testing.expect(allocations.peak_bytes <= base_bytes + pool.budget.state);
    try pool.prepare(device, small, false);
    try std.testing.expectEqual(pool.budget.state / 4, pool.before.?.length());
    const keep = small.keeps[1];
    small.keeps[1].next_slot = 0;
    try std.testing.expectError(error.BadGdnPoolPlan, pool.prepare(device, small, false));
    small.keeps[1] = keep;
    try std.testing.expectEqual(@as(usize, 2), pool.before_slots);
    pool.releaseBefore();
    for (state) |word| try std.testing.expectEqual(@as(u8, 0x59), word);
}

const Fixture = struct {
    pool: gpu.Pool,
    metadata: gpu.Metadata,
    scratch: metal.Buffer,
    output: metal.Buffer,
    fn init(device: metal.Device) !Fixture {
        var f: Fixture = undefined;
        f.pool = try gpu.Pool.init(device, try shape(), 2, 8, 16);
        errdefer f.pool.deinit();
        inline for (@typeInfo(gpu.Metadata).@"struct".field_names) |name|
            @field(f.metadata, name) = .{ .buffer = try device.buffer(4096, 0) };
        f.scratch = try device.buffer(4096, 0);
        f.output = try device.buffer(4096, 0);
        return f;
    }
    fn deinit(f: *Fixture) void {
        f.pool.deinit();
        inline for (@typeInfo(gpu.Metadata).@"struct".field_names) |name| @field(f.metadata, name).buffer.deinit();
        f.scratch.deinit();
        f.output.deinit();
    }
    fn refs(f: *Fixture) gpu.Refs {
        const span = @import("families/qwen27/gdn.zig").Span{ .buffer = f.scratch, .bytes = f.scratch.length() };
        return .{ .z = .{ .buffer = f.scratch }, .a = .{ .buffer = f.scratch }, .b = .{ .buffer = f.scratch }, .output = .{ .buffer = f.output }, .scalars = .{ .conv = span, .a_log = span, .dt = span, .norm = span, .kinds = kinds } };
    }
};

test "qwen27 bank: real two-layer verify and empty keep use partial strides and retain rollback until commit" {
    var allocations: metal.Allocations = .{};
    const device = metal.Device{ .allocations = &allocations };
    var f = try Fixture.init(device);
    defer f.deinit();
    var one = try Window.init(a, 3, &.{-1}, 2, 64);
    defer one.deinit();
    var wide = try Window.init(a, 9, &.{ -1, 0 }, 2, 64);
    defer wide.deinit();
    var plan = try makePlan(&f.pool, &.{
        .{ .window = one, .state_slot = 1, .next_slot = 1, .path = &.{0} },
        .{ .window = wide, .state_slot = 7, .next_slot = 7, .path = &.{ 0, 1 } },
    });
    defer plan.deinit();
    var record: metal.Recorder = .{ .segments = 2 };
    const e = metal.ComputeEncoder{ .record = &record };
    try std.testing.expectError(error.MissingGdnBefore, f.pool.forward(0, .{ .host = plan, .metadata = f.metadata }, f.refs(), e));
    try f.pool.prepare(device, plan, false);
    try std.testing.expectEqual(@as(usize, 2), f.pool.before_slots);
    for (0..2) |layer| {
        const words = try f.pool.committed(layer, 1, true);
        @memset(@constCast(words), @intCast(0x21 + layer));
        try f.pool.forward(layer, .{ .host = plan, .metadata = f.metadata }, f.refs(), e);
        try std.testing.expect(words[0] != 0x21 + layer);
    }
    var keep = try makePlan(&f.pool, &.{
        .{ .window = one, .state_slot = 1, .next_slot = 1, .path = &.{} },
        .{ .window = wide, .state_slot = 7, .next_slot = 7, .path = &.{} },
    });
    defer keep.deinit();
    const bank = f.pool.before.?.id;
    for (0..2) |layer| {
        try std.testing.expectEqual(bank, f.pool.before.?.id);
        try f.pool.commit(layer, .{ .host = keep, .metadata = f.metadata }, e);
        for (try f.pool.committed(layer, 1, true)) |word| try std.testing.expectEqual(@as(u8, @intCast(0x21 + layer)), word);
    }
    try std.testing.expectEqual(@as(usize, 2), record.restores);
    try f.pool.prepare(device, plan, true);
    try std.testing.expect(f.pool.before == null);
    for (0..2) |layer| try f.pool.forwardChain(layer, .{ .host = plan, .metadata = f.metadata }, f.refs(), e, true);
}

test {
    _ = @import("qwen27_frame_test.zig");
}
