//! Borrowed bounded GPU route grouping; padded sixty-four-row tiles preserve each original output slot.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const exl3 = @import("exl3_projection.zig");
pub const Field = enum { counts, offsets, ranks, expert_ids, token_ids, slot_ids, destination };
const field_count = std.enums.values(Field).len;

/// The grouped plan's tile rows and the routed GEMM's tile rows are ONE number: the plan pads each expert's block
/// to a multiple of it and the GEMM's tile must match it, or a tile would straddle two experts' weights. 128 rows
/// (TF_DSV41_MOE_TILE_ROWS=128) halves the number of tiles -- and so the number of weight decodes -- for the
/// experts with more than sixty-four rows.
pub var tile_rows_cache: ?u32 = null;
pub fn tileRows() u32 {
    if (tile_rows_cache) |v| return v;
    var v: u32 = 64;
    if (std.c.getenv("TF_DSV41_MOE_TILE_ROWS")) |e| v = std.fmt.parseInt(u32, std.mem.span(e), 10) catch 64;
    if (v != 64 and v != 128) v = 64;
    tile_rows_cache = v;
    return v;
}

pub const Plan = struct {
    rows: u32,
    experts: u32,
    padded: u32,
    offsets: [field_count]usize,
    bytes: usize,

    pub fn init(rows: usize, experts: usize, tile: usize) !Plan {
        if (rows == 0 or experts == 0 or tile == 0 or rows > std.math.maxInt(u32) or experts > std.math.maxInt(u32))
            return error.InvalidExpertGrouping;
        const active = @min(rows, experts);
        const tiles = try std.math.add(usize, active, (rows - active) / tile);
        const padded = try std.math.mul(usize, tiles, tile);
        if (padded > std.math.maxInt(u32)) return error.ExpertGroupingTooLarge;
        const sizes = [_]usize{ experts * 4, experts * 4, rows * 4, padded * 4, padded * 4, padded * 4, padded * 4 };
        var plan = Plan{ .rows = @intCast(rows), .experts = @intCast(experts), .padded = @intCast(padded), .offsets = undefined, .bytes = 0 };
        for (sizes, 0..) |size, i| {
            plan.offsets[i] = std.mem.alignForward(usize, plan.bytes, 16);
            plan.bytes = try std.math.add(usize, plan.offsets[i], size);
        }
        plan.bytes = std.mem.alignForward(usize, plan.bytes, 16);
        return plan;
    }
};

pub const Scratch = struct {
    buffer: mtl.Buffer,
    offset: usize = 0,
    plan: Plan,

    pub fn view(self: Scratch, field: Field) exl3.View {
        return .{ .buffer = self.buffer, .offset = self.offset + self.plan.offsets[@backingInt(field)] };
    }
};

pub const Pipelines = struct {
    stages: [4]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_expert_groups, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var stages: [4]mtl.Pipeline = undefined;
        var loaded: usize = 0;
        errdefer for (stages[0..loaded]) |pipeline| pipeline.deinit();
        inline for (.{ "tf_expert_group_clear", "tf_expert_group_count", "tf_expert_group_prefix", "tf_expert_group_scatter" }, 0..) |name, i| {
            stages[i] = try mtl.Pipeline.init(device, lib, name, false);
            loaded += 1;
        }
        return .{ .stages = stages };
    }
    pub fn deinit(self: Pipelines) void {
        for (self.stages) |pipeline| pipeline.deinit();
    }
};
const Args = extern struct { rows: u32, experts: u32, repeat: u32, padded: u32, block: u32, tile: u32 };

/// `block` is the slot count of one input block when the pick table is more than one block (0: the whole table
/// is one block). A fused gate+up table is two blocks of `repeat` slots a token.
pub fn encode(enc: mtl.ComputeEncoder, pipelines: *const Pipelines, ids: exl3.View, scratch: Scratch, repeat: u32, block: u32) !void {
    // Plan must come from Plan.init; route IDs remain GPU-resident and are bounds-checked by the kernels.
    if (repeat == 0 or scratch.plan.rows == 0 or scratch.plan.experts == 0 or
        scratch.plan.rows % repeat != 0) return error.InvalidExpertGrouping;
    if (ids.offset % 4 != 0 or scratch.offset % 4 != 0 or
        ids.offset > ids.buffer.length() or @as(usize, scratch.plan.rows) * 4 > ids.buffer.length() - ids.offset or
        scratch.offset > scratch.buffer.length() or scratch.plan.bytes > scratch.buffer.length() - scratch.offset)
        return error.InvalidExpertGroupingStorage;
    const args = Args{ .rows = scratch.plan.rows, .experts = scratch.plan.experts, .repeat = repeat, .padded = scratch.plan.padded, .block = block, .tile = tileRows() };
    for (pipelines.stages, 0..) |pipeline, phase| {
        enc.setPipeline(pipeline);
        enc.setBuffer(ids.buffer, ids.offset, 0);
        inline for (std.enums.values(Field), 1..) |field, i| {
            const view = scratch.view(field);
            enc.setBuffer(view.buffer, view.offset, i);
        }
        enc.setValue(args, 8);
        enc.dispatchThreads(.{ .width = switch (phase) {
            0 => @max(args.experts, args.padded),
            2 => 1,
            else => args.rows,
        } }, .{ .width = 256 });
        enc.barrier();
    }
}

test "expert grouping bounds padding without context-sized allocation" {
    const small = try Plan.init(6, 384, 64);
    try std.testing.expectEqual(@as(u32, 384), small.padded);
    const full = try Plan.init(1536, 384, 64);
    try std.testing.expectEqual(@as(u32, 25728), full.padded);
    try std.testing.expect(full.bytes < 512 * 1024);
    // The parameterised grouping tile (8da7b80eb, the priced-out 128-row bench instrument) computes tile > 64
    // padding as its own plan: active rows group to the tile, the remainder adds one tile - 384 active + 9 tiles of
    // 128 = 50,304. The shipped 64-row tile keeps the bounds above.
    const tall = try Plan.init(1536, 384, 128);
    try std.testing.expectEqual(@as(u32, 50304), tall.padded);
    try std.testing.expectError(error.InvalidExpertGrouping, Plan.init(0, 384, 64));
    try std.testing.expectError(error.ExpertGroupingTooLarge, Plan.init(std.math.maxInt(u32), 384, 64));
}
