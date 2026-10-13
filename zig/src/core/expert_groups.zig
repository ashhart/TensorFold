//! Borrowed bounded GPU route grouping; padded eight-row tiles preserve each original output slot.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const exl3 = @import("exl3_projection.zig");
pub const Field = enum { counts, offsets, ranks, expert_ids, token_ids, slot_ids };
const field_count = std.enums.values(Field).len;

pub const Plan = struct {
    rows: u32,
    experts: u32,
    padded: u32,
    offsets: [field_count]usize,
    bytes: usize,

    pub fn init(rows: usize, experts: usize) !Plan {
        if (rows == 0 or experts == 0 or rows > std.math.maxInt(u32) or experts > std.math.maxInt(u32))
            return error.InvalidExpertGrouping;
        const active = @min(rows, experts);
        const tiles = try std.math.add(usize, active, (rows - active) / 8);
        const padded = try std.math.mul(usize, tiles, 8);
        if (padded > std.math.maxInt(u32)) return error.ExpertGroupingTooLarge;
        const sizes = [_]usize{ experts * 4, experts * 4, rows * 4, padded * 4, padded * 4, padded * 4 };
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
const Args = extern struct { rows: u32, experts: u32, repeat: u32, padded: u32 };

pub fn encode(enc: mtl.ComputeEncoder, pipelines: *const Pipelines, ids: exl3.View, scratch: Scratch, repeat: u32) !void {
    // Plan must come from Plan.init; route IDs remain GPU-resident and are bounds-checked by the kernels.
    if (repeat == 0 or scratch.plan.rows == 0 or scratch.plan.experts == 0 or
        scratch.plan.rows % repeat != 0) return error.InvalidExpertGrouping;
    if (ids.offset % 4 != 0 or scratch.offset % 4 != 0 or
        ids.offset > ids.buffer.length() or @as(usize, scratch.plan.rows) * 4 > ids.buffer.length() - ids.offset or
        scratch.offset > scratch.buffer.length() or scratch.plan.bytes > scratch.buffer.length() - scratch.offset)
        return error.InvalidExpertGroupingStorage;
    const args = Args{ .rows = scratch.plan.rows, .experts = scratch.plan.experts, .repeat = repeat, .padded = scratch.plan.padded };
    for (pipelines.stages, 0..) |pipeline, phase| {
        enc.setPipeline(pipeline);
        enc.setBuffer(ids.buffer, ids.offset, 0);
        inline for (std.enums.values(Field), 1..) |field, i| {
            const view = scratch.view(field);
            enc.setBuffer(view.buffer, view.offset, i);
        }
        enc.setValue(args, 7);
        enc.dispatchThreads(.{ .width = switch (phase) {
            0 => @max(args.experts, args.padded),
            2 => 1,
            else => args.rows,
        } }, .{ .width = 256 });
        enc.barrier();
    }
}

test "expert grouping bounds padding without context-sized allocation" {
    const small = try Plan.init(6, 384);
    try std.testing.expectEqual(@as(u32, 48), small.padded);
    const full = try Plan.init(1536, 384);
    try std.testing.expectEqual(@as(u32, 4224), full.padded);
    try std.testing.expect(full.bytes < 64 * 1024);
    try std.testing.expectError(error.InvalidExpertGrouping, Plan.init(0, 384));
    try std.testing.expectError(error.ExpertGroupingTooLarge, Plan.init(std.math.maxInt(u32), 384));
}
