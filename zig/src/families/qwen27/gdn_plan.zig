//! Flattened windows retain local parent indices and global projected-row indices in separate buffers.
const std = @import("std");
const abi = @import("gdn_contract.zig");
const Window = @import("state.zig").Window;

pub const Input = struct { window: Window, state_slot: u32, next_slot: u32, path: []const u32 };

pub const Plan = struct {
    gpa: std.mem.Allocator,
    parents: []i32,
    windows: []u32,
    row_slots: []u32,
    segments: []abi.Segment,
    keeps: []abi.Keep,
    kept_rows: []u32,

    pub fn init(gpa: std.mem.Allocator, p: abi.Params, inputs: []const Input) !Plan {
        if (inputs.len == 0 or p.slots == 0 or p.slots > 64 or inputs.len > p.slots or p.rows == 0 or p.rows > 128 or p.taps < 2 or p.taps > 8) return error.BadGdnPlan;
        var rows: usize = 0;
        var retained: usize = 0;
        var destinations: u64 = 0;
        for (inputs) |input| {
            if (input.window.taps != p.taps or input.state_slot >= p.slots or input.next_slot >= p.slots) return error.BadGdnPlan;
            const bit = @as(u64, 1) << @intCast(input.next_slot);
            if (destinations & bit != 0) return error.OverlappingGdnCommit;
            destinations |= bit;
            _ = try input.window.keep(input.path);
            rows = try std.math.add(usize, rows, input.window.parents.len);
            retained = try std.math.add(usize, retained, input.path.len);
        }
        if (rows != p.rows) return error.BadGdnPlan;
        const parents = try gpa.alloc(i32, rows);
        errdefer gpa.free(parents);
        const windows = try gpa.alloc(u32, rows * p.taps);
        errdefer gpa.free(windows);
        const slots = try gpa.alloc(u32, rows);
        errdefer gpa.free(slots);
        const segments = try gpa.alloc(abi.Segment, inputs.len);
        errdefer gpa.free(segments);
        const keeps = try gpa.alloc(abi.Keep, inputs.len);
        errdefer gpa.free(keeps);
        const kept_rows = try gpa.alloc(u32, retained);
        errdefer gpa.free(kept_rows);
        var first: u32 = 0;
        var keep_first: u32 = 0;
        for (inputs, 0..) |input, segment| {
            const n: u32 = @intCast(input.window.parents.len);
            @memcpy(parents[first..][0..n], input.window.parents);
            @memset(slots[first..][0..n], input.state_slot);
            for (input.window.conv, 0..) |index, at| windows[first * p.taps + at] = if (index < p.taps - 1) index else index + first;
            for (input.path, 0..) |row, at| kept_rows[keep_first + at] = first + row;
            segments[segment] = .{ .first = first, .rows = n, .state_slot = input.state_slot, .next_slot = input.next_slot };
            keeps[segment] = .{ .first = keep_first, .rows = @intCast(input.path.len), .state_slot = input.state_slot, .next_slot = input.next_slot };
            first += n;
            keep_first += @intCast(input.path.len);
        }
        return .{ .gpa = gpa, .parents = parents, .windows = windows, .row_slots = slots, .segments = segments, .keeps = keeps, .kept_rows = kept_rows };
    }

    pub fn deinit(plan: *Plan) void {
        plan.gpa.free(plan.parents);
        plan.gpa.free(plan.windows);
        plan.gpa.free(plan.row_slots);
        plan.gpa.free(plan.segments);
        plan.gpa.free(plan.keeps);
        plan.gpa.free(plan.kept_rows);
        plan.* = undefined;
    }
};
