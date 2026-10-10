//! State persists per stream and layer; round logs persist per layer while one snapshot slab is reused.
const std = @import("std");
const abi = @import("gdn_contract.zig");
const Plan = @import("gdn_plan.zig").Plan;

pub fn validatePlan(shape: abi.Shape, slots: usize, capacity: usize, plan: Plan) !void {
    const rows = plan.parents.len;
    const taps = shape.c.conv_kernel;
    if (rows == 0 or rows > capacity or slots == 0 or slots > 64 or plan.windows.len != rows * taps or plan.row_slots.len != rows or plan.segments.len == 0 or plan.segments.len > slots or plan.keeps.len != plan.segments.len) return error.BadGdnPoolPlan;
    var first: usize = 0;
    var kept: usize = 0;
    var destinations: u64 = 0;
    for (plan.segments, plan.keeps) |segment, keep| {
        if (segment.first != first or segment.rows == 0 or segment.rows > rows - first or segment.state_slot >= slots or segment.next_slot >= slots or keep.state_slot != segment.state_slot or keep.next_slot != segment.next_slot or keep.first != kept or keep.rows > segment.rows or keep.rows > plan.kept_rows.len - kept) return error.BadGdnPoolPlan;
        const bit = @as(u64, 1) << @intCast(segment.next_slot);
        if (destinations & bit != 0) return error.BadGdnPoolPlan;
        destinations |= bit;
        for (0..segment.rows) |local| {
            const row = first + local;
            const parent = plan.parents[row];
            if ((local == 0 and parent != -1) or (local > 0 and (parent < 0 or parent >= local)) or plan.row_slots[row] != segment.state_slot) return error.BadGdnPoolPlan;
            for (plan.windows[row * taps ..][0..taps], 0..) |source, tap| {
                const behind = taps - 1 - tap;
                var ancestor: i32 = @intCast(local);
                var hops: usize = 0;
                while (hops < behind and ancestor >= 0) : (hops += 1) ancestor = plan.parents[first + @as(usize, @intCast(ancestor))];
                const expected: usize = if (ancestor >= 0) taps - 1 + first + @as(usize, @intCast(ancestor)) else taps - 1 - (behind - hops + 1);
                if (source != expected) return error.BadGdnPoolPlan;
            }
        }
        var parent: i32 = -1;
        for (plan.kept_rows[kept..][0..keep.rows]) |global| {
            if (global < first or global >= first + segment.rows or plan.parents[global] != parent) return error.BadGdnPoolPlan;
            parent = @intCast(global - first);
        }
        first += segment.rows;
        kept += keep.rows;
    }
    if (first != rows or kept != plan.kept_rows.len) return error.BadGdnPoolPlan;
}

pub const Budget = struct {
    shape: abi.Shape,
    layers: usize,
    slots: usize,
    rows: usize,
    state: usize,
    history: usize,
    projections: usize,
    query: usize,
    key: usize,
    value: usize,
    decay: usize,
    mixing: usize,
    snapshots: usize,
    conv_tails: usize,
    scratch: usize,

    pub fn init(shape: abi.Shape, layers: usize, slots: usize, rows: usize) !Budget {
        if (layers == 0 or layers > shape.c.layers or slots == 0 or slots > 64 or rows == 0 or rows > 128) return error.BadGdnPool;
        const layer_slots = try std.math.mul(usize, layers, slots);
        const layer_rows = try std.math.mul(usize, layers, rows);
        const query = try abi.countBytes(u16, shape.c.k_heads * shape.c.dk, layer_rows);
        const value = try abi.countBytes(u16, shape.value, layer_rows);
        const decay = try abi.countBytes(f32, shape.c.v_heads, layer_rows);
        const mixing = try abi.countBytes(u16, shape.c.v_heads, layer_rows);
        const snapshot = try abi.countBytes(f32, shape.state, rows);
        const tails = try abi.countBytes(u16, shape.conv, rows);
        const outputs = try abi.countBytes(u16, shape.value, rows * 2);
        return .{
            .shape = shape,
            .layers = layers,
            .slots = slots,
            .rows = rows,
            .state = try abi.countBytes(f32, shape.state, layer_slots),
            .history = try abi.countBytes(u16, shape.conv, layer_slots),
            .projections = try abi.countBytes(u16, shape.qkv, layer_rows),
            .query = query,
            .key = query,
            .value = value,
            .decay = decay,
            .mixing = mixing,
            .snapshots = snapshot,
            .conv_tails = tails,
            .scratch = try std.math.add(usize, try std.math.add(usize, snapshot, tails), outputs),
        };
    }

    pub fn stateOffset(b: Budget, layer: usize, slot: usize) !usize {
        if (layer >= b.layers or slot >= b.slots) return error.BadGdnPoolIndex;
        return abi.countBytes(f32, b.shape.state, layer * b.slots + slot);
    }

    pub fn historyOffset(b: Budget, layer: usize, slot: usize) !usize {
        if (layer >= b.layers or slot >= b.slots) return error.BadGdnPoolIndex;
        return abi.countBytes(u16, b.shape.conv, layer * b.slots + slot);
    }

    pub fn logBytes(b: Budget) usize {
        return b.query + b.key + b.value + b.decay + b.mixing;
    }

    pub fn totalBytes(b: Budget) usize {
        return 2 * b.state + 2 * b.history + b.projections + b.logBytes() + b.scratch;
    }
};
