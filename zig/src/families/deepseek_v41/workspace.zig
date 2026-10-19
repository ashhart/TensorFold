//! Fixed chunk-sized activations: no layer history or context-by-query score matrices.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("../../core/exl3_projection.zig");
const View = @import("view.zig").View;
const hash = @import("engram_hash.zig");
const groups = @import("../../core/expert_groups.zig");

pub const Field = enum {
    ids,
    token_map,
    history,
    stream,
    collapsed,
    normalized,
    branch,
    previous_pre,
    attention_pre,
    post,
    comb,
    mixes,
    inv,
    engram_ids_0,
    engram_ids_1,
    engram_gather,
    engram_kv,
    router_logits,
    route_ids,
    route_weights,
    gate,
    up,
    activation,
    routed,
    shared_gate,
    shared_up,
    shared_activation,
    shared_down,
    logits,
    projection,
    grouping,
    /// Streamed Engram staging, one set per table layer (empty in resident mode): compact rows, scales, local IDs.
    engram_stage_w_0,
    engram_stage_w_1,
    engram_stage_s_0,
    engram_stage_s_1,
    engram_local_0,
    engram_local_1,
    /// Merged MoE products (rows below the plan): each token's six routed IDs then the shared expert's.
    moe_routes,
    /// Where merged row 7 r + j lands in the gate, up, activation and routed fields (moeSlot).
    moe_slots,
};

/// The shared expert as a seventh row per token: index of the routes table, and of descriptor tables.
pub const shared_expert: u32 = 384;
/// Merged slot (r,j) maps to routed row 6*r+j or shared row 6*plan_rows+r, preserving routed row positions.
pub fn moeSlot(plan_rows: usize, r: usize, j: usize) u32 {
    return @intCast(if (j < 6) r * 6 + j else plan_rows * 6 + r);
}
const count = std.enums.values(Field).len;

pub const Plan = struct {
    rows: u32,
    lanes: u32,
    streamed: bool = false,
    offsets: [count]usize,
    bytes: usize,
    /// Size of the projection scratch field: concurrent product groups split it between their jobs.
    projection_bytes: usize = 0,

    pub fn init(rows: u32) !Plan {
        return initLanes(rows, 1);
    }

    pub fn initLanes(rows: u32, lanes: u32) !Plan {
        return initStreamed(rows, lanes, false);
    }

    /// `streamed` adds the Engram staging fields; resident offsets are unchanged either way.
    pub fn initStreamed(rows: u32, lanes: u32, streamed: bool) !Plan {
        if (rows == 0 or rows > 4096) return error.InvalidPrefillChunk;
        if (lanes == 0 or lanes > 256) return error.InvalidLaneCount;
        const t: usize = rows;
        const stage: usize = if (streamed) t * hash.columns else 0;
        const grouped = if (rows > 1) try groups.Plan.init(t * 6, 384) else null;
        const projection_bytes = @max(try exl3.scratchBytes(5120, 129280, t), if (grouped) |g| @max(try exl3.scratchBytes(5120, 2304, g.padded), try exl3.scratchBytes(2304, 5120, g.padded)) else try exl3.scratchBytes(2304, 5120, t * 6));
        const sizes = [_]usize{
            t * 4,            hash.raw_vocab * 4, @as(usize, lanes) * 16,          t * 20480 * 2, t * 5120 * 2, t * 5120 * 2, t * 5120 * 2,
            t * 4 * 4,        t * 4 * 4,          t * 4 * 4,                       t * 16 * 4,    t * 24 * 4,   t * 4,        t * 24 * 4,
            t * 24 * 4,       t * 6144 * 2,       t * 25600 * 2,                   t * 384 * 4,   t * 6 * 4,    t * 6 * 4,    t * 7 * 2304 * 2,
            t * 7 * 2304 * 2, t * 7 * 2304 * 2,   t * 7 * 5120 * 2,                t * 2304 * 2,  t * 2304 * 2, t * 2304 * 2, t * 5120 * 2,
            t * 129280 * 4,   projection_bytes,   if (grouped) |g| g.bytes else 0, stage * 256,   stage * 256,  stage * 8,    stage * 8,
            stage * 4,        stage * 4,          t * 7 * 4,                       t * 7 * 4,
        };
        var out = Plan{ .rows = rows, .lanes = lanes, .streamed = streamed, .offsets = undefined, .bytes = 0, .projection_bytes = projection_bytes };
        for (sizes, 0..) |size, i| {
            out.offsets[i] = std.mem.alignForward(usize, out.bytes, 16);
            out.bytes = try std.math.add(usize, out.offsets[i], size);
        }
        out.bytes = std.mem.alignForward(usize, out.bytes, 16384);
        return out;
    }
};

pub const Workspace = struct {
    buffer: mtl.Buffer,
    plan: Plan,

    pub fn init(device: mtl.Device, plan: Plan, token_map: []const u8) !Workspace {
        if (token_map.len != hash.raw_vocab * 4) return error.InvalidTokenMap;
        const buffer = try device.buffer(plan.bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        const w = Workspace{ .buffer = buffer, .plan = plan };
        @memcpy(buffer.contents()[plan.offsets[@backingInt(Field.token_map)]..][0..token_map.len], token_map);
        @memset(buffer.contents()[plan.offsets[@backingInt(Field.history)]..][0 .. @as(usize, plan.lanes) * 16], 0);
        const slots: [*]u32 = @ptrCast(@alignCast(buffer.contents() + plan.offsets[@backingInt(Field.moe_slots)]));
        for (0..plan.rows) |r| for (0..7) |j| {
            slots[r * 7 + j] = moeSlot(plan.rows, r, j);
        };
        return w;
    }

    pub fn deinit(w: *Workspace) void {
        w.buffer.deinit();
    }
    pub fn view(w: *const Workspace, field: Field) View {
        return .{ .buffer = w.buffer, .offset = w.plan.offsets[@backingInt(field)] };
    }
    /// Slot s's Engram cursor: lane 0 keeps the single-lane layout byte for byte.
    pub fn historyView(w: *const Workspace, slot: u32) !View {
        if (slot >= w.plan.lanes) return error.InvalidLaneSlot;
        return .{ .buffer = w.buffer, .offset = w.plan.offsets[@backingInt(Field.history)] + @as(usize, slot) * 16 };
    }
    pub fn projection(w: *const Workspace) exl3.View {
        const v = w.view(.projection);
        return .{ .buffer = v.buffer, .offset = v.offset };
    }
    /// The projection scratch at `offset` bytes into its field.
    pub fn projectionAt(w: *const Workspace, offset: usize) exl3.View {
        const v = w.view(.projection);
        return .{ .buffer = v.buffer, .offset = v.offset + offset };
    }
    pub fn grouping(w: *const Workspace, rows: u32) !groups.Scratch {
        if (rows < 2 or rows > w.plan.rows) return error.InvalidPrefillChunk;
        const v = w.view(.grouping);
        return .{ .buffer = v.buffer, .offset = v.offset, .plan = try groups.Plan.init(@as(usize, rows) * 6, 384) };
    }
    pub fn bytes(w: *const Workspace, field: Field, len: usize) []u8 {
        const v = w.view(field);
        return (v.buffer.contents() + v.offset)[0..len];
    }
    pub fn words(w: *const Workspace, field: Field, len: usize) []u32 {
        const v = w.view(field);
        return @as([*]u32, @ptrCast(@alignCast(v.buffer.contents() + v.offset)))[0..len];
    }
    pub fn tokenIds(w: *const Workspace, rows: usize) []u32 {
        const v = w.view(.ids);
        return @as([*]u32, @ptrCast(@alignCast(v.buffer.contents() + v.offset)))[0..rows];
    }
    pub fn logits(w: *const Workspace, rows: usize) []const f32 {
        const v = w.view(.logits);
        return @as([*]const f32, @ptrCast(@alignCast(v.buffer.contents() + v.offset)))[0 .. rows * 129280];
    }
};
