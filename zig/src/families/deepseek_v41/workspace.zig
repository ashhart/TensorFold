//! Fixed chunk-sized activations: no layer history or context-by-query score matrices.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("core").exl3_projection;
const View = @import("view.zig").View;
const hash = @import("engram_hash.zig");
const groups = @import("core").expert_groups;

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
    /// The fused gate+up dispatch's plan: twelve picks a token (six gate, six up one table entry further).
    grouping_fused,
    /// The fused gate+up dispatch's pick table: two six-slot rows a token.
    fused_ids,
    /// The input row each fused pick reads: rows 0..6T-1 and 6T..12T-1 both read their own token.
    fused_sources,
    /// The hyper-connection boundary's second stream buffer: the fused expand writes here and the views swap.
    stream_next,
    /// The boundary's partial sums: 20 slices of 1024 columns x 25 (24 mixes and the squares) a row.
    hc_part,
};
const count = std.enums.values(Field).len;

pub const Plan = struct {
    rows: u32,
    lanes: u32,
    offsets: [count]usize,
    bytes: usize,

    pub fn init(rows: u32) !Plan {
        return initLanes(rows, 1);
    }

    pub fn initLanes(rows: u32, lanes: u32) !Plan {
        if (rows == 0 or rows > 16384) return error.InvalidPrefillChunk;
        if (lanes == 0 or lanes > 256) return error.InvalidLaneCount;
        const t: usize = rows;
        const grouped = if (rows > 1) try groups.Plan.init(t * 6, 384, groups.tileRows()) else null;
        // The fused plan only ever covers a window (the fused call is taken below eight rows).
        const window_rows: usize = @min(t, 8);
        const fused_group = if (rows > 1) try groups.Plan.init(window_rows * 12, 768, groups.tileRows()) else null;
        const dense_scratch = try exl3.scratchBytes(5120, 129280, t);
        const grouped_scratch = if (grouped) |g| @max(try exl3.scratchBytes(5120, 2304, g.padded), try exl3.scratchBytes(2304, 5120, g.padded)) else @max(try exl3.scratchBytes(2304, 5120, t * 6), try exl3.scratchBytes(5120, 2304, t * 12));
        // The fused plan pads twelve slots a token to 64-row tiles (five rows -> 60 slots -> 3,840 padded).
        const fused_scratch = if (fused_group) |g| @max(try exl3.scratchBytes(5120, 2304, g.padded), try exl3.scratchBytes(2304, 5120, g.padded)) else 0;
        const projection_bytes = @max(dense_scratch, @max(grouped_scratch, fused_scratch));
        const sizes = [_]usize{
            t * 4,            hash.raw_vocab * 4, @as(usize, lanes) * 16,          t * 20480 * 2,                       t * 5120 * 2, t * 5120 * 2, t * 5120 * 2,
            t * 4 * 4,        t * 4 * 4,          t * 4 * 4,                       t * 16 * 4,                          t * 24 * 4,   t * 4,        t * 24 * 4,
            t * 24 * 4,       t * 6144 * 2,       t * 25600 * 2,                   t * 384 * 4,                         t * 6 * 4,    t * 6 * 4,    t * 6 * 2304 * 2,
            t * 6 * 2304 * 2, t * 6 * 2304 * 2,   t * 6 * 5120 * 2,                t * 2304 * 2,                        t * 2304 * 2, t * 2304 * 2, t * 5120 * 2,
            t * 129280 * 4,   projection_bytes,   if (grouped) |g| g.bytes else 0, if (fused_group) |g| g.bytes else 0, t * 12 * 4,   t * 12 * 4,   t * 20480 * 2,
            t * 2000,
        };
        var out = Plan{ .rows = rows, .lanes = lanes, .offsets = undefined, .bytes = 0 };
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
    /// The shared gate/up fusion's two maps (input row and descriptor id).
    shared_maps: mtl.Buffer,
    plan: Plan,

    pub const shared_map_slots: usize = 72;

    pub fn init(device: mtl.Device, plan: Plan, token_map: []const u8) !Workspace {
        if (token_map.len != hash.raw_vocab * 4) return error.InvalidTokenMap;
        const buffer = try device.buffer(plan.bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buffer.deinit();
        const shared_maps = try device.buffer(shared_map_slots * 2 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer shared_maps.deinit();
        {
            const src: [*]u32 = @ptrCast(@alignCast(shared_maps.contents()));
            var off: usize = 0;
            for (1..9) |rows| {
                for (0..2 * rows) |r| {
                    src[off + r] = @intCast(r % rows);
                    src[shared_map_slots + off + r] = @intCast(r / rows);
                }
                off += 2 * rows;
            }
        }
        const w = Workspace{ .buffer = buffer, .shared_maps = shared_maps, .plan = plan };
        @memcpy(buffer.contents()[plan.offsets[@backingInt(Field.token_map)]..][0..token_map.len], token_map);
        @memset(buffer.contents()[plan.offsets[@backingInt(Field.history)]..][0 .. @as(usize, plan.lanes) * 16], 0);
        return w;
    }

    pub fn deinit(w: *Workspace) void {
        w.buffer.deinit();
        w.shared_maps.deinit();
    }
    /// The input-row map for a `rows`-row window (rows 1..8): the first `2*rows` entries of the window's block.
    pub fn sharedMapInputs(w: *const Workspace, rows: u32) View {
        const off: usize = @as(usize, rows) * (@as(usize, rows) - 1);
        return .{ .buffer = w.shared_maps, .offset = off * 4 };
    }
    /// The descriptor-id map for a `rows`-row window: the second half of the window's block.
    pub fn sharedMapRoutes(w: *const Workspace, rows: u32) View {
        const off: usize = @as(usize, rows) * (@as(usize, rows) - 1);
        return .{ .buffer = w.shared_maps, .offset = (shared_map_slots + off) * 4 };
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
    pub fn grouping(w: *const Workspace, rows: u32, picks: u32, experts: u32) !groups.Scratch {
        if (rows < 2 or rows > w.plan.rows) return error.InvalidPrefillChunk;
        const v = w.view(.grouping);
        return .{ .buffer = v.buffer, .offset = v.offset, .plan = try groups.Plan.init(@as(usize, rows) * picks, experts, groups.tileRows()) };
    }
    /// The fused gate+up call's plan: one token reads twelve slots.
    pub fn groupingFused(w: *const Workspace, rows: u32, picks: u32, experts: u32) !groups.Scratch {
        if (rows < 2 or rows > w.plan.rows) return error.InvalidPrefillChunk;
        const v = w.view(.grouping_fused);
        return .{ .buffer = v.buffer, .offset = v.offset, .plan = try groups.Plan.init(@as(usize, rows) * 2 * picks, 2 * experts, groups.tileRows()) };
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
