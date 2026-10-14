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
};
const count = std.enums.values(Field).len;

pub const Plan = struct {
    rows: u32,
    offsets: [count]usize,
    bytes: usize,

    pub fn init(rows: u32) !Plan {
        if (rows == 0 or rows > 4096) return error.InvalidPrefillChunk;
        const t: usize = rows;
        const grouped = if (rows > 1) try groups.Plan.init(t * 6, 384) else null;
        const projection_bytes = @max(try exl3.scratchBytes(5120, 129280, t), if (grouped) |g| @max(try exl3.scratchBytes(5120, 2304, g.padded), try exl3.scratchBytes(2304, 5120, g.padded)) else try exl3.scratchBytes(2304, 5120, t * 6));
        const sizes = [_]usize{
            t * 4,            hash.raw_vocab * 4, 16,                              t * 20480 * 2, t * 5120 * 2, t * 5120 * 2, t * 5120 * 2,
            t * 4 * 4,        t * 4 * 4,          t * 4 * 4,                       t * 16 * 4,    t * 24 * 4,   t * 4,        t * 24 * 4,
            t * 24 * 4,       t * 6144 * 2,       t * 25600 * 2,                   t * 384 * 4,   t * 6 * 4,    t * 6 * 4,    t * 6 * 2304 * 2,
            t * 6 * 2304 * 2, t * 6 * 2304 * 2,   t * 6 * 5120 * 2,                t * 2304 * 2,  t * 2304 * 2, t * 2304 * 2, t * 5120 * 2,
            t * 129280 * 4,   projection_bytes,   if (grouped) |g| g.bytes else 0,
        };
        var out = Plan{ .rows = rows, .offsets = undefined, .bytes = 0 };
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
        @memset(buffer.contents()[plan.offsets[@backingInt(Field.history)]..][0..16], 0);
        return w;
    }

    pub fn deinit(w: *Workspace) void {
        w.buffer.deinit();
    }
    pub fn view(w: *const Workspace, field: Field) View {
        return .{ .buffer = w.buffer, .offset = w.plan.offsets[@backingInt(field)] };
    }
    pub fn projection(w: *const Workspace) exl3.View {
        const v = w.view(.projection);
        return .{ .buffer = v.buffer, .offset = v.offset };
    }
    pub fn grouping(w: *const Workspace, rows: u32) !groups.Scratch {
        if (rows < 2 or rows > w.plan.rows) return error.InvalidPrefillChunk;
        const v = w.view(.grouping);
        return .{ .buffer = v.buffer, .offset = v.offset, .plan = try groups.Plan.init(@as(usize, rows) * 6, 384) };
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
