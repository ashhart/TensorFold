//! A DFlash2 block whose window was accepted on the GPU (glm_draft_follow.metal): ids, positions, mask from the count.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const cfg = @import("../qwen27/dflash/config.zig");
const dattn = @import("metal_dflash_attention.zig");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// glm_follow_stream: window rows from `first`, context keys [lo, lo + context) then block rows; `begin` its start.
pub const Stream = extern struct { first: u64, lo: u64, begin: u64, context: u32, keys: u32 };
/// glm_follow_args: `streams` blocks of `block` rows, the first `trained` blind to the rest; `mask_stride` keys a row.
pub const Args = extern struct { block: u32, streams: u32, trained: u32, window: u32, mask_token: u32, mask_stride: u32, rows: u32, pad: u32 = 0 };

/// A stream's window: its context's start before the window, the window's first position and its rows.
pub const Window = struct { begin: u64, first: u64, rows: u32 };

/// A pass planned at each stream's longest context (every window row kept): the kernel's and the attention's tables.
pub const Plan = struct { args: Args, streams: [dattn.max_streams]Stream, keys: [dattn.max_streams]dattn.Stream };

/// The plan for `windows` at `block` rows a stream over K/V rings of `ring` rows.
pub fn plan(c: cfg.Config, windows: []const Window, block: u32, ring: u32) !Plan {
    if (windows.len == 0 or windows.len > dattn.max_streams or block < 2 or block > dattn.max_block) return error.BadFollowPlan;
    var p: Plan = .{ .args = .{ .block = block, .streams = @intCast(windows.len), .trained = c.block, .window = c.window, .mask_token = c.mask, .mask_stride = 0, .rows = @intCast(windows.len * block) }, .streams = undefined, .keys = undefined };
    for (windows, 0..) |w, k| {
        if (w.rows == 0 or w.begin > w.first) return error.BadFollowPlan;
        const lo = @max(w.begin, (w.first + 1) -| (@as(u64, c.window) - 1)); // context start with zero kept drafts
        const context: u32 = @intCast(w.first + w.rows - lo);
        if (context > ring) return error.BadFollowPlan;
        p.streams[k] = .{ .first = w.first, .lo = lo, .begin = w.begin, .context = context, .keys = context + block };
        p.keys[k] = .{ .keys = context + block, .context = context, .at = @intCast(lo % ring) };
        p.args.mask_stride = @max(p.args.mask_stride, context + block);
    }
    return p;
}

/// The kernel's output on the host: `accepted` two words a stream (drafts kept, anchor) into ids, positions, mask.
pub fn fill(p: Plan, accepted: []const u32, ids: []u32, positions: []u64, mask: []u8) void {
    const a = p.args;
    for (0..a.rows) |row| {
        const s = row / a.block;
        const q: u32 = @intCast(row % a.block);
        const t = p.streams[s];
        const end = t.first + accepted[2 * s] + 1;
        ids[row] = if (q == 0) accepted[2 * s + 1] else a.mask_token;
        positions[row] = end + q;
        const reach = end + q + 1;
        const from = @max(t.begin, reach -| a.window);
        for (0..a.mask_stride) |key| {
            const visible = if (key < t.context) blk: {
                const pos = t.lo + key;
                break :blk pos >= from and pos < end;
            } else if (key < t.keys) !(q < a.trained and a.block > a.trained and key - t.context >= a.trained) else false;
            mask[row * a.mask_stride + key] = @intFromBool(visible);
        }
    }
}

pub const Follow = struct {
    pipe: mtl.Pipeline,

    /// `edit` replaces one text of the source (a contract control's planted fault), else null.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, edit: ?[2][]const u8) !Follow {
        const text = if (edit) |x| blk: {
            if (std.mem.indexOf(u8, sources.glm_draft_follow, x[0]) == null) return error.ControlEditMissing;
            break :blk try std.mem.replaceOwned(u8, a, sources.glm_draft_follow, x[0], x[1]);
        } else try a.dupe(u8, sources.glm_draft_follow);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{ .pipe = try mtl.Pipeline.init(device, lib, "glm_draft_follow", false) };
    }

    pub fn deinit(x: Follow) void {
        x.pipe.deinit();
    }

    /// `p`'s ids, positions and mask from `accepted` (two words a stream), a thread a key of a row.
    pub fn encode(x: Follow, e: mtl.ComputeEncoder, p: Plan, accepted: Ref, ids: Ref, positions: Ref, mask: Ref) !void {
        const a = p.args;
        if (a.rows == 0 or a.mask_stride == 0 or positions.off % 8 != 0 or accepted.off % 4 != 0 or ids.off % 4 != 0) return error.BadFollowBuffer;
        if (mask.off + @as(usize, a.rows) * a.mask_stride > mask.buf.length() or positions.off + @as(usize, a.rows) * 8 > positions.buf.length() or ids.off + @as(usize, a.rows) * 4 > ids.buf.length()) return error.BadFollowBuffer;
        e.setPipeline(x.pipe);
        e.setBuffer(accepted.buf, accepted.off, 0);
        e.setValue(a, 1);
        e.setValue(p.streams, 2);
        e.setBuffer(ids.buf, ids.off, 3);
        e.setBuffer(positions.buf, positions.off, 4);
        e.setBuffer(mask.buf, mask.off, 5);
        e.dispatchThreads(mtl.Size.of(a.mask_stride, a.rows, 1), mtl.Size.of(32, 1, 1));
    }
};

test "a plan at the longest context, filled at a kept count, masks as the context's own plan at that count does" {
    var c: cfg.Config = .{};
    c.window = 64;
    const fill_mask = @import("metal_dflash_context.zig").fillMask;
    for ([_]u64{ 0, 30, 100 }) |first| for ([_]u32{ 8, 16 }) |block| for (0..8) |kept| {
        const begin: u64 = if (first == 100) 40 else 0;
        const p = try plan(c, &.{.{ .begin = begin, .first = first, .rows = 8 }}, block, c.window + 16);
        var ids: [16]u32 = undefined;
        var positions: [16]u64 = undefined;
        var mask: [16 * 96]u8 = undefined;
        const accepted = [2]u32{ @intCast(kept), 77 };
        fill(p, &accepted, ids[0..block], positions[0..block], mask[0 .. block * p.args.mask_stride]);
        // the context's own plan at this count: [start, end) then the block
        const end = first + kept + 1;
        const start = @max(begin, end -| (c.window - 1));
        const rows: u32 = @intCast(end - start);
        var own: [16 * 96]u8 = undefined;
        try fill_mask(own[0 .. block * (rows + block)], rows, block, c.window, c.block);
        for (0..block) |q| {
            try std.testing.expectEqual(if (q == 0) @as(u32, 77) else c.mask, ids[q]);
            try std.testing.expectEqual(end + q, positions[q]);
            for (0..p.args.mask_stride) |key| {
                const want: u8 = if (key < p.streams[0].context) blk: {
                    const pos = p.streams[0].lo + key;
                    break :blk if (pos < start or pos >= end) 0 else own[q * (rows + block) + (pos - start)];
                } else own[q * (rows + block) + rows + (key - p.streams[0].context)];
                try std.testing.expectEqual(want, mask[q * p.args.mask_stride + key]);
            }
        }
    };
}
