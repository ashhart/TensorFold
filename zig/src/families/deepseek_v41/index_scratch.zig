//! Bounds for index scratch at intermediate starts and ragged row counts, before workspace sharing and pricing.

const std = @import("std");
const buffers = @import("buffers.zig");
const block = @import("block.zig");
const Config = @import("config.zig").Config;
const longpf = @import("longpf.zig");

/// Dispatch samples seed reachable branches; enlarge their roles and include every possible row-block position.
pub fn plan(p: *buffers.Plan, rows: u32, o: block.Options, cfg: *const Config, comptime tag: []const u8) !void {
    // Ratio one bounds every compressed layer, including a row block's end.
    const keys = @max(1, o.limit + rows);
    const topk: i64 = cfg.index_topk;
    const step = @min(@as(i64, 16), longpf.plan(rows, keys, topk));
    const count: usize = @intCast(@divFloor(@as(i64, rows) + step - 1, step));
    for (0..count) |j| {
        var name: [96]u8 = undefined;
        const role = try std.fmt.bufPrint(&name, "L." ++ tag ++ "ix.pos{d}", .{j});
        const g = try p.sizes.getOrPut(p.a, role);
        if (!g.found_existing) {
            g.key_ptr.* = try p.a.dupe(u8, role);
            g.value_ptr.* = 0;
        }
        g.value_ptr.* = @max(g.value_ptr.*, 4);
    }
    const candidate_keys = @as(i64, cfg.candidate_blocks) * cfg.candidate_block_size;
    const nk = @max(keys, candidate_keys);
    // Materialised rows use budget/12 elements, with a minimum block of 16 rows.
    const elements = @max(@divFloor(o.index_budget, 12), 16 * nk);
    // Reindex additionally holds candidate positions, so its row budget divides by 16.
    const candidate_elements = @max(@divFloor(o.index_budget, 16), 16 * candidate_keys);
    const stream_bytes = @max(longpf.budget, longpf.splits(keys) * 2 * topk * 8);
    const sizes = [_]struct { []const u8, i64 }{
        .{ "L." ++ tag ++ "ix.scores", elements * 4 },
        .{ "L." ++ tag ++ "ix.keys", @max(elements * 8, @divFloor(stream_bytes, 2)) },
        .{ "L." ++ tag ++ "ix.ckeys", candidate_elements * 4 },
        .{ "L." ++ tag ++ "ix.bkeys", (@divFloor(elements, cfg.candidate_block_size) + rows) * 8 },
        .{ "L." ++ tag ++ "ix.stream", stream_bytes },
    };
    for (sizes) |entry| if (p.sizes.getPtr(entry[0])) |size| {
        size.* = @max(size.*, @as(u64, @intCast(entry[1])));
    };
}
