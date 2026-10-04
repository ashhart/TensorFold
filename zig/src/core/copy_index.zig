//! decode.CopyIndex: the longest continuation of an earlier copy of the last `min_match` tokens (n-gram starts).

const std = @import("std");

pub const min_match = 8;
const Key = [min_match]u32;

pub const CopyIndex = struct {
    gpa: std.mem.Allocator,
    ctx: std.ArrayList(u32) = .empty,
    starts: std.AutoHashMapUnmanaged(Key, std.ArrayList(u32)) = .empty,

    pub fn init(gpa: std.mem.Allocator, context: []const u32) !CopyIndex {
        var x: CopyIndex = .{ .gpa = gpa };
        errdefer x.deinit();
        try x.extend(context);
        return x;
    }

    pub fn deinit(x: *CopyIndex) void {
        var it = x.starts.valueIterator();
        while (it.next()) |v| v.deinit(x.gpa);
        x.starts.deinit(x.gpa);
        x.ctx.deinit(x.gpa);
    }

    pub fn extend(x: *CopyIndex, tokens: []const u32) !void {
        for (tokens) |t| {
            try x.ctx.append(x.gpa, t);
            if (x.ctx.items.len < min_match) continue;
            const s = x.ctx.items.len - min_match;
            const key: Key = x.ctx.items[s..][0..min_match].*;
            const got = try x.starts.getOrPut(x.gpa, key);
            if (!got.found_existing) got.value_ptr.* = .empty;
            try got.value_ptr.append(x.gpa, @intCast(s));
        }
    }

    /// Up to `out.len` tokens that followed the latest earlier copy of the last 8 (none shorter than 8), copied out.
    pub fn chain(x: *const CopyIndex, out: []u32) []const u32 {
        const max_nodes = out.len;
        const ctx = x.ctx.items;
        if (ctx.len < 2 * min_match or max_nodes < 1) return &.{};
        const key: Key = ctx[ctx.len - min_match ..][0..min_match].*;
        const list = x.starts.get(key) orelse return &.{};
        var best: []const u32 = &.{};
        var i = list.items.len;
        while (i > 0) {
            i -= 1;
            const start = list.items[i];
            if (start > ctx.len - min_match - 1) continue;
            const from = start + min_match;
            const cont = ctx[from..@min(ctx.len, from + max_nodes)];
            if (cont.len > best.len) {
                best = cont;
                if (best.len == max_nodes) break;
            }
        }
        if (best.len < min_match) return &.{};
        @memcpy(out[0..best.len], best);
        return out[0..best.len];
    }
};

test "copies the continuation of the latest earlier match" {
    const a = std.testing.allocator;
    var seq: [40]u32 = undefined;
    for (&seq, 0..) |*v, i| v.* = @intCast(i % 20);
    var x = try CopyIndex.init(a, &seq);
    defer x.deinit();
    var buf: [15]u32 = undefined;
    const got = x.chain(&buf);
    try std.testing.expectEqual(@as(usize, 15), got.len);
    try std.testing.expectEqual(@as(u32, 0), got[0]);
    try std.testing.expectEqual(@as(usize, 0), x.chain(buf[0..0]).len);
}
