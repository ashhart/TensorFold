//! A shared round packs independent stream windows; only their stateful mixers split at stream boundaries.
const std = @import("std");
pub fn Batch(comptime S: type) type {
    return struct {
        const Self = @This();
        pub const Segment = struct { state: *S, tokens: []const u32 };
        items: []const Segment,
        pub fn segments(b: *const Self) []const Segment {
            return b.items;
        }
        pub fn count(b: *const Self) !usize {
            if (b.items.len == 0) return error.Qwen4EmptyBatch;
            var total: usize = 0;
            for (b.items, 0..) |item, i| {
                if (item.tokens.len == 0 or item.tokens.len > 16) return error.Qwen4BatchWindow;
                for (b.items[0..i]) |prior| if (prior.state == item.state) return error.Qwen4DuplicateStream;
                total = try std.math.add(usize, total, item.tokens.len);
            }
            if (total > 128) return error.Qwen4BatchRows;
            return total;
        }
        pub fn begin(b: *Self) !void {
            _ = try b.count();
            for (b.items) |item| try item.state.canBegin(item.tokens.len);
            errdefer b.poison();
            for (b.items) |item| try item.state.begin(item.tokens.len);
        }
        pub fn layers(b: *const Self) usize {
            return if (b.items.len == 0) 0 else b.items[0].state.layers();
        }
        pub fn admit(b: *const Self, rows: usize) !void {
            if (try b.count() != rows) return error.Qwen4BatchRows;
            for (b.items) |item| {
                if (item.state.layers() != b.layers()) return error.Qwen4BatchLayers;
                try item.state.admit(item.tokens.len);
            }
        }
        pub fn poison(b: *Self) void {
            for (b.items) |item| item.state.poison();
        }
        pub fn captured(b: *Self, layer: usize) !void {
            for (b.items) |item| try item.state.captured(layer);
        }
        pub fn advanced(b: *Self, tokens: []const u32) !void {
            try b.admit(tokens.len);
            var off: usize = 0;
            for (b.items) |item| {
                if (!std.mem.eql(u32, item.tokens, tokens[off..][0..item.tokens.len])) return error.Qwen4BatchTokens;
                off += item.tokens.len;
            }
            for (b.items) |item| try item.state.advanced(item.tokens);
        }
    };
}
