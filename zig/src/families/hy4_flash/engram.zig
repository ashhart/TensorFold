//! Hy4-flash's Engram hashes: a position's 2- and 3-gram mixed into one id a hash head, as the reference reads them.
const std = @import("std");

/// The release's n-gram sizes and hash heads: 2- and 3-grams, 8 heads each.
pub const max_ngram = 3;
pub const heads_per_ngram = 8;
pub const heads = (max_ngram - 1) * heads_per_ngram;

/// A layer's odd n-gram multipliers: NumPy's default_rng(seed + 10007 layer_id) draw (engram_hashes.py).
pub const Multipliers = struct { layer_id: u32, m: [max_ngram]u64 };

pub const release = [_]Multipliers{
    .{ .layer_id = 2, .m = .{ 81853575799317, 39690085502883, 24229915018165 } },
    .{ .layer_id = 12, .m = .{ 19457948045719, 44729556510419, 84779352922315 } },
    .{ .layer_id = 22, .m = .{ 72974230057307, 21530602121221, 55998057243097 } },
    .{ .layer_id = 32, .m = .{ 29128509562787, 16745155128523, 22474588302733 } },
};

pub fn multipliers(layer_id: u32) ?[max_ngram]u64 {
    for (release) |r| if (r.layer_id == layer_id) return r.m;
    return null;
}

/// A position's hash ids from the table ids of its token and the two before it (pad where none).
pub fn hash(compressed: [max_ngram]u32, m: [max_ngram]u64, moduli: []const u32, out: []u32) void {
    var mixed: [max_ngram]u64 = undefined;
    for (compressed, m, 0..) |c, k, j| mixed[j] = @as(u64, c) * k;
    var head: usize = 0;
    for (2..max_ngram + 1) |size| {
        var mix = mixed[0];
        for (1..size) |j| mix ^= mixed[j];
        for (0..heads_per_ngram) |_| {
            out[head] = @intCast(mix % moduli[head]);
            head += 1;
        }
    }
}

const t = std.testing;

test "the release's hash ids at each Engram layer equal the reference's, start and out-of-table tokens included" {
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, @embedFile("fixtures/engram_hashes.json"), .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const config = try std.json.parseFromSlice(std.json.Value, t.allocator, @embedFile("fixtures/config.json"), .{});
    defer config.deinit();
    const text = config.value.object.get("text_config").?.object;
    const sizes = text.get("engram_layer_head_vocab_sizes").?.object;
    const pad: u32 = @intCast(text.get("engram_pad_id").?.integer);
    const table = root.get("compressed").?.object;
    const tokens = root.get("tokens").?.array.items;
    var key: [16]u8 = undefined;
    const pad_c: u32 = @intCast(table.get(try std.fmt.bufPrint(&key, "{d}", .{pad})).?.integer);
    var it = root.get("layers").?.object.iterator();
    var checked: u32 = 0;
    while (it.next()) |layer| {
        const id = try std.fmt.parseInt(u32, layer.key_ptr.*, 10);
        const m = multipliers(id).?;
        for (layer.value_ptr.object.get("multipliers").?.array.items, m) |want, got| try t.expectEqual(@as(u64, @intCast(want.integer)), got);
        var moduli: [heads]u32 = undefined;
        var n: usize = 0;
        for (sizes.get(layer.key_ptr.*).?.array.items) |order| for (order.array.items) |s| {
            moduli[n] = @intCast(s.integer);
            n += 1;
        };
        for (layer.value_ptr.object.get("hash_ids").?.array.items, 0..) |want_row, pos| {
            var c: [max_ngram]u32 = undefined;
            for (0..max_ngram) |j| {
                if (pos < j) {
                    c[j] = pad_c;
                    continue;
                }
                const v = table.get(try std.fmt.bufPrint(&key, "{d}", .{tokens[pos - j].integer})).?;
                c[j] = if (v == .null) pad_c else @intCast(v.integer);
            }
            var got: [heads]u32 = undefined;
            hash(c, m, &moduli, &got);
            for (want_row.array.items, got) |w, g| try t.expectEqual(@as(u32, @intCast(w.integer)), g);
            checked += 1;
        }
    }
    try t.expectEqual(@as(u32, 4 * 18), checked);
}
