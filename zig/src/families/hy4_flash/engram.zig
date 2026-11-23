//! Engram row selection for Tencent's released Hy4-flash vocabulary map.
const std = @import("std");
const config = @import("config.zig");
const hashValues = @import("engram_hash_core").hashValues;

pub const order = 3;
pub const heads = 8;
pub const columns = (order - 1) * heads;
const lookup_sha = "87c7b2e1c2fc227ba7fdec4dd7a867533072eb2b73a85cc54189174ccfaf5f3f";

pub const TokenMap = struct {
    bytes: []const u8,
    pad: u32,

    pub fn open(bytes: []const u8, text: config.Text) !TokenMap {
        if (bytes.len != @as(usize, text.engram_lookup_table_size) * 4 or
            text.engram_pad_id >= text.engram_lookup_table_size) return error.BadEngramMap;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        if (!std.mem.eql(u8, &hex, lookup_sha)) return error.UnknownEngramVocabulary;
        return .{ .bytes = bytes, .pad = text.engram_pad_id };
    }

    pub fn normalize(self: TokenMap, id: u32) u32 {
        const slot = if (id < self.bytes.len / 4) id else self.pad;
        return std.mem.readInt(u32, self.bytes[@as(usize, slot) * 4 ..][0..4], .little);
    }
};

pub const Layer = struct {
    multipliers: [order]u64,
    primes: [columns]u32,
    offsets: [columns]u32,

    pub fn open(text: config.Text, number: u32) !Layer {
        if (text.engram_seed != 0 or text.engram_max_ngram_size != order or text.engram_n_head_per_ngram != heads)
            return error.UnsupportedEngramHash;
        // The reference PCG64 draws use the checked vocabulary's maximum ID plus one, which is 92620.
        const multipliers: [order]u64 = switch (number) {
            2 => .{ 81853575799317, 39690085502883, 24229915018165 },
            12 => .{ 19457948045719, 44729556510419, 84779352922315 },
            22 => .{ 72974230057307, 21530602121221, 55998057243097 },
            32 => .{ 29128509562787, 16745155128523, 22474588302733 },
            else => return error.UnknownEngramLayer,
        };
        _ = try text.engramRows(number);
        var key_storage: [10]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_storage, "{d}", .{number});
        const groups = text.engram_layer_head_vocab_sizes.map.get(key) orelse return error.UnknownEngramLayer;
        var result: Layer = .{ .multipliers = multipliers, .primes = undefined, .offsets = undefined };
        var row: u32 = 0;
        for (groups, 0..) |group, n| for (group, 0..) |prime, head| {
            const col = n * heads + head;
            result.primes[col] = prime;
            result.offsets[col] = row;
            row = try std.math.add(u32, row, prime);
        };
        return result;
    }

    pub fn hash(self: Layer, map: TokenMap, tokens: [order]u32) [columns]u32 {
        var normalized: [order]u32 = undefined;
        for (tokens, &normalized) |token, *value| value.* = map.normalize(token);
        return self.hashNormalized(normalized);
    }

    pub fn hashNormalized(self: Layer, normalized: [order]u32) [columns]u32 {
        var rows: [columns]u32 = undefined;
        hashValues(&normalized, &self.multipliers, &self.primes, &self.offsets, heads, &rows);
        return rows;
    }
};

test "the native Engram hash returns the upstream rows for a released token window" {
    const parsed = try config.parse(std.testing.allocator, @embedFile("config_fixture.json"));
    defer parsed.deinit();
    const layer = try Layer.open(parsed.value.text_config, 2);
    const rows = layer.hashNormalized(.{ 1, 0, 91804 });
    const expected = [_]u32{ 9395452, 11699939, 23502176, 32861672, 43851032, 53212976, 68274127, 73702352, 91770608, 93761842, 110295210, 116456011, 133373173, 137621161, 154112274, 155288835 };
    try std.testing.expectEqualSlices(u32, &expected, &rows);
}
