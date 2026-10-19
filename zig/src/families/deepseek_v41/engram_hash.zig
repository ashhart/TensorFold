//! Exact DeepSeek AI inference/engram.py hashes; metadata from the released Flash configuration.
const std = @import("std");

pub const order = 4;
pub const heads = 8;
pub const columns = 24;
pub const dead: i32 = -1;
pub const raw_vocab = 129280;
pub const compressed_vocab = 99092;
pub const pad_id = 2;

pub const Layer = extern struct {
    layer_id: u32,
    rows: u32,
    multipliers: [order]u64,
    primes: [columns]u32,
    offsets: [columns]u32,
};

fn layer(id: u32, rows: u32, multipliers: [order]u64, primes: [columns]u32) Layer {
    var offsets: [columns]u32 = undefined;
    var offset: u32 = 0;
    for (primes, 0..) |p, i| {
        offsets[i] = offset;
        offset += p;
    }
    std.debug.assert(offset == rows);
    return .{ .layer_id = id, .rows = rows, .multipliers = multipliers, .primes = primes, .offsets = offsets };
}

pub const layers = [_]Layer{
    layer(1, 384006168, .{ 76632096046245, 4839876093313, 35959672319349, 73987337458391 }, .{
        16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183,
        16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321,
        16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463,
    }),
    layer(14, 384016682, .{ 67716810739261, 51510806800915, 30921347202721, 82619226485591 }, .{
        16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627,
        16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769,
        16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889,
    }),
};

fn word(bytes: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, bytes[at..][0..4], .little);
}

/// Borrowed packed metadata; the caller owns the blob; adapters own any device upload.
pub const TokenMap = struct {
    bytes: []const u8,

    pub fn parse(blob: []const u8) !TokenMap {
        const header = 96;
        const layer_size = 232;
        const map_offset = header + layers.len * layer_size;
        if (blob.len != map_offset + raw_vocab * 4 or !std.mem.eql(u8, blob[0..8], "DSENGR01"))
            return error.InvalidEngramMetadata;
        const expected = [_]u32{ raw_vocab, compressed_vocab, pad_id, order, heads, layers.len };
        for (expected, 0..) |value, i| if (word(blob, 8 + i * 4) != value) return error.InvalidEngramMetadata;
        // Reject a tokenizer which happened to have the right size but normalized differently.
        const tokenizer_sha = [_]u8{
            0xc9, 0x0d, 0xfa, 0x01, 0x24, 0x9d, 0xb1, 0xbe, 0x42, 0x45, 0x78, 0x0a, 0x05, 0x2e, 0xde, 0x75,
            0x2e, 0x13, 0x61, 0xc6, 0x12, 0xac, 0x6d, 0x08, 0xe2, 0xbd, 0xad, 0xa7, 0xd5, 0x99, 0x47, 0x6b,
        };
        if (!std.mem.eql(u8, blob[32..64], &tokenizer_sha)) return error.InvalidEngramTokenizer;
        for (layers, 0..) |meta, i| {
            const at = header + i * layer_size;
            if (word(blob, at) != meta.layer_id or word(blob, at + 4) != meta.rows) return error.InvalidEngramMetadata;
            for (meta.multipliers, 0..) |v, j| {
                if (std.mem.readInt(u64, blob[at + 8 + j * 8 ..][0..8], .little) != v)
                    return error.InvalidEngramMetadata;
            }
            for (meta.primes, 0..) |v, j| if (word(blob, at + 40 + j * 4) != v) return error.InvalidEngramMetadata;
            for (meta.offsets, 0..) |v, j| if (word(blob, at + 136 + j * 4) != v) return error.InvalidEngramMetadata;
        }
        const result = TokenMap{ .bytes = blob[map_offset..] };
        for (0..raw_vocab) |id| if (word(result.bytes, id * 4) >= compressed_vocab) return error.InvalidEngramTokenMap;
        if (try result.get(2) != pad_id) return error.InvalidEngramTokenMap;
        const map_sha = [_]u8{
            0xc6, 0x0a, 0x86, 0x32, 0x2e, 0xc1, 0x7b, 0x41, 0x42, 0xbf, 0xef, 0x3c, 0x57, 0xa8, 0xd8, 0x1f,
            0xb4, 0x28, 0x55, 0x0c, 0xdf, 0x48, 0x7f, 0x88, 0xa4, 0x32, 0x0c, 0xb5, 0x9f, 0xe4, 0x64, 0x81,
        };
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(result.bytes, &digest, .{});
        if (!std.mem.eql(u8, &digest, &map_sha)) return error.InvalidEngramTokenMap;
        return result;
    }

    pub fn get(self: TokenMap, token: u32) !u32 {
        if (token >= self.bytes.len / 4) return error.InvalidEngramToken;
        return word(self.bytes, @as(usize, token) * 4);
    }
};

/// Only the previous three positions are retained, including DEAD positions; newest is first.
pub const History = struct {
    tail: [order - 1]i32 = .{ dead, dead, dead },
    position: u64 = 0,

    pub fn reset(self: *History) void {
        self.* = .{};
    }

    pub fn snapshot(self: History) History {
        return self;
    }

    /// Restore before replaying the accepted prefix of a speculative chunk.
    pub fn rollback(self: *History, saved: History) void {
        self.* = saved;
    }

    pub fn push(self: *History, compressed: i32, meta: Layer) ![columns]u32 {
        if (self.position == std.math.maxInt(u64)) return error.InvalidEngramChunk;
        if (compressed < dead or compressed >= compressed_vocab) return error.InvalidEngramCompressedId;
        var tokens: [order]u32 = @splat(pad_id);
        var blocked = compressed == dead;
        if (!blocked) tokens[0] = @intCast(compressed);
        for (0..order - 1) |i| {
            blocked = blocked or self.position <= i or self.tail[i] == dead;
            if (!blocked) tokens[i + 1] = @intCast(self.tail[i]);
        }
        var result: [columns]u32 = undefined;
        hashValues(&tokens, &meta.multipliers, &meta.primes, &meta.offsets, heads, &result);
        self.tail[2] = self.tail[1];
        self.tail[1] = self.tail[0];
        self.tail[0] = compressed;
        self.position += 1;
        return result;
    }
};

/// Single sequence, token mask bytes (0=DEAD), output [tokens,24], no allocations or lost chunk history.
pub fn hashChunk(history: *History, map: TokenMap, meta: Layer, ids: []const u32, mask: ?[]const u8, out: [][columns]u32) !void {
    if (out.len != ids.len or (mask != null and mask.?.len != ids.len)) return error.InvalidEngramChunk;
    if (ids.len > std.math.maxInt(u64) - history.position) return error.InvalidEngramChunk;
    // Validate the entire chunk before mutating history.
    for (ids) |token| _ = try map.get(token);
    for (ids, 0..) |token, i| {
        const compressed: i32 = if (mask != null and mask.?[i] == 0) dead else @intCast(try map.get(token));
        out[i] = try history.push(compressed, meta);
    }
}

/// One hash equation for canonical history and the chain's window/lookback adapter.
pub fn hashValues(tokens: []const u32, mult: anytype, primes: anytype, offsets: anytype, head_count: usize, out: anytype) void {
    var rolling: u64 = 0;
    for (tokens, 0..) |token, shift| {
        rolling ^= @as(u64, token) *% @as(u64, @intCast(mult[shift]));
        if (shift == 0) continue;
        for (0..head_count) |head| {
            const col = (shift - 1) * head_count + head;
            out[col] = @intCast(rolling % @as(u64, @intCast(primes[col])) + @as(u64, @intCast(offsets[col])));
        }
    }
}
