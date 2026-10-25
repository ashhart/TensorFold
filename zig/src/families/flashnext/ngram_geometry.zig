//! Shared Flash Next/Qwen4 n-gram table geometry. All arithmetic is checked before allocation.
const std = @import("std");
pub fn isPrime(n: u64) bool {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    var d: u64 = 3;
    while (d <= n / d) : (d += 2) if (n % d == 0) return false;
    return true;
}
pub fn nextPrime(after: u64) !u64 {
    var n = try std.math.add(u64, after, 1);
    while (!isPrime(n)) n = try std.math.add(u64, n, 1);
    return n;
}
pub fn tableRows(base: usize, heads: usize, ordinal: usize, divisor: usize) !usize {
    if (base == 0 or heads == 0 or divisor == 0) return error.InvalidNgramGeometry;
    const skip = try std.math.mul(usize, heads, ordinal);
    var prime: u64 = base - 1;
    for (0..skip) |_| prime = try nextPrime(prime);
    var total: usize = 0;
    for (0..heads) |_| {
        prime = try nextPrime(prime);
        total = try std.math.add(usize, total, std.math.cast(usize, prime) orelse return error.Overflow);
    }
    return std.math.mul(usize, try std.math.divCeil(usize, total, divisor), divisor);
}
pub fn splitmix64(value: u64) u64 {
    var v = value +% 0x9E3779B97F4A7C15;
    v = (v ^ (v >> 30)) *% 0xBF58476D1CE4E5B9;
    v = (v ^ (v >> 27)) *% 0x94D049BB133111EB;
    return v ^ (v >> 31);
}
pub fn multiplier(seed: u64, ordinal: usize, position: usize, vocab: usize) !i64 {
    if (vocab == 0) return error.InvalidNgramGeometry;
    const half = @max(1, ((@as(u64, 1) << 63) - 1) / vocab / 2);
    const layer = seed +% (@as(u64, @intCast(ordinal)) *% 10007);
    const value = layer +% 0x9E3779B97F4A7C15 *% (@as(u64, @intCast(position)) +% 1);
    return @intCast(2 * (splitmix64(value) % half) + 1);
}
test "ngram geometry: tiny successive layer primes and source multipliers" {
    try std.testing.expectEqual(@as(usize, 88), try tableRows(17, 4, 0, 8));
    try std.testing.expectEqual(@as(usize, 152), try tableRows(17, 4, 1, 8));
    try std.testing.expect(try multiplier(1234, 0, 0, 32) != try multiplier(1234, 1, 0, 32));
    try std.testing.expectError(error.InvalidNgramGeometry, tableRows(0, 4, 0, 8));
}

pub const Policy = struct {
    a: std.mem.Allocator,
    ngram: usize,
    heads_per_ngram: usize,
    vocab: usize,
    eos: u32,
    multipliers: []u64,
    sizes: []u64,
    offsets: []usize,
    pub fn init(a: std.mem.Allocator, ngram: usize, heads_per_ngram: usize, vocab: usize, eos: u32, base: usize, seed: u64, ordinal: usize) !Policy {
        if (ngram < 2 or ngram > 16 or heads_per_ngram == 0 or heads_per_ngram > 16 or vocab == 0 or eos >= vocab or base == 0) return error.InvalidNgramGeometry;
        const heads = try std.math.mul(usize, ngram - 1, heads_per_ngram);
        const multipliers = try a.alloc(u64, ngram);
        errdefer a.free(multipliers);
        const sizes = try a.alloc(u64, heads);
        errdefer a.free(sizes);
        const offsets = try a.alloc(usize, heads);
        errdefer a.free(offsets);
        for (multipliers, 0..) |*m, i| m.* = @bitCast(try multiplier(seed, ordinal, i, vocab));
        var prime: u64 = base - 1;
        for (0..try std.math.mul(usize, ordinal, heads)) |_| prime = try nextPrime(prime);
        var offset: usize = 0;
        for (sizes, offsets) |*size, *at| {
            prime = try nextPrime(prime);
            if (prime > std.math.maxInt(i64)) return error.InvalidNgramGeometry;
            size.* = prime;
            at.* = offset;
            offset = try std.math.add(usize, offset, std.math.cast(usize, prime) orelse return error.Overflow);
        }
        return .{ .a = a, .ngram = ngram, .heads_per_ngram = heads_per_ngram, .vocab = vocab, .eos = eos, .multipliers = multipliers, .sizes = sizes, .offsets = offsets };
    }
    pub fn deinit(p: Policy) void {
        p.a.free(p.offsets);
        p.a.free(p.sizes);
        p.a.free(p.multipliers);
    }
    pub fn view(p: Policy) View {
        return .{ .ngram = p.ngram, .heads_per_ngram = p.heads_per_ngram, .vocab = p.vocab, .eos = p.eos, .multipliers = p.multipliers, .sizes = p.sizes, .offsets = p.offsets };
    }
    pub fn step(p: Policy, history: []u32, token: u32, ids: []usize) !void {
        return p.view().step(history, token, ids);
    }
};

/// Borrowed policy data retains no allocator pointer, so it can follow its arena owner when that owner moves.
pub const View = struct {
    ngram: usize,
    heads_per_ngram: usize,
    vocab: usize,
    eos: u32,
    multipliers: []const u64,
    sizes: []const u64,
    offsets: []const usize,
    /// History is oldest first. Hashing and validation complete before that stream's history advances.
    pub fn step(p: View, history: []u32, token: u32, ids: []usize) !void {
        if (history.len != p.ngram - 1 or ids.len != p.sizes.len) return error.InvalidNgramGeometry;
        if (token >= p.vocab) return error.NgramTokenOutOfRange;
        for (history) |old| if (old >= p.vocab) return error.NgramTokenOutOfRange;
        var shifted: [16]u64 = undefined;
        shifted[0] = token;
        var cut = false;
        for (1..p.ngram) |j| {
            const old = history[history.len - j];
            shifted[j] = if (cut) p.eos else old;
            cut = cut or old == p.eos;
        }
        for (ids, p.sizes, p.offsets, 0..) |*id, size, offset, head| {
            var hash: u64 = 0;
            for (0..2 + head / p.heads_per_ngram) |j| hash ^= shifted[j] *% p.multipliers[j];
            const remainder = @mod(@as(i64, @bitCast(hash)), @as(i64, @intCast(size)));
            id.* = try std.math.add(usize, offset, @as(usize, @intCast(remainder)));
        }
        std.mem.copyForwards(u32, history[0 .. history.len - 1], history[1..]);
        history[history.len - 1] = token;
    }
};
