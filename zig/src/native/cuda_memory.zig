//! CUDA memory admission arithmetic and host-pool validation, without a device.
const std = @import("std");
const gib: f64 = 1 << 30;

/// What /proc/meminfo says: MemTotal and MemAvailable in bytes (the page cache counts as available).
pub const MemInfo = struct { total: u64, available: u64 };

pub fn meminfo(text: []const u8) ?MemInfo {
    var total: ?u64 = null;
    var available: ?u64 = null;
    var rows = std.mem.tokenizeScalar(u8, text, '\n');
    while (rows.next()) |row| {
        var words = std.mem.tokenizeAny(u8, row, ": \t");
        const key = words.next() orelse continue;
        const is_total = std.mem.eql(u8, key, "MemTotal");
        if (!is_total and !std.mem.eql(u8, key, "MemAvailable")) continue;
        const kib = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
        if (!std.mem.eql(u8, words.next() orelse return null, "kB") or words.next() != null) return null;
        const bytes = std.math.mul(u64, kib, 1024) catch return null;
        if (is_total) {
            if (total != null) return null;
            total = bytes;
        } else {
            if (available != null) return null;
            available = bytes;
        }
    }
    const t = total orelse return null;
    const a = available orelse return null;
    return if (t > 0 and a <= t) .{ .total = t, .available = a } else null;
}

/// CUDA allocation room after the reserve and optional process-memory cap.
pub const Pool = struct {
    free: u64,
    total: u64,
    reserve: u64,
    limit: ?u64,
    unified: bool,

    /// Bytes the engine may still allocate on top of `held`.
    pub fn room(p: Pool, held: u64) u64 {
        const left = p.free -| p.reserve;
        return if (p.limit) |cap| @min(left, cap -| held) else left;
    }
};

/// TENSORFOLD_MEMORY_RESERVE_GIB (2 GiB up to the pool's size), else a tenth of the pool and at least 4 GiB.
pub fn reserveBytes(text: ?[]const u8, total: u64) error{Invalid}!u64 {
    const t = std.mem.trim(u8, text orelse "", " ");
    if (t.len == 0) return @max(4 << 30, total / 10);
    const g = std.fmt.parseFloat(f64, t) catch return error.Invalid;
    if (!(g >= 2)) return error.Invalid;
    const bytes = try scaledBytes(g);
    return if (bytes <= total) bytes else error.Invalid;
}

/// TENSORFOLD_CUDA_MEMORY_LIMIT_GB in bytes; null when unset.
pub fn limitBytes(text: ?[]const u8) error{Invalid}!?u64 {
    const t = std.mem.trim(u8, text orelse return null, " ");
    if (t.len == 0) return null;
    const g = std.fmt.parseFloat(f64, t) catch return error.Invalid;
    return try scaledBytes(g);
}

fn scaledBytes(g: f64) error{Invalid}!u64 {
    const bytes = g * gib;
    const upper: f64 = 0x1p64;
    if (!std.math.isFinite(bytes) or bytes < 1 or bytes >= upper) return error.Invalid;
    return @intFromFloat(bytes);
}

/// Streams the room fits, as many as asked when they all fit; an error when none does, or a fixed --parallel doesn't.
pub fn admit(room: u64, stream: u64, asked: u32, fixed: bool) error{ NoStream, TooMany }!u32 {
    const fits = if (stream == 0) asked else std.math.cast(u32, room / stream) orelse std.math.maxInt(u32);
    if (fits == 0) return error.NoStream;
    if (fits >= asked) return asked;
    return if (fixed) error.TooMany else fits;
}

/// Integrated GPUs require validated host counts; discrete devices use CUDA's counts.
pub fn counts(unified: bool, card: MemInfo, text: ?[]const u8) error{HostMemoryUnavailable}!MemInfo {
    if (unified) return meminfo(text orelse return error.HostMemoryUnavailable) orelse error.HostMemoryUnavailable;
    return card;
}

test "the memory plan: reserve, cap, and admission that fails closed" {
    const g: u64 = 1 << 30;
    try std.testing.expectEqual(4 * g, try reserveBytes(null, 24 * g)); // at least 4 GiB
    try std.testing.expectEqual(12 * g, try reserveBytes("", 120 * g)); // a tenth
    try std.testing.expectEqual(3 * g, try reserveBytes("3", 120 * g));
    try std.testing.expectError(error.Invalid, reserveBytes("1", 120 * g));
    try std.testing.expectError(error.Invalid, reserveBytes("200", 120 * g));
    try std.testing.expectEqual(@as(?u64, null), try limitBytes(null));
    try std.testing.expectEqual(@as(?u64, 40 * g), try limitBytes("40"));
    try std.testing.expectError(error.Invalid, limitBytes("0"));
    const p: Pool = .{ .free = 70 * g, .total = 96 * g, .reserve = 10 * g, .limit = null, .unified = false };
    try std.testing.expectEqual(60 * g, p.room(20 * g));
    const capped: Pool = .{ .free = 70 * g, .total = 96 * g, .reserve = 10 * g, .limit = 30 * g, .unified = false };
    try std.testing.expectEqual(10 * g, capped.room(20 * g)); // the cap less what the engine holds
    try std.testing.expectEqual(@as(u32, 8), try admit(60 * g, g, 8, false));
    try std.testing.expectEqual(@as(u32, 5), try admit(5 * g + 1, g, 8, false)); // auto serves what fits
    try std.testing.expectError(error.TooMany, admit(5 * g, g, 8, true)); // a fixed --parallel refuses
    try std.testing.expectError(error.NoStream, admit(g - 1, g, 8, false));
}

test "meminfo: total and available in bytes" {
    const m = meminfo("MemTotal:       131072 kB\nMemFree:          1024 kB\nMemAvailable:    65536 kB\n").?;
    try std.testing.expectEqual(@as(u64, 131072 * 1024), m.total);
    try std.testing.expectEqual(@as(u64, 65536 * 1024), m.available);
    try std.testing.expect(meminfo("MemTotal: 5 kB\n") == null);
}

test "unified admission refuses missing or malformed host memory" {
    const card: MemInfo = .{ .total = 128 << 30, .available = 100 << 30 };
    for ([_]?[]const u8{
        null,
        "",
        "MemTotal: 131072 kB\n",
        "MemTotal: bad kB\nMemAvailable: 1024 kB\n",
        "MemTotal: 131072 MB\nMemAvailable: 1024 MB\n",
        "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
        "MemTotal: 1024 kB\nMemAvailable: 2048 kB\n",
        "MemTotal: 18014398509481984 kB\nMemAvailable: 1024 kB\n",
        "MemTotal: 131072 kB\nMemTotal: 131072 kB\nMemAvailable: 1024 kB\n",
    }) |text| {
        try std.testing.expectError(error.HostMemoryUnavailable, counts(true, card, text));
    }
    try std.testing.expectEqual(card, try counts(false, card, null));
    try std.testing.expectEqual(MemInfo{ .total = 131072 * 1024, .available = 65536 * 1024 }, try counts(true, card, "MemTotal: 131072 kB\nMemAvailable: 65536 kB\n"));
}

test "memory limit refuses an oversized finite value and the rounded u64 boundary" {
    try std.testing.expectError(error.Invalid, limitBytes("1e300"));
    try std.testing.expectError(error.Invalid, limitBytes("17179869184"));
    try std.testing.expectError(error.Invalid, limitBytes("17179869183.9999995")); // parses to 2^34: 2^64 bytes
    try std.testing.expectEqual(@as(?u64, std.math.maxInt(u64) - 2047), try limitBytes("17179869183.9999980926513671875"));
}
