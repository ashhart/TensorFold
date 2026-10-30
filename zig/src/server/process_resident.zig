//! Linux resident set readings from /proc/self/status; neither a physical footprint nor device memory.
const std = @import("std");
const builtin = @import("builtin");

/// Bytes, each null when the file has no such line or its value does not read as kB.
pub const Snapshot = struct {
    resident_bytes: ?u64 = null, // VmRSS
    anon_bytes: ?u64 = null, // RssAnon: heap, stacks and private maps
    file_bytes: ?u64 = null, // RssFile: the binary, libraries and mapped files
    shmem_bytes: ?u64 = null, // RssShmem: System V shared memory, tmpfs maps and shared anonymous maps
    peak_bytes: ?u64 = null, // VmHWM: the kernel's lifetime peak of VmRSS
};

const fields = [_]struct { key: []const u8, name: []const u8 }{
    .{ .key = "VmRSS", .name = "resident_bytes" },
    .{ .key = "RssAnon", .name = "anon_bytes" },
    .{ .key = "RssFile", .name = "file_bytes" },
    .{ .key = "RssShmem", .name = "shmem_bytes" },
    .{ .key = "VmHWM", .name = "peak_bytes" },
};

/// The five counts in `text`; null when it has none of them. A repeated line keeps its first value.
pub fn parse(text: []const u8) ?Snapshot {
    var out: Snapshot = .{};
    var found = false;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        inline for (fields) |f| if (std.mem.eql(u8, line[0..colon], f.key) and @field(out, f.name) == null) {
            @field(out, f.name) = kib(line[colon + 1 ..]);
            found = found or @field(out, f.name) != null;
        };
    }
    return if (found) out else null;
}

/// "   2032 kB" in bytes; any other unit, a missing number or an overflow is null.
fn kib(value: []const u8) ?u64 {
    var words = std.mem.tokenizeAny(u8, value, " \t");
    const number = words.next() orelse return null;
    const unit = words.next() orelse return null;
    if (!std.mem.eql(u8, unit, "kB") or words.next() != null) return null;
    const n = std.fmt.parseUnsigned(u64, number, 10) catch return null;
    return std.math.mul(u64, n, 1024) catch null;
}

/// The longest /proc/self/status read; the five lines sit in its first kilobyte.
const max_bytes = 4096;

/// One kernel sample into a stack buffer; null off Linux or when the file does not read. Procfs reports size 0.
pub fn read(io: std.Io) ?Snapshot {
    if (builtin.os.tag != .linux) return null;
    var file = std.Io.Dir.cwd().openFile(io, "/proc/self/status", .{}) catch return null;
    defer file.close(io);
    var text: [max_bytes]u8 = undefined;
    var n: usize = 0;
    while (n < text.len) {
        const got = file.readStreaming(io, &.{text[n..]}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return null,
        };
        if (got == 0) break;
        n += got;
    }
    // a read cut at max_bytes ends inside a line: parse only whole lines
    const whole = if (n < text.len) n else (std.mem.lastIndexOfScalar(u8, &text, '\n') orelse return null) + 1;
    return parse(text[0..whole]);
}

test "status lines parse to bytes" {
    const text = "Name:\ttensorfold\nVmPeak:\t 9437184 kB\nVmHWM:\t 4508876 kB\nVmRSS:\t 4194304 kB\n" ++
        "RssAnon:\t 3145728 kB\nRssFile:\t  786432 kB\nRssShmem:\t  262144 kB\nVmData:\t 3500000 kB\nThreads:\t42\n";
    const s = parse(text).?;
    try std.testing.expectEqual(@as(?u64, 4194304 * 1024), s.resident_bytes);
    try std.testing.expectEqual(@as(?u64, 3145728 * 1024), s.anon_bytes);
    try std.testing.expectEqual(@as(?u64, 786432 * 1024), s.file_bytes);
    try std.testing.expectEqual(@as(?u64, 262144 * 1024), s.shmem_bytes);
    try std.testing.expectEqual(@as(?u64, 4508876 * 1024), s.peak_bytes);
}

test "a 64 KiB-page kernel still counts in kB" {
    const s = parse("VmHWM:\t  131136 kB\nVmRSS:\t  131072 kB\nRssAnon:\t   65536 kB\nRssFile:\t   65472 kB\nRssShmem:\t      64 kB\n").?;
    try std.testing.expectEqual(@as(?u64, 131072 << 10), s.resident_bytes);
    try std.testing.expectEqual(@as(?u64, 64 << 10), s.shmem_bytes);
    try std.testing.expectEqual(@as(?u64, 131136 << 10), s.peak_bytes);
}

test "missing lines, other units and bad numbers are unknown, not zero" {
    // kernels before 4.5 have no RssAnon, RssFile or RssShmem
    const old = parse("VmHWM:\t 2048 kB\nVmRSS:\t 1024 kB\nVmData:\t 9 kB\n").?;
    try std.testing.expectEqual(@as(?u64, 1024 << 10), old.resident_bytes);
    try std.testing.expectEqual(@as(?u64, 2048 << 10), old.peak_bytes);
    try std.testing.expectEqual(@as(?u64, null), old.anon_bytes);
    try std.testing.expectEqual(@as(?u64, null), old.file_bytes);
    try std.testing.expectEqual(@as(?u64, null), old.shmem_bytes);
    const bad = parse("VmRSS:\t 1024 MB\nRssAnon:\t kB\nRssFile:\t 12x kB\nRssShmem:\t 99999999999999999999 kB\nVmHWM:\t 4 kB extra\n");
    try std.testing.expectEqual(@as(?Snapshot, null), bad);
    const overflow = parse("VmRSS:\t 18014398509481984 kB\nRssAnon:\t 1 kB\n").?; // 2^54 kB is 2^64 bytes
    try std.testing.expectEqual(@as(?u64, null), overflow.resident_bytes);
    try std.testing.expectEqual(@as(?u64, 1024), overflow.anon_bytes);
    try std.testing.expectEqual(@as(?Snapshot, null), parse(""));
    try std.testing.expectEqual(@as(?Snapshot, null), parse("MemTotal:\t 1 kB\nVmRSSX:\t 1 kB\n"));
}

test "a repeated line keeps its first value" {
    const s = parse("VmRSS:\t 1 kB\nVmRSS:\t 2 kB\n").?;
    try std.testing.expectEqual(@as(?u64, 1024), s.resident_bytes);
}

test "this process's status reads whole and parses (Linux)" {
    if (builtin.os.tag != .linux) {
        try std.testing.expect(read(std.testing.io) == null);
        return;
    }
    const s = read(std.testing.io) orelse return error.Unreadable;
    try std.testing.expect(s.resident_bytes.? > 0);
    try std.testing.expect(s.peak_bytes.? >= s.resident_bytes.?);
    if (s.anon_bytes) |anon| try std.testing.expect(anon > 0 and anon <= s.peak_bytes.?); // Linux 4.5 and later
}
