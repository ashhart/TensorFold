//! Bounded host streaming of kept-state probe images, preserving each family's gather boundaries.
const std = @import("std");
const api = @import("engine_api");
const rows = api.prompt_cache.rows;
const Allocator = std.mem.Allocator;
const Saved = api.prompt_cache.Saved;

const chunk_bytes = 1 << 20;
pub const Zero = struct { off: usize, len: usize };

/// Hash every part in order through one bounded chunk, synchronously gathering on the caller's thread.
pub fn hash(a: Allocator, seed: u64, image: *const rows.Image, ptr: *anyopaque, saved: Saved, parts: []const usize, zero: ?Zero) !u64 {
    var total: usize = 0;
    for (parts) |bytes| total += bytes;
    const buf = try a.alloc(u8, @min(total, chunk_bytes));
    defer a.free(buf);
    var h = std.hash.Wyhash.init(seed);
    var off: usize = 0;
    for (parts) |bytes| {
        const end = off + bytes;
        while (off < end) {
            const n = @min(buf.len, end - off);
            const range = [_]rows.image.Range{.{ .off = off, .len = n }};
            try image.gather(ptr, saved, &range, buf[0..n]);
            if (zero) |z| {
                const lo = @max(off, z.off);
                const hi = @min(off + n, z.off + z.len);
                if (lo < hi) @memset(buf[lo - off .. hi - off], 0);
            }
            h.update(buf[0..n]);
            off += n;
        }
    }
    return h.final();
}

const Peak = @import("engine_api").rows_peak_allocator.Peak;
const t = std.testing;

const Host = struct {
    source: []const u8,
    seen: []u8,
    fail_at: ?usize = null,
    calls: usize = 0,

    fn gather(p: *anyopaque, _: Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
        const h: *@This() = @ptrCast(@alignCast(p));
        for (ranges) |g| {
            const off: usize = @intCast(g.off);
            const n: usize = @intCast(g.len);
            if (h.fail_at) |at| if (off >= at) return error.TestGatherFailure;
            @memcpy(out[0..n], h.source[off..][0..n]);
            @memcpy(h.seen[off..][0..n], h.source[off..][0..n]);
            h.calls += 1;
        }
    }
};

fn stateParts(_: *anyopaque, _: u32, _: *std.ArrayList(rows.image.Part), _: Allocator) anyerror!void {}
fn blank(_: *anyopaque, _: u32) anyerror!Saved {
    unreachable;
}
fn scatter(_: *anyopaque, _: Saved, _: []const rows.image.Range, _: []const u8) anyerror!void {}
const test_image: rows.Image = .{ .parts = stateParts, .gather = Host.gather, .blank = blank, .scatter = scatter };

fn fill(bytes: []u8) void {
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% 197 + i / 251 + 29);
}

test "kept probe: chunks preserve bytes and Wyhash with bounded allocation and cleared words" {
    const n = 2 * chunk_bytes + 137;
    var source: [n]u8 = undefined;
    var seen: [n]u8 = @splat(0);
    fill(&source);
    var host: Host = .{ .source = &source, .seen = &seen };
    var peak: Peak = .{ .child = t.allocator };
    const got = try hash(peak.allocator(), 0x6375, &test_image, &host, &host, &.{ 16, chunk_bytes - 19, chunk_bytes + 140 }, .{ .off = 8, .len = 8 });
    var expected = source;
    @memset(expected[8..16], 0);
    try t.expectEqualSlices(u8, &source, &seen);
    try t.expectEqual(std.hash.Wyhash.hash(0x6375, &expected), got);
    try t.expectEqual(@as(usize, 4), host.calls);
    try t.expectEqual(@as(usize, chunk_bytes), peak.peak);
    try t.expectEqual(@as(usize, 0), peak.live);
}

test "kept probe: a mid-image gather failure propagates without changing source bytes" {
    const n = chunk_bytes + 33;
    var source: [n]u8 = undefined;
    var seen: [n]u8 = @splat(0);
    fill(&source);
    const before = source;
    var host: Host = .{ .source = &source, .seen = &seen, .fail_at = chunk_bytes };
    try t.expectError(error.TestGatherFailure, hash(t.allocator, 7, &test_image, &host, &host, &.{n}, null));
    try t.expectEqual(@as(usize, 1), host.calls);
    try t.expectEqualSlices(u8, &before, &source);
}
