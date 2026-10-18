//! TF_DSV41_ACCEPT_LOG=<path>: one JSON line per draft-policy event (rank 0) for draft-depth studies; off when unset.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

var fd: c_int = -2; // -2: not looked up yet, -1: off
var round: u64 = 0;
var buf: [16384]u8 = undefined;

fn on() bool {
    if (fd == -2) {
        fd = -1;
        if (std.c.getenv("TF_DSV41_ACCEPT_LOG")) |p| {
            const f = std.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
            if (f >= 0) fd = f else if (!quiet) std.log.warn("TF_DSV41_ACCEPT_LOG: cannot open {s}", .{std.mem.span(p)});
        }
    }
    return fd >= 0;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn put(s: []const u8) void {
    var at: usize = 0;
    while (at < s.len) {
        const n = std.c.write(fd, s[at..].ptr, s.len - at);
        if (n <= 0) return;
        at += @intCast(n);
    }
}

pub const Ask = struct { slot: u32, start: u64, most: usize, k: usize, conf: []const f64, q: []const f64 };

pub fn choose(windows: usize, rows: u64, others: u64, asks: []const Ask) void {
    if (!on()) return;
    round += 1;
    var w = std.Io.Writer.fixed(&buf);
    w.print("{{\"ev\":\"choose\",\"r\":{d},\"t\":{d},\"windows\":{d},\"rows\":{d},\"others\":{d},\"asks\":[", .{ round, nowNs(), windows, rows, others }) catch return;
    for (asks, 0..) |a, i| {
        w.print("{s}{{\"slot\":{d},\"start\":{d},\"most\":{d},\"k\":{d},\"conf\":[", .{ if (i > 0) "," else "", a.slot, a.start, a.most, a.k }) catch return;
        for (a.conf, 0..) |v, j| w.print("{s}{d:.5}", .{ if (j > 0) "," else "", v }) catch return;
        w.writeAll("],\"q\":[") catch return;
        for (a.q, 0..) |v, j| w.print("{s}{d:.5}", .{ if (j > 0) "," else "", v }) catch return;
        w.writeAll("]}") catch return;
    }
    w.writeAll("]}\n") catch return;
    put(w.buffered());
}

pub fn commit(slot: u32, rows: u32, keep: usize, drafted: bool, finished: bool, generated: usize, max_new: u32) void {
    if (!on()) return;
    var w = std.Io.Writer.fixed(&buf);
    w.print("{{\"ev\":\"commit\",\"r\":{d},\"t\":{d},\"slot\":{d},\"rows\":{d},\"keep\":{d},\"drafted\":{},\"finished\":{},\"generated\":{d},\"max_new\":{d}}}\n", .{ round, nowNs(), slot, rows, keep, drafted, finished, generated, max_new }) catch return;
    put(w.buffered());
}
