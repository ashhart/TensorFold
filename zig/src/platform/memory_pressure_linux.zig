//! Bounded Linux pressure readings; a healthy reclaimable page cache is not itself memory pressure.
const std = @import("std");
const builtin = @import("builtin");

pub const Reading = struct {
    total: u64,
    available: u64,
    some_avg10: f64,
    full_avg10: f64,

    /// Experimental PSI levels: 1 normal, 2 warning, 4 critical; the governor owns recovery timing.
    pub fn level(r: Reading) u32 {
        if (r.full_avg10 >= 10) return 4;
        if (r.some_avg10 >= 10 or r.full_avg10 >= 1) return 2;
        return 1;
    }
};

fn avg10(line: []const u8) ?f64 {
    var words = std.mem.tokenizeAny(u8, line, " \t\r");
    _ = words.next();
    var found: ?f64 = null;
    while (words.next()) |word| {
        if (!std.mem.startsWith(u8, word, "avg10=")) continue;
        if (found != null) return null;
        const n = std.fmt.parseFloat(f64, word[6..]) catch return null;
        if (!std.math.isFinite(n) or n < 0 or n > 100) return null;
        found = n;
    }
    return found;
}

/// Read both PSI rows and both meminfo fields; incomplete, duplicate or malformed required data stays unknown.
pub fn parse(psi: []const u8, meminfo: []const u8) ?Reading {
    var some: ?f64 = null;
    var full: ?f64 = null;
    var rows = std.mem.tokenizeScalar(u8, psi, '\n');
    while (rows.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, " \t\r");
        const name = words.next() orelse continue;
        const dst = if (std.mem.eql(u8, name, "some")) &some else if (std.mem.eql(u8, name, "full")) &full else continue;
        if (dst.* != null) return null;
        dst.* = avg10(line) orelse return null;
    }
    var total: ?u64 = null;
    var available: ?u64 = null;
    rows = std.mem.tokenizeScalar(u8, meminfo, '\n');
    while (rows.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, ": \t\r");
        const name = words.next() orelse continue;
        const dst = if (std.mem.eql(u8, name, "MemTotal")) &total else if (std.mem.eql(u8, name, "MemAvailable")) &available else continue;
        if (dst.* != null) return null;
        const kib = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
        if (!std.mem.eql(u8, words.next() orelse return null, "kB") or words.next() != null) return null;
        dst.* = std.math.mul(u64, kib, 1024) catch return null;
    }
    const t = total orelse return null;
    const a = available orelse return null;
    if (t == 0 or a > t) return null;
    return .{ .total = t, .available = a, .some_avg10 = some orelse return null, .full_avg10 = full orelse return null };
}

fn readFile(root: []const u8, name: []const u8, buf: []u8) ?[]const u8 {
    var path: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const z = std.fmt.bufPrintSentinel(&path, "{s}/{s}", .{ root, name }, 0) catch return null;
    const fd = std.c.open(z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var used: usize = 0;
    while (used < buf.len) {
        const n = std.c.read(fd, buf[used..].ptr, buf.len - used);
        if (n < 0) return null;
        if (n == 0) return buf[0..used];
        used += @intCast(n);
    }
    return null;
}

/// The same production reader on a planted proc root; no allocation and no retained bytes between readings.
pub fn readAt(root: []const u8) ?Reading {
    if (comptime builtin.os.tag == .windows) return null;
    var psi: [1024]u8 = undefined;
    var mem: [16384]u8 = undefined;
    const p = readFile(root, "pressure/memory", &psi) orelse return null;
    const m = readFile(root, "meminfo", &mem) orelse return null;
    return parse(p, m);
}

pub fn pressure() ?u32 {
    return (readAt("/proc") orelse return null).level();
}
