//! Server output: ``[tensorfold]`` lines on stdout, and the live status line they clear first.
const std = @import("std");
const Conn = @import("http_conn.zig").Conn;

var global_io: ?std.Io = null;
var mutex: std.Io.Mutex = .init;
var shown = false; // the live line is on screen
var line_start = true; // the newest write ended its line, so a redraw cannot split it
var quiet = false;

pub fn init(io_: std.Io, silent: bool) void {
    global_io = io_;
    quiet = silent;
}

pub fn io() std.Io {
    return global_io orelse std.Io.Threaded.global_single_threaded.io();
}

const clear = "\r\x1b[2K";

fn writeOut(bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const rc = std.posix.system.write(1, bytes[off..].ptr, bytes.len - off);
        if (std.posix.errno(rc) != .SUCCESS or rc <= 0) return;
        off += @intCast(rc);
    }
}

/// One ``[tensorfold] ...`` line.
pub fn line(comptime fmt: []const u8, args: anytype) void {
    if (quiet) return;
    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.print("[tensorfold] " ++ fmt ++ "\n", args) catch {
        buf[buf.len - 1] = '\n';
        w.end = buf.len;
    };
    raw(w.buffered());
}

/// Text as given, the live line cleared first.
pub fn raw(text: []const u8) void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (shown) {
        writeOut(clear);
        shown = false;
    }
    writeOut(text);
    if (text.len > 0) line_start = text[text.len - 1] == '\n';
}

/// Redraws the status line unless a log line is half written.
pub fn status(text: []const u8, width: usize) void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (!line_start) return;
    writeOut(clear);
    writeOut(text[0..@min(text.len, @max(20, width) - 1)]);
    shown = true;
}

pub fn clearStatus() void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (shown) writeOut(clear);
    shown = false;
}

pub fn keySuffix(c: *const Conn) []const u8 {
    if (!c.auth_enabled) return "";
    return if (c.key_label) |label| label_suffix(label) else " key=unauthenticated";
}

threadlocal var suffix_buf: [80]u8 = undefined;

fn label_suffix(label: []const u8) []const u8 {
    return std.fmt.bufPrint(&suffix_buf, " key={s}", .{label}) catch " key=?";
}

/// The access line ``send_response`` prints: ``ip "GET / HTTP/1.1" 200 -``.
pub fn request(c: *const Conn, code: u16) void {
    line("{s} \"{s}\" {d} -{s}", .{ c.peer, c.requestline, code, keySuffix(c) });
}
