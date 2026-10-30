//! Server output: ``[tensorfold]`` lines on stdout, and the live status line they clear first.
const std = @import("std");
const platform = @import("sys");
const Conn = @import("http_conn.zig").Conn;
const afterChars = @import("reply_text.zig").afterChars;

var global_io: ?std.Io = null;
var mutex: std.Io.Mutex = .init;
var shown = false; // the live line is on screen
var line_start = true; // the newest write ended its line, so a redraw cannot split it
var quiet = false;
var stamped = false; // TENSORFOLD_LOG_TIME: each log line starts with its UTC time

pub const Test = if (@import("builtin").is_test) struct {
    pub threadlocal var sink: ?struct { ctx: *anyopaque, call: *const fn (*anyopaque, []const u8) void } = null;
} else struct {};

pub fn init(io_: std.Io, silent: bool) void {
    global_io = io_;
    quiet = silent;
    stamped = timeWanted(platform.getenv("TENSORFOLD_LOG_TIME"));
}

/// TENSORFOLD_LOG_TIME: 1 or on stamps the lines; unset, 0 or anything else leaves them as before.
pub fn timeWanted(text: ?[]const u8) bool {
    const s = std.mem.trim(u8, text orelse return false, " \t");
    return std.mem.eql(u8, s, "1") or std.ascii.eqlIgnoreCase(s, "on");
}

/// ``2026-10-30T18:54:04.123Z``: a UTC time given in milliseconds since the epoch, an earlier one as the epoch.
pub fn utcTime(buf: *[24]u8, ms: i64) []const u8 {
    const t: u64 = @intCast(@max(ms, 0));
    const es: std.time.epoch.EpochSeconds = .{ .secs = t / 1000 };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    const date = .{ yd.year, md.month.numeric(), @as(u8, md.day_index) + 1 };
    const time = .{ ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute(), t % 1000 };
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}Z", date ++ time) catch unreachable;
}

pub fn io() std.Io {
    return global_io orelse std.Io.Threaded.global_single_threaded.io();
}

fn now() i64 {
    return std.Io.Clock.real.now(io()).toMilliseconds();
}

const clear = "\r\x1b[2K";

fn writeOut(bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const rc = platform.c.write(1, bytes[off..].ptr, bytes.len - off);
        if (platform.c.errno(rc) != .SUCCESS or rc <= 0) return;
        off += @intCast(rc);
    }
}

/// One ``[tensorfold] ...`` line.
pub fn line(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) {
        if (Test.sink == null) return;
    } else if (quiet) return;
    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    if (stamped) {
        var t: [24]u8 = undefined;
        w.print("{s} ", .{utcTime(&t, now())}) catch unreachable;
    }
    w.print("[tensorfold] " ++ fmt ++ "\n", args) catch {
        buf[buf.len - 1] = '\n';
        w.end = buf.len;
    };
    if (@import("builtin").is_test) Test.sink.?.call(Test.sink.?.ctx, w.buffered()) else raw(w.buffered());
}

/// std.log for every engine the server runs: the default line, after its UTC time when TENSORFOLD_LOG_TIME is on.
pub fn stdLog(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    if (!stamped) return std.log.defaultLog(level, scope, format, args);
    const ms = now();
    const dio = std.Options.debug_io;
    const prev = dio.swapCancelProtection(.blocked);
    defer _ = dio.swapCancelProtection(prev);
    var buffer: [64]u8 = undefined;
    const t = std.debug.lockStderr(&buffer).terminal();
    defer std.debug.unlockStderr();
    stampedLine(level, scope, format, args, t, ms) catch {};
}

/// The default std.log line after its UTC time, written to `t`.
pub fn stampedLine(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
    t: std.Io.Terminal,
    ms: i64,
) std.Io.Writer.Error!void {
    var s: [24]u8 = undefined;
    try t.writer.print("{s} ", .{utcTime(&s, ms)});
    try std.log.defaultLogFileTerminal(level, scope, format, args, t);
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

/// ``refused <id>: <message>``, the message cut to 300 characters: why a request got an error reply.
pub fn refused(buf: []u8, id: []const u8, message: []const u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("refused {s}: {s}", .{ id, message[0 .. message.len - afterChars(message, 300).len] }) catch {};
    return w.buffered();
}

/// ``ended <id> reason=...``: a request that ends without its reply, because its client left (``why`` null) or it failed.
pub fn ended(buf: []u8, id: []const u8, why: ?[]const u8, prompt: usize, tokens: usize, after: f64) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("ended {s} reason=", .{id}) catch {};
    if (why) |e| w.print("error ({s})", .{e}) catch {} else w.writeAll("client-left") catch {};
    w.print(" prompt={d} tokens={d} after={d:.2}s", .{ prompt, tokens, after }) catch {};
    return w.buffered();
}

test "a request's refused and ended lines" {
    var buf: [1400]u8 = undefined;
    const accents: [301][2]u8 = @splat(.{ 0xc3, 0xa9 }); // two bytes a character, so a byte cut would split one
    const cut = refused(&buf, "cmpl-1", std.mem.sliceAsBytes(&accents));
    try std.testing.expectEqualStrings("refused cmpl-1: ", cut[0..16]);
    try std.testing.expectEqualStrings(std.mem.sliceAsBytes(accents[0..300]), cut[16..]);
    try std.testing.expectEqualStrings("refused cmpl-1: short", refused(&buf, "cmpl-1", "short"));
    try std.testing.expectEqualStrings("ended chatcmpl-1 reason=client-left prompt=12 tokens=3 after=0.50s", ended(&buf, "chatcmpl-1", null, 12, 3, 0.5));
    try std.testing.expectEqualStrings("ended chatcmpl-1 reason=error (OutOfMemory) prompt=12 tokens=0 after=1.25s", ended(&buf, "chatcmpl-1", "OutOfMemory", 12, 0, 1.25));
}

test "TENSORFOLD_LOG_TIME, and the UTC time a line starts with" {
    try std.testing.expect(timeWanted("1") and timeWanted(" on ") and timeWanted("ON"));
    try std.testing.expect(!timeWanted(null) and !timeWanted("") and !timeWanted("0") and !timeWanted("yes"));
    var t: [24]u8 = undefined;
    try std.testing.expectEqualStrings("1970-01-01T00:00:00.000Z", utcTime(&t, 0));
    try std.testing.expectEqualStrings("2026-10-30T21:47:03.512Z", utcTime(&t, 1_793_396_823_512));
    try std.testing.expectEqualStrings("2028-02-29T23:59:59.999Z", utcTime(&t, 1_835_481_599_999));
    try std.testing.expectEqualStrings("1970-01-01T00:00:00.000Z", utcTime(&t, -1));
}

test "a [tensorfold] line with the time off is as before, and with it on starts with the time" {
    const Capture = struct {
        bytes: [64]u8 = undefined,
        len: usize = 0,
        fn call(ctx: *anyopaque, text: []const u8) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(c.bytes[0..text.len], text);
            c.len = text.len;
        }
    };
    var c: Capture = .{};
    Test.sink = .{ .ctx = &c, .call = Capture.call };
    defer Test.sink = null;
    line("serving {s}", .{"x"});
    try std.testing.expectEqualStrings("[tensorfold] serving x\n", c.bytes[0..c.len]);
    stamped = true;
    defer stamped = false;
    line("serving {s}", .{"x"});
    const out = c.bytes[0..c.len];
    for ([_]usize{ 4, 7 }) |i| try std.testing.expectEqual(@as(u8, '-'), out[i]);
    for ([_]usize{ 13, 16 }) |i| try std.testing.expectEqual(@as(u8, ':'), out[i]);
    try std.testing.expectEqualStrings("T", out[10..11]);
    try std.testing.expectEqualStrings(".", out[19..20]);
    try std.testing.expectEqualStrings("Z [tensorfold] serving x\n", out[23..]);
}

test "a std.log line with the time on: the time, then the default line" {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    const t: std.Io.Terminal = .{ .writer = &w, .mode = .no_color };
    try stampedLine(.warn, .default, "hi {d}", .{1}, t, 1_793_396_823_512);
    try stampedLine(.info, .glm, "rank {d} up", .{0}, t, 1_793_396_823_513);
    const want = "2026-10-30T21:47:03.512Z warning: hi 1\n2026-10-30T21:47:03.513Z info(glm): rank 0 up\n";
    try std.testing.expectEqualStrings(want, w.buffered());
}
