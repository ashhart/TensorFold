const std = @import("std");

const window = 2.0;
const clear = "\r\x1b[2K";
var output_mutex: std.Io.Mutex = .init;
var active_display: ?*Display = null;

pub fn print(comptime format: []const u8, args: anytype) void {
    const io = std.Options.debug_io;
    output_mutex.lockUncancelable(io);
    defer output_mutex.unlock(io);
    if (active_display) |display| display.print(format, args) else std.debug.print(format, args);
}

pub fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / std.time.ns_per_s;
}

const Meter = struct {
    const Event = struct { time: f64, tokens: usize };
    events: std.ArrayList(Event) = .empty,

    fn rate(m: *Meter, instant: f64) f64 {
        var expired: usize = 0;
        while (expired < m.events.items.len and m.events.items[expired].time < instant - window) : (expired += 1) {}
        std.mem.copyForwards(Event, m.events.items, m.events.items[expired..]);
        m.events.items.len -= expired;
        var tokens: f64 = 0;
        for (m.events.items) |event| tokens += @floatFromInt(event.tokens);
        return tokens / window;
    }

    fn add(m: *Meter, a: std.mem.Allocator, tokens: usize, instant: f64) !void {
        _ = m.rate(instant);
        if (tokens > 0) try m.events.append(a, .{ .time = instant, .tokens = tokens });
    }
};

const ChunkRate = struct {
    time: f64 = -std.math.inf(f64),
    value: f64 = 0,

    fn add(m: *ChunkRate, tokens: usize, seconds: f64, instant: f64) void {
        if (tokens > 0 and seconds > 0) m.* = .{ .time = instant, .value = @as(f64, @floatFromInt(tokens)) / seconds };
    }

    fn rate(m: ChunkRate, instant: f64) f64 {
        return if (instant - m.time <= window) m.value else 0;
    }
};

pub const Snapshot = struct {
    neural_proposed: u64 = 0,
    neural_accepted: u64 = 0,
    proposed_tokens: u64 = 0,
    accepted_tokens: u64 = 0,
    structural_proposed: u64 = 0,
    structural_accepted: u64 = 0,
    connections: usize,
    waiting_requests: usize,
    decode_tokens_per_second: f64,
    prefill_tokens_per_second: f64,
    decoded_tokens: u64 = 0,
    prefilled_tokens: u64 = 0,
    available: bool = true,

    pub fn render(s: Snapshot, out: *std.Io.Writer) !void {
        if (!s.available) return out.writeAll("[tensorfold] status unavailable: OutOfMemory");
        try out.print("[tensorfold] {d} connection{s}", .{ s.connections, if (s.connections == 1) "" else "s" });
        if (s.waiting_requests > 0) try out.print(" ({d} waiting)", .{s.waiting_requests});
        try out.writeAll(" · decode ");
        try number(out, s.decode_tokens_per_second);
        try out.writeAll(" tok/s · prefill ");
        try number(out, s.prefill_tokens_per_second);
        try out.writeAll(" tok/s");
    }
};

fn number(out: *std.Io.Writer, rate: f64) !void {
    const floor = @floor(rate);
    const rounded = floor + @as(f64, if (rate - floor > 0.5 or (rate - floor == 0.5 and @mod(floor, 2) == 1)) 1 else 0);
    var value: u64 = @intFromFloat(@max(0, @min(rounded, 18446744073709549568.0)));
    var buffer: [26]u8 = undefined;
    var offset: usize = buffer.len;
    var digits: usize = 0;
    while (true) {
        if (digits > 0 and digits % 3 == 0) {
            offset -= 1;
            buffer[offset] = ',';
        }
        offset -= 1;
        buffer[offset] = '0' + @as(u8, @intCast(value % 10));
        value /= 10;
        digits += 1;
        if (value == 0) break;
    }
    try out.writeAll(buffer[offset..]);
}

pub const Stats = struct {
    neural_proposed: u64 = 0,
    neural_accepted: u64 = 0,
    proposed_tokens: u64 = 0,
    accepted_tokens: u64 = 0,
    structural_proposed: u64 = 0,
    structural_accepted: u64 = 0,
    io: std.Io,
    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    connections: usize = 0,
    waiting: usize = 0,
    decoded: Meter = .{},
    prefilled: ChunkRate = .{},
    decoded_tokens: u64 = 0,
    prefilled_tokens: u64 = 0,
    available: bool = true,

    pub fn deinit(s: *Stats) void {
        s.decoded.events.deinit(s.allocator);
    }

    pub fn enqueue(s: *Stats) void {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.connections += 1;
        s.waiting += 1;
    }

    pub fn activate(s: *Stats) void {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.waiting -= 1;
    }

    pub fn finish(s: *Stats, activated: bool) void {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.connections -= 1;
        if (!activated) s.waiting -= 1;
    }

    pub fn record(s: *Stats, prefilled: usize, decoded: usize, started: f64, ended: f64) void {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.decoded.add(s.allocator, decoded, ended) catch {
            s.available = false;
        };
        s.prefilled.add(prefilled, ended - started, ended);
        s.decoded_tokens +|= decoded;
        s.prefilled_tokens +|= prefilled;
    }

    pub fn snapshot(s: *Stats) Snapshot {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        const instant = now(s.io);
        return .{ .connections = s.connections, .waiting_requests = s.waiting, .decode_tokens_per_second = s.decoded.rate(instant), .prefill_tokens_per_second = s.prefilled.rate(instant), .decoded_tokens = s.decoded_tokens, .prefilled_tokens = s.prefilled_tokens, .available = s.available, .proposed_tokens = s.proposed_tokens, .accepted_tokens = s.accepted_tokens, .structural_proposed = s.structural_proposed, .structural_accepted = s.structural_accepted, .neural_proposed = s.neural_proposed, .neural_accepted = s.neural_accepted };
    }

    pub fn recordNeural(s: *Stats, proposed: usize, accepted: usize) void {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.neural_proposed +|= proposed;
        s.neural_accepted +|= accepted;
    }

    pub fn recordDrafts(s: *Stats, proposed: usize, accepted: usize, structural_proposed: usize, structural_accepted: usize) void {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.proposed_tokens +|= proposed;
        s.accepted_tokens +|= accepted;
        s.structural_proposed +|= structural_proposed;
        s.structural_accepted +|= structural_accepted;
    }
};

pub const Line = struct {
    shown: bool = false,
    line_start: bool = true,

    pub fn erase(l: *Line, out: *std.Io.Writer) !void {
        if (l.shown) {
            try out.writeAll(clear);
            try out.flush();
            l.shown = false;
        }
    }

    pub fn write(l: *Line, out: *std.Io.Writer, real: *std.Io.Writer, text: []const u8) !void {
        try l.erase(out);
        try real.writeAll(text);
        try real.flush();
        if (text.len > 0) l.line_start = text[text.len - 1] == '\n';
    }

    pub fn draw(l: *Line, out: *std.Io.Writer, text: []const u8, columns: usize) !void {
        if (!l.line_start) return;
        var utf8 = (try std.unicode.Utf8View.init(text)).iterator();
        const clipped = utf8.peek(@max(20, columns -| 1));
        try out.writeAll(clear);
        try out.writeAll(clipped);
        try out.flush();
        l.shown = true;
    }
};

pub const Display = struct {
    io: std.Io,
    stats: *Stats,
    mutex: std.Io.Mutex = .init,
    line: Line = .{},
    enabled: bool = false,
    columns_hint: ?usize = null,
    stopping: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub fn start(d: *Display, disabled: ?[]const u8, columns: ?[]const u8) void {
        if (disabled) |value| if (std.mem.eql(u8, value, "1")) return;
        if (std.c.isatty(std.Io.File.stdout().handle) != 1) return;
        d.columns_hint = columnHint(columns);
        d.enabled = true;
        d.thread = std.Thread.spawn(.{}, tick, .{d}) catch {
            d.enabled = false;
            return;
        };
        output_mutex.lockUncancelable(std.Options.debug_io);
        defer output_mutex.unlock(std.Options.debug_io);
        active_display = d;
    }

    pub fn stop(d: *Display) void {
        d.stopping.store(true, .release);
        if (d.thread) |thread| thread.join();
        output_mutex.lockUncancelable(std.Options.debug_io);
        defer output_mutex.unlock(std.Options.debug_io);
        if (active_display == d) active_display = null;
        d.mutex.lockUncancelable(d.io);
        defer d.mutex.unlock(d.io);
        var output = std.Io.File.stdout().writerStreaming(d.io, &.{});
        d.line.erase(&output.interface) catch {};
        d.enabled = false;
    }

    pub fn print(d: *Display, comptime format: []const u8, args: anytype) void {
        d.mutex.lockUncancelable(d.io);
        defer d.mutex.unlock(d.io);
        var output = std.Io.File.stdout().writerStreaming(d.io, &.{});
        if (d.enabled) d.line.erase(&output.interface) catch {};
        std.debug.print(format, args);
        if (format.len > 0) d.line.line_start = format[format.len - 1] == '\n';
    }

    fn tick(d: *Display) void {
        while (true) {
            std.Io.sleep(d.io, .fromMilliseconds(500), .awake) catch return;
            if (d.stopping.load(.acquire)) return;
            var buffer: [256]u8 = undefined;
            var text = std.Io.Writer.fixed(&buffer);
            d.stats.snapshot().render(&text) catch continue;
            var size: std.c.winsize = undefined;
            const columns: usize = d.columns_hint orelse if (std.c.ioctl(std.Io.File.stdout().handle, std.c.T.IOCGWINSZ, &size) == 0 and size.col > 0) size.col else 100;
            d.mutex.lockUncancelable(d.io);
            defer d.mutex.unlock(d.io);
            var output = std.Io.File.stdout().writerStreaming(d.io, &.{});
            d.line.draw(&output.interface, text.buffered(), columns) catch {};
        }
    }
};

fn columnHint(value: ?[]const u8) ?usize {
    const number_value = std.fmt.parseInt(usize, std.mem.trim(u8, value orelse return null, " \t\n\r"), 10) catch return null;
    return if (number_value > 0) number_value else null;
}

test "terminal width respects positive COLUMNS and otherwise probes the terminal" {
    try std.testing.expectEqual(@as(?usize, 80), columnHint(" 80 "));
    for ([_]?[]const u8{ null, "", "0", "-1", "invalid" }) |value| try std.testing.expectEqual(null, columnHint(value));
}

test "queued rejection, memory waiting and cancellation release request counters" {
    var stats = Stats{ .io = std.testing.io, .allocator = std.testing.allocator };
    defer stats.deinit();
    stats.enqueue();
    stats.enqueue();
    stats.enqueue();
    stats.finish(false);
    stats.activate();
    try std.testing.expectEqual(@as(usize, 2), stats.snapshot().connections);
    try std.testing.expectEqual(@as(usize, 1), stats.snapshot().waiting_requests);
    stats.finish(true);
    stats.finish(false);
    try std.testing.expectEqual(@as(usize, 0), stats.snapshot().connections);
    try std.testing.expectEqual(@as(usize, 0), stats.snapshot().waiting_requests);
}

pub fn check(io: std.Io, path: []const u8) !void {
    const a = std.heap.page_allocator;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(8 * 1024 * 1024));
    defer a.free(bytes);
    const parsed = try std.json.parseFromSlice(struct {
        events: []const struct { time: f64, tokens: usize, seconds: f64, prefill: usize, connections: usize, waiting: usize, decode_rate: f64, prefill_rate: f64, text: []const u8 },
        lines: []const struct { action: []const u8, text: []const u8, columns: usize, stdout: []const u8, stderr: []const u8 },
    }, a, bytes, .{});
    defer parsed.deinit();
    var decoded = Meter{};
    defer decoded.events.deinit(a);
    var prefilled = ChunkRate{};
    for (parsed.value.events) |event| {
        try decoded.add(a, event.tokens, event.time);
        prefilled.add(event.prefill, event.seconds, event.time);
        const snapshot = Snapshot{ .connections = event.connections, .waiting_requests = event.waiting, .decode_tokens_per_second = decoded.rate(event.time), .prefill_tokens_per_second = prefilled.rate(event.time) };
        try std.testing.expectEqual(event.decode_rate, snapshot.decode_tokens_per_second);
        try std.testing.expectEqual(event.prefill_rate, snapshot.prefill_tokens_per_second);
        var text: std.Io.Writer.Allocating = .init(a);
        defer text.deinit();
        try snapshot.render(&text.writer);
        try std.testing.expectEqualStrings(event.text, text.written());
    }
    var line = Line{};
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    var err: std.Io.Writer.Allocating = .init(a);
    defer err.deinit();
    for (parsed.value.lines) |event| {
        if (std.mem.eql(u8, event.action, "draw")) try line.draw(&out.writer, event.text, event.columns) else if (std.mem.eql(u8, event.action, "stop")) try line.erase(&out.writer) else try line.write(&out.writer, if (std.mem.eql(u8, event.action, "stdout")) &out.writer else &err.writer, event.text);
        try std.testing.expectEqualStrings(event.stdout, out.written());
        try std.testing.expectEqualStrings(event.stderr, err.written());
    }
    std.debug.print("PASS: {d} upstream live rate/status cases and {d} terminal redraw/write cases\n", .{ parsed.value.events.len, parsed.value.lines.len });
}
