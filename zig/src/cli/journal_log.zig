//! `tensorfold log`: reads the journal's or request log's kept files oldest first; never writes, locks or truncates.
const std = @import("std");
const show = @import("journal_show.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const Writer = std.Io.Writer;

pub const usage =
    \\usage: tensorfold log [list] [PATH] [--last N] [--route chat|completions] [--model M] [--json]
    \\       tensorfold log show ID [PATH] [--json]
    \\       tensorfold log follow [PATH] [--json]
    \\PATH defaults to $TENSORFOLD_REQUEST_JOURNAL, or $TENSORFOLD_REQUEST_LOG when that is unset.
    \\PATH.3, PATH.2, PATH.1 and PATH are read, oldest first, whatever the file's shape.
    \\
;

pub const Command = enum { list, show, follow };

pub const Options = struct {
    command: Command = .list,
    id: ?[]const u8 = null,
    path: ?[]const u8 = null,
    last: ?usize = null,
    route: ?[]const u8 = null,
    model: ?[]const u8 = null,
    json: bool = false,
    /// Tests only: how often `follow` looks for new records, and after how many finished lines it returns.
    poll_ms: u64 = 250,
    follow_limit: ?usize = null,
};

/// The arguments after `log`; null on anything it doesn't take.
pub fn parse(args: []const []const u8) ?Options {
    var o: Options = .{};
    var i: usize = 0;
    if (args.len > 0) {
        if (std.meta.stringToEnum(Command, args[0])) |c| {
            o.command = c;
            i = 1;
        }
    }
    if (o.command == .show) {
        if (i >= args.len or std.mem.startsWith(u8, args[i], "--")) return null;
        o.id = args[i];
        i += 1;
    }
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--json")) {
            o.json = true;
            continue;
        }
        const takes = [_][]const u8{ "--last", "--route", "--model" };
        const flag = for (takes) |f| {
            if (std.mem.eql(u8, arg, f)) break f;
        } else null;
        if (flag) |f| {
            if (o.command != .list or i + 1 >= args.len) return null;
            i += 1;
            if (std.mem.eql(u8, f, "--last")) o.last = std.fmt.parseInt(usize, args[i], 10) catch return null;
            if (std.mem.eql(u8, f, "--route")) o.route = args[i];
            if (std.mem.eql(u8, f, "--model")) o.model = args[i];
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--") or o.path != null) return null;
        o.path = arg;
    }
    return o;
}

/// One request: its records as journaled, in the order the journal first named it.
pub const Entry = struct {
    id: []const u8,
    start: ?Value = null,
    prepared: ?Value = null,
    finish: ?Value = null,
    quarantined: bool = false,
    bare_body: bool = false,

    pub fn body(e: Entry) ?Value {
        const s = e.start orelse return null;
        return if (e.bare_body) s else s.object.get("body");
    }

    pub fn startedAt(e: Entry) ?i64 {
        if (e.bare_body) return null;
        return if (e.start) |s| int(s, "at") else null;
    }
};

const Record = enum { bare, journal, request, reply };

fn recordOf(v: Value) Record {
    const fields = v.object;
    // A chat or completion body can carry arbitrary client fields alongside its input.
    if (fields.contains("messages") or fields.contains("prompt")) return .bare;
    if ((fields.get("version") orelse .null) == .integer and (fields.get("at") orelse .null) == .integer and
        str(v, "kind") != null) return .journal;
    const tag = str(v, "type") orelse return .bare;
    const id = fields.get("id") orelse return .bare;
    if (eql(tag, "reply")) return if (id == .string and isReply(v)) .reply else .bare;
    // Keep unknown tags that use the same envelope as a request.
    return if ((id == .string or id == .null) and fields.contains("body")) .request else .bare;
}

fn isReply(v: Value) bool {
    const fields = v.object;
    const status = fields.get("status") orelse return false;
    if (status != .integer) return false;
    if (status.integer >= 400) return str(v, "error") != null;
    const reason = fields.get("finish_reason") orelse return false;
    return (reason == .string or reason == .null) and
        (fields.get("stream") orelse .null) == .bool and str(v, "text") != null and
        (fields.get("usage") orelse .null) == .object and (fields.get("timings") orelse .null) == .object and
        (fields.get("truncated") orelse .null) == .bool;
}

/// Every request the kept files name, oldest first; a torn last line is left for the next read.
pub const Log = struct {
    a: Allocator,
    entries: std.ArrayList(Entry) = .empty,
    index: std.StringHashMapUnmanaged(usize) = .empty,
    skipped: usize = 0,
    /// The number of request-log lines that have been assigned an id by the reader.
    numbered: usize = 0,

    /// One complete line as (request index, was its finish); an unreadable record is passed over and counted.
    pub fn add(l: *Log, text: []const u8) !?struct { usize, bool } {
        if (std.mem.trim(u8, text, " \r").len == 0) return null;
        const v = std.json.parseFromSliceLeaky(Value, l.a, text, .{}) catch {
            l.skipped += 1;
            return null;
        };
        if (v != .object) {
            l.skipped += 1;
            return null;
        }
        const record = recordOf(v);
        const id = switch (record) {
            .bare => try l.numberedId(),
            .journal => str(v, "id") orelse return null,
            .request => str(v, "id") orelse try l.numberedId(),
            .reply => str(v, "id").?,
        };
        const at = l.index.get(id) orelse blk: {
            try l.entries.append(l.a, .{ .id = id });
            try l.index.put(l.a, id, l.entries.items.len - 1);
            break :blk l.entries.items.len - 1;
        };
        const e = &l.entries.items[at];
        switch (record) {
            .bare => {
                e.start = v;
                e.bare_body = true;
                return .{ at, true };
            },
            .request => e.start = v,
            .reply => {
                e.finish = .{ .object = liftedFinish(l.a, v) };
                return .{ at, true };
            },
            .journal => {
                const kind = str(v, "kind").?;
                if (eql(kind, "finish")) {
                    e.finish = v;
                    return .{ at, true };
                }
                if (eql(kind, "prepared")) e.prepared = v else if (eql(kind, "quarantine")) e.quarantined = true else e.start = v;
            },
        }
        return .{ at, false };
    }

    /// Number request-log lines without a server id in the order they are read.
    fn numberedId(l: *Log) ![]const u8 {
        l.numbered += 1;
        return std.fmt.allocPrint(l.a, "{d}", .{l.numbered});
    }

    /// The complete lines of `bytes`; returns how many bytes they span (the rest is a line still being written).
    pub fn feed(l: *Log, bytes: []const u8, done: ?*std.ArrayList(usize)) !usize {
        var used: usize = 0;
        while (std.mem.indexOfScalarPos(u8, bytes, used, '\n')) |nl| {
            if (try l.add(bytes[used..nl])) |got| if (got[1]) if (done) |found| try found.append(l.a, got[0]);
            used = nl + 1;
        }
        return used;
    }
};

/// The path to read: the argument, else $TENSORFOLD_REQUEST_JOURNAL, else $TENSORFOLD_REQUEST_LOG.
pub fn journalPath(o: Options, env: ?*const std.process.Environ.Map) ?[]const u8 {
    if (o.path) |p| return p;
    const m = env orelse return null;
    if (m.get("TENSORFOLD_REQUEST_JOURNAL")) |v| if (v.len != 0) return v;
    const r = m.get("TENSORFOLD_REQUEST_LOG") orelse return null;
    return if (r.len == 0) null else r;
}

/// Whether the chosen `PATH` came from `$TENSORFOLD_REQUEST_LOG`, whose writer keeps two files, not four.
fn fromRequestLog(o: Options, env: ?*const std.process.Environ.Map) bool {
    if (o.path) |_| return false;
    const m = env orelse return false;
    if (m.get("TENSORFOLD_REQUEST_JOURNAL")) |v| if (v.len != 0) return false;
    const r = m.get("TENSORFOLD_REQUEST_LOG") orelse return false;
    return r.len != 0;
}

const suffixes = [_][]const u8{ ".3", ".2", ".1", "" };

/// Reads PATH.3 to PATH into `l`; returns how many of the four files exist.
pub fn load(a: Allocator, io: std.Io, path: []const u8, l: *Log) !usize {
    var found: usize = 0;
    for (suffixes) |s| {
        const name = try std.fmt.allocPrint(a, "{s}{s}", .{ path, s });
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, name, a, .limited(1 << 30)) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => return e,
        };
        found += 1;
        _ = try l.feed(bytes, null);
    }
    return found;
}

pub fn run(a: Allocator, io: std.Io, out: *Writer, err: *Writer, env: ?*const std.process.Environ.Map, args: []const []const u8) !u8 {
    const o = parse(args) orelse {
        try err.writeAll(usage);
        return 2;
    };
    return runWith(a, io, out, err, env, o);
}

pub fn runWith(a: Allocator, io: std.Io, out: *Writer, err: *Writer, env: ?*const std.process.Environ.Map, o: Options) !u8 {
    const path = journalPath(o, env) orelse {
        try err.writeAll("tensorfold log: name the journal (tensorfold log list PATH) or set TENSORFOLD_REQUEST_JOURNAL\n");
        return 2;
    };
    var l: Log = .{ .a = a };
    if (try load(a, io, path, &l) == 0 and o.command != .follow) {
        if (fromRequestLog(o, env)) // its writer keeps the current file and one archive, no `.2` or `.3`
            try err.print("tensorfold log: no journal at {s} (nor {s}.1)\n", .{ path, path })
        else
            try err.print("tensorfold log: no journal at {s} (nor {s}.1 to {s}.3)\n", .{ path, path, path });
        return 1;
    }
    switch (o.command) {
        .list => try list(&l, out, o),
        .show => {
            const at = l.index.get(o.id.?) orelse {
                try err.print("tensorfold log: no request {s} in {s}\n", .{ o.id.?, path });
                return 1;
            };
            const e = l.entries.items[at];
            if (o.json) try writeJson(out, e) else try show.entry(a, out, e);
        },
        .follow => try follow(a, io, path, &l, out, o),
    }
    if (l.skipped > 0) try err.print("tensorfold log: {d} unreadable record(s) passed over\n", .{l.skipped});
    return 0;
}

fn list(l: *Log, out: *Writer, o: Options) !void {
    var picked: std.ArrayList(Entry) = .empty;
    for (l.entries.items) |e| {
        if (e.start == null and e.finish == null and !e.quarantined) continue; // a clear or stop record
        if (o.route) |r| if (!eql(routeOf(e), r) and !eql(pathOf(e), r)) continue;
        if (o.model) |m| if (!eql(modelOf(e), m)) continue;
        try picked.append(l.a, e);
    }
    const from = if (o.last) |n| picked.items.len -| n else 0;
    for (picked.items[from..]) |e| if (o.json) try writeJson(out, e) else try line(out, e);
}

/// The list line: start time (UTC), reply id, route, model, status and outcome, tokens, first token and decode rate.
pub fn line(out: *Writer, e: Entry) !void {
    try stamp(out, if (e.start != null) e.startedAt() else if (e.finish) |f| int(f, "at") else null);
    try out.print("  {s}  {s}  {s}  ", .{ e.id, routeOf(e), modelOf(e) });
    if (e.finish) |f| {
        try out.print("{d} {s}", .{ int(f, "status") orelse 0, str(f, "outcome") orelse "finished" });
    } else try out.writeAll(if (e.quarantined) "quarantined (in flight at a crash)" else "in flight");
    if (e.quarantined and e.finish != null) try out.writeAll(" (quarantined)");
    const prompt = if (e.finish) |f| int(f, "prompt_tokens") else null;
    try out.writeAll("  prompt ");
    try optInt(out, prompt orelse if (e.prepared) |p| int(p, "prompt_tokens") else null);
    try out.writeAll("  out ");
    try optInt(out, if (e.finish) |f| int(f, "completion_tokens") else null);
    try out.writeAll("  ttft ");
    if (if (e.finish) |f| num(f, "ttft_ms") else null) |v| try out.print("{d:.1} ms", .{v}) else try out.writeAll("-");
    try out.writeAll("  ");
    if (if (e.finish) |f| num(f, "decode_tps") else null) |v| try out.print("{d:.1} tok/s", .{v}) else try out.writeAll("- tok/s");
    try out.writeAll("\n");
}

/// `--json`: the request's records as journaled, one object a line.
pub fn writeJson(out: *Writer, e: Entry) !void {
    try out.print("{{\"id\":{f},\"quarantined\":{s},\"start\":", .{ std.json.fmt(e.id, .{}), if (e.quarantined) "true" else "false" });
    try std.json.Stringify.value(e.start orelse Value.null, .{}, out);
    try out.writeAll(",\"prepared\":");
    try std.json.Stringify.value(e.prepared orelse Value.null, .{}, out);
    try out.writeAll(",\"finish\":");
    try std.json.Stringify.value(e.finish orelse Value.null, .{}, out);
    try out.writeAll("}\n");
}

/// Prints each newly finished request's list line, following PATH across rotations until stopped.
fn follow(a: Allocator, io: std.Io, path: []const u8, l: *Log, out: *Writer, o: Options) !void {
    var file: ?std.Io.File = null;
    defer if (file) |f| f.close(io);
    var offset: u64 = 0;
    var pending: std.ArrayList(u8) = .empty;
    var printed: usize = 0;
    // the records already kept were read by `load`: follow starts at the current file's end
    if (std.Io.Dir.cwd().openFile(io, path, .{})) |f| {
        file = f;
        offset = (try f.stat(io)).size;
        offset = lastNewline(try readFrom(a, io, f, 0, offset)); // a line still being written is read once whole
    } else |_| {}
    while (true) {
        try out.flush();
        if (o.follow_limit) |n| if (printed >= n) return;
        std.Io.sleep(io, .fromMilliseconds(@intCast(o.poll_ms)), .awake) catch return;
        if (file == null) {
            file = std.Io.Dir.cwd().openFile(io, path, .{}) catch continue;
            offset = 0;
        }
        const f = file.?;
        const size = (try f.stat(io)).size;
        if (size > offset) {
            try pending.appendSlice(a, try readFrom(a, io, f, offset, size));
            offset = size;
            printed += try drain(l, &pending, out, o);
        }
        // rotation: PATH is a new file once the server renamed this one to PATH.1; this one is read to its end first
        const now = std.Io.Dir.cwd().statFile(io, path, .{}) catch continue;
        if (now.inode != (try f.stat(io)).inode) {
            f.close(io);
            file = null;
        }
    }
}

fn drain(l: *Log, pending: *std.ArrayList(u8), out: *Writer, o: Options) !usize {
    var finished: std.ArrayList(usize) = .empty;
    const used = try l.feed(pending.items, &finished);
    pending.replaceRangeAssumeCapacity(0, used, &.{});
    for (finished.items) |i| if (o.json) try writeJson(out, l.entries.items[i]) else try line(out, l.entries.items[i]);
    return finished.items.len;
}

fn readFrom(a: Allocator, io: std.Io, f: std.Io.File, from: u64, to: u64) ![]u8 {
    const buf = try a.alloc(u8, @intCast(to - from));
    const n = try f.readPositionalAll(io, buf, from);
    return buf[0..n];
}

fn lastNewline(bytes: []const u8) usize {
    return if (std.mem.lastIndexOfScalar(u8, bytes, '\n')) |i| i + 1 else 0;
}

pub fn stamp(out: *Writer, at: ?i64) !void {
    const t = at orelse return out.writeAll("-------------------");
    const s: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(0, t)) };
    const yd = s.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = s.getDaySeconds();
    try out.print("{d:0>4}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2}:{d:0>2}", .{ yd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute() });
}

pub fn pathOf(e: Entry) []const u8 {
    const s = e.start orelse return "-";
    if (!e.bare_body) if (str(s, "path")) |p| return p;
    const body = e.body() orelse return "-";
    if (body != .object) return "-";
    // the request log keeps no path, so the body's shape names the route: a prompt list is a completion
    if (body.object.get("messages") != null) return "chat";
    if (body.object.get("prompt") != null) return "completions";
    return "-";
}

pub fn routeOf(e: Entry) []const u8 {
    const p = pathOf(e);
    if (eql(p, "/v1/chat/completions")) return "chat";
    if (eql(p, "/v1/completions")) return "completions";
    return p;
}

pub fn modelOf(e: Entry) []const u8 {
    const s = e.start orelse return "-";
    const body = e.body() orelse s;
    return if (body == .object) str(body, "model") orelse "-" else "-";
}

pub fn optInt(out: *Writer, v: ?i64) !void {
    if (v) |n| try out.print("{d}", .{n}) else try out.writeAll("-");
}

pub fn str(v: Value, key: []const u8) ?[]const u8 {
    if (v != .object) return null;
    const f = v.object.get(key) orelse return null;
    return if (f == .string) f.string else null;
}

pub fn int(v: Value, key: []const u8) ?i64 {
    if (v != .object) return null;
    const f = v.object.get(key) orelse return null;
    return switch (f) {
        .integer => |n| n,
        .float => |x| @intFromFloat(x),
        else => null,
    };
}

pub fn num(v: Value, key: []const u8) ?f64 {
    if (v != .object) return null;
    const f = v.object.get(key) orelse return null;
    return switch (f) {
        .integer => |n| @floatFromInt(n),
        .float => |x| x,
        else => null,
    };
}

/// The measures `line` and `show` read, which a request-log reply record nests under `usage` and `timings`.
const lifted = [_][]const u8{ "prompt_tokens", "cached_tokens", "completion_tokens", "queue_ms", "ttft_ms", "decode_tps" };

/// A reply record with its nested measures put beside the rest of its fields, so one reader prints both sources.
fn liftedFinish(a: Allocator, v: Value) std.json.ObjectMap {
    // the copy is what the entry keeps, as `put` may move the entries array: `v` is not read after this
    var fields = v.object;
    for ([_][]const u8{ "usage", "timings" }) |group| {
        const nested = fields.get(group) orelse continue;
        if (nested != .object) continue;
        for (lifted) |k| if (fields.get(k) == null) if (nested.object.get(k)) |x| fields.put(a, k, x) catch return fields;
    }
    return fields;
}

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

test {
    _ = @import("journal_log_test.zig");
    _ = @import("journal_log_shape_test.zig");
}
