//! TENSORFOLD_REQUEST_JOURNAL: an opt-in, owner-only, rotated request journal; crashed requests are quarantined.
const std = @import("std");
const platform = @import("sys");
const json = @import("json");
const log = @import("log.zig");
const private = @import("request_log.zig").private;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Sha256 = std.crypto.hash.sha2.Sha256;
const c = platform.c;

/// Past this a file becomes `.1`, each older file moves up one, and the oldest of `files` is removed.
pub const file_limit: usize = 64 << 20;
pub const files = 4;
/// The most one record holds: a larger body or reply is journaled by its size and SHA-256 only.
pub const record_limit: usize = 8 << 20;
/// A quarantine lapses a day after the crash that set it, unless cleared sooner; at most `quarantine_limit` hold.
pub const quarantine_s: i64 = 24 * 60 * 60;
pub const quarantine_limit = 4096;
/// Requests in flight at one crash are only flagged; one in flight at a second crash too is refused.
const refuse_at = 2;
const path_max = 4000;

/// The API key's label, the route and the whole body: one key's crashed request never blocks another key's.
pub const Fingerprint = [32]u8;

pub fn fingerprint(key: []const u8, chat: bool, body: Value) Fingerprint {
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer.Hashing(Sha256) = .init(&buf);
    w.writer.print("tensorfold-quarantine-1\x00{s}\x00{s}\x00", .{ key, if (chat) "chat" else "completion" }) catch unreachable;
    json.write(&w.writer, body, .{ .compact = true }) catch unreachable; // hashing drains never fail
    w.writer.flush() catch unreachable;
    return w.hasher.finalResult();
}

/// What serve prints when the journal can't be opened.
pub fn why(e: anyerror) []const u8 {
    return switch (e) {
        error.JournalBusy => "another server holds it (its .lock file is locked)",
        error.JournalPermissions => "it or its .lock file is not a regular file only its owner can read and write",
        error.JournalOpen => "it or its .lock file cannot be opened, or is a symbolic link",
        error.JournalPath => "the path is empty or too long",
        else => @errorName(e),
    };
}

pub const Journal = struct {
    gpa: Allocator,
    io: std.Io,
    path: [:0]const u8,
    lock: c.fd_t,
    fd: c.fd_t,
    mutex: std.Io.Mutex = .init,
    bytes: usize = 0,
    limit: usize = file_limit,
    lost: std.atomic.Value(u64) = .init(0),
    blocked: std.AutoHashMapUnmanaged(Fingerprint, i64) = .empty, // the second each refusal lapses; fixed once open

    /// Takes `path.lock`, reads every kept file and quarantines the requests a crash left without a finish.
    pub fn open(gpa: Allocator, io: std.Io, path: []const u8) !*Journal {
        if (path.len == 0 or path.len > path_max or std.mem.indexOfScalar(u8, path, 0) != null) return error.JournalPath;
        const j = try gpa.create(Journal);
        errdefer gpa.destroy(j);
        j.* = .{ .gpa = gpa, .io = io, .path = try gpa.dupeSentinel(u8, path, 0), .lock = -1, .fd = -1 };
        errdefer j.release();
        var buf: [path_max + 16]u8 = undefined;
        j.lock = try openPrivate(try sibling(&buf, path, ".lock"), .{ .ACCMODE = .RDWR, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true });
        if (c.flock(j.lock, c.LOCK.EX | c.LOCK.NB) != 0) return error.JournalBusy;
        j.fd = try openPrivate(j.path, .{ .ACCMODE = .RDWR, .APPEND = true, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true });
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var r: Recovery = .{ .a = arena.allocator(), .gpa = gpa, .now = j.now() };
        var older: usize = files;
        while (older > 1) {
            older -= 1;
            const fd = openPrivate(try sibling(&buf, path, suffix(older)), .{ .NOFOLLOW = true, .CLOEXEC = true }) catch continue;
            defer _ = c.close(fd);
            try r.scan(fd);
        }
        try r.scan(j.fd);
        j.bytes = r.whole;
        // a crash mid-write leaves a torn last line; the next record starts on a fresh one
        if (r.size > r.whole and c.ftruncate(j.fd, @intCast(r.whole)) != 0) return error.JournalWrite;
        try r.settle(j);
        if (r.skipped > 0) log.line("request journal {s}: {d} unreadable record(s) passed over", .{ path, r.skipped });
        if (j.blocked.count() > 0) log.line("request journal {s}: {d} request(s) quarantined after crashes", .{ path, j.blocked.count() });
        return j;
    }

    pub fn close(j: *Journal) void {
        const lost = j.lost.load(.monotonic);
        if (lost > 0) log.line("request journal {s}: {d} record(s) were not written", .{ j.path, lost });
        j.release();
        j.gpa.destroy(j);
    }

    fn release(j: *Journal) void {
        if (j.fd >= 0) _ = c.close(j.fd);
        if (j.lock >= 0) _ = c.close(j.lock);
        j.blocked.deinit(j.gpa);
        j.gpa.free(j.path);
    }

    /// A clean stop: requests still open now were not in flight at a crash.
    pub fn stopped(j: *Journal) void {
        var arena: std.heap.ArenaAllocator = .init(j.gpa);
        defer arena.deinit();
        const o = j.record(arena.allocator(), "stop", null) catch return j.lose();
        j.append(arena.allocator(), .{ .object = o });
    }

    /// Whether this fingerprint crashed the server and is refused until it lapses or the operator clears it.
    pub fn quarantined(j: *Journal, fp: Fingerprint) bool {
        const until = j.blocked.get(fp) orelse return false;
        return j.now() < until;
    }

    pub fn record(j: *Journal, a: Allocator, kind: []const u8, id: ?[]const u8) Allocator.Error!*json.Object {
        const o = try json.newObject(a);
        try o.put(a, "version", .{ .int = "1" });
        try o.put(a, "kind", .{ .string = kind });
        if (id) |s| try o.put(a, "id", .{ .string = s });
        try o.put(a, "at", try json.intValue(a, j.now()));
        return o;
    }

    /// One record as one line in one write, with no sync; a failure or a full file loses the record, never the request.
    pub fn append(j: *Journal, a: Allocator, value: Value) void {
        var out: std.Io.Writer.Allocating = .init(a);
        json.write(&out.writer, value, .{ .compact = true }) catch return j.lose();
        out.writer.writeByte('\n') catch return j.lose();
        const line = out.written();
        if (line.len > record_limit) return j.lose();
        j.mutex.lockUncancelable(j.io);
        defer j.mutex.unlock(j.io);
        if (j.fd >= 0 and j.bytes > 0 and j.bytes + line.len > j.limit) j.rotate();
        if (j.fd < 0) return j.lose();
        const n = c.write(j.fd, line.ptr, line.len);
        if (n == @as(isize, @intCast(line.len))) {
            j.bytes += line.len;
            return;
        }
        if (n > 0) _ = c.ftruncate(j.fd, @intCast(j.bytes)); // no torn line for the next record to follow
        j.lose();
    }

    /// The current file becomes `.1`, older ones move up, and a new one starts; a failure stops the journal.
    fn rotate(j: *Journal) void {
        _ = c.close(j.fd);
        j.fd = -1;
        var from: [path_max + 16]u8 = undefined;
        var to: [path_max + 16]u8 = undefined;
        var i: usize = files - 1;
        while (i > 0) : (i -= 1) _ = c.rename(sibling(&from, j.path, suffix(i - 1)) catch return, sibling(&to, j.path, suffix(i)) catch return);
        j.fd = openPrivate(j.path, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }) catch -1;
        j.bytes = 0;
    }

    fn lose(j: *Journal) void {
        if (j.lost.fetchAdd(1, .monotonic) == 0) log.line("request journal {s}: a record was not written (a full disk, an I/O error or a record past {d} MiB); requests go on", .{ j.path, record_limit >> 20 });
    }

    fn now(j: *const Journal) i64 {
        return std.Io.Clock.real.now(j.io).toSeconds();
    }
};

fn suffix(n: usize) []const u8 {
    return ([_][]const u8{ "", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8" })[n];
}

fn sibling(buf: []u8, path: []const u8, tail: []const u8) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}{s}", .{ path, tail }, 0) catch error.JournalPath;
}

/// Opened only as a regular file no one but its owner can read or write; a link or a shared file is refused.
fn openPrivate(path: [*:0]const u8, flags: c.O) !c.fd_t {
    const fd = c.open(path, flags, @as(c.mode_t, 0o600));
    if (fd < 0) return error.JournalOpen;
    if (private(fd)) return fd;
    _ = c.close(fd);
    return error.JournalPermissions;
}

/// The kept files read oldest first: starts still open at the end were in flight when the server last stopped.
const Recovery = struct {
    a: Allocator,
    gpa: Allocator,
    now: i64,
    pending: std.StringArrayHashMapUnmanaged(Fingerprint) = .empty,
    crashes: std.AutoArrayHashMapUnmanaged(Fingerprint, Crashes) = .empty,
    skipped: usize = 0,
    size: usize = 0,
    whole: usize = 0, // the last file's bytes up to and including its last newline
    const Crashes = struct { n: u32 = 0, until: i64 = 0 };

    fn scan(r: *Recovery, fd: c.fd_t) !void {
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(r.gpa);
        var scratch: std.heap.ArenaAllocator = .init(r.gpa);
        defer scratch.deinit();
        var chunk: [1 << 16]u8 = undefined;
        var over = false; // past a record's size: the line is passed over
        r.size = 0;
        r.whole = 0;
        while (true) {
            const n = c.read(fd, &chunk, chunk.len);
            if (n < 0) return error.JournalRead;
            if (n == 0) break;
            var rest: []const u8 = chunk[0..@intCast(n)];
            r.size += rest.len;
            while (std.mem.indexOfScalar(u8, rest, '\n')) |at| {
                if (!over) try line.appendSlice(r.gpa, rest[0..at]);
                if (over) r.skipped += 1 else try r.entry(scratch.allocator(), line.items);
                _ = scratch.reset(.retain_capacity);
                line.clearRetainingCapacity();
                over = false;
                rest = rest[at + 1 ..];
                r.whole = r.size - rest.len;
            }
            if (!over and line.items.len + rest.len > record_limit) {
                over = true;
                line.clearRetainingCapacity();
            } else if (!over) try line.appendSlice(r.gpa, rest);
        }
    }

    fn entry(r: *Recovery, a: Allocator, text: []const u8) !void {
        if (text.len == 0) return;
        const v = switch (try json.parse(a, text)) {
            .ok => |v| v,
            .err => return r.skip(),
        };
        if (((v.get("version") orelse return r.skip()).int64() orelse return r.skip()) != 1) return r.skip();
        const kind = v.strField("kind") orelse return r.skip();
        if (eql(kind, "stop")) return r.pending.clearRetainingCapacity();
        if (eql(kind, "clear")) {
            if (json.truthyField(v, "all")) return r.crashes.clearRetainingCapacity();
            _ = r.crashes.orderedRemove(hexOf(v) orelse return r.skip());
            return;
        }
        const id = v.strField("id") orelse return r.skip();
        if (eql(kind, "start")) return r.pending.put(r.a, try r.a.dupe(u8, id), hexOf(v) orelse return r.skip());
        if (!eql(kind, "finish") and !eql(kind, "quarantine")) return;
        _ = r.pending.orderedRemove(id);
        if (!eql(kind, "quarantine")) return;
        const at = (v.get("at") orelse return r.skip()).int64() orelse return r.skip();
        try r.add(hexOf(v) orelse return r.skip(), at + quarantine_s);
    }

    fn add(r: *Recovery, fp: Fingerprint, until: i64) !void {
        if (until <= r.now) return;
        const w = try r.crashes.getOrPut(r.a, fp);
        if (!w.found_existing) w.value_ptr.* = .{};
        w.value_ptr.n += 1;
        w.value_ptr.until = @max(w.value_ptr.until, until);
    }

    /// The last session's open starts are quarantined, then every fingerprint in flight at two crashes is refused.
    fn settle(r: *Recovery, j: *Journal) !void {
        for (r.pending.keys(), r.pending.values()) |id, fp| {
            const o = try j.record(r.a, "quarantine", id);
            const hex = std.fmt.bytesToHex(fp, .lower);
            try o.put(r.a, "fingerprint", .{ .string = try r.a.dupe(u8, &hex) });
            j.append(r.a, .{ .object = o });
            try r.add(fp, r.now + quarantine_s);
        }
        for (r.crashes.keys(), r.crashes.values()) |fp, w| {
            if (w.n >= refuse_at and j.blocked.count() < quarantine_limit) try j.blocked.put(j.gpa, fp, w.until);
        }
    }

    fn skip(r: *Recovery) void {
        r.skipped += 1;
    }
};

fn hexOf(v: Value) ?Fingerprint {
    const text = v.strField("fingerprint") orelse return null;
    var fp: Fingerprint = undefined;
    if (text.len != 64) return null;
    _ = std.fmt.hexToBytes(&fp, text) catch return null;
    return fp;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
