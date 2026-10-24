//! The notes' journal next to the snapshot: each write one JSON line, on disk before its reply, replayed at startup.
const std = @import("std");
const libc = @import("sys").c;
const builtin = @import("builtin");
const json = @import("json");
const write = @import("write.zig");
const intent = @import("erase_intent.zig");
const tmp_file = @import("journal_tmp.zig");
const Record = write.Record;
const Value = json.Value;
const Allocator = std.mem.Allocator;
pub const format = "tf-notes-journal/3";
pub const format_v2 = "tf-notes-journal/2";
/// The first format: no commit lines, every whole line a write (read, then rewritten as format 3, at open).
pub const format_v1 = "tf-notes-journal/1";

pub const OpenError = error{ OutOfMemory, JournalIo, BadJournal, OtherSnapshot, Locked };
pub const IoError = error{ OutOfMemory, JournalIo };
/// JournalPoisoned: an append failed and so did putting the file back; it may hold part of a record until a restart.
pub const AppendError = IoError || error{JournalPoisoned};
/// RenamedUnsynced: the new file is renamed in but its folder's sync failed; a power cut may leave either file, whole.
pub const RewriteError = IoError || error{RenamedUnsynced};

/// Tests only: the Nth (from 0) write, truncate, file sync, folder sync or F_FULLFSYNC call fails, counted apart.
pub const Faults = struct {
    write: ?u32 = null, // a failing write puts down half its bytes
    truncate: ?u32 = null,
    sync: ?u32 = null,
    dir: ?u32 = null,
    full: ?u32 = null, // this F_FULLFSYNC returns `full_errno`, and fullSync's fallback rule decides
    full_errno: libc.E = .IO,
    writes: u32 = 0,
    truncates: u32 = 0,
    syncs: u32 = 0,
    dirs: u32 = 0,
    fulls: u32 = 0,
    fallbacks: u32 = 0, // plain fsyncs made because F_FULLFSYNC was unsupported

    fn hit(at: ?u32, count: *u32) bool {
        const n = count.*;
        count.* += 1;
        return at != null and at.? == n;
    }
};

/// Tests only: the faults every journal opened from now on starts with (its cut and its format-1 rewrite at open).
pub var faults_at_open: ?*Faults = null;

/// One server start's writer flags, a journal line: older notes prove "off" from these, not from today's flags.
pub const Run = struct {
    learn: bool = false,
    recall: ?bool = false, // null on older manifests: historical recall is unknown

    compact_memory: []const u8 = "", // empty: compaction was off
    request_log: []const u8 = "", // empty: the request log was off
    slide: bool = false,
    pocket: bool = false, // a Sliding Weights pocket existed (graph facts or saved pocket rows)
    next_source: u64 = 1, // the first source this start would assign
};

/// One snapshot's journal: its path, the open file (after the first write or a replay), and its size.
pub const Journal = struct {
    gpa: Allocator,
    path: [:0]u8,
    snapshot_sha: [64]u8,
    fd: c_int = -1,
    lock_fd: c_int = -1,
    next_id: u32 = 1,
    size: u64 = 0,
    truncated: u64 = 0, // bytes of a torn last line cut at open
    next_source: u64 = 1, // the header's: an erase rewrites it, so an erased write's source is never used again
    stale: usize = 0, // temporary files of a rewrite that died before its rename, removed at open (lock held)
    hook: ?*const fn (ctx: *anyopaque, tmp: [:0]const u8) void = null, // tests: between a rewrite's sync and rename
    faults: ?*Faults = null, // tests: a write, truncate, sync, folder sync or F_FULLFSYNC made to fail
    poisoned: bool = false, // writes stop: a failed rollback, an unsynced rename, or a start whose run line did not persist
    doubt: bool = false, // with poisoned: the failed append's commit line was written, so a restart may apply it
    upgraded: bool = false, // the journal was an older format and was rewritten as format 3 at open
    hook_ctx: *anyopaque = undefined,
    erases: std.ArrayList(intent.Intent) = .empty,
    runs: std.ArrayList(Run) = .empty, // start manifests; strings owned by gpa

    pub fn deinit(j: *Journal) void {
        if (j.fd >= 0) _ = libc.close(j.fd);
        if (j.lock_fd >= 0) _ = libc.close(j.lock_fd);
        for (j.erases.items) |i| i.deinit(j.gpa);
        j.erases.deinit(j.gpa);
        for (j.runs.items) |r| freeRun(j.gpa, r);
        j.runs.deinit(j.gpa);
        j.gpa.free(j.path);
    }

    /// The journal at `path`, read when it exists: its records (strings owned by `a`), checked against the snapshot.
    pub fn open(gpa: Allocator, a: Allocator, path: []const u8, snapshot_sha: [64]u8, records: *std.ArrayList(Record), problem: *[]const u8) OpenError!Journal {
        var j: Journal = .{ .gpa = gpa, .path = try gpa.dupeSentinel(u8, path, 0), .snapshot_sha = snapshot_sha, .faults = faults_at_open };
        errdefer j.deinit();
        const lock_path = try std.fmt.allocPrintSentinel(a, "{s}.lock", .{path}, 0);
        defer a.free(lock_path);
        j.lock_fd = libc.open(lock_path, .{ .ACCMODE = .RDWR, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(libc.mode_t, 0o600));
        if (j.lock_fd < 0) return error.JournalIo;
        if (libc.flock(j.lock_fd, libc.LOCK.EX | libc.LOCK.NB) != 0) {
            problem.* = "another process holds the notes journal: one server per snapshot";
            return error.Locked;
        }
        const fd = libc.open(j.path, .{ .ACCMODE = .RDWR, .NOFOLLOW = true, .CLOEXEC = true }, @as(libc.mode_t, 0o600));
        if (fd < 0) {
            if (libc.errno(fd) == .NOENT) return j; // no write yet: the file comes with the first one
            problem.* = "the notes journal exists but cannot be opened for reading and writing";
            return error.JournalIo;
        }
        j.fd = fd;
        if (libc.flock(fd, libc.LOCK.EX | libc.LOCK.NB) != 0) {
            problem.* = "another process holds the notes journal: one server per snapshot";
            return error.Locked;
        }
        // ours now: a temporary file left by a rewrite that never reached its rename is stale (the journal holds all)
        j.stale = try tmp_file.cleanStale(a, path);
        const text = try readAll(a, fd, problem); // in `a`: the records' strings are slices of it
        const head_end = (std.mem.indexOfScalar(u8, text, '\n') orelse {
            problem.* = "the notes journal has no whole header line";
            return error.BadJournal;
        }) + 1;
        const v = try checkHeader(a, text[0 .. head_end - 1], snapshot_sha, problem);
        j.next_source = v.next_source;
        j.next_id = v.next_id;
        const parsed_head = (try json.parse(a, text[0 .. head_end - 1])).ok;
        if (parsed_head.get("erase_intents")) |ev| {
            const items = ev.items() orelse return error.BadJournal;
            for (items) |x| {
                const owned = try (try intent.decode(a, x)).dupe(gpa);
                j.erases.append(gpa, owned) catch {
                    owned.deinit(gpa);
                    return error.OutOfMemory;
                };
            }
        }
        // a write counts only with its commit line after it; without one (never answered as stored) it is cut, unread
        var good = head_end;
        var pos = head_end;
        while (pos < text.len) {
            const nl = std.mem.indexOfScalarPos(u8, text, pos, '\n') orelse break; // a torn last line
            if (try readRun(a, text[pos..nl], problem)) |run| {
                const owned = try dupeRun(j.gpa, run);
                errdefer freeRun(j.gpa, owned);
                try j.runs.append(j.gpa, owned);
                pos = nl + 1;
                good = pos;
                continue;
            }
            const r = try readRecord(a, text[pos..nl], problem);
            if (v.version == 1) {
                try records.append(a, r);
                pos = nl + 1;
                good = pos;
                continue;
            }
            const cnl = std.mem.indexOfScalarPos(u8, text, nl + 1, '\n') orelse break; // no whole commit line
            if (!try isCommit(a, text[nl + 1 .. cnl], r.source)) {
                problem.* = "a write in the notes journal is followed by something other than its commit line";
                return error.BadJournal;
            }
            try records.append(a, r);
            pos = cnl + 1;
            good = pos;
        }
        if (good < text.len) {
            j.truncated = text.len - good;
            if (!j.cut(good) or !j.sync()) {
                problem.* = "the notes journal ends in a write without its commit line that cannot be cut";
                return error.JournalIo;
            }
        }
        j.size = good;
        if (v.version < 3) {
            // Older formats are rewritten under the lock before stable record identities are persisted.
            j.rewrite(a, records.items, j.next_source) catch |e| {
                problem.* = switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.JournalIo => "the notes journal (older format) could not be rewritten as format 3; nothing changed",
                    error.RenamedUnsynced => upgrade_unsynced, // the start stops either way
                };
                return error.JournalIo;
            };
            j.upgraded = true;
        }
        return j;
    }

    /// Persist start flags now or at create(); failure poisons writes so every note has a recorded start.
    pub fn addRun(j: *Journal, a: Allocator, run: Run) void {
        const owned = dupeRun(j.gpa, run) catch {
            j.poisoned = true;
            return;
        };
        j.runs.append(j.gpa, owned) catch {
            freeRun(j.gpa, owned);
            j.poisoned = true;
            return;
        };
        if (j.fd < 0) return;
        const line = encodeRun(a, owned) catch {
            j.poisoned = true;
            return;
        };
        defer a.free(line);
        if (j.put(line, j.size) and j.sync()) {
            j.size += line.len;
            return;
        }
        _ = j.cut(j.size);
        _ = j.sync();
        j.poisoned = true;
    }

    /// Appends one record, then its commit line, each synced by fullSync; on any failure the file is cut back.
    pub fn append(j: *Journal, a: Allocator, r: Record) AppendError!void {
        if (j.poisoned) return error.JournalPoisoned;
        if (j.fd < 0) try j.create(a);
        const line = try encode(a, r);
        defer a.free(line);
        const mark = try commitLine(a, r.source);
        defer a.free(mark);
        // the write's line, synced; then its commit line, synced: without that line a replay drops the write
        if (j.put(line, j.size) and j.sync()) {
            const marked = j.put(mark, j.size + line.len);
            if (marked and j.sync()) {
                j.size += line.len + mark.len;
                return;
            }
            j.doubt = marked; // the commit line is down and may reach the disk although its sync failed
        }
        // not stored: cut back to the last committed write and sync; failing that, poisoned until a restart
        if (!j.cut(j.size) or !j.sync()) {
            j.poisoned = true;
            return error.JournalPoisoned;
        }
        j.doubt = false;
        return error.JournalIo;
    }

    fn put(j: *Journal, data: []const u8, at: u64) bool {
        if (j.faults) |f| if (Faults.hit(f.write, &f.writes)) {
            _ = writeAll(j.fd, data[0 .. data.len / 2], at);
            return false;
        };
        return writeAll(j.fd, data, at);
    }

    fn cut(j: *Journal, len: u64) bool {
        if (j.faults) |f| if (Faults.hit(f.truncate, &f.truncates)) return false;
        return libc.ftruncate(j.fd, @intCast(len)) == 0;
    }

    fn sync(j: *Journal) bool {
        return j.syncFile(j.fd);
    }

    fn syncFile(j: *Journal, fd: c_int) bool {
        if (j.faults) |f| if (Faults.hit(f.sync, &f.syncs)) return false;
        return fullSync(fd, j.faults);
    }

    /// The folder's entry for the journal (a create or a rename) synced as its file is; false when it cannot be.
    fn syncDir(j: *Journal) bool {
        if (j.faults) |f| if (Faults.hit(f.dir, &f.dirs)) return false;
        var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
        const dir = std.fs.path.dirname(j.path) orelse ".";
        if (dir.len >= buf.len) return false;
        @memcpy(buf[0..dir.len], dir);
        buf[dir.len] = 0;
        const fd = libc.open(@ptrCast(&buf), .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true });
        if (fd < 0) return false;
        defer _ = libc.close(fd);
        return fullSync(fd, j.faults);
    }

    /// The journal as `records` alone: a new file of our own, synced, renamed over it, the folder synced.
    pub fn rewrite(j: *Journal, a: Allocator, records: []const Record, next_source: u64) RewriteError!void {
        const t = try tmp_file.create(a, j.path);
        defer a.free(t.path);
        var ok = false;
        defer if (!ok) tmp_file.removeOwn(t);
        const fd = t.fd;
        const head = try j.header(a, next_source);
        defer a.free(head);
        var size: u64 = 0;
        if (!writeAll(fd, head, size)) return error.JournalIo;
        size += head.len;
        size += try writeRuns(j, a, fd, size);
        for (records) |r| {
            const line = try encode(a, r);
            defer a.free(line);
            const mark = try commitLine(a, r.source);
            defer a.free(mark);
            if (!writeAll(fd, line, size) or !writeAll(fd, mark, size + line.len)) return error.JournalIo;
            size += line.len + mark.len;
        }
        if (!j.syncFile(fd)) return error.JournalIo;
        if (libc.flock(fd, libc.LOCK.EX | libc.LOCK.NB) != 0) return error.JournalIo;
        if (j.hook) |h| h(j.hook_ctx, t.path);
        // up to here a failure leaves the old journal as it was: nothing changed (rename either happens or not)
        if (libc.rename(t.path, j.path) != 0) return error.JournalIo;
        ok = true;
        // the path names the new file now; until its folder is synced a power cut may bring back the old one
        if (j.fd >= 0) _ = libc.close(j.fd);
        j.fd = fd;
        j.size = size;
        j.next_source = next_source;
        if (!j.syncDir()) {
            j.poisoned = true;
            return error.RenamedUnsynced;
        }
    }

    fn header(j: *Journal, a: Allocator, next_source: u64) Allocator.Error![]u8 {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const intents = try intent.encode(arena.allocator(), j.erases.items);
        return std.fmt.allocPrint(a, "{{\"format\":\"{s}\",\"snapshot_sha256\":\"{s}\",\"next_source\":{d},\"next_id\":{d},\"erase_intents\":{s}}}\n", .{ format, &j.snapshot_sha, next_source, j.next_id, intents });
    }

    /// The file with its header, made at the first write.
    fn create(j: *Journal, a: Allocator) IoError!void {
        const head = try j.header(a, j.next_source);
        defer a.free(head);
        const fd = libc.open(j.path, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(libc.mode_t, 0o600));
        if (fd < 0) return error.JournalIo;
        if (libc.flock(fd, libc.LOCK.EX | libc.LOCK.NB) != 0) {
            _ = libc.close(fd);
            return error.JournalIo;
        }
        // the header synced and the new name in the folder synced, or no journal (it holds no write yet)
        if (!writeAll(fd, head, 0)) {
            _ = libc.close(fd);
            _ = libc.unlink(j.path);
            return error.JournalIo;
        }
        const extra = writeRuns(j, a, fd, head.len) catch {
            _ = libc.close(fd);
            _ = libc.unlink(j.path);
            return error.JournalIo;
        };
        if (!j.syncFile(fd) or !j.syncDir()) {
            _ = libc.close(fd);
            _ = libc.unlink(j.path);
            return error.JournalIo;
        }
        j.fd = fd;
        j.size = head.len + extra;
    }
};

fn writeRuns(j: *Journal, a: Allocator, fd: c_int, at: u64) IoError!u64 {
    var size: u64 = 0;
    for (j.runs.items) |r| {
        const line = try encodeRun(a, r);
        defer a.free(line);
        if (!writeAll(fd, line, at + size)) return error.JournalIo;
        size += line.len;
    }
    return size;
}

fn encodeRun(a: Allocator, r: Run) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "{\"run\":{\"learn\":");
    try out.appendSlice(a, if (r.learn) "true" else "false");
    try out.appendSlice(a, ",\"recall\":");
    try out.appendSlice(a, if (r.recall) |v| (if (v) "true" else "false") else "null");
    try out.appendSlice(a, ",\"compact_memory\":");
    if (r.compact_memory.len == 0) try out.appendSlice(a, "null") else try str(a, &out, r.compact_memory);
    try out.appendSlice(a, ",\"request_log\":");
    if (r.request_log.len == 0) try out.appendSlice(a, "null") else try str(a, &out, r.request_log);
    try out.appendSlice(a, ",\"slide\":");
    try out.appendSlice(a, if (r.slide) "true" else "false");
    try out.appendSlice(a, ",\"pocket\":");
    try out.appendSlice(a, if (r.pocket) "true" else "false");
    try out.print(a, ",\"next_source\":{d}", .{r.next_source});
    try out.appendSlice(a, "}}\n");
    return out.toOwnedSlice(a);
}

fn readRun(a: Allocator, line: []const u8, problem: *[]const u8) OpenError!?Run {
    const v = switch (try json.parse(a, line)) {
        .ok => |x| x,
        .err => return null,
    };
    const run = v.get("run") orelse return null;
    if (run != .object) {
        problem.* = "a run line in the notes journal is not a start manifest";
        return error.BadJournal;
    }
    const next = (run.get("next_source") orelse {
        problem.* = "a run line in the notes journal is not a start manifest";
        return error.BadJournal;
    }).int64() orelse {
        problem.* = "a run line in the notes journal is not a start manifest";
        return error.BadJournal;
    };
    return .{
        .learn = try runFlag(run, "learn", problem),
        .recall = if (run.get("recall")) |rv| (if (rv == .null) null else try runFlag(run, "recall", problem)) else null,
        .compact_memory = try runPath(run, "compact_memory", problem),
        .request_log = try runPath(run, "request_log", problem),
        .slide = try runFlag(run, "slide", problem),
        .pocket = try runFlag(run, "pocket", problem),
        .next_source = @intCast(@max(next, 1)),
    };
}

fn runFlag(obj: Value, key: []const u8, problem: *[]const u8) OpenError!bool {
    const x = obj.get(key) orelse {
        problem.* = "a run line in the notes journal is not a start manifest";
        return error.BadJournal;
    };
    if (x == .bool) return x.bool;
    problem.* = "a run line in the notes journal is not a start manifest";
    return error.BadJournal;
}

fn runPath(obj: Value, key: []const u8, problem: *[]const u8) OpenError![]const u8 {
    const x = obj.get(key) orelse {
        problem.* = "a run line in the notes journal is not a start manifest";
        return error.BadJournal;
    };
    if (x == .null) return "";
    if (x == .string) return x.string;
    problem.* = "a run line in the notes journal is not a start manifest";
    return error.BadJournal;
}

fn dupeRun(g: Allocator, r: Run) Allocator.Error!Run {
    const c = try g.dupe(u8, r.compact_memory);
    errdefer g.free(c);
    const l = try g.dupe(u8, r.request_log);
    return .{ .learn = r.learn, .recall = r.recall, .compact_memory = c, .request_log = l, .slide = r.slide, .pocket = r.pocket, .next_source = r.next_source };
}

fn freeRun(g: Allocator, r: Run) void {
    g.free(r.compact_memory);
    g.free(r.request_log);
}

fn readRecord(a: Allocator, line: []const u8, problem: *[]const u8) OpenError!Record {
    const v = switch (try json.parse(a, line)) {
        .ok => |x| x,
        .err => {
            problem.* = "a line of the notes journal is not JSON";
            return error.BadJournal;
        },
    };
    return decode(v) orelse {
        problem.* = "a line of the notes journal is not a write";
        return error.BadJournal;
    };
}

/// Whether `line` is the commit line of the write `source`.
fn isCommit(a: Allocator, line: []const u8, source: []const u8) OpenError!bool {
    const v = switch (try json.parse(a, line)) {
        .ok => |x| x,
        .err => return false,
    };
    return std.mem.eql(u8, v.strField("commit") orelse return false, source);
}

/// A write's commit line: written and synced after the write's own line, it makes the write count.
pub fn commitLine(a: Allocator, source: []const u8) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "{\"commit\":");
    try str(a, &out, source);
    try out.appendSlice(a, "}\n");
    return out.toOwnedSlice(a);
}

const Header = struct { next_source: u64, next_id: u32, version: u8 };

/// The header checked against the snapshot: its next_source and its format (1: no commit lines).
fn checkHeader(a: Allocator, line: []const u8, sha: [64]u8, problem: *[]const u8) OpenError!Header {
    const v = switch (try json.parse(a, line)) {
        .ok => |x| x,
        .err => {
            problem.* = "the notes journal's first line is not its header";
            return error.BadJournal;
        },
    };
    const f = v.strField("format") orelse "";
    const version: u8 = if (std.mem.eql(u8, f, format)) 3 else if (std.mem.eql(u8, f, format_v2)) 2 else if (std.mem.eql(u8, f, format_v1)) 1 else {
        problem.* = "the notes journal's header names another format";
        return error.BadJournal;
    };
    if (!std.mem.eql(u8, v.strField("snapshot_sha256") orelse "", &sha)) {
        problem.* = "the notes journal was written over another snapshot (its snapshot_sha256 differs)";
        return error.OtherSnapshot;
    }
    const next = if (v.get("next_source")) |n| n.int64() orelse 1 else 1;
    const next_id = if (v.get("next_id")) |n| std.math.cast(u32, n.int64() orelse return error.BadJournal) orelse return error.BadJournal else 1;
    return .{ .next_source = @intCast(@max(next, 1)), .next_id = @max(next_id, 1), .version = version };
}

/// A record as one journal line: fixed keys, the text as UTF-8 (not escaped), a newline.
pub fn encode(a: Allocator, r: Record) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.print(a, "{{\"op\":\"{s}\",\"source\":", .{@tagName(r.op)});
    try str(a, &out, r.source);
    try out.print(a, ",\"id\":{d},\"told\":{d},\"subject\":", .{ r.id, r.told });
    try str(a, &out, r.subject);
    try out.appendSlice(a, ",\"attribute\":");
    try str(a, &out, r.attribute);
    switch (r.target) {
        .none => try out.appendSlice(a, ",\"target\":null"),
        .note => |n| try out.print(a, ",\"target\":{{\"note\":{d}}}", .{n}),
        .source => |s| {
            try out.appendSlice(a, ",\"target\":{\"source\":");
            try str(a, &out, s);
            try out.append(a, '}');
        },
    }
    try out.appendSlice(a, ",\"text\":");
    try str(a, &out, r.text);
    try out.appendSlice(a, "}\n");
    return out.toOwnedSlice(a);
}

fn str(a: Allocator, out: *std.ArrayList(u8), s: []const u8) Allocator.Error!void {
    const q = try json.quote(a, s, .{ .ascii = false });
    defer a.free(q);
    try out.appendSlice(a, q);
}

/// A journal line read back, or null when it is not one encode() writes.
pub fn decode(v: Value) ?Record {
    const op = std.meta.stringToEnum(write.Op, v.strField("op") orelse return null) orelse return null;
    const told = (v.get("told") orelse return null).int64() orelse return null;
    var target: write.Target = .none;
    if (v.field("target")) |t| {
        if (t.field("note")) |n| {
            target = .{ .note = std.math.cast(u32, n.int64() orelse return null) orelse return null };
        } else target = .{ .source = t.strField("source") orelse return null };
    }
    if ((op == .add) != (target == .none)) return null;
    const source = v.strField("source") orelse return null;
    if (source.len == 0 or source.len > std.math.maxInt(u16)) return null; // a note keeps its source's length in 16 bits; none is 0
    const id = if (v.get("id")) |n| std.math.cast(u32, n.int64() orelse return null) orelse return null else 0;
    return .{ .op = op, .id = id, .source = source, .told = told, .subject = v.strField("subject") orelse return null, .attribute = v.strField("attribute") orelse return null, .target = target, .text = v.strField("text") orelse return null };
}

fn readAll(gpa: Allocator, fd: c_int, problem: *[]const u8) OpenError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var buf: [65536]u8 = undefined;
    var at: i64 = 0;
    while (true) {
        const n = libc.pread(fd, &buf, buf.len, at);
        if (n < 0) {
            problem.* = "the notes journal cannot be read";
            return error.JournalIo;
        }
        if (n == 0) break;
        try out.appendSlice(gpa, buf[0..@intCast(n)]);
        at += n;
    }
    return out.toOwnedSlice(gpa);
}

fn writeAll(fd: c_int, data: []const u8, at: u64) bool {
    var done: usize = 0;
    while (done < data.len) {
        const n = libc.pwrite(fd, data[done..].ptr, data.len - done, @intCast(at + done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

/// The reply that stops a start whose format-1 upgrade renamed the new file in but could not sync its folder.
const upgrade_unsynced = "the notes journal (older format) was rewritten as format 3 and renamed in, but its folder could not be synced: after a power cut either file may be there, each whole and with the same writes; start again";

/// An F_FULLFSYNC errno that means the filesystem does not do it: XNU's own test (kern_descrip.c, F_BARRIERFSYNC).
fn unsupported(e: libc.E) bool {
    return e == .NOTTY or e == .OPNOTSUPP or e == .INVAL; // OPNOTSUPP is ENOTSUP (45) on Darwin
}

/// Darwin: F_FULLFSYNC (fsync, then the drive's cache flushed); plain fsync only where unsupported. Elsewhere: fsync.
pub fn fullSync(fd: c_int, faults: ?*Faults) bool {
    if (builtin.os.tag.isDarwin()) {
        const e = if (faults) |f| (if (Faults.hit(f.full, &f.fulls)) f.full_errno else fcntlFull(fd)) else fcntlFull(fd);
        if (e == .SUCCESS) return true;
        if (!unsupported(e)) return false; // EIO, ENODEV, EBADF and the rest: a failed sync, no fallback
        if (faults) |f| f.fallbacks += 1;
    }
    // Linux fsync(2): "includes writing through or flushing a disk cache if present" (not tested on Linux here)
    return libc.fsync(fd) == 0;
}

fn fcntlFull(fd: c_int) libc.E {
    return libc.errno(libc.fcntl(fd, libc.F.FULLFSYNC));
}
