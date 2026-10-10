//! Rank 0's request log on the compat wire: one JSON line a request, rotated, with Python's id hashes.
const std = @import("std");
const json = @import("json.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const compat = @import("compat.zig");
const LogSettings = compat.LogSettings;
const round = compat.round;
const Outcome = compat.Outcome;

// -- the request log ------------------------------------------------------------------------------------------------

/// Rank 0's request log, one JSON line a request (no text: counts, timings, hashes of ids).
pub const RequestLog = struct {
    gpa: Allocator,
    io: std.Io,
    s: LogSettings,
    /// the chat template's role tokens the tokenizer knows (``<|user|>``, ``<|assistant|>``, ``<|observation|>``)
    roles: Roles,
    mutex: std.Io.Mutex = .init,
    n: u64 = 0,
    live: std.AutoArrayHashMapUnmanaged(u64, *Ticket) = .empty,
    /// the last ``window`` prompts, oldest first
    kept: std.ArrayList(Kept) = .empty,
    fd: ?std.posix.fd_t = null,
    written: u64 = 0,
    warned: bool = false,

    pub const hash_tokens = 4096;
    pub const Roles = struct { user: ?u32 = null, assistant: ?u32 = null, observation: ?u32 = null };
    const Kept = struct { n: u64, ids: []u32, conv: [16]u8 };

    pub const Ticket = struct {
        n: u64,
        start: f64,
        ids: []u32,
        first: ?f64 = null,
        conv: ?[16]u8 = null,
    };

    pub fn init(gpa: Allocator, io: std.Io, s: LogSettings, roles: Roles) RequestLog {
        return .{ .gpa = gpa, .io = io, .s = s, .roles = roles };
    }

    pub fn deinit(l: *RequestLog) void {
        for (l.kept.items) |k| l.gpa.free(k.ids);
        l.kept.deinit(l.gpa);
        for (l.live.values()) |t| l.free(t);
        l.live.deinit(l.gpa);
        if (l.fd) |fd| _ = std.posix.system.close(fd);
    }

    fn free(l: *RequestLog, t: *Ticket) void {
        l.gpa.free(t.ids);
        l.gpa.destroy(t);
    }

    fn unix(l: *const RequestLog) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.real.now(l.io).toNanoseconds())) / 1e9;
    }

    /// A request's prompt ids, before it runs.
    pub fn begin(l: *RequestLog, ids: []const u32) ?*Ticket {
        const t = l.gpa.create(Ticket) catch return null;
        t.* = .{ .n = 0, .start = l.unix(), .ids = l.gpa.dupe(u32, ids) catch {
            l.gpa.destroy(t);
            return null;
        } };
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        l.n += 1;
        t.n = l.n;
        l.live.put(l.gpa, t.n, t) catch {};
        return t;
    }

    /// The first streamed delta.
    pub fn first(l: *const RequestLog, t: ?*Ticket) void {
        const x = t orelse return;
        if (x.first == null) x.first = l.unix();
    }

    /// blake2b-64 of the ids as little-endian int32, keyed with the salt when set; 16 hex digits.
    pub fn hash(l: *const RequestLog, ids: []const u32) [16]u8 {
        const B = std.crypto.hash.blake2.Blake2b(64);
        var h = B.init(.{ .key = if (l.s.salt.len > 0) l.s.salt[0..@min(l.s.salt.len, 64)] else null });
        if (@import("builtin").cpu.arch.endian() == .little) h.update(std.mem.sliceAsBytes(ids)) else for (ids) |x| {
            var le: [4]u8 = undefined;
            std.mem.writeInt(u32, &le, x, .little);
            h.update(&le);
        }
        var d: [8]u8 = undefined;
        h.final(&d);
        return std.fmt.bytesToHex(d, .lower);
    }

    fn firstOf(ids: []const u32, targets: []const ?u32) ?usize {
        var best: ?usize = null;
        for (targets) |t| if (t) |x| if (std.mem.indexOfScalar(u32, ids[0..(best orelse ids.len)], x)) |i| {
            best = i;
        };
        return best;
    }

    fn conv(l: *const RequestLog, ids: []const u32) [16]u8 {
        if (firstOf(ids, &.{l.roles.assistant})) |at| return l.hash(ids[0 .. at + 1]);
        return l.hash(ids[0..@min(ids.len, hash_tokens)]);
    }

    /// How a request ended, for its line.
    pub const Result = struct {
        chat: bool,
        /// null: no result (it raised)
        finish: ?[]const u8,
        completion_tokens: ?usize,
        outcome: ?Outcome,
        /// the exception class it raised
        err: ?[]const u8 = null,
        thinking: ?bool,
        max_tokens_eff: ?i64,
    };

    /// Writes the request's line (never fails a request) and keeps its prompt for the prefix fields.
    pub fn end(l: *RequestLog, t: ?*Ticket, body: Value, r: Result) void {
        const x = t orelse return;
        var arena = std.heap.ArenaAllocator.init(l.gpa);
        defer arena.deinit();
        if (l.record(arena.allocator(), x, body, r)) |line| l.write(line) else |_| l.warn("the record could not be built");
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        _ = l.live.swapRemove(x.n);
        if (l.s.window == 0) return l.free(x);
        if (l.kept.items.len >= l.s.window) l.gpa.free(l.kept.orderedRemove(0).ids);
        l.kept.append(l.gpa, .{ .n = x.n, .ids = x.ids, .conv = x.conv orelse l.conv(x.ids) }) catch {
            return l.free(x);
        };
        l.gpa.destroy(x);
    }

    fn warn(l: *RequestLog, what: []const u8) void {
        if (l.warned) return;
        l.warned = true;
        std.debug.print("[tensorfold] request log (TF_COMPAT_REQUEST_LOG={s}): {s}; further errors not shown\n", .{ l.s.path, what });
    }

    const Prefix = struct { lcp: usize = 0, same: usize = 0, other: usize = 0 };

    /// The longest common prefix with every earlier prompt still kept or in flight, by conversation.
    fn analyse(l: *RequestLog, t: *Ticket) Prefix {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        if (t.conv == null) t.conv = l.conv(t.ids);
        var p: Prefix = .{};
        const fold = struct {
            fn one(q: *Prefix, mine: []const u32, conv_: [16]u8, other: []const u32, oconv: [16]u8) void {
                if (other.len == 0 or mine.len == 0 or other[0] != mine[0]) return;
                const d = std.mem.indexOfDiff(u32, mine, other) orelse mine.len;
                q.lcp = @max(q.lcp, d);
                if (std.mem.eql(u8, &conv_, &oconv)) q.same = @max(q.same, d) else q.other = @max(q.other, d);
            }
        }.one;
        for (l.kept.items) |k| if (k.n < t.n) fold(&p, t.ids, t.conv.?, k.ids, k.conv);
        for (l.live.values()) |o| if (o.n < t.n) {
            if (o.conv == null) o.conv = l.conv(o.ids);
            fold(&p, t.ids, t.conv.?, o.ids, o.conv.?);
        };
        return p;
    }

    fn num(a: Allocator, v: ?f64, comptime digits: u8) Allocator.Error!Value {
        _ = a;
        return if (v) |x| .{ .float = round(x, digits) } else .null;
    }

    fn record(l: *RequestLog, a: Allocator, t: *Ticket, body: Value, r: Result) ![]const u8 {
        const o = try json.newObject(a);
        const st = r.outcome;
        const prompt = t.ids.len;
        const cached: usize = if (st) |s| s.cached else 0;
        const prefill_s: ?f64 = if (st) |s| s.prefill_s else null;
        const decode_s: ?f64 = if (st) |s| s.decode_s else null;
        const done: ?usize = r.completion_tokens;
        const finish: Value = if (r.err != null) .{ .string = "error" } else if (st != null and st.?.cancelled) .{ .string = "cancelled" } else if (r.finish) |f| .{ .string = f } else .null;
        const kw = body.get("chat_template_kwargs") orelse Value.null;
        const asked = (body.field("max_tokens") orelse body.field("max_completion_tokens"));
        const now_s = l.unix();
        try o.put(a, "ts", .{ .float = round(now_s, 3) });
        try o.put(a, "start", .{ .float = round(t.start, 3) });
        try o.put(a, "n", try json.intValue(a, t.n));
        try o.put(a, "kind", .{ .string = if (r.chat) "chat" else "completion" });
        try o.put(a, "prompt", try json.intValue(a, prompt));
        try o.put(a, "cached", try json.intValue(a, cached));
        try o.put(a, "cache_src", .{ .string = if (cached > 0) "slot" else "none" });
        try o.put(a, "prefill_s", try num(a, prefill_s, 4));
        try o.put(a, "prefill_tps", if (prefill_s) |p| (if (p != 0) Value{ .float = round(@as(f64, @floatFromInt(prompt - cached)) / round(p, 4), 1) } else .null) else .null);
        try o.put(a, "queue_s", .null);
        try o.put(a, "first_s", if (t.first) |f| .{ .float = round(f - t.start, 3) } else .null);
        try o.put(a, "decode_tokens", if (done) |d| try json.intValue(a, d) else .null);
        try o.put(a, "decode_s", try num(a, decode_s, 4));
        try o.put(a, "decode_tps", if (decode_s != null and decode_s.? != 0 and done != null and done.? != 0) .{ .float = round(@as(f64, @floatFromInt(done.?)) / round(decode_s.?, 4), 2) } else .null);
        try o.put(a, "tokens_per_round", .null);
        try o.put(a, "rounds", if (st) |s| try json.intValue(a, s.rounds) else .null);
        for ([_][]const u8{ "pieces", "slot", "kv_pages", "kv_free", "marks", "multi" }) |k| try o.put(a, k, .null);
        try o.put(a, "thinking", if (r.thinking) |b| .{ .bool = b } else .null);
        try o.put(a, "effort", if (r.thinking orelse false) (kw.get("reasoning_effort") orelse .null) else .null);
        try o.put(a, "effort_asked", body.get("reasoning_effort") orelse .null);
        try o.put(a, "max_tokens", if (asked != null and asked.?.truthy()) (if (asked.?.int64()) |n| try json.intValue(a, n) else asked.?) else .null);
        try o.put(a, "max_tokens_eff", if (r.max_tokens_eff) |n| try json.intValue(a, n) else .null);
        try o.put(a, "finish", finish);
        try o.put(a, "error", if (r.err) |e| .{ .string = e } else .null);
        try o.put(a, "tools", try json.intValue(a, if (body.field("tools")) |x| (if (x == .array) x.array.len else 0) else 0));
        try o.put(a, "messages", if (r.chat) try json.intValue(a, if (body.field("messages")) |x| (if (x == .array) x.array.len else 0) else 0) else .null);
        try o.put(a, "policy", .null);
        try o.put(a, "fast_prefill", .null);
        const p = l.analyse(t);
        const ids = t.ids;
        const sys_len = firstOf(ids, &.{ l.roles.user, l.roles.assistant, l.roles.observation });
        const head = ids[0..@min(ids.len, hash_tokens)];
        try o.put(a, "head_hash", .{ .string = try a.dupe(u8, &l.hash(head)) });
        try o.put(a, "head_len", try json.intValue(a, head.len));
        try o.put(a, "sys_len", if (sys_len) |s| try json.intValue(a, s) else .null);
        try o.put(a, "sys_hash", if (sys_len != null and sys_len.? > 0) .{ .string = try a.dupe(u8, &l.hash(ids[0..sys_len.?])) } else .null);
        try o.put(a, "conv", .{ .string = try a.dupe(u8, &t.conv.?) });
        try o.put(a, "lcp", try json.intValue(a, p.lcp));
        try o.put(a, "lcp_same", try json.intValue(a, p.same));
        try o.put(a, "lcp_other", try json.intValue(a, p.other));
        const text = try json.stringify(a, .{ .object = o }, .{ .compact = true });
        return std.mem.concat(a, u8, &.{ text, "\n" });
    }

    fn write(l: *RequestLog, line: []const u8) void {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        l.writeLocked(line) catch |e| l.warn(@errorName(e));
    }

    fn open(l: *RequestLog) !void {
        if (std.fs.path.dirname(l.s.path)) |d| std.Io.Dir.cwd().createDirPath(l.io, d) catch {};
        const pathz = try l.gpa.dupeSentinel(u8, l.s.path, 0);
        defer l.gpa.free(pathz);
        const fd = try std.posix.openatZ(std.posix.AT.FDCWD, pathz, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true, .CLOEXEC = true }, 0o644);
        l.fd = fd;
        const end_at = std.posix.system.lseek(fd, 0, std.posix.SEEK.END);
        l.written = if (end_at < 0) 0 else @intCast(end_at);
    }

    fn writeLocked(l: *RequestLog, line: []const u8) !void {
        if (l.fd == null) try l.open();
        if (l.written > 0 and l.written + line.len > l.s.max_bytes) try l.rotate();
        var off: usize = 0;
        while (off < line.len) {
            const rc = std.posix.system.write(l.fd.?, line.ptr + off, line.len - off);
            if (std.posix.errno(rc) != .SUCCESS) return error.WriteFailed;
            off += @intCast(rc);
        }
        l.written += line.len;
    }

    /// ``<file>`` -> ``<file>.1`` (``.1`` -> ``.2`` ..., ``keep`` old files).
    fn rotate(l: *RequestLog) !void {
        _ = std.posix.system.close(l.fd.?);
        l.fd = null;
        const dir = std.Io.Dir.cwd();
        var buf_a: [std.fs.max_path_bytes]u8 = undefined;
        var buf_b: [std.fs.max_path_bytes]u8 = undefined;
        if (l.s.keep == 0) {
            dir.deleteFile(l.io, l.s.path) catch {};
        } else {
            var i: u32 = l.s.keep - 1;
            while (i > 0) : (i -= 1) {
                const src = try std.fmt.bufPrint(&buf_a, "{s}.{d}", .{ l.s.path, i });
                const dst = try std.fmt.bufPrint(&buf_b, "{s}.{d}", .{ l.s.path, i + 1 });
                dir.rename(src, dir, dst, l.io) catch {};
            }
            dir.rename(l.s.path, dir, try std.fmt.bufPrint(&buf_b, "{s}.1", .{l.s.path}), l.io) catch {};
        }
        try l.open();
        l.written = 0;
    }
};

test "the request log's hash is Python's blake2b(digest_size=8) of int32 ids" {
    // hashlib.blake2b(array("i", [0, 1, 2]).tobytes(), digest_size=8).hexdigest() and with key=b"salt"
    var l = RequestLog.init(std.testing.allocator, std.testing.io, .{ .path = "/dev/null" }, .{});
    defer l.deinit();
    try std.testing.expectEqualStrings("99a5826197698e6d", &l.hash(&.{ 0, 1, 2 }));
    l.s.salt = "salt";
    try std.testing.expectEqualStrings("c90ad2de02f9487a", &l.hash(&.{ 0, 1, 2 }));
}
