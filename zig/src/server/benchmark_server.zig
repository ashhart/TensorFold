//! A command's own serve child, the benchmark's lock, flags and name by default: port, readiness and a stop that reaps.
const std = @import("std");
const platform = @import("sys");
const builtin = @import("builtin");
const json = @import("json");
const cli = @import("cli.zig");
const stream = @import("benchmark_stream.zig");
const sse = @import("benchmark_sse.zig");
const stop_flag = @import("benchmark_stop.zig");
const Sink = @import("benchmark_log.zig").Sink;
const Allocator = std.mem.Allocator;
const posix = std.posix;

pub const served_name = ".tensorfold-benchmark";

/// What the child serves: this executable, the checkpoint directory and the benchmark's settings.
pub const Plan = struct { exe: []const u8, dir: []const u8, backend: []const u8, tokens: u32, context: ?i64, drafter_dir: ?[]const u8, serial: bool, kv_cache: []const u8 = "bf16" };

/// What a failed start or stop says: the benchmark's words unless a command gives its own.
pub const Words = struct {
    log: []const u8 = "Benchmark server could not start; check local checkpoint and directory permissions",
    spawn: []const u8 = "Could not start the benchmark server; check the TensorFold installation",
    exited: []const u8 = "Benchmark server exited during startup; check the checkpoint and backend toolchain",
    timed_out: []const u8 = "Benchmark server startup timed out after",
    stuck: []const u8 = "Benchmark server did not exit after termination; cleanup is incomplete",
};

/// ``argv`` replaces the child's command line (tests use a stand-in); ``tail`` gets an exited child's last log lines.
pub const Options = struct {
    startup_timeout_s: f64,
    grace_ms: u32 = 10_000,
    bench_dir: []const u8,
    argv: ?[]const []const u8 = null,
    tail: ?*[]const []const u8 = null,
    /// The name the serving line and ``/v1/models`` must show.
    served: []const u8 = served_name,
    /// Hold ``bench_dir/.lock`` while the child runs; without it ``bench_dir`` only holds the starting child's log.
    lock: bool = true,
    /// The child in a process group of its own, so a terminal's Ctrl-C reaches only this process.
    own_group: bool = false,
    /// The probes' bearer; normally also the child's only API key (none for a benchmark).
    key: ?[]const u8 = null,
    /// The complete caller credential list for the child environment when it differs from ``key``.
    environment_key: ?[]const u8 = null,
    /// All configured credential forms to remove before a log line is cleaned or truncated.
    secrets: []const []const u8 = &.{},
    words: Words = .{},
};

/// The startup timeout without --startup-timeout: 600 s, plus 6 s per GB of weights past 20 GB, at most 3600 s.
pub fn startupTimeout(weight_bytes: u64) f64 {
    const gb = @as(f64, @floatFromInt(weight_bytes)) / 1e9;
    return @min(3600, @ceil(600 + 6 * @max(0, gb - 20)));
}

/// This build's serve backend: ``--backend``'s native values without ``auto`` (mlx on macOS, cuda or rocm on Linux).
pub fn buildBackend() []const u8 {
    for (cli.flags) |f| if (std.mem.eql(u8, f.name, "--backend")) for (f.native_values.?) |v| {
        if (!std.mem.eql(u8, v, "auto")) return v;
    };
    return "auto";
}

/// The child's command line: loopback, a free port, one request at a time, no prompt reuse, no snapshots.
pub fn argv(a: Allocator, plan: Plan) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    try out.appendSlice(a, &.{ plan.exe, "serve", plan.dir, "--host", "127.0.0.1", "--port", "0", "--name", served_name, "--backend", plan.backend });
    try out.appendSlice(a, &.{ "--snapshot-dir", "none", "--max-snapshots", "0", "--prompt-cache-gib", "0", "--parallel", "1", "--no-update-check" });
    try out.appendSlice(a, &.{ "--max-tokens", try std.fmt.allocPrint(a, "{d}", .{plan.tokens}) });
    try out.appendSlice(a, &.{ "--drafter", plan.drafter_dir orelse "none" });
    if (plan.context) |c| try out.appendSlice(a, &.{ "--context", try std.fmt.allocPrint(a, "{d}", .{c}) });
    try out.appendSlice(a, &.{ "--kv-cache", plan.kv_cache });
    if (plan.serial) try out.append(a, "--no-drafts");
    return out.items;
}

/// Parent variables minus the API key, request log, journal, seed salt and upload token; no live line or update check.
pub fn environ(a: Allocator, parent: *const std.process.Environ.Map) !std.process.Environ.Map {
    var env = try parent.clone(a);
    for ([_][]const u8{ "TENSORFOLD_API_KEY", "TENSORFOLD_REQUEST_LOG", "TENSORFOLD_REQUEST_JOURNAL", "TENSORFOLD_SEED_SALT", "TENSORFOLD_BENCHMARK_TOKEN" }) |k| _ = env.orderedRemove(k);
    try env.put("TENSORFOLD_NO_LIVE", "1");
    try env.put("TENSORFOLD_NO_UPDATE_CHECK", "1");
    try env.put("HF_HUB_OFFLINE", "1");
    return env;
}

/// The port in serve's first ``serving SERVED at http://127.0.0.1:PORT/v1`` line.
pub fn servingPort(log: []const u8, served: []const u8) ?u16 {
    const head = "[tensorfold] serving ";
    const tail = " at http://127.0.0.1:";
    var from: usize = 0;
    const rest = while (std.mem.indexOfPos(u8, log, from, head)) |at| : (from = at + 1) {
        const after = log[at + head.len ..];
        if (std.mem.startsWith(u8, after, served) and std.mem.startsWith(u8, after[served.len..], tail)) break after[served.len + tail.len ..];
    } else return null;
    var n: usize = 0;
    while (n < rest.len and std.ascii.isDigit(rest[n])) n += 1;
    if (n == 0 or !std.mem.startsWith(u8, rest[n..], "/v1")) return null;
    return std.fmt.parseInt(u16, rest[0..n], 10) catch null;
}

/// Opens ``path`` without following a symlink, creating it 0600 when missing; losing that race opens the winner's file.
fn openLock(io: std.Io, path: []const u8) !std.Io.File {
    const cwd = std.Io.Dir.cwd();
    const existing: std.Io.Dir.OpenFileOptions = .{ .mode = .read_write, .follow_symlinks = false, .allow_directory = false };
    return cwd.openFile(io, path, existing) catch |e| switch (e) {
        error.FileNotFound => cwd.createFile(io, path, .{ .exclusive = true, .truncate = false, .permissions = .fromMode(0o600) }) catch |c| switch (c) {
            error.PathAlreadyExists => cwd.openFile(io, path, existing),
            else => c,
        },
        else => e,
    };
}

/// Holds ``bench_dir/.lock``, a regular file, for this run; the OS releases it if the benchmark dies.
pub fn lockRun(io: std.Io, bench_dir: []const u8, problem: *[]const u8) ?std.Io.File {
    const cwd = std.Io.Dir.cwd();
    const cannot = "Could not create the benchmark lock; check the TensorFold directory permissions";
    _ = cwd.createDirPathStatus(io, bench_dir, .fromMode(0o700)) catch {
        problem.* = cannot;
        return null;
    };
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/.lock", .{bench_dir}) catch {
        problem.* = cannot;
        return null;
    };
    const file = openLock(io, path) catch {
        problem.* = cannot;
        return null;
    };
    const regular = if (file.stat(io)) |st| st.kind == .file else |_| false;
    if (regular and (file.tryLock(io, .exclusive) catch false)) return file;
    file.close(io);
    problem.* = if (regular) "Another TensorFold benchmark is running; wait for it to finish" else cannot;
    return null;
}

pub const Managed = struct {
    io: std.Io,
    pid: posix.pid_t,
    log: std.Io.File,
    lock: ?std.Io.File,
    grace_ms: u32,
    served: []const u8 = served_name,
    stuck: []const u8 = (Words{}).stuck,
    key: ?[]const u8 = null,
    secrets: []const []const u8 = &.{},
    /// The child's output, redacted before it is written to ``log``.
    sink: ?*Sink = null,
    base_url: []const u8 = "",
    reaped: bool = false,
    closed: bool = false,

    /// Spawns the benchmark's child and waits until it serves its name; null with ``problem`` set, or "" when stopped.
    pub fn start(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, plan: Plan, o: Options, problem: *[]const u8) !?Managed {
        return spawn(a, io, env, o.argv orelse try argv(a, plan), o, problem);
    }

    /// Spawns ``child_argv`` and waits until it serves ``o.served``; null with ``problem`` set, or "" when stopped.
    pub fn spawn(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, child_argv: []const []const u8, o: Options, problem: *[]const u8) !?Managed {
        var rand: [8]u8 = undefined;
        io.random(&rand);
        const log_path = try std.fmt.allocPrint(a, "{s}/.serve-{s}.log", .{ o.bench_dir, std.fmt.bytesToHex(rand, .lower) });
        var child_env = try environ(a, env);
        if (o.environment_key orelse o.key) |k| try child_env.put("TENSORFOLD_API_KEY", k);
        const lock = if (o.lock) lockRun(io, o.bench_dir, problem) orelse return null else null;
        const cwd = std.Io.Dir.cwd();
        const log = cwd.createFile(io, log_path, .{ .exclusive = true, .read = true, .permissions = .fromMode(0o600) }) catch {
            if (lock) |l| l.close(io);
            problem.* = o.words.log;
            return null;
        };
        const group: ?posix.pid_t = if (o.own_group) 0 else null;
        // the child writes into a pipe: keys are replaced before any byte reaches the log file
        const sink: ?*Sink = Sink.open(std.heap.page_allocator, io, log, try sinkSecrets(a, o), o.served) catch {
            cwd.deleteFile(io, log_path) catch {};
            log.close(io);
            if (lock) |l| l.close(io);
            problem.* = o.words.spawn;
            return null;
        };
        const out: std.process.SpawnOptions.StdIo = .{ .file = sink.?.child };
        const child = std.process.spawn(io, .{ .argv = child_argv, .environ_map = &child_env, .stdin = .ignore, .stdout = out, .stderr = out, .pgid = group }) catch {
            if (sink) |k| k.close(std.heap.page_allocator);
            cwd.deleteFile(io, log_path) catch {};
            log.close(io);
            if (lock) |l| l.close(io);
            problem.* = o.words.spawn;
            return null;
        };
        stop_flag.guardChild(child.id.?);
        cwd.deleteFile(io, log_path) catch {};
        var m: Managed = .{ .io = io, .pid = child.id.?, .log = log, .lock = lock, .grace_ms = o.grace_ms, .served = o.served, .stuck = o.words.stuck, .key = o.key, .secrets = o.secrets, .sink = sink };
        if (sink) |k| k.started() catch {
            _ = m.stop();
            problem.* = o.words.spawn;
            return null;
        };
        const ok = m.ready(a, o.startup_timeout_s, o.tail, o.words, problem) catch |e| {
            _ = m.stop();
            return e;
        };
        if (ok) return m;
        if (m.stop()) |cleanup| problem.* = cleanup;
        return null;
    }

    fn ready(m: *Managed, a: Allocator, timeout_s: f64, tail: ?*[]const []const u8, words: Words, problem: *[]const u8) !bool {
        var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = m.io };
        client.connection_pool.free_size = 0;
        defer client.deinit();
        const deadline = stream.deadlineAfter(m.io, timeout_s);
        var port: ?u16 = null;
        var buf: [64 << 10]u8 = undefined;
        while (stream.now(m.io) < deadline) {
            if (stop_flag.requested() != null) {
                problem.* = "";
                return false;
            }
            if (m.exited()) {
                problem.* = words.exited;
                if (m.sink) |k| k.settle(1000);
                if (m.sink) |k| k.cutHeld(1000); // a descendant may hold the pipe: the last line is written now
                if (tail) |t| t.* = try lastLinesRedacted(a, m.logEnd(&buf), m.secrets);
                return false;
            }
            if (port == null) {
                // the sink reads it from the raw line: a key may have changed the file's copy
                port = if (m.sink) |k| k.servingPort() else servingPort(m.logEnd(&buf), m.served);
                if (port) |p| m.base_url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{p});
            }
            if (port != null and try probe(a, m.io, &client, m.base_url, m.served, m.key)) return true;
            m.io.sleep(.fromMilliseconds(200), .awake) catch {};
        }
        problem.* = try std.fmt.allocPrint(a, "{s} {d} s (--startup-timeout, at most 3600); check the checkpoint, GPU memory and backend toolchain", .{ words.timed_out, timeout_s });
        return false;
    }

    /// The log's last ``buf.len`` bytes, excluding a first line cut in the middle.
    fn logEnd(m: *Managed, buf: []u8) []const u8 {
        const size = if (m.log.stat(m.io)) |st| st.size else |_| 0;
        var bytes = buf[0 .. m.log.readPositional(m.io, &.{buf}, size -| buf.len) catch 0];
        if (size > buf.len) {
            const cut = std.mem.indexOfAny(u8, bytes, "\r\n") orelse return bytes[bytes.len..];
            var next = cut + 1;
            if (bytes[cut] == '\r' and next < bytes.len and bytes[next] == '\n') next += 1;
            bytes = bytes[next..];
        }
        return bytes;
    }

    /// ``http://127.0.0.1:PORT``.
    pub fn base(m: *const Managed) []const u8 {
        return m.base_url;
    }

    /// The child's last log lines once it has exited on its own, else null.
    pub fn exitedTail(m: *Managed, a: Allocator) !?[]const []const u8 {
        if (m.closed or !m.exited()) return null;
        if (m.sink) |k| k.settle(1000);
        if (m.sink) |k| k.cutHeld(1000);
        var buf: [64 << 10]u8 = undefined;
        return try lastLinesRedacted(a, m.logEnd(&buf), m.secrets);
    }

    /// TERM, a grace period, KILL, another; always closes the log and the lock. Null when reaped.
    pub fn stop(m: *Managed) ?[]const u8 {
        if (m.closed) return null;
        m.closed = true;
        defer {
            if (m.sink) |k| k.close(std.heap.page_allocator);
            m.sink = null;
            m.log.close(m.io);
            if (m.lock) |l| l.close(m.io);
        }
        if (m.reaped or m.exited()) return null;
        posix.kill(m.pid, .TERM) catch {};
        if (m.waitExit()) return null;
        posix.kill(m.pid, .KILL) catch {};
        if (m.waitExit()) return null;
        return m.stuck;
    }

    fn waitExit(m: *Managed) bool {
        var waited: u32 = 0;
        while (waited <= m.grace_ms) : (waited += 50) {
            if (m.exited()) return true;
            m.io.sleep(.fromMilliseconds(50), .awake) catch {};
        }
        return m.exited();
    }

    /// The child is released from a second Ctrl-C while its exited pid is still reserved, and only then reaped.
    fn exited(m: *Managed) bool {
        if (m.reaped) return true;
        if (!ended(m.pid)) return false;
        stop_flag.releaseChild(m.pid);
        var status: c_int = 0;
        while (true) {
            const rc = platform.c.waitpid(m.pid, &status, 0);
            if (rc >= 0 or posix.errno(rc) != .INTR) break;
        }
        m.reaped = true;
        return true;
    }
};

const wait = struct {
    extern "c" fn waitid(idtype: c_uint, id: c_uint, info: *platform.c.siginfo_t, options: c_int) c_int;
    const p_pid = 1;
    const exited: c_int = if (builtin.os.tag == .linux) std.os.linux.W.EXITED else 0x4;
    const no_wait: c_int = if (builtin.os.tag == .linux) std.os.linux.W.NOWAIT else 0x20;
};

/// True once ``pid`` has exited, or is no child of this process, without reaping it.
pub fn ended(pid: posix.pid_t) bool {
    var info: platform.c.siginfo_t = undefined;
    @memset(std.mem.asBytes(&info), 0);
    const rc = wait.waitid(wait.p_pid, @intCast(pid), &info, wait.exited | wait.no_wait | platform.c.W.NOHANG);
    if (rc < 0) return posix.errno(rc) == .CHILD;
    const got = if (builtin.os.tag == .linux) info.fields.common.first.piduid.pid else info.pid;
    return got == pid;
}

/// The forms the log sink replaces: every configured credential, and the probes' key when it is not among them.
fn sinkSecrets(a: Allocator, o: Options) ![]const []const u8 {
    const key = o.key orelse return o.secrets;
    for (o.secrets) |secret| if (std.mem.eql(u8, secret, key)) return o.secrets;
    return std.mem.concat(a, []const u8, &.{ o.secrets, &.{key} });
}

/// Up to three last ``tensorfold:`` or ``error:`` lines in order (an earlier variable hint first), else the last line.
pub fn lastLines(a: Allocator, log: []const u8) ![]const []const u8 {
    return lastLinesRedacted(a, log, &.{});
}

/// Select managed log lines after sanitization, redaction, limiting and final containment.
pub fn lastLinesRedacted(a: Allocator, log: []const u8, secrets: []const []const u8) ![]const []const u8 {
    var picked: std.ArrayList([]const u8) = .empty; // newest first
    var fallback: []const u8 = "";
    var hint: ?[]const u8 = null;
    var lines = std.mem.splitBackwardsAny(u8, log, "\r\n");
    while (lines.next()) |line| {
        const text = (try sse.cleanContained(a, line, secrets, sse.line_limit)) orelse continue;
        if (text.len == 0 or leakNoise(text)) continue;
        if (fallback.len == 0) fallback = text;
        if ((std.mem.startsWith(u8, text, "tensorfold:") or std.mem.startsWith(u8, text, "error:")) and picked.items.len < 3) {
            try picked.append(a, text);
        } else if (hint == null and picked.items.len > 0 and namesVariable(text)) hint = text;
    }
    if (picked.items.len == 0) return if (fallback.len > 0) a.dupe([]const u8, &.{fallback}) else &.{};
    const named = for (picked.items) |p| {
        if (namesVariable(p)) break true;
    } else false;
    if (hint != null and !named) {
        if (picked.items.len == 3) _ = picked.pop();
        try picked.append(a, hint.?);
    }
    std.mem.reverse([]const u8, picked.items);
    return picked.items;
}

/// A line of an allocator's leak report: ``error(gpa): ... leaked`` or ``(empty stack trace)``.
fn leakNoise(text: []const u8) bool {
    if (std.mem.eql(u8, text, "(empty stack trace)")) return true;
    return std.mem.startsWith(u8, text, "error(") and std.mem.indexOf(u8, text, "leaked") != null;
}

/// A line that starts with a letter and names an environment variable, such as ``GLM_LOAD_LIMIT_GB``.
fn namesVariable(text: []const u8) bool {
    if (text.len == 0 or !std.ascii.isAlphabetic(text[0])) return false;
    var words = std.mem.tokenizeAny(u8, text, " \t=,;:()[]'\"`.");
    while (words.next()) |w| {
        if (w.len < 4 or !std.ascii.isUpper(w[0]) or std.mem.indexOfScalar(u8, w, '_') == null) continue;
        for (w) |ch| {
            if (!std.ascii.isUpper(ch) and !std.ascii.isDigit(ch) and ch != '_') break;
        } else return true;
    }
    return false;
}

/// Ready: ``/health`` not warming and ok, and ``/v1/models`` lists ``served``; ``key`` goes as both requests' bearer.
pub fn probe(a: Allocator, io: std.Io, client: *std.http.Client, base_url: []const u8, served: []const u8, key: ?[]const u8) !bool {
    const health = try getObject(a, io, client, base_url, "/health", key) orelse return false;
    if (health.get("warming")) |w| if (w == .bool and w.bool) return false;
    const status_ok = if (health.get("status")) |s| s == .string and std.mem.eql(u8, s.string, "ok") else false;
    const ok_true = if (health.get("ok")) |o| o == .bool and o.bool else false;
    if (!status_ok and !ok_true) return false;
    const models = try getObject(a, io, client, base_url, "/v1/models", key) orelse return false;
    const data = models.get("data") orelse return false;
    if (data != .array) return false;
    for (data.array) |d| if (d.get("id")) |id| if (id == .string and std.mem.eql(u8, id.string, served)) return true;
    return false;
}

fn getObject(a: Allocator, io: std.Io, client: *std.http.Client, base_url: []const u8, route: []const u8, key: ?[]const u8) !?json.Value {
    const url = try std.mem.concat(a, u8, &.{ base_url, route });
    const got = stream.fetch(a, io, client, .{ .url = url, .limit = 64 << 10, .timeout_s = 1, .token = key });
    if (got.failed != .none or got.status < 200 or got.status > 299) return null;
    return switch (try json.parse(a, got.body)) {
        .ok => |v| if (v == .object) v else null,
        .err => null,
    };
}
