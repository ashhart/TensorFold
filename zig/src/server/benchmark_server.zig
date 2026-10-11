//! The benchmark's own serve child: lock, flags, environment, port, readiness and a stop that always reaps.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json.zig");
const cli = @import("cli.zig");
const stream = @import("benchmark_stream.zig");
const stop_flag = @import("benchmark_stop.zig");
const Allocator = std.mem.Allocator;
const posix = std.posix;

pub const served_name = ".tensorfold-benchmark";
const serving_prefix = "[tensorfold] serving " ++ served_name ++ " at http://127.0.0.1:";

/// What the child serves: this executable, the checkpoint directory and the benchmark's settings.
pub const Plan = struct { exe: []const u8, dir: []const u8, backend: []const u8, tokens: u32, context: ?i64, drafter_dir: ?[]const u8, serial: bool };

/// ``argv`` replaces the child's command line (tests use a stand-in); ``tail`` gets an exited child's last log line.
pub const Options = struct { startup_timeout_s: f64, grace_ms: u32 = 10_000, bench_dir: []const u8, argv: ?[]const []const u8 = null, tail: ?*[]const u8 = null };

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
    if (plan.drafter_dir) |d| try out.appendSlice(a, &.{ "--drafter", d });
    if (plan.context) |c| try out.appendSlice(a, &.{ "--context", try std.fmt.allocPrint(a, "{d}", .{c}) });
    if (plan.serial) try out.append(a, "--no-drafts");
    return out.items;
}

/// The parent's variables without the API key, request log, seed salt and upload token; no live line or update check.
pub fn environ(a: Allocator, parent: *const std.process.Environ.Map) !std.process.Environ.Map {
    var env = try parent.clone(a);
    for ([_][]const u8{ "TENSORFOLD_API_KEY", "TENSORFOLD_REQUEST_LOG", "TENSORFOLD_SEED_SALT", "TENSORFOLD_BENCHMARK_TOKEN" }) |k| _ = env.orderedRemove(k);
    try env.put("TENSORFOLD_NO_LIVE", "1");
    try env.put("TENSORFOLD_NO_UPDATE_CHECK", "1");
    try env.put("HF_HUB_OFFLINE", "1");
    return env;
}

/// The port in serve's ``serving .tensorfold-benchmark at http://127.0.0.1:PORT/v1`` line.
pub fn servingPort(log: []const u8) ?u16 {
    const at = std.mem.indexOf(u8, log, serving_prefix) orelse return null;
    const rest = log[at + serving_prefix.len ..];
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
    lock: std.Io.File,
    grace_ms: u32,
    base_url: []const u8 = "",
    reaped: bool = false,
    closed: bool = false,

    /// Spawns the child and waits until it serves the benchmark name; null with ``problem`` set, or "" when stopped.
    pub fn start(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, plan: Plan, o: Options, problem: *[]const u8) !?Managed {
        var rand: [8]u8 = undefined;
        io.random(&rand);
        const log_path = try std.fmt.allocPrint(a, "{s}/.serve-{s}.log", .{ o.bench_dir, std.fmt.bytesToHex(rand, .lower) });
        var child_env = try environ(a, env);
        const child_argv = o.argv orelse try argv(a, plan);
        const lock = lockRun(io, o.bench_dir, problem) orelse return null;
        const cwd = std.Io.Dir.cwd();
        const log = cwd.createFile(io, log_path, .{ .exclusive = true, .read = true, .permissions = .fromMode(0o600) }) catch {
            lock.close(io);
            problem.* = "Benchmark server could not start; check local checkpoint and directory permissions";
            return null;
        };
        const child = std.process.spawn(io, .{ .argv = child_argv, .environ_map = &child_env, .stdin = .ignore, .stdout = .{ .file = log }, .stderr = .{ .file = log } }) catch {
            cwd.deleteFile(io, log_path) catch {};
            log.close(io);
            lock.close(io);
            problem.* = "Could not start the benchmark server; check the TensorFold installation";
            return null;
        };
        stop_flag.guardChild(child.id.?);
        cwd.deleteFile(io, log_path) catch {};
        var m: Managed = .{ .io = io, .pid = child.id.?, .log = log, .lock = lock, .grace_ms = o.grace_ms };
        const ok = m.ready(a, o.startup_timeout_s, o.tail, problem) catch |e| {
            _ = m.stop();
            return e;
        };
        if (ok) return m;
        if (m.stop()) |cleanup| problem.* = cleanup;
        return null;
    }

    fn ready(m: *Managed, a: Allocator, timeout_s: f64, tail: ?*[]const u8, problem: *[]const u8) !bool {
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
                problem.* = "Benchmark server exited during startup; check the checkpoint and backend toolchain";
                if (tail) |t| t.* = try lastLine(a, m.logEnd(&buf));
                return false;
            }
            if (port == null) {
                port = servingPort(m.logEnd(&buf));
                if (port) |p| m.base_url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{p});
            }
            if (port != null and try probe(a, m.io, &client, m.base_url)) return true;
            m.io.sleep(.fromMilliseconds(200), .awake) catch {};
        }
        problem.* = "Benchmark server startup timed out; check the checkpoint, GPU memory and backend toolchain";
        return false;
    }

    /// The log's last ``buf.len`` bytes.
    fn logEnd(m: *Managed, buf: []u8) []const u8 {
        const size = if (m.log.stat(m.io)) |st| st.size else |_| 0;
        return buf[0 .. m.log.readPositional(m.io, &.{buf}, size -| buf.len) catch 0];
    }

    /// ``http://127.0.0.1:PORT``.
    pub fn base(m: *const Managed) []const u8 {
        return m.base_url;
    }

    /// TERM, a grace period, KILL, another; always closes the log and the lock. Null when reaped.
    pub fn stop(m: *Managed) ?[]const u8 {
        if (m.closed) return null;
        m.closed = true;
        defer {
            m.log.close(m.io);
            m.lock.close(m.io);
        }
        if (m.reaped or m.exited()) return null;
        posix.kill(m.pid, .TERM) catch {};
        if (m.waitExit()) return null;
        posix.kill(m.pid, .KILL) catch {};
        if (m.waitExit()) return null;
        return "Benchmark server did not exit after termination; cleanup is incomplete";
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
            const rc = std.c.waitpid(m.pid, &status, 0);
            if (rc >= 0 or posix.errno(rc) != .INTR) break;
        }
        m.reaped = true;
        return true;
    }
};

const wait = struct {
    extern "c" fn waitid(idtype: c_uint, id: c_uint, info: *std.c.siginfo_t, options: c_int) c_int;
    const p_pid = 1;
    const exited: c_int = if (builtin.os.tag == .linux) std.os.linux.W.EXITED else 0x4;
    const no_wait: c_int = if (builtin.os.tag == .linux) std.os.linux.W.NOWAIT else 0x20;
};

/// True once ``pid`` has exited, or is no child of this process, without reaping it.
pub fn ended(pid: posix.pid_t) bool {
    var info: std.c.siginfo_t = undefined;
    @memset(std.mem.asBytes(&info), 0);
    const rc = wait.waitid(wait.p_pid, @intCast(pid), &info, wait.exited | wait.no_wait | std.c.W.NOHANG);
    if (rc < 0) return posix.errno(rc) == .CHILD;
    const got = if (builtin.os.tag == .linux) info.fields.common.first.piduid.pid else info.pid;
    return got == pid;
}

const line_limit = 200;

/// The log's last ``tensorfold:`` or ``error:`` line, else its last visible line, cleaned and cut to 200 bytes.
pub fn lastLine(a: Allocator, log: []const u8) ![]const u8 {
    var fallback: []const u8 = "";
    var lines = std.mem.splitBackwardsAny(u8, log, "\r\n");
    while (lines.next()) |line| {
        const text = try visible(a, line);
        if (text.len == 0) continue;
        if (std.mem.startsWith(u8, text, "tensorfold:") or std.mem.startsWith(u8, text, "error:")) return text;
        if (fallback.len == 0) fallback = text;
    }
    return fallback;
}

/// ``line`` without escape sequences, control characters or broken UTF-8, trimmed and cut to 200 bytes.
fn visible(a: Allocator, line: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < line.len) {
        if (line[i] == 0x1b) {
            i = afterEscape(line, i);
            continue;
        }
        const len = std.unicode.utf8ByteSequenceLength(line[i]) catch 0;
        const cp: u21 = if (len > 0 and i + len <= line.len) std.unicode.utf8Decode(line[i .. i + len]) catch 0 else 0;
        i += if (cp == 0) 1 else len;
        if (cp == '\t') try out.append(a, ' ');
        if (cp < 0x20 or (cp >= 0x7f and cp < 0xa0)) continue;
        if (out.items.len + len > line_limit) {
            try out.appendSlice(a, "...");
            break;
        }
        try out.appendSlice(a, line[i - len .. i]);
    }
    return std.mem.trim(u8, out.items, " ");
}

/// The index past the escape sequence at ``i``: a CSI sequence through its final byte, else ESC and one more byte.
fn afterEscape(line: []const u8, i: usize) usize {
    if (i + 1 >= line.len or line[i + 1] != '[') return @min(i + 2, line.len);
    var j = i + 2;
    while (j < line.len and (line[j] < 0x40 or line[j] > 0x7e)) j += 1;
    return @min(j + 1, line.len);
}

/// Ready: ``/health`` not warming and ok, and ``/v1/models`` lists the benchmark name.
pub fn probe(a: Allocator, io: std.Io, client: *std.http.Client, base_url: []const u8) !bool {
    const health = try getObject(a, io, client, base_url, "/health") orelse return false;
    if (health.get("warming")) |w| if (w == .bool and w.bool) return false;
    const status_ok = if (health.get("status")) |s| s == .string and std.mem.eql(u8, s.string, "ok") else false;
    const ok_true = if (health.get("ok")) |o| o == .bool and o.bool else false;
    if (!status_ok and !ok_true) return false;
    const models = try getObject(a, io, client, base_url, "/v1/models") orelse return false;
    const data = models.get("data") orelse return false;
    if (data != .array) return false;
    for (data.array) |d| if (d.get("id")) |id| if (id == .string and std.mem.eql(u8, id.string, served_name)) return true;
    return false;
}

fn getObject(a: Allocator, io: std.Io, client: *std.http.Client, base_url: []const u8, route: []const u8) !?json.Value {
    const url = try std.mem.concat(a, u8, &.{ base_url, route });
    const got = stream.fetch(a, io, client, .{ .url = url, .limit = 64 << 10, .timeout_s = 1 });
    if (got.failed != .none or got.status < 200 or got.status > 299) return null;
    return switch (try json.parse(a, got.body)) {
        .ok => |v| if (v == .object) v else null,
        .err => null,
    };
}
