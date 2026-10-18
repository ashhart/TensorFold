//! The hardware allowlist per OS: CPU and GPU names, memory, cores, and the CUDA driver version.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const receipt = @import("benchmark_receipt.zig");
const Allocator = std.mem.Allocator;

pub const Gpu = receipt.Gpu;
pub const Hardware = receipt.Hardware;
pub const unavailable: Hardware = .{};

/// What the probes found: the hardware block, and on Linux the single driver version nvidia-smi reports.
pub const Probe = struct { hardware: Hardware, driver: ?[]const u8 = null };

/// The host's allowlisted facts for a backend; no hostname, serials, UUIDs or model identifier is read.
pub fn detect(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, backend: []const u8) Probe {
    if (builtin.os.tag == .macos) {
        const cpu = modelName(a, sysctlString("machdep.cpu.brand_string") orelse "");
        const memory = sysctlU64("hw.memsize");
        const out = run(a, io, &.{ "/usr/sbin/system_profiler", "SPDisplaysDataType", "-json" });
        return .{ .hardware = summarize(cpu, memory, macGpus(a, out orelse "", memory), true) };
    }
    const cpu = cpuinfo(a, readProc(a, io, "/proc/cpuinfo") orelse "");
    const memory = meminfo(readProc(a, io, "/proc/meminfo") orelse "") orelse (std.process.totalSystemMemory() catch null);
    const csv = if (gpuQuery(backend)) |argv| run(a, io, argv) orelse "" else "";
    return linux(a, cpu, memory, csv, env.get("CUDA_VISIBLE_DEVICES"));
}

/// The GPU query behind a backend's facts: nvidia-smi for cuda; none for rocm, whose GPUs and driver stay unknown.
pub fn gpuQuery(backend: []const u8) ?[]const []const u8 {
    if (!std.mem.eql(u8, backend, "cuda")) return null;
    return &.{ "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits" };
}

/// Linux facts from the CPU and memory probes and the nvidia-smi rows, empty when no query ran.
pub fn linux(a: Allocator, cpu: ?[]const u8, memory: ?u64, csv: []const u8, visible: ?[]const u8) Probe {
    const rows = nvidiaRows(a, csv, visible);
    const gpus = a.alloc(Gpu, rows.len) catch return .{ .hardware = unavailable };
    for (rows, gpus) |r, *g| g.* = .{ .name = r.name, .memory_bytes = r.memory_bytes };
    return .{ .hardware = summarize(cpu, memory, gpus, false), .driver = driver(rows) };
}

/// The evidence rule: ``detected`` needs CPU, memory and GPUs with their memory; any fact is ``partial``.
pub fn summarize(cpu: ?[]const u8, memory: ?u64, gpus: []const Gpu, mlx: bool) Hardware {
    var memory_type: []const u8 = "unknown";
    if (mlx) {
        for (gpus) |g| if (std.mem.startsWith(u8, g.name, "Apple M")) {
            memory_type = "unified";
        };
    } else if (gpus.len > 0) {
        memory_type = for (gpus) |g| {
            if (!std.mem.eql(u8, g.name, "NVIDIA GB10") and !std.mem.startsWith(u8, g.name, "NVIDIA RTX Spark")) break "dedicated";
        } else "unified";
    }
    const complete = cpu != null and memory != null and gpus.len > 0 and for (gpus) |g| {
        if (g.memory_bytes == null) break false;
    } else true;
    const evidence = cpu != null or memory != null or gpus.len > 0;
    return .{ .cpu_model = cpu, .system_memory_bytes = memory, .memory_type = memory_type, .gpus = gpus, .source = if (complete) "detected" else if (evidence) "partial" else "unavailable" };
}

/// A processor or GPU description from the allowlist, whitespace collapsed and a clock suffix dropped; else null.
pub fn modelName(a: Allocator, text: []const u8) ?[]const u8 {
    var words: std.ArrayList(u8) = .empty;
    var it = std.mem.tokenizeAny(u8, text, " \t\n\r\x0b\x0c");
    while (it.next()) |w| {
        if (words.items.len > 0) words.append(a, ' ') catch return null;
        words.appendSlice(a, w) catch return null;
    }
    const name = dropClock(words.items);
    if (name.len > 120 or std.mem.eql(u8, name, "NVIDIA NPU")) return null;
    if (appleChip(name)) return name;
    for ([_][]const u8{ "Intel", "AMD", "NVIDIA", "Tesla", "Quadro", "GeForce", "ARM", "Ampere", "Qualcomm" }) |prefix| {
        if (!std.mem.startsWith(u8, name, prefix)) continue;
        const rest = name[prefix.len..];
        if (rest.len > 110) continue;
        for (rest) |ch| {
            if (!std.ascii.isAlphanumeric(ch) and std.mem.indexOfScalar(u8, " ().+_-", ch) == null) break;
        } else return name;
    }
    return null;
}

fn appleChip(name: []const u8) bool {
    if (!std.mem.startsWith(u8, name, "Apple M")) return false;
    var i: usize = "Apple M".len;
    const digits = i;
    while (i < name.len and std.ascii.isDigit(name[i])) i += 1;
    if (i == digits) return false;
    const tail = name[i..];
    return tail.len == 0 or std.mem.eql(u8, tail, " Pro") or std.mem.eql(u8, tail, " Max") or std.mem.eql(u8, tail, " Ultra");
}

/// Drops a trailing `` @ 2.40GHz`` (any case, G or M) after collapsing whitespace.
fn dropClock(name: []const u8) []const u8 {
    if (name.len < 3 or !std.ascii.eqlIgnoreCase(name[name.len - 2 ..], "hz")) return name;
    const unit = std.ascii.toLower(name[name.len - 3]);
    if (unit != 'g' and unit != 'm') return name;
    var i = name.len - 3;
    if (i > 0 and name[i - 1] == ' ') i -= 1;
    const number_end = i;
    while (i > 0 and (std.ascii.isDigit(name[i - 1]) or name[i - 1] == '.')) i -= 1;
    if (i == number_end or i < 3 or !std.mem.eql(u8, name[i - 3 .. i], " @ ")) return name;
    return name[0 .. i - 3];
}

/// Python's ``int(v)`` for profiler fields, kept only when positive.
fn positiveInt(v: ?json.Value) ?u64 {
    const x = v orelse return null;
    const n: i64 = switch (x) {
        .int => x.int64() orelse return null,
        .string => |s| std.fmt.parseInt(i64, std.mem.trim(u8, s, " \t\n\r"), 10) catch return null,
        .float => |f| if (std.math.isFinite(f) and @abs(f) < 9.2e18) @intFromFloat(f) else return null,
        .bool => |b| @intFromBool(b),
        else => return null,
    };
    return if (n > 0) @intCast(n) else null;
}

/// GPUs from ``system_profiler SPDisplaysDataType -json``: only the model and core count leave this function.
pub fn macGpus(a: Allocator, profiler_json: []const u8, memory: ?u64) []const Gpu {
    const parsed = json.parse(a, profiler_json) catch return &.{};
    const doc = switch (parsed) {
        .ok => |v| v,
        .err => return &.{},
    };
    const displays = doc.get("SPDisplaysDataType") orelse return &.{};
    if (displays != .array) return &.{};
    var out: std.ArrayList(Gpu) = .empty;
    for (displays.array) |d| {
        if (d != .object) continue;
        const name = modelName(a, json.strOr(d.get("sppci_model"))) orelse continue;
        out.append(a, .{ .name = name, .cores = positiveInt(d.get("sppci_cores")) }) catch return &.{};
    }
    var unified = false;
    for (out.items) |g| unified = unified or std.mem.startsWith(u8, g.name, "Apple M");
    if (unified) for (out.items) |*g| {
        g.memory_bytes = memory;
    };
    return out.items;
}

pub const Row = struct { name: []const u8, memory_bytes: ?u64, driver: ?[]const u8 };

/// nvidia-smi rows of exactly name, memory MiB and driver, filtered by ``CUDA_VISIBLE_DEVICES`` indices.
pub fn nvidiaRows(a: Allocator, csv: []const u8, visible: ?[]const u8) []const Row {
    var rows: std.ArrayList(Row) = .empty;
    var lines = std.mem.splitScalar(u8, csv, '\n');
    while (lines.next()) |raw| {
        const l = std.mem.trimEnd(u8, raw, "\r");
        if (l.len == 0 or std.mem.count(u8, l, ",") != 2) continue;
        var cols = std.mem.splitScalar(u8, l, ',');
        const name = modelName(a, cols.next().?) orelse continue;
        const mib = std.fmt.parseInt(u64, std.mem.trim(u8, cols.next().?, " "), 10) catch 0;
        const ver = std.mem.trim(u8, cols.next().?, " ");
        rows.append(a, .{ .name = name, .memory_bytes = if (mib > 0 and mib < 1 << 43) mib << 20 else null, .driver = if (driverVersion(ver)) ver else null }) catch return &.{};
    }
    const v = visible orelse return rows.items;
    const sel = std.mem.trim(u8, v, " \t\n\r");
    if (sel.len == 0 or std.mem.eql(u8, sel, "-1")) return &.{};
    var kept: std.ArrayList(Row) = .empty;
    var it = std.mem.splitScalar(u8, v, ',');
    while (it.next()) |raw| {
        const index = std.mem.trim(u8, raw, " \t");
        const n = std.fmt.parseInt(usize, index, 10) catch return &.{};
        for (index) |ch| if (!std.ascii.isDigit(ch)) return &.{};
        if (n < rows.items.len) kept.append(a, rows.items[n]) catch return &.{};
    }
    return kept.items;
}

fn driverVersion(s: []const u8) bool {
    if (s.len == 0 or s.len > 80 or !std.ascii.isDigit(s[0])) return false;
    for (s[1..]) |ch| if (!std.ascii.isAlphanumeric(ch) and std.mem.indexOfScalar(u8, ".+_-", ch) == null) return false;
    return true;
}

/// The one driver version every kept row agrees on, else null.
pub fn driver(rows: []const Row) ?[]const u8 {
    var found: ?[]const u8 = null;
    for (rows) |r| if (r.driver) |d| {
        if (found) |f| if (!std.mem.eql(u8, f, d)) return null;
        found = d;
    };
    return found;
}

/// The first ``model name`` or ``Hardware`` value in /proc/cpuinfo that passes the allowlist.
pub fn cpuinfo(a: Allocator, text: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |l| {
        const colon = std.mem.indexOfScalar(u8, l, ':') orelse continue;
        const key = std.mem.trim(u8, l[0..colon], " \t");
        if (!std.mem.eql(u8, key, "model name") and !std.mem.eql(u8, key, "Hardware")) continue;
        if (modelName(a, l[colon + 1 ..])) |n| return n;
    }
    return null;
}

/// /proc/meminfo's MemTotal in bytes.
pub fn meminfo(text: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |l| {
        if (!std.mem.startsWith(u8, l, "MemTotal:")) continue;
        var parts = std.mem.tokenizeAny(u8, l, " \t");
        _ = parts.next();
        const n = std.fmt.parseInt(u64, parts.next() orelse return null, 10) catch return null;
        const unit = parts.next() orelse return null;
        return if (std.mem.eql(u8, unit, "kB") and n > 0 and n < 1 << 53) n * 1024 else null;
    }
    return null;
}

fn readProc(a: Allocator, io: std.Io, path: []const u8) ?[]const u8 {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &buf);
    return r.interface.allocRemaining(a, .limited(4 << 20)) catch null;
}

fn run(a: Allocator, io: std.Io, argv: []const []const u8) ?[]const u8 {
    const r = std.process.run(a, io, .{
        .argv = argv,
        .stdout_limit = .limited(4 << 20),
        .stderr_limit = .limited(64 << 10),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(15), .clock = .awake } },
    }) catch return null;
    return if (r.term.success()) r.stdout else null;
}

fn sysctlString(comptime name: [:0]const u8) ?[]const u8 {
    const S = struct {
        var buf: [256]u8 = undefined;
    };
    var len: usize = S.buf.len;
    if (std.c.sysctlbyname(name, &S.buf, &len, null, 0) != 0 or len == 0) return null;
    return std.mem.trimEnd(u8, S.buf[0..len], "\x00");
}

fn sysctlU64(comptime name: [:0]const u8) ?u64 {
    var v: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname(name, &v, &len, null, 0) != 0 or v == 0) return null;
    return v;
}
