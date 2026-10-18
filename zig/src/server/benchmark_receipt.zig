//! A receipt as JSON: its value, canonical bytes, private atomic save, bounded load and the indented preview.
const std = @import("std");
const json = @import("json");
const suite = @import("benchmark_suite.zig");
const check = @import("benchmark_check.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const max_bytes = 1 << 20;

pub const Model = struct {
    repo_id: ?[]const u8 = null,
    revision: ?[]const u8 = null,
    family: ?[]const u8 = null,
    config_sha256: ?[]const u8 = null,
    tokenizer_sha256: ?[]const u8 = null,
    quantization: ?[]const u8 = null,
    drafter_repo_id: ?[]const u8 = null,
    local: bool,
};

pub const Gpu = struct { name: []const u8, memory_bytes: ?u64 = null, cores: ?u64 = null };

pub const Hardware = struct {
    cpu_model: ?[]const u8 = null,
    system_memory_bytes: ?u64 = null,
    memory_type: []const u8 = "unknown",
    gpus: []const Gpu = &.{},
    source: []const u8 = "unavailable",
};

pub const Runtime = struct {
    tensorfold_version: []const u8,
    backend: []const u8,
    python_version: []const u8,
    platform: []const u8,
    driver: ?[]const u8 = null,
};

pub const Settings = struct {
    tokens: u32,
    repetitions: u32,
    temperatures: []const f64,
    serial: bool,
    context_tokens: ?i64,
    managed: bool,
    /// 1 or 2 when attested. A managed run is one rank. An attached run stays null until the server manifest names it.
    rank_count: ?u8 = null,
};

const Builder = struct {
    a: Allocator,
    o: *json.Object,

    fn init(a: Allocator) !Builder {
        return .{ .a = a, .o = try json.newObject(a) };
    }

    fn put(b: Builder, key: []const u8, v: Value) !void {
        try b.o.put(b.a, key, v);
    }

    fn str(b: Builder, key: []const u8, s: ?[]const u8) !void {
        try b.put(key, if (s) |x| .{ .string = x } else .null);
    }

    fn int(b: Builder, key: []const u8, n: anytype) !void {
        try b.put(key, if (n) |x| try json.intValue(b.a, x) else .null);
    }

    fn float(b: Builder, key: []const u8, f: ?f64) !void {
        try b.put(key, if (f) |x| .{ .float = x } else .null);
    }

    fn value(b: Builder) Value {
        return .{ .object = b.o };
    }
};

/// The receipt in its field order, then sorted at every level as it is saved and shown.
pub fn value(a: Allocator, run_id: []const u8, created_at: []const u8, model: Model, runtime: Runtime, hardware: Hardware, settings: Settings, samples: []const suite.Sample) !Value {
    const top = try Builder.init(a);
    try top.int("schema_version", @as(?u8, 1));
    const s = try Builder.init(a);
    try s.str("id", suite.id);
    try s.str("sha256", suite.sha256_hex);
    try top.put("suite", s.value());
    try top.str("run_id", run_id);
    try top.str("created_at", created_at);
    const m = try Builder.init(a);
    inline for (.{ "repo_id", "revision", "family", "config_sha256", "tokenizer_sha256", "quantization", "drafter_repo_id" }) |k| try m.str(k, @field(model, k));
    try m.put("local", .{ .bool = model.local });
    try top.put("model", m.value());
    const r = try Builder.init(a);
    try r.str("tensorfold_version", runtime.tensorfold_version);
    try r.str("backend", runtime.backend);
    try r.str("python_version", runtime.python_version);
    try r.str("platform", runtime.platform);
    const deps = try Builder.init(a);
    if (runtime.driver) |d| try deps.str("driver", d);
    try r.put("dependencies", deps.value());
    try top.put("runtime", r.value());
    const h = try Builder.init(a);
    try h.str("cpu_model", hardware.cpu_model);
    try h.int("system_memory_bytes", hardware.system_memory_bytes);
    try h.str("memory_type", hardware.memory_type);
    const gpus = try a.alloc(Value, hardware.gpus.len);
    for (hardware.gpus, gpus) |g, *slot| {
        const gb = try Builder.init(a);
        try gb.str("name", g.name);
        try gb.int("memory_bytes", g.memory_bytes);
        try gb.int("cores", g.cores);
        slot.* = gb.value();
    }
    try h.put("gpus", .{ .array = gpus });
    try h.int("gpu_count", if (gpus.len > 0) @as(?usize, gpus.len) else null);
    try h.str("source", hardware.source);
    try top.put("hardware", h.value());
    const st = try Builder.init(a);
    try st.int("tokens", @as(?u32, settings.tokens));
    try st.int("repetitions", @as(?u32, settings.repetitions));
    const temps = try a.alloc(Value, settings.temperatures.len);
    for (settings.temperatures, temps) |t, *slot| slot.* = .{ .float = t };
    try st.put("temperatures", .{ .array = temps });
    try st.put("serial", .{ .bool = settings.serial });
    try st.int("context_tokens", settings.context_tokens);
    try st.put("managed", .{ .bool = settings.managed });
    try st.int("rank_count", settings.rank_count orelse if (settings.managed) @as(?u8, 1) else null);
    try top.put("settings", st.value());
    const rows = try a.alloc(Value, samples.len);
    for (samples, rows) |x, *slot| slot.* = try sampleValue(a, x);
    try top.put("samples", .{ .array = rows });
    const v = top.value();
    suite.sortKeys(v);
    return v;
}

fn sampleValue(a: Allocator, x: suite.Sample) !Value {
    const b = try Builder.init(a);
    try b.str("fixture_id", x.fixture_id);
    try b.float("temperature", x.temperature);
    try b.int("repeat", @as(?i32, x.repeat));
    try b.int("seed", @as(?u32, x.seed));
    try b.str("status", @tagName(x.status));
    try b.str("finish_reason", x.finish_reason);
    try b.int("prompt_tokens", x.prompt_tokens);
    try b.int("completion_tokens", x.completion_tokens);
    try b.int("cached_tokens", x.cached_tokens);
    inline for (.{ "delivery_seconds", "delivery_tps", "ttft_seconds", "end_to_end_seconds", "server_decode_tps", "server_decode_seconds", "prefill_seconds" }) |k| try b.float(k, @field(x, k));
    try b.str("token_sha", x.token_sha);
    try b.str("error_code", if (x.error_code) |c| @tagName(c) else null);
    return b.value();
}

/// The canonical bytes: sorted keys, compact separators, ASCII escapes, no newline.
pub fn canonical(a: Allocator, v: Value) ![]u8 {
    suite.sortKeys(v);
    return json.stringify(a, v, .{ .compact = true, .ascii = true });
}

/// Checks then writes ``canonical`` plus a newline through a 0600 temp file renamed over ``path``.
pub fn save(a: Allocator, io: std.Io, v: Value, path: []const u8, now_ms: i64, problem: *[]const u8) !bool {
    if (!check.receipt(v, .save, now_ms, problem)) return false;
    const bytes = try std.mem.concat(a, u8, &.{ try canonical(a, v), "\n" });
    if (bytes.len > max_bytes) {
        problem.* = "Benchmark receipt exceeds 1 MiB";
        return false;
    }
    const dir = std.fs.path.dirname(path) orelse ".";
    const cwd = std.Io.Dir.cwd();
    cwd.createDirPath(io, dir) catch {
        problem.* = try std.fmt.allocPrint(a, "Could not create the receipt directory {s}", .{dir});
        return false;
    };
    var name: [16]u8 = undefined;
    io.random(&name);
    const temp = try std.fmt.allocPrint(a, "{s}/.benchmark-{s}", .{ dir, std.fmt.bytesToHex(name[0..8], .lower) });
    const failed = try std.fmt.allocPrint(a, "Could not save the benchmark receipt to {s}", .{path});
    var file = cwd.createFile(io, temp, .{ .exclusive = true, .permissions = .fromMode(0o600) }) catch {
        problem.* = failed;
        return false;
    };
    const written = if (file.writeStreamingAll(io, bytes)) true else |_| false;
    file.close(io);
    if (!written or (if (cwd.rename(temp, cwd, path, io)) false else |_| true)) {
        cwd.deleteFile(io, temp) catch {};
        problem.* = failed;
        return false;
    }
    return true;
}

/// What ``save`` needs, checked before a run: a file name, not a directory; its directory is made; a file fits there.
pub fn writable(a: Allocator, io: std.Io, path: []const u8, problem: *[]const u8) !bool {
    const cwd = std.Io.Dir.cwd();
    problem.* = try std.fmt.allocPrint(a, "Could not save the benchmark receipt to {s}", .{path});
    const base = std.fs.path.basename(path);
    if (path.len == 0 or std.fs.path.isSep(path[path.len - 1]) or std.mem.eql(u8, base, ".") or std.mem.eql(u8, base, "..")) return false;
    const is_dir = if (cwd.statFile(io, path, .{})) |st| st.kind == .directory else |_| false;
    if (is_dir) return false;
    const dir = std.fs.path.dirname(path) orelse ".";
    cwd.createDirPath(io, dir) catch {
        problem.* = try std.fmt.allocPrint(a, "Could not create the receipt directory {s}", .{dir});
        return false;
    };
    var name: [8]u8 = undefined;
    io.random(&name);
    const temp = try std.fmt.allocPrint(a, "{s}/.benchmark-{s}", .{ dir, std.fmt.bytesToHex(name, .lower) });
    const file = cwd.createFile(io, temp, .{ .exclusive = true, .permissions = .fromMode(0o600) }) catch return false;
    file.close(io);
    cwd.deleteFile(io, temp) catch {};
    problem.* = "";
    return true;
}

/// A saved receipt, refused past 1 MiB, with duplicate keys, non-finite numbers or a broken rule.
pub fn load(a: Allocator, io: std.Io, path: []const u8, now_ms: i64, problem: *[]const u8) !?Value {
    const cwd = std.Io.Dir.cwd();
    const cannot = try std.fmt.allocPrint(a, "Could not read the benchmark receipt {s}", .{path});
    const st = cwd.statFile(io, path, .{}) catch {
        problem.* = cannot;
        return null;
    };
    if (st.size > max_bytes) {
        problem.* = "Benchmark receipt exceeds 1 MiB";
        return null;
    }
    const bytes = cwd.readFileAlloc(io, path, a, .limited(max_bytes + 1)) catch |e| {
        problem.* = if (e == error.StreamTooLong) "Benchmark receipt exceeds 1 MiB" else cannot;
        return null;
    };
    const v = switch (try json.parse(a, bytes)) {
        .ok => |x| x,
        .err => {
            problem.* = "Invalid benchmark JSON";
            return null;
        },
    };
    if (std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{ .duplicate_field_behavior = .@"error" })) |_| {} else |e| if (e == error.DuplicateField) {
        problem.* = "Duplicate JSON field";
        return null;
    }
    if (!finite(v)) {
        problem.* = "Nonfinite JSON number";
        return null;
    }
    if (!check.receipt(v, .save, now_ms, problem)) return null;
    return v;
}

fn finite(v: Value) bool {
    return switch (v) {
        .float => |f| std.math.isFinite(f),
        .array => |items| for (items) |item| {
            if (!finite(item)) break false;
        } else true,
        .object => |o| for (o.values()) |item| {
            if (!finite(item)) break false;
        } else true,
        else => true,
    };
}

/// Python's ``json.dumps(v, indent=2)``: one item per line, two spaces per level.
pub fn preview(w: *std.Io.Writer, v: Value) std.Io.Writer.Error!void {
    try indented(w, v, 0);
    try w.writeByte('\n');
}

fn indented(w: *std.Io.Writer, v: Value, depth: usize) std.Io.Writer.Error!void {
    switch (v) {
        .array => |items| {
            if (items.len == 0) return w.writeAll("[]");
            try w.writeByte('[');
            for (items, 0..) |item, i| {
                try w.writeAll(if (i > 0) ",\n" else "\n");
                try w.splatByteAll(' ', 2 * (depth + 1));
                try indented(w, item, depth + 1);
            }
            try w.writeByte('\n');
            try w.splatByteAll(' ', 2 * depth);
            try w.writeByte(']');
        },
        .object => |o| {
            if (o.count() == 0) return w.writeAll("{}");
            try w.writeByte('{');
            for (o.keys(), o.values(), 0..) |k, item, i| {
                try w.writeAll(if (i > 0) ",\n" else "\n");
                try w.splatByteAll(' ', 2 * (depth + 1));
                try json.writeString(w, k, .{});
                try w.writeAll(": ");
                try indented(w, item, depth + 1);
            }
            try w.writeByte('\n');
            try w.splatByteAll(' ', 2 * depth);
            try w.writeByte('}');
        },
        else => try json.write(w, v, .{}),
    }
}

/// A random lowercase UUID v4, as ``str(uuid.uuid4())``.
pub fn runId(io: std.Io) [36]u8 {
    var b: [16]u8 = undefined;
    io.random(&b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    const hex = std.fmt.bytesToHex(b, .lower);
    var out: [36]u8 = undefined;
    _ = std.fmt.bufPrint(&out, "{s}-{s}-{s}-{s}-{s}", .{ hex[0..8], hex[8..12], hex[12..16], hex[16..20], hex[20..32] }) catch unreachable;
    return out;
}

/// UTC wall time as ``datetime.isoformat()`` with Z: microseconds only when not zero.
pub fn createdAt(ns: i96, buf: *[32]u8) []const u8 {
    const n: i64 = @intCast(ns);
    const secs: u64 = @intCast(@max(0, @divFloor(n, std.time.ns_per_s)));
    const micros: u64 = @intCast(@divFloor(@mod(n, std.time.ns_per_s), std.time.ns_per_us));
    const es: std.time.epoch.EpochSeconds = .{ .secs = secs };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    const head = std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}", .{
        yd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute(),
    }) catch unreachable;
    const tail = if (micros == 0) std.fmt.bufPrint(buf[head.len..], "Z", .{}) else std.fmt.bufPrint(buf[head.len..], ".{d:0>6}Z", .{micros});
    return buf[0 .. head.len + (tail catch unreachable).len];
}
