//! The served engine's draft pricing: a stored calibration table picked by engine shape (TF_DSV41_CALIB), or defaults.
const std = @import("std");
const Allocator = std.mem.Allocator;
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");
const joint = @import("joint.zig");
const measure_mod = @import("calib_measure.zig");
const Costs = costs_mod.Costs;

pub const Mode = enum { cached, default, real, measure, zig };

/// The engine the table must have been measured on (slots, world, DSpark; context preferred).
pub const Shape = struct { slots: u32, world: u32, dspark: bool, context: ?u64 = null };

/// `measure`: the engine must measure (defaults until it has); `measured`: this boot's own measurement.
pub const Source = enum { default, cached, measure, measured };

/// Whose measurement an entry is: the reference engine's (no "engine" key) or Zig's (`"engine": "zig"`).
pub const Engine = enum { python, zig };

pub const Loaded = struct {
    costs: Costs,
    source: Source,
    /// the entry read (gpa-owned; null for the defaults)
    path: ?[]u8 = null,

    pub fn deinit(l: *Loaded, gpa: Allocator) void {
        l.costs.deinit(gpa);
        if (l.path) |p| gpa.free(p);
    }
};

/// What the environment asks for (strings as read; null: unset).
pub const Knobs = struct {
    mode: ?[]const u8 = null,
    dir: ?[]const u8 = null,
    file: ?[]const u8 = null,
    home: ?[]const u8 = null,

    pub fn fromEnv() Knobs {
        return .{ .mode = env("TF_DSV41_CALIB"), .dir = env("TF_DSV41_CALIB_DIR"), .file = env("TF_DSV41_CALIB_FILE"), .home = env("HOME") };
    }
};

pub fn mode(raw: ?[]const u8) !Mode {
    const v = trimmed(raw) orelse return .cached;
    var buf: [16]u8 = undefined;
    if (v.len > buf.len) return error.CalibMode;
    return std.meta.stringToEnum(Mode, std.ascii.lowerString(&buf, v)) orelse error.CalibMode;
}

/// The cost table for `want` per `k`; `default_rows`: the defaults' row count (16, the served engine prices 64).
pub fn load(gpa: Allocator, io: std.Io, k: Knobs, want: Shape, default_rows: usize) !Loaded {
    const m = try mode(k.mode);
    switch (m) {
        .default => return defaults(gpa, default_rows),
        .measure => return measure(gpa, default_rows),
        .cached, .real, .zig => {}, // real: the newest stored reference measurement
    }
    if (trimmed(k.file)) |f| return .{ .costs = try readEntry(gpa, io, f), .source = .cached, .path = try gpa.dupe(u8, f) };
    // nothing to read: the defaults, or (zig) a measurement
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dirOf(k, &dir_buf) orelse return if (m == .zig) measure(gpa, default_rows) else defaults(gpa, default_rows);
    const path = (try pickOf(gpa, io, dir, want, if (m == .zig) .zig else .python)) orelse return if (m == .zig) measure(gpa, default_rows) else defaults(gpa, default_rows);
    errdefer gpa.free(path);
    return .{ .costs = try readEntry(gpa, io, path), .source = .cached, .path = path };
}

/// TF_DSV41_CALIB_DIR, else ~/.cache/tensorfold/dsv41-calib (null: neither).
pub fn dirOf(k: Knobs, buf: []u8) ?[]const u8 {
    return trimmed(k.dir) orelse std.fmt.bufPrint(buf, "{s}/.cache/tensorfold/dsv41-calib", .{trimmed(k.home) orelse return null}) catch null;
}

fn measure(gpa: Allocator, rows: usize) !Loaded {
    return .{ .costs = try costs_mod.defaults(gpa, rows), .source = .measure };
}

fn defaults(gpa: Allocator, rows: usize) !Loaded {
    return .{ .costs = try costs_mod.defaults(gpa, rows), .source = .default };
}

/// One entry's table as the ranks share it (calib.load, then Costs.encode / decode).
pub fn readEntry(gpa: Allocator, io: std.Io, path: []const u8) !Costs {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    var raw = try Costs.parseJson(gpa, bytes);
    defer raw.deinit(gpa);
    return shared(gpa, raw);
}

/// `c` after rank 0's share: microsecond ints and back (`Costs.decode(costs.encode())`).
pub fn shared(gpa: Allocator, c: Costs) !Costs {
    const ints = try c.encode(gpa);
    defer gpa.free(ints);
    return Costs.decode(gpa, ints);
}

const Meta = struct {
    engine: []const u8 = "",
    meta: struct {
        shape: struct { slots: u32 = 0, world: u32 = 0, dspark: bool = false, context: ?u64 = null } = .{},
        time: []const u8 = "",
    } = .{},
};

/// The directory's reference-engine `calib-*.json` for `want`'s engine shape; null: none.
pub fn pick(gpa: Allocator, io: std.Io, dir: []const u8, want: Shape) !?[]u8 {
    return pickOf(gpa, io, dir, want, .python);
}

/// `pick` over one engine's entries (`"engine": "zig"` or not).
pub fn pickOf(gpa: Allocator, io: std.Io, dir: []const u8, want: Shape, engine: Engine) !?[]u8 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return null;
    defer d.close(io);
    var best: ?[]u8 = null;
    errdefer if (best) |b| gpa.free(b);
    var best_rank: u8 = 0;
    var best_time: [32]u8 = undefined;
    var best_time_len: usize = 0;
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file and e.kind != .sym_link) continue;
        if (!std.mem.startsWith(u8, e.name, "calib-") or !std.mem.endsWith(u8, e.name, ".json")) continue;
        const path = try std.fs.path.join(gpa, &.{ dir, e.name });
        var keep = false;
        defer if (!keep) gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24)) catch continue;
        defer gpa.free(bytes);
        const parsed = std.json.parseFromSlice(Meta, gpa, bytes, .{ .ignore_unknown_fields = true }) catch continue;
        defer parsed.deinit();
        const m = parsed.value.meta;
        if (std.mem.eql(u8, parsed.value.engine, "zig") != (engine == .zig)) continue;
        if (m.shape.world != want.world or m.shape.slots < want.slots or m.shape.dspark != want.dspark) continue;
        // a usable table (calib.load's rule) before it can win
        var c = Costs.parseJson(gpa, bytes) catch continue;
        c.deinit(gpa);
        const rank: u8 = 1 + @as(u8, if (m.shape.slots == want.slots) 2 else 0) + @intFromBool(want.context != null and m.shape.context == want.context);
        const t = m.time[0..@min(m.time.len, best_time.len)];
        const newer = std.mem.order(u8, t, best_time[0..best_time_len]) == .gt;
        if (best == null or rank > best_rank or (rank == best_rank and newer)) {
            if (best) |b| gpa.free(b);
            best = path;
            keep = true;
            best_rank = rank;
            @memcpy(best_time[0..t.len], t);
            best_time_len = t.len;
        }
    }
    return best;
}

/// A Zig measurement's file: calib-zig-<shape SHA-256, 16 hex>.json (one per shape; a new one replaces it).
pub fn zigName(buf: []u8, want: Shape) ![]const u8 {
    var key: [96]u8 = undefined;
    const k = try std.fmt.bufPrint(&key, "slots={d} world={d} dspark={d} context={d}", .{ want.slots, want.world, @intFromBool(want.dspark), want.context orelse 0 });
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(k, &h, .{});
    return std.fmt.bufPrint(buf, "calib-zig-{x}.json", .{h[0..8]});
}

/// What a Zig measurement stores beside its table (meta: method, seconds; rank 0's kept rows).
pub const Record = struct {
    method: measure_mod.Method,
    seconds: f64,
    /// rank 0's rows 1..16 as kept (the lower of sweep and recheck), ms
    kept: []const f64 = &.{},
    /// the calibration prompt's ids (rank 0's text, before the slot cuts)
    tokens: usize = 0,
};

/// Writes `c` for `want` into `dir` as zigName atomically; returns its path (gpa-owned); `now_s` stamps meta.time.
pub fn store(gpa: Allocator, io: std.Io, dir: []const u8, want: Shape, c: Costs, rec: Record, now_s: i64) ![]u8 {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    var nb: [64]u8 = undefined;
    const path = try std.fs.path.join(gpa, &.{ dir, try zigName(&nb, want) });
    errdefer gpa.free(path);
    const tmp = try std.fmt.allocPrint(gpa, "{s}.{d}.tmp", .{ path, std.c.getpid() });
    defer gpa.free(tmp);
    var tb: [32]u8 = undefined;
    const es: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(0, now_s)) };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    const time = try std.fmt.bufPrint(&tb, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}", .{ yd.year, @backingInt(md.month), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute() });
    const m = rec.method;
    const doc = .{
        .verify = c.verify,
        .draft = c.draft,
        .slot = c.slot,
        .engine = "zig",
        .meta = .{
            .shape = .{ .slots = want.slots, .world = want.world, .dspark = want.dspark, .context = want.context },
            .time = time,
            .method = .{ .reps = m.reps, .warm = m.warm, .stat = @tagName(m.stat), .recheck = m.recheck, .shape = @tagName(m.shape), .cycle = m.cycle, .deep = m.deep },
            .seconds = rec.seconds,
            .kept_rank0 = rec.kept,
            .prompt_tokens = rec.tokens,
            .engine = "zig",
        },
    };
    const bytes = try std.json.Stringify.valueAlloc(gpa, doc, .{ .whitespace = .indent_1 });
    defer gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp, .data = bytes });
    errdefer std.Io.Dir.cwd().deleteFile(io, tmp) catch {};
    try std.Io.Dir.cwd().rename(tmp, std.Io.Dir.cwd(), path, io);
    return path;
}

/// The depth's knobs (TF_DSV41_DEPTH, _DRAFT_DEPTH, _DRAFT_SKIP, _DEPTH_JOINT, _DEPTH_FAIR, _DEPTH_HOST_MS, rows cap).
pub fn depthSettings(get: *const fn ([*:0]const u8) ?[]const u8) !depth.Settings {
    var s: depth.Settings = .{};
    if (trimmed(get("TF_DSV41_CODE_ACCEPT"))) |v| s.code_accept = try flag(v, error.CodeAccept);
    if (s.code_accept) if (trimmed(get("TF_DSV41_GRAPH_BUCKET"))) |v| {
        s.context_bucket = try std.fmt.parseInt(u32, v, 10);
        if (s.context_bucket == 0) return error.BadGraphBucket;
    };
    const m = trimmed(get("TF_DSV41_DEPTH")) orelse "cost";
    s.mode = if (std.ascii.eqlIgnoreCase(m, "cost")) .cost else if (std.ascii.eqlIgnoreCase(m, "static")) .static else return error.DepthMode;
    if (trimmed(get("TF_DSV41_DRAFT_DEPTH"))) |v| {
        s.cap = try std.fmt.parseInt(usize, v, 10);
        if (s.cap > 15) return error.DraftDepth;
    } else s.cap = if (s.mode == .cost) 5 else 3;
    if (trimmed(get("TF_DSV41_DRAFT_SKIP"))) |v| s.skip = try flag(v, error.DraftSkip);
    if (trimmed(get("TF_DSV41_DEPTH_JOINT"))) |v| s.joint = if (std.mem.eql(u8, v, "2")) 2 else @intFromBool(try flag(v, error.DepthJoint));
    s.fair = try float(get("TF_DSV41_DEPTH_FAIR"), joint.fair, 0.0, 4.0);
    s.host_ms = try float(get("TF_DSV41_DEPTH_HOST_MS"), joint.host_ms, 0.0, 100.0);
    if (trimmed(get("TF_DSV41_GRAPH_ROWS_MAX"))) |v| s.max_rows = @max(1, @min(try std.fmt.parseInt(u64, v, 10), costs_mod.max_rows));
    return s;
}

fn flag(v: []const u8, bad: anyerror) !bool {
    for ([_][]const u8{ "1", "on", "true" }) |t| if (std.ascii.eqlIgnoreCase(v, t)) return true;
    for ([_][]const u8{ "0", "off", "false" }) |t| if (std.ascii.eqlIgnoreCase(v, t)) return false;
    return bad;
}

fn float(raw: ?[]const u8, default: f64, lo: f64, hi: f64) !f64 {
    const v = trimmed(raw) orelse return default;
    const x = try std.fmt.parseFloat(f64, v);
    if (x < lo or x > hi) return error.DepthKnob;
    return x;
}

pub fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

fn trimmed(v: ?[]const u8) ?[]const u8 {
    const t = std.mem.trim(u8, v orelse return null, " \t\r\n");
    return if (t.len == 0) null else t;
}
