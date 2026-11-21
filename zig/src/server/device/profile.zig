//! ~/.tensorfold/device.json: the last scan, versioned, read back until the hardware or this binary changes.
const std = @import("std");
const sys = @import("sys");
const machine = @import("machine.zig");
const support = @import("support.zig");
const Allocator = std.mem.Allocator;

pub const schema: u32 = 1;

/// quick: the OS's own listing, a card's compute capability from its name; full: the GPU's driver was asked too.
pub const Scan = enum { quick, full };

pub const Doc = struct {
    schema: u32 = schema,
    scanned_at: []const u8,
    binary_version: []const u8,
    /// machine.fingerprint of the quick listing, which both scans make.
    fingerprint: []const u8,
    scan: Scan,
    machine: machine.Machine,
    /// What the binary that scanned derived, kept as a record; readers derive their own.
    support: support.Support,
};

/// ~/.tensorfold/device.json (HOME, or USERPROFILE on Windows); null when neither is set.
pub fn path(a: Allocator, env: ?*const std.process.Environ.Map) !?[]const u8 {
    const e = env orelse return null;
    const home = e.get("HOME") orelse e.get("USERPROFILE") orelse return null;
    if (home.len == 0) return null;
    return try std.fs.path.join(a, &.{ home, ".tensorfold", "device.json" });
}

/// The profile at `file`; null when it is missing, unreadable, malformed or of another schema.
pub fn load(a: Allocator, io: std.Io, file: []const u8) ?Doc {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(1 << 20)) catch return null;
    return parse(a, bytes);
}

pub fn parse(a: Allocator, bytes: []const u8) ?Doc {
    const parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{ .parse_numbers = false }) catch return null;
    defer parsed.deinit();
    const doc = readValue(Doc, a, parsed.value) catch return null;
    return if (doc.schema == schema) doc else null;
}

// The typed integer parser's quad-float fallback pulls compiler-rt's memcpy into the native server.
fn readValue(comptime T: type, a: Allocator, source: std.json.Value) std.json.ParseFromValueError!T {
    switch (@typeInfo(T)) {
        .int => switch (source) {
            .number_string, .string => |raw| return decimalInteger(T, raw),
            else => return error.UnexpectedToken,
        },
        .optional => |info| {
            if (source == .null) return null;
            return try readValue(info.child, a, source);
        },
        .@"struct" => |info| {
            if (source != .object) return error.UnexpectedToken;
            var result: T = undefined;
            inline for (info.field_names, info.field_types, info.field_attrs) |name, Field, attrs| {
                @field(result, name) = if (source.object.get(name)) |v|
                    try readValue(Field, a, v)
                else
                    attrs.defaultValue(Field) orelse return error.MissingField;
            }
            return result;
        },
        .pointer => |info| {
            if (info.size != .slice) @compileError("device profiles only contain slice pointers");
            if (info.child == u8 and source == .string) return a.dupe(u8, source.string);
            if (source != .array) return error.UnexpectedToken;
            const values = try a.alloc(info.child, source.array.items.len);
            for (values, source.array.items) |*out, item| out.* = try readValue(info.child, a, item);
            return values;
        },
        .bool, .@"enum" => return std.json.innerParseFromValue(T, a, source, .{}),
        else => @compileError("unsupported device profile type " ++ @typeName(T)),
    }
}

fn decimalInteger(comptime T: type, raw: []const u8) std.json.ParseFromValueError!T {
    if (std.mem.indexOfAny(u8, raw, ".eE") == null) return std.fmt.parseInt(T, raw, 10);
    const negative = raw[0] == '-';
    const unsigned = if (negative or raw[0] == '+') raw[1..] else raw;
    const split = std.mem.indexOfAny(u8, unsigned, "eE") orelse unsigned.len;
    const mantissa = unsigned[0..split];
    const exponent = if (split == unsigned.len) 0 else try std.fmt.parseInt(i64, unsigned[split + 1 ..], 10);
    const point = std.mem.indexOfScalar(u8, mantissa, '.') orelse mantissa.len;
    const decimal_places: i64 = @intCast(mantissa.len -| (point + 1));
    const significant = std.mem.trimEnd(u8, mantissa, "0");
    const zeros: i64 = @intCast(mantissa.len - significant.len);
    var seen_digit = zeros != 0;
    var value: T = 0;
    for (significant, 0..) |ch, index| {
        if (index == point) continue;
        if (ch < '0' or ch > '9') return error.InvalidNumber;
        seen_digit = true;
        value = try std.math.add(T, try std.math.mul(T, value, 10), ch - '0');
    }
    if (!seen_digit) return error.InvalidNumber;
    if (value == 0) return 0;
    if (negative) return error.Overflow;
    const places = try std.math.add(i64, exponent, zeros - decimal_places);
    if (places < 0) return error.InvalidNumber;
    const power = std.math.cast(T, places) orelse return error.Overflow;
    const scale = std.math.powi(T, 10, power) catch return error.Overflow;
    return std.math.mul(T, value, scale);
}

pub fn write(w: *std.Io.Writer, doc: Doc) !void {
    try std.json.Stringify.value(doc, .{ .whitespace = .indent_2 }, w);
    try w.writeByte('\n');
}

/// Writes `doc` to `file` through a unique temporary file renamed into place, its folder made first.
pub fn save(a: Allocator, io: std.Io, file: []const u8, doc: Doc) !void {
    if (std.fs.path.dirname(file)) |dir| try std.Io.Dir.cwd().createDirPath(io, dir);
    var out: std.Io.Writer.Allocating = .init(a);
    try write(&out.writer, doc);
    try sys.atomic_file.replace(a, file, &.{out.written()});
}

/// Why a scan is due, or null when the saved profile still describes this machine and binary.
pub const Due = enum { missing, asked, binary, hardware, quick };

/// `want_full`: the caller can ask the driver (`tensorfold device`) and a quick profile is not enough for it.
pub fn due(old: ?Doc, fingerprint: []const u8, version: []const u8, asked: bool, want_full: bool) ?Due {
    const d = old orelse return .missing;
    if (asked) return .asked;
    if (!std.mem.eql(u8, d.binary_version, version)) return .binary;
    if (!std.mem.eql(u8, d.fingerprint, fingerprint)) return .hardware;
    if (want_full and d.scan == .quick) return .quick;
    return null;
}

/// UTC time as ISO 8601 to the second ("2026-11-13T01:22:01Z").
pub fn stamp(buf: *[20]u8, seconds: i64) []const u8 {
    const es: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(0, seconds)) };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        yd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute(),
    }) catch unreachable;
}
