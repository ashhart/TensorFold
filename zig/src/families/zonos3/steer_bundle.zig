//! A steering bundle (the reference's --steer-bundle file and its direction sets) and a request's controls resolved.
const std = @import("std");
const cfg = @import("config.zig");
const npy = @import("npy.zig");
const steering = @import("steering.zig");
const Allocator = std.mem.Allocator;

const D = cfg.dim;
const E = cfg.speaker_dim;
/// steering.RAMP_FRAMES: 0.2 s at 15 frames a second.
const ramp_frames = 3;
/// The codec's frames a second, which turn steer_release's seconds into frames.
const frame_rate = 15.0;
/// Bipolar axes by name, besides a manifest entry of kind "axis" (directions.AXIS_NAMES).
const axis_names = [_][]const u8{ "valence", "arousal" };

pub const Direction = struct {
    /// "<set>.<name>", as a request's controls name it.
    id: []const u8,
    set: []const u8,
    /// A raw direction moves the speaker embedding (192 values); a hidden one adds to bank source `source` (1024).
    raw: bool,
    source: u32,
    axis: bool,
    vector: []const f32,
};

/// A control's expected duration multiplier by dose: the bundle's `timing` table, rising in dose.
pub const Timing = struct { id: []const u8, doses: []const f64, factors: []const f64 };

/// The bundle's steer_release: the request fields' defaults and the hidden directions that never release.
pub const ReleaseDefault = struct { onset_seconds: f64, ramp_seconds: f64, hold: f64, exempt: []const []const u8 };

pub const Bundle = struct {
    arena: std.heap.ArenaAllocator,
    name: []const u8 = "",
    directions: []const Direction = &.{},
    /// The bank sources the hidden sets steer, rising: every delta holds a row for each.
    sources: []const u32 = &.{},
    timing: []const Timing = &.{},
    release: ?ReleaseDefault = null,

    pub fn deinit(b: *Bundle) void {
        b.arena.deinit();
    }

    pub fn find(b: *const Bundle, id: []const u8) ?*const Direction {
        for (b.directions) |*d| if (std.mem.eql(u8, d.id, id)) return d;
        return null;
    }

    fn exempt(b: *const Bundle, id: []const u8) bool {
        const r = b.release orelse return false;
        for (r.exempt) |e| if (std.mem.eql(u8, e, id)) return true;
        return false;
    }

    /// steering._duration_factor: the product over the timed controls of their table at the dose, in table order.
    fn durationFactor(b: *const Bundle, doses: []const Dose) f64 {
        var factor: f64 = 1.0;
        for (b.timing) |t| for (doses) |d| {
            if (d.value != 0 and std.mem.eql(u8, d.id, t.id)) factor *= interp(d.value, t.doses, t.factors);
        };
        return factor;
    }
};

/// numpy.interp: the table's ends outside it, a straight line between its points.
fn interp(x: f64, xs: []const f64, ys: []const f64) f64 {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.len - 1]) return ys[ys.len - 1];
    var j: usize = 0;
    while (xs[j + 1] <= x) j += 1;
    if (x == xs[j]) return ys[j];
    return (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]) * (x - xs[j]) + ys[j];
}

fn number(v: std.json.Value) ?f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => null,
    };
}

fn field(o: std.json.ObjectMap, name: []const u8, comptime tag: std.meta.Tag(std.json.Value)) !@FieldType(std.json.Value, @tagName(tag)) {
    const v = o.get(name) orelse return error.BadSteerBundle;
    return if (v == tag) @field(v, @tagName(tag)) else error.BadSteerBundle;
}

/// steering.load_bundle and build_bundle: the bundle file, each set's manifest.json beside it and its vectors.
pub fn load(gpa: Allocator, io: std.Io, path: []const u8) !Bundle {
    var b: Bundle = .{ .arena = .init(gpa) };
    errdefer b.deinit();
    const a = b.arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const text = try cwd.readFileAlloc(io, path, a, .limited(1 << 24));
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}));
    if (root != .object) return error.BadSteerBundle;
    const spec = root.object;
    b.name = if (spec.get("name")) |v| (if (v == .string) v.string else return error.BadSteerBundle) else std.fs.path.stem(path);
    const dir = std.fs.path.dirname(path) orelse ".";
    var directions: std.ArrayList(Direction) = .empty;
    var sources: std.ArrayList(u32) = .empty;
    const sets = try field(spec, "sets", .object);
    for (sets.keys(), sets.values()) |set, rel| {
        if (rel != .string or std.mem.indexOfScalar(u8, set, '.') != null) return error.BadSteerBundle;
        const folder = try std.fs.path.join(a, &.{ dir, rel.string });
        const manifest_text = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ folder, "manifest.json" }), a, .limited(1 << 24));
        const manifest_root = try std.json.parseFromSliceLeaky(std.json.Value, a, manifest_text, .{});
        if (manifest_root != .object) return error.BadSteerBundle;
        const manifest = manifest_root.object;
        const dim = try field(manifest, "dim", .integer);
        const space: []const u8 = if (manifest.get("space")) |v| (if (v == .string) v.string else return error.BadSteerBundle) else "raw";
        const raw = std.mem.eql(u8, space, "raw");
        if (!raw and !std.mem.eql(u8, space, "attnres_source")) return error.BadSteerBundle;
        if (dim != @as(i64, if (raw) E else D)) return error.SteerBundleOfAnotherModel;
        var source: u32 = 0;
        if (!raw) {
            source = std.math.cast(u32, try field(manifest, "source_index", .integer)) orelse return error.BadSteerBundle;
            if (source >= cfg.sources) return error.SteerBundleOfAnotherModel;
            if (std.mem.indexOfScalar(u32, sources.items, source) == null) try sources.append(a, source);
        }
        const entries = try field(manifest, "directions", .object);
        if (entries.count() == 0) return error.BadSteerBundle;
        // {**named, **axes}: the set's named directions, then its axes, each in the manifest's order
        for ([_]bool{ false, true }) |axes| for (entries.keys(), entries.values()) |name, entry| {
            if (entry != .object) return error.BadSteerBundle;
            var axis = for (axis_names) |n| {
                if (std.mem.eql(u8, n, name)) break true;
            } else false;
            if (entry.object.get("kind")) |k| axis = axis or (k == .string and std.mem.eql(u8, k.string, "axis"));
            if (axis != axes) continue;
            const file = try field(entry.object, "file", .string);
            const bytes = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ folder, file }), a, .limited(1 << 20));
            const arr = try npy.parse(bytes);
            if (arr.kind != .f32 or arr.numel() != dim) return error.BadSteerBundle;
            const vector = try arr.floats(a);
            for (vector) |x| if (!std.math.isFinite(x)) return error.BadSteerBundle;
            try directions.append(a, .{ .id = try std.fmt.allocPrint(a, "{s}.{s}", .{ set, name }), .set = set, .raw = raw, .source = source, .axis = axis, .vector = vector });
        };
    }
    std.mem.sort(u32, sources.items, {}, std.sort.asc(u32));
    b.directions = directions.items;
    b.sources = sources.items;
    if (spec.get("timing")) |t| {
        if (t != .object) return error.BadSteerBundle;
        const timing = try a.alloc(Timing, t.object.count());
        for (timing, t.object.keys(), t.object.values()) |*out, id, table| {
            if (b.find(id) == null or table != .array or table.array.items.len == 0) return error.BadSteerBundle;
            const Point = struct { x: f64, y: f64 };
            const points = try a.alloc(Point, table.array.items.len);
            for (points, table.array.items) |*p, row| {
                if (row != .array or row.array.items.len != 2) return error.BadSteerBundle;
                p.* = .{ .x = number(row.array.items[0]) orelse return error.BadSteerBundle, .y = number(row.array.items[1]) orelse return error.BadSteerBundle };
            }
            std.mem.sort(Point, points, {}, struct {
                fn less(_: void, l: Point, r: Point) bool {
                    return l.x < r.x or (l.x == r.x and l.y < r.y);
                }
            }.less);
            const xs = try a.alloc(f64, points.len);
            const ys = try a.alloc(f64, points.len);
            for (points, xs, ys) |p, *x, *y| {
                x.* = p.x;
                y.* = p.y;
            }
            out.* = .{ .id = id, .doses = xs, .factors = ys };
        }
        b.timing = timing;
    }
    if (spec.get("steer_release")) |r| if (r != .null) {
        if (r != .object) return error.BadSteerBundle;
        var release: ReleaseDefault = .{
            .onset_seconds = number(r.object.get("onset_seconds") orelse return error.BadSteerBundle) orelse return error.BadSteerBundle,
            .ramp_seconds = number(r.object.get("ramp_seconds") orelse return error.BadSteerBundle) orelse return error.BadSteerBundle,
            .hold = number(r.object.get("hold") orelse return error.BadSteerBundle) orelse return error.BadSteerBundle,
            .exempt = &.{},
        };
        if (!releaseOk(release.hold, release.onset_seconds, release.ramp_seconds)) return error.BadSteerBundle;
        if (r.object.get("exempt")) |list| {
            if (list != .array) return error.BadSteerBundle;
            const exempt = try a.alloc([]const u8, list.array.items.len);
            for (exempt, list.array.items) |*e, v| {
                if (v != .string) return error.BadSteerBundle;
                const d = b.find(v.string) orelse return error.BadSteerBundle;
                if (d.raw) return error.BadSteerBundle;
                e.* = v.string;
            }
            release.exempt = exempt;
        }
        b.release = release;
    };
    return b;
}

/// steering.check_release: a hold from 0 through 1, times that are finite and not negative.
fn releaseOk(hold: f64, onset: f64, ramp: f64) bool {
    return std.math.isFinite(hold) and hold >= 0 and hold <= 1 and std.math.isFinite(onset) and onset >= 0 and std.math.isFinite(ramp) and ramp >= 0;
}

pub const Dose = struct { id: []const u8, value: f64 };
/// Hidden controls over the characters [start, end) of the request's text.
pub const SpanRequest = struct { start: i64, end: i64, controls: []const Dose };

pub const Request = struct {
    controls: []const Dose = &.{},
    spans: []const SpanRequest = &.{},
    /// The text's length in characters (code points), which span positions count.
    chars: usize = 0,
    /// The speaker embedding of an embedding voice; null for a voice by reference codes or a clip.
    embedding: ?*const [E]f32 = null,
    /// control_span_rate: frames a character; null is 1.
    rate: ?f64 = null,
    hold: ?f64 = null,
    onset_seconds: ?f64 = null,
    ramp_seconds: ?f64 = null,
};

pub const Resolved = struct {
    /// The speaker embedding after the raw controls; null when none moved it.
    embedding: ?[E]f32 = null,
    plan: steering.Plan = .{},
    /// Each kept span's characters [start, end), in the plan's span order: what reading-position spans follow.
    span_chars: []const [2]usize = &.{},
};

pub const Error = error{ OutOfMemory, Refused };

fn refuse(a: Allocator, message: *[]const u8, comptime fmt: []const u8, args: anytype) Error {
    message.* = try std.fmt.allocPrint(a, fmt, args);
    return error.Refused;
}

/// steering.request_release: the request's steer-then-hold schedule over the bundle's defaults; null: constant.
fn requestRelease(a: Allocator, b: *const Bundle, r: Request, message: *[]const u8) Error!?struct { onset: u32, ramp: u32, hold: f64 } {
    const default = b.release orelse {
        if (r.hold == null or r.hold.? == 1.0) return null;
        return refuse(a, message, "steer_hold below 1 needs a steering bundle with steer_release; this server's bundle has none", .{});
    };
    const onset = r.onset_seconds orelse default.onset_seconds;
    const ramp = r.ramp_seconds orelse default.ramp_seconds;
    const hold = r.hold orelse default.hold;
    if (!releaseOk(hold, onset, ramp)) return refuse(a, message, "steer_hold must be from 0 through 1, and steer_onset_seconds and steer_ramp_seconds finite and not negative", .{});
    if (onset > 600 or ramp > 600) return refuse(a, message, "steer_onset_seconds and steer_ramp_seconds must be at most 600", .{});
    if (hold == 1.0) return null;
    return .{ .onset = @intFromFloat(roundHalfEven(onset * frame_rate)), .ramp = @intFromFloat(roundHalfEven(ramp * frame_rate)), .hold = hold };
}

/// Python's round(): to the nearest whole number, a half to the even one.
fn roundHalfEven(x: f64) f64 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(x) else r;
}

/// hidden_vector: dose * vector summed in the doses' order in fp32, a row a bundle source; empty: no hidden dose.
fn hiddenVector(a: Allocator, b: *const Bundle, doses: []const Dose, keep: enum { all, exempt, released }) Error![]f32 {
    var out: []f32 = &.{};
    for (doses) |d| {
        const dir = b.find(d.id).?;
        if (dir.raw or d.value == 0) continue;
        if (keep != .all and (keep == .exempt) != b.exempt(d.id)) continue;
        if (out.len == 0) {
            out = try a.alloc(f32, b.sources.len * D);
            @memset(out, 0);
        }
        const row = std.mem.indexOfScalar(u32, b.sources, dir.source).?;
        const v: f32 = @floatCast(d.value);
        for (out[row * D ..][0..D], dir.vector) |*o, x| o.* += v * x;
    }
    return out;
}

/// torch.linalg.vector_norm of 192 fp32 values as the reference's CPU build sums: eight running sums, added in order.
fn norm(x: *const [E]f32) f32 {
    var acc: [8]f32 = @splat(0);
    for (0..E / 8) |i| for (&acc, x[i * 8 ..][0..8]) |*s, v| {
        s.* += v * v;
    };
    var sum = acc[0];
    for (acc[1..]) |s| sum += s;
    return @sqrt(sum);
}

fn checkDoses(a: Allocator, b: *const Bundle, doses: []const Dose, message: *[]const u8) Error!void {
    for (doses) |d| {
        if (b.find(d.id) == null) return refuse(a, message, "unknown control '{s}'; see /v1/audio/controls", .{d.id});
        if (!std.math.isFinite(d.value)) return refuse(a, message, "control '{s}' must be finite", .{d.id});
    }
}

/// steering.resolve_controls with request_release, from the arena `a`; on error.Refused `message` says why.
pub fn resolve(a: Allocator, b: *const Bundle, r: Request, message: *[]const u8) Error!Resolved {
    const release = try requestRelease(a, b, r, message);
    var out: Resolved = .{};
    if (r.controls.len == 0 and r.spans.len == 0) return out;
    try checkDoses(a, b, r.controls, message);
    for (r.spans) |s| try checkDoses(a, b, s.controls, message);
    // the speaker embedding: every raw control at once, the norm kept
    var moved = false;
    var delta: [E]f32 = @splat(0);
    for (r.controls) |d| {
        const dir = b.find(d.id).?;
        if (!dir.raw or d.value == 0) continue;
        if (r.embedding == null) return refuse(a, message, "control '{s}' moves the speaker embedding and needs a voice by embedding (speaker_mode \"embedding\")", .{d.id});
        const v: f32 = @floatCast(d.value);
        for (&delta, dir.vector) |*o, x| o.* += v * x;
        moved = true;
    }
    if (moved) {
        const base = r.embedding.?;
        var sum: [E]f32 = undefined;
        for (&sum, base, delta) |*o, x, y| o.* = x + y;
        const scale = norm(base) / @max(norm(&sum), 1e-8);
        for (&sum) |*o| o.* *= scale;
        out.embedding = sum;
    }
    out.plan.sources = b.sources;
    if (release) |rel| {
        out.plan.constant = try hiddenVector(a, b, r.controls, .exempt);
        out.plan.released = try hiddenVector(a, b, r.controls, .released);
        if (out.plan.released.len != 0) out.plan.release = .{ .onset = rel.onset, .ramp = rel.ramp, .hold = rel.hold };
    } else out.plan.constant = try hiddenVector(a, b, r.controls, .all);
    if (r.spans.len == 0) return out;
    // spans: frames placed by character position, a span's characters at its own controls' pace
    const rate = (r.rate orelse 1.0) * b.durationFactor(r.controls);
    if (!(rate > 0) or !std.math.isFinite(rate)) return refuse(a, message, "control_span_rate must be positive", .{});
    const spans = try a.dupe(SpanRequest, r.spans);
    std.mem.sort(SpanRequest, spans, {}, struct {
        fn less(_: void, l: SpanRequest, h: SpanRequest) bool {
            return l.start < h.start or (l.start == h.start and l.end < h.end);
        }
    }.less);
    for (spans, 0..) |s, i| {
        if (s.start < 0 or s.start >= s.end or s.end > r.chars) return refuse(a, message, "control span [{d}, {d}) is outside the text (0..{d})", .{ s.start, s.end, r.chars });
        for (s.controls) |d| if (d.value != 0 and b.find(d.id).?.raw)
            return refuse(a, message, "control '{s}' moves the speaker embedding and cannot vary inside a request; only hidden controls take spans", .{d.id});
        if (i > 0 and s.start < spans[i - 1].end) return refuse(a, message, "control spans [{d}, {d}) and [{d}, ...) overlap", .{ spans[i - 1].start, spans[i - 1].end, s.start });
    }
    const frame_at = try a.alloc(f64, r.chars + 1);
    frame_at[0] = 0;
    var next: usize = 0;
    for (frame_at[1..], 0..) |*f, c| {
        while (next < spans.len and spans[next].end <= c) next += 1;
        const inside = next < spans.len and spans[next].start <= c;
        f.* = frame_at[c] + if (inside) rate * b.durationFactor(spans[next].controls) else rate;
    }
    var rows: std.ArrayList(steering.Span) = .empty;
    var chars: std.ArrayList([2]usize) = .empty;
    for (spans) |s| {
        const vec = try hiddenVector(a, b, s.controls, .all);
        if (vec.len == 0) continue;
        try rows.append(a, .{ .start = @floatCast(roundHalfEven(frame_at[@intCast(s.start)])), .end = @floatCast(roundHalfEven(frame_at[@intCast(s.end)])), .ramp = ramp_frames, .delta = vec });
        try chars.append(a, .{ @intCast(s.start), @intCast(s.end) });
    }
    out.plan.spans = rows.items;
    out.span_chars = chars.items;
    return out;
}

test "interp follows numpy: the table's ends outside it, straight lines inside" {
    const t = std.testing;
    const xs = [_]f64{ -2, 0, 2 };
    const ys = [_]f64{ 2.2, 1.0, 0.63 };
    try t.expectEqual(@as(f64, 2.2), interp(-5, &xs, &ys));
    try t.expectEqual(@as(f64, 0.63), interp(2, &xs, &ys));
    try t.expectEqual(@as(f64, 1.0), interp(0, &xs, &ys));
    try t.expectApproxEqAbs(@as(f64, 1.6), interp(-1, &xs, &ys), 1e-12);
    try t.expectApproxEqAbs(@as(f64, 0.815), interp(1, &xs, &ys), 1e-12);
}

test "round follows Python: a half goes to the even number" {
    const t = std.testing;
    for ([_]f64{ 0.5, 1.5, 2.5, 3.49, -0.5, -1.5, 7 }, [_]f64{ 0, 2, 2, 3, 0, -2, 7 }) |x, want| try t.expectEqual(want, roundHalfEven(x));
}

fn testBundle(a: Allocator, raw: []const f32, hidden_a: []const f32, hidden_b: []const f32) Bundle {
    const directions = a.dupe(Direction, &.{
        .{ .id = "who.age", .set = "who", .raw = true, .source = 0, .axis = true, .vector = raw },
        .{ .id = "mood.joy", .set = "mood", .raw = false, .source = 19, .axis = false, .vector = hidden_a },
        .{ .id = "pace.tempo", .set = "pace", .raw = false, .source = 15, .axis = true, .vector = hidden_b },
    }) catch unreachable;
    const timing = a.dupe(Timing, &.{.{ .id = "pace.tempo", .doses = &.{ 0, 2 }, .factors = &.{ 1, 0.5 } }}) catch unreachable;
    return .{ .arena = .init(a), .directions = directions, .sources = &.{ 15, 19 }, .timing = timing, .release = .{ .onset_seconds = 1, .ramp_seconds = 1, .hold = 0.5, .exempt = &.{"pace.tempo"} } };
}

test "controls resolve to a moved embedding of the same norm, constant and released deltas, and span rows" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = try a.alloc(f32, E);
    const ha = try a.alloc(f32, D);
    const hb = try a.alloc(f32, D);
    for (raw, 0..) |*x, i| x.* = if (i == 0) 1 else 0;
    @memset(ha, 2);
    @memset(hb, 4);
    const b = testBundle(a, raw, ha, hb);
    var message: []const u8 = "";
    var base: [E]f32 = @splat(0);
    base[1] = 3;
    const got = try resolve(a, &b, .{ .controls = &.{ .{ .id = "who.age", .value = 4 }, .{ .id = "mood.joy", .value = 0.5 }, .{ .id = "pace.tempo", .value = 2 } }, .spans = &.{.{ .start = 2, .end = 4, .controls = &.{.{ .id = "mood.joy", .value = 1 }} }}, .chars = 10, .embedding = &base }, &message);
    // (4, 3, 0...) scaled back to norm 3
    try t.expectApproxEqAbs(@as(f32, 2.4), got.embedding.?[0], 1e-6);
    try t.expectApproxEqAbs(@as(f32, 1.8), got.embedding.?[1], 1e-6);
    // tempo is exempt: constant on source 15's row; joy releases on source 19's
    try t.expectEqual(@as(f32, 8), got.plan.constant[0]);
    try t.expectEqual(@as(f32, 0), got.plan.constant[D]);
    try t.expectEqual(@as(f32, 1), got.plan.released[D]);
    try t.expectEqual(steering.Release{ .onset = 15, .ramp = 15, .hold = 0.5 }, got.plan.release);
    // tempo 2 halves the pace: characters 2 and 4 start at frames 1 and 2
    try t.expectEqual(@as(usize, 1), got.plan.spans.len);
    try t.expectEqual(@as(f32, 1), got.plan.spans[0].start);
    try t.expectEqual(@as(f32, 2), got.plan.spans[0].end);
    try t.expectEqual(@as(f32, 2), got.plan.spans[0].delta[D]);
    // hold 1 keeps every control constant
    const held = try resolve(a, &b, .{ .controls = &.{.{ .id = "mood.joy", .value = 0.5 }}, .hold = 1 }, &message);
    try t.expectEqual(@as(usize, 0), held.plan.released.len);
    try t.expectEqual(@as(f32, 1), held.plan.constant[D]);
}

test "controls are refused with the reason" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = try a.alloc(f32, E);
    const h = try a.alloc(f32, D);
    @memset(raw, 1);
    @memset(h, 1);
    const b = testBundle(a, raw, h, h);
    var message: []const u8 = "";
    const joy = [_]Dose{.{ .id = "mood.joy", .value = 1 }};
    const cases = [_]struct { r: Request, want: []const u8 }{
        .{ .r = .{ .controls = &.{.{ .id = "mood.nope", .value = 1 }} }, .want = "unknown control 'mood.nope'" },
        .{ .r = .{ .controls = &.{.{ .id = "who.age", .value = 1 }} }, .want = "needs a voice by embedding" },
        .{ .r = .{ .controls = &joy, .hold = 1.5 }, .want = "steer_hold must be" },
        .{ .r = .{ .spans = &.{.{ .start = 3, .end = 12, .controls = &joy }}, .chars = 10 }, .want = "outside the text" },
        .{ .r = .{ .spans = &.{ .{ .start = 0, .end = 5, .controls = &joy }, .{ .start = 4, .end = 6, .controls = &joy } }, .chars = 10 }, .want = "overlap" },
        .{ .r = .{ .spans = &.{.{ .start = 0, .end = 5, .controls = &.{.{ .id = "who.age", .value = 1 }} }}, .chars = 10 }, .want = "cannot vary inside a request" },
        .{ .r = .{ .spans = &.{.{ .start = 0, .end = 5, .controls = &joy }}, .chars = 10, .rate = 0 }, .want = "control_span_rate must be positive" },
    };
    for (cases) |c| {
        try t.expectError(error.Refused, resolve(a, &b, c.r, &message));
        try t.expect(std.mem.indexOf(u8, message, c.want) != null);
    }
}
