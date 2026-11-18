//! ZONOS3's steering fields of a speech request: controls from the server's bundle, resolved before the engine runs.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const req = @import("zonos_request");
const Allocator = std.mem.Allocator;

const bundle_mod = req.steer_bundle;
const Bundle = bundle_mod.Bundle;
const prefix = "ZONOS3 request: ";

pub const fields = struct {
    pub const controls = "controls";
    pub const spans = "control_spans";
    pub const rate = "control_span_rate";
    pub const follow = "control_span_follow";
    pub const hold = "steer_hold";
    pub const onset = "steer_onset_seconds";
    pub const ramp = "steer_ramp_seconds";
    pub const guidance = "controls_cfg_scale";
    pub const lead = "control_span_lead";
    pub const lag = "control_span_lag";
    pub const all = [_][]const u8{ controls, spans, rate, follow, hold, onset, ramp, guidance, lead, lag };
};

/// The ZONOS3 paths and decoders a server may be started with (Args.zonos).
pub const Paths = struct {
    steer_bundle: ?[]const u8 = null,
    laci_heads: ?[]const u8 = null,
    short_text: ?[]const u8 = null,
};

pub const flags = [_]@import("cli.zig").Flag{
    .{ .name = "--steer-bundle", .path = true, .native = true },
    .{ .name = "--laci-heads", .path = true, .native = true },
    .{ .name = "--short-text", .path = true, .native = true },
};

/// Sets a ZONOS3 flag on `out` (the server's Args); false: not one of them.
pub fn flag(out: anytype, name: []const u8, value: []const u8) bool {
    if (std.mem.eql(u8, name, "--steer-bundle")) out.zonos.steer_bundle = value else if (std.mem.eql(u8, name, "--laci-heads")) out.zonos.laci_heads = value else if (std.mem.eql(u8, name, "--short-text")) out.zonos.short_text = value else return false;
    return true;
}

pub const Made = struct {
    /// The speaker embedding after the raw controls; null: as it was.
    embedding: ?[req.config.speaker_dim]f32 = null,
    steering: ?*api.SpeechSteering = null,
    /// `control_span_follow: "reading"` with spans that steer: each span's characters, and the gate's lead and lag.
    follow: ?struct { chars: []const [2]usize, lead: f64, lag: f64 } = null,
};

fn number(v: json.Value) ?f64 {
    return switch (v) {
        .float => |f| f,
        .int => |t| std.fmt.parseFloat(f64, t) catch null,
        else => null,
    };
}

fn optional(cx: *errors.Cx, raw: json.Value, comptime name: []const u8) errors.Refused!?f64 {
    const v = raw.field(name) orelse return null;
    return number(v) orelse cx.refuse(prefix ++ name ++ " must be a number");
}

/// `{"<set>.<direction>": dose}` in the request's order.
fn doses(cx: *errors.Cx, comptime what: []const u8, value: json.Value) errors.Refused![]const bundle_mod.Dose {
    if (value != .object) return cx.refuse(prefix ++ what ++ " must be an object of control names and doses");
    const out = try cx.a.alloc(bundle_mod.Dose, value.object.keys().len);
    for (out, value.object.keys(), value.object.values()) |*d, id, v| d.* = .{ .id = id, .value = number(v) orelse return cx.refuse(prefix ++ what ++ ": every dose must be a number") };
    return out;
}

/// The request's steering resolved with the bundle; `chars`: the text's length, `speaker_guided`: its twin is taken.
pub fn plan(bundle: ?*const Bundle, cx: *errors.Cx, raw: json.Value, chars: usize, embedding: ?*const [req.config.speaker_dim]f32, speaker_guided: bool) errors.Refused!Made {
    var r: bundle_mod.Request = .{ .chars = chars, .embedding = embedding };
    r.hold = try optional(cx, raw, fields.hold);
    r.onset_seconds = try optional(cx, raw, fields.onset);
    r.ramp_seconds = try optional(cx, raw, fields.ramp);
    r.rate = try optional(cx, raw, fields.rate);
    const guidance = try optional(cx, raw, fields.guidance) orelse 1;
    if (!std.math.isFinite(guidance) or guidance < 0) return cx.refuse(prefix ++ "controls_cfg_scale must be finite and not negative");
    if (guidance != 1 and speaker_guided) return cx.refuse(prefix ++ "speaker_cfg_scale and controls_cfg_scale cannot be combined: a request has one guidance twin");
    var reading = false;
    if (raw.field(fields.follow)) |v| {
        const name = v.str() orelse return cx.refuse(prefix ++ "control_span_follow must be \"rate\" or \"reading\"");
        reading = std.mem.eql(u8, name, "reading");
        if (!reading and !std.mem.eql(u8, name, "rate")) return cx.refuse(prefix ++ "control_span_follow must be \"rate\" or \"reading\"");
    }
    const lead = try optional(cx, raw, fields.lead) orelse 1.0;
    const lag = try optional(cx, raw, fields.lag) orelse 0.0;
    if (!(lead >= 0 and lead <= 8)) return cx.refuse(prefix ++ "control_span_lead must be from 0 through 8 tokens");
    if (!(lag >= -4 and lag <= 8)) return cx.refuse(prefix ++ "control_span_lag must be from -4 through 8 tokens");
    if (raw.field(fields.controls)) |v| r.controls = try doses(cx, fields.controls, v);
    if (raw.field(fields.spans)) |v| {
        const list = v.items() orelse return cx.refuse(prefix ++ "control_spans must be a list of {start, end, controls}");
        const spans = try cx.a.alloc(bundle_mod.SpanRequest, list.len);
        for (spans, list) |*s, entry| {
            const message = prefix ++ "a control span is {\"start\": character, \"end\": character, \"controls\": {...}}";
            if (entry != .object or entry.object.keys().len != 3) return cx.refuse(message);
            const start = entry.field("start") orelse return cx.refuse(message);
            const end = entry.field("end") orelse return cx.refuse(message);
            if (start != .int or end != .int) return cx.refuse(message);
            s.* = .{
                .start = std.fmt.parseInt(i64, start.int, 10) catch return cx.refuse(message),
                .end = std.fmt.parseInt(i64, end.int, 10) catch return cx.refuse(message),
                .controls = try doses(cx, "a control span's controls", entry.field("controls") orelse return cx.refuse(message)),
            };
        }
        r.spans = spans;
    }
    const b = bundle orelse {
        if (r.controls.len != 0 or r.spans.len != 0) return cx.refuse(prefix ++ "this server has no steering bundle (start it with --steer-bundle); controls are unavailable");
        if (r.hold != null and r.hold.? != 1) return cx.refuse(prefix ++ "this server has no steering bundle (start it with --steer-bundle); steer_hold below 1 needs its steer_release");
        return .{};
    };
    var message: []const u8 = "";
    const got = bundle_mod.resolve(cx.a, b, r, &message) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Refused => cx.fail(.request, prefix ++ "{s}", .{message}),
    };
    var out: Made = .{ .embedding = got.embedding };
    if (reading and got.span_chars.len != 0) out.follow = .{ .chars = got.span_chars, .lead = lead, .lag = lag };
    if (!got.plan.active()) return out;
    const spans = try cx.a.alloc(api.SpeechSteering.Span, got.plan.spans.len);
    for (spans, got.plan.spans) |*to, from| to.* = .{ .start = from.start, .end = from.end, .ramp = from.ramp, .delta = from.delta };
    const steering = try cx.a.create(api.SpeechSteering);
    // plan.request_rules R12: guidance on the controls needs a constant or released delta and a speaker row
    const guided = guidance != 1 and embedding != null and (got.plan.constant.len != 0 or got.plan.released.len != 0);
    steering.* = .{ .sources = got.plan.sources, .constant = got.plan.constant, .released = got.plan.released, .onset = got.plan.release.onset, .ramp = got.plan.release.ramp, .hold = got.plan.release.hold, .spans = spans, .guidance = if (guided) @floatCast(guidance) else 1 };
    out.steering = steering;
    return out;
}

/// GET /v1/audio/controls: the directions as a request names them, the timing tables and the release defaults.
pub fn document(a: Allocator, b: *const Bundle) !json.Value {
    const directions = try json.newObject(a);
    for (b.directions) |d| {
        const entry = try json.newObject(a);
        try entry.put(a, "set", .{ .string = d.set });
        try entry.put(a, "space", .{ .string = if (d.raw) "speaker_embedding" else "hidden" });
        try entry.put(a, "kind", .{ .string = if (d.axis) "axis" else "named" });
        if (!d.raw) try entry.put(a, "source_index", .{ .int = try std.fmt.allocPrint(a, "{d}", .{d.source}) });
        var sum: f64 = 0;
        for (d.vector) |x| sum += @as(f64, x) * x;
        try entry.put(a, "norm", .{ .float = @sqrt(sum) });
        try directions.put(a, d.id, .{ .object = entry });
    }
    const timing = try json.newObject(a);
    for (b.timing) |t| {
        const rows = try a.alloc(json.Value, t.doses.len);
        for (rows, t.doses, t.factors) |*row, x, y| {
            const pair = try a.alloc(json.Value, 2);
            pair[0] = .{ .float = x };
            pair[1] = .{ .float = y };
            row.* = .{ .array = pair };
        }
        try timing.put(a, t.id, .{ .array = rows });
    }
    const object = try json.newObject(a);
    try object.put(a, "bundle", .{ .string = b.name });
    try object.put(a, "directions", .{ .object = directions });
    try object.put(a, "timing", .{ .object = timing });
    if (b.release) |r| {
        const release = try json.newObject(a);
        try release.put(a, "onset_seconds", .{ .float = r.onset_seconds });
        try release.put(a, "ramp_seconds", .{ .float = r.ramp_seconds });
        try release.put(a, "hold", .{ .float = r.hold });
        const exempt = try a.alloc(json.Value, r.exempt.len);
        for (exempt, r.exempt) |*e, id| e.* = .{ .string = id };
        try release.put(a, "exempt", .{ .array = exempt });
        try object.put(a, "steer_release", .{ .object = release });
    } else try object.put(a, "steer_release", .null);
    return .{ .object = object };
}

test "without a bundle controls are refused, and a hold below 1; plain requests pass" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const plain = (try json.parse(a, "{\"steer_hold\": 1, \"controls_cfg_scale\": 2, \"controls\": null}")).ok;
    const made = try plan(null, &cx, plain, 5, null, false);
    try t.expect(made.steering == null and made.embedding == null);
    const cases = [_][2][]const u8{
        .{ "{\"controls\": {\"mood.valence\": 1}}", "this server has no steering bundle (start it with --steer-bundle); controls are unavailable" },
        .{ "{\"control_spans\": [{\"start\": 0, \"end\": 2, \"controls\": {\"a.b\": 1}}]}", "this server has no steering bundle (start it with --steer-bundle); controls are unavailable" },
        .{ "{\"steer_hold\": 0.5}", "steer_hold below 1 needs its steer_release" },
        .{ "{\"controls\": [1]}", "controls must be an object of control names and doses" },
        .{ "{\"controls_cfg_scale\": -1}", "controls_cfg_scale must be finite and not negative" },
        .{ "{\"control_span_follow\": \"words\"}", "must be \"rate\" or \"reading\"" },
        .{ "{\"control_span_lead\": 9}", "control_span_lead must be" },
        .{ "{\"control_span_lag\": -5}", "control_span_lag must be" },
        .{ "{\"control_spans\": [{\"start\": 0, \"end\": 2}]}", "a control span is" },
    };
    for (cases) |c| {
        try t.expectError(error.Refused, plan(null, &cx, (try json.parse(a, c[0])).ok, 5, null, false));
        try t.expect(std.mem.indexOf(u8, cx.message, c[1]) != null);
    }
    try t.expectError(error.Refused, plan(null, &cx, (try json.parse(a, "{\"controls_cfg_scale\": 2}")).ok, 5, null, true));
    try t.expect(std.mem.indexOf(u8, cx.message, "cannot be combined") != null);
}

test "--steer-bundle and --laci-heads name their files" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const cli = @import("cli.zig");
    var u: cli.Usage = .{};
    try std.testing.expect((try cli.parse(arena.allocator(), &.{"m"}, &u)).zonos.steer_bundle == null);
    try std.testing.expectEqualStrings("b/bundle.json", (try cli.parse(arena.allocator(), &.{ "m", "--steer-bundle", "b/bundle.json" }, &u)).zonos.steer_bundle.?);
    try std.testing.expectEqualStrings("heads", (try cli.parse(arena.allocator(), &.{ "m", "--laci-heads=heads" }, &u)).zonos.laci_heads.?);
}
