//! The decode-time note lookup's request fields and its report: the reply's notes.decode field and the log line.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const log = @import("log.zig");
const notes_route = @import("notes_route.zig");
const notes_decode = @import("notes_decode.zig");
const index = @import("notes/index.zig");
const Live = notes_decode.Live;

/// The weight of a key only one note has, log(1 + (N - 0.5) / 1.5): the lookup's subject boost at subject 1.
pub fn topWeight(idx: *const index.Index, past: bool) f64 {
    const n: u32 = idx.n_cur + (if (past) idx.n_hist else 0);
    if (n == 0) return 0;
    return idx.log(1.0 + (@as(f64, @floatFromInt(n)) - 0.5) / 1.5);
}

/// One note put in while the reply decoded.
pub const Injection = struct {
    ids: []const u32,
    shares: []const f64, // each one's score over topWeight
    at: usize, // reply tokens before it
    place: Place,
    block_tokens: usize, // every note the model sees after it, in tokens
    lookup_ns: i96,
    build_ns: i96 = 0, // the continuation's prompt rendered and its request made
    drain_ns: i96 = 0, // the cancel of the request before it, until the engine let it go
    wait_ns: i96 = 0, // the continuation submitted, until its first token (its prefill, most of it)
    prompt: usize, // the continuation's prompt tokens (the answer so far included)
    cached: ?u32 = null, // of them, restored from a kept state
    pause_ns: ?i96 = null, // last token before it to the first after it, as the reply's thread saw them
};

/// One check's best new note (0: none), and for "notes_echo" replies its first few candidates.
pub const Check = struct { at: usize, best: u32 = 0, share: f64 = 0, ids: []const u32 = &.{}, shares: []const f64 = &.{}, named: []const bool = &.{} };

/// The candidates a "notes_echo" reply lists per check.
pub const echo_top = 8;

/// What a check found: the new notes that qualify and fit, best first, and the block that holds them.
pub const Hit = struct { ids: []const u32, shares: []const f64, block: []const u8, tokens: usize, lookup_ns: i96 };
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Where new notes go: (a) into the first block, (b) a block before the assistant turn, (c) one before the question.
pub const Place = enum { first, turn, user };

/// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 24.
pub const Ask = struct {
    on: bool = true,
    place: Place = default_place,
    dry: bool = false,
    names: bool = default_names,
    checks: u32 = notes_decode.default_checks,
    budget_ms: f64 = notes_decode.default_budget_ms,
    gap: u32 = notes_decode.default_gap,
};

/// The defaults, measured (NOTES-DECODE.md): a hit's notes before the question, a name the reply writes a check point.
pub const default_place: Place = .user;
pub const default_names = true;
/// The first lookup's hop is off unless the request sends notes_value_hop: who first stays the default.
pub const default_value_hop = false;
pub const default_before_user = false;

/// A request's notes_value_hop and notes_position ("first" or "user"): how the first lookup is made and placed.
pub fn shapeAsked(cx: *Cx, body: Value) errors.Refused!notes_route.Shape {
    const on = (try notes_route.asked(cx, body)) != null;
    const hop = body.field("notes_value_hop");
    const position = body.field("notes_position");
    const subjects = body.field("notes_subjects");
    if (!on) {
        if (hop != null or position != null or subjects != null) return cx.refuse("notes_value_hop, notes_position and notes_subjects need \"notes\": true");
        return .{};
    }
    var value_hop = default_value_hop;
    if (hop) |v| {
        if (v != .bool) return cx.refuse("notes_value_hop must be a boolean");
        value_hop = v.bool;
    }
    var before = default_before_user;
    if (position) |v| {
        const s = v.str() orelse "";
        before = if (std.mem.eql(u8, s, "user")) true else if (std.mem.eql(u8, s, "first")) false else return cx.refuse("notes_position must be \"first\" or \"user\"");
    }
    var resolve_subjects = false;
    if (subjects) |v| {
        if (v != .bool) return cx.refuse("notes_subjects must be a boolean");
        resolve_subjects = v.bool;
    }
    return .{ .hop = if (value_hop) .subject_value else null, .before_user = before, .resolve_subjects = resolve_subjects };
}

/// The decode fields of a request; `notes_on`: it asked for notes, which they need.
pub fn asked(cx: *Cx, body: Value, notes_on: bool) errors.Refused!Ask {
    var ask: Ask = .{};
    const on = body.field("notes_decode");
    const place = body.field("notes_decode_place");
    const dry = body.field("notes_decode_dry");
    const names = body.field("notes_decode_names");
    const checks = body.field("notes_decode_checks");
    const budget = body.field("notes_decode_budget_ms");
    const gap = body.field("notes_decode_gap");
    if (!notes_on) {
        if (on != null or place != null or dry != null or names != null or checks != null or budget != null or gap != null) return cx.refuse("the notes_decode fields need \"notes\": true");
        return ask;
    }
    if (checks) |v| ask.checks = std.math.cast(u32, v.int64() orelse -1) orelse return cx.refuse("notes_decode_checks must be a whole number, 0 or more");
    if (gap) |v| ask.gap = std.math.cast(u32, v.int64() orelse -1) orelse return cx.refuse("notes_decode_gap must be a whole number of tokens, 0 or more");
    if (budget) |v| {
        const ms = if (v == .bool) null else v.float64();
        if (ms == null or !(ms.? >= 0 and ms.? <= 3.6e6)) return cx.refuse("notes_decode_budget_ms must be a number of milliseconds, 0 to 3600000");
        ask.budget_ms = ms.?;
    }
    if (names) |v| {
        if (v != .bool) return cx.refuse("notes_decode_names must be a boolean");
        ask.names = v.bool;
    }
    if (on) |v| {
        if (v != .bool) return cx.refuse("notes_decode must be a boolean");
        ask.on = v.bool;
    }
    if (dry) |v| {
        if (v != .bool) return cx.refuse("notes_decode_dry must be a boolean");
        ask.dry = v.bool;
    }
    if (place) |v| {
        const s = v.str() orelse "";
        ask.place = std.meta.stringToEnum(Place, s) orelse return cx.refuse("notes_decode_place must be \"first\", \"turn\" or \"user\"");
    }
    return ask;
}

/// The reply's notes.decode field.
pub fn value(l: *const Live, a: Allocator) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "on", .{ .bool = l.on });
    if (!l.on) {
        if (l.stopped) |s| try o.put(a, "stopped", .{ .string = s });
        return .{ .object = o };
    }
    try o.put(a, "place", .{ .string = @tagName(l.place) });
    try o.put(a, "names", .{ .bool = l.names });
    try o.put(a, "value_hop", .{ .bool = l.shape.hop != null });
    try o.put(a, "position", .{ .string = if (l.shape.before_user) "user" else "first" });
    if (l.dry) try o.put(a, "dry", .{ .bool = true });
    try o.put(a, "notes_decode.min_share", .{ .float = notes_decode.min_share });
    try o.put(a, "checks", try json.intValue(a, l.n_checks));
    try o.put(a, "attempts", try json.intValue(a, l.attempts));
    try o.put(a, "triggers", try json.intValue(a, l.triggers));
    try o.put(a, "pending", .{ .bool = l.pending });
    try o.put(a, "trigger_us", .{ .float = micro(l.trigger_ns) });
    try o.put(a, "scans", try json.intValue(a, l.scans));
    const limits = try json.newObject(a);
    try limits.put(a, "checks", try json.intValue(a, l.max_checks));
    try limits.put(a, "budget_ms", .{ .float = micro(l.budget_ns) / 1e3 });
    try limits.put(a, "gap", try json.intValue(a, l.gap));
    try o.put(a, "limits", .{ .object = limits });
    try o.put(a, "capped", try json.intValue(a, l.capped));
    try o.put(a, "check_us", .{ .float = micro(l.check_ns) });
    try o.put(a, "best_share", .{ .float = l.best.share });
    try o.put(a, "best_id", if (l.best.best > 0) try json.intValue(a, l.best.best) else .null);
    const list = try a.alloc(Value, l.injections.items.len);
    for (l.injections.items, list) |*inj, *slot| slot.* = try injectionValue(a, inj);
    try o.put(a, "injected", .{ .array = list });
    try o.put(a, "ids", try intsValue(a, l.held.items));
    try o.put(a, "stopped", if (l.stopped) |s| .{ .string = s } else .null);
    if (!l.echo) return .{ .object = o };
    const checks = try a.alloc(Value, l.checks.items.len);
    for (l.checks.items, checks) |c, *slot| {
        const x = try json.newObject(a);
        try x.put(a, "at", try json.intValue(a, c.at));
        try x.put(a, "best_id", if (c.best > 0) try json.intValue(a, c.best) else .null);
        try x.put(a, "share", .{ .float = c.share });
        try x.put(a, "ids", try intsValue(a, c.ids));
        const ln = try a.alloc(Value, c.named.len);
        for (c.named, ln) |b, *v| v.* = .{ .bool = b };
        try x.put(a, "named", .{ .array = ln });
        const sh = try a.alloc(Value, c.shares.len);
        for (c.shares, sh) |f, *v| v.* = .{ .float = f };
        try x.put(a, "shares", .{ .array = sh });
        slot.* = .{ .object = x };
    }
    try o.put(a, "check_list", .{ .array = checks });
    try o.put(a, "block", .{ .string = if (l.place == .first) l.block else l.late_block });
    return .{ .object = o };
}

fn micro(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}

fn intsValue(a: Allocator, xs: []const u32) Allocator.Error!Value {
    const out = try a.alloc(Value, xs.len);
    for (xs, out) |x, *slot| slot.* = try json.intValue(a, x);
    return .{ .array = out };
}

fn injectionValue(a: Allocator, inj: *const Injection) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "ids", try intsValue(a, inj.ids));
    const shares = try a.alloc(Value, inj.shares.len);
    for (inj.shares, shares) |s, *slot| slot.* = .{ .float = s };
    try o.put(a, "shares", .{ .array = shares });
    try o.put(a, "at_token", try json.intValue(a, inj.at));
    try o.put(a, "place", .{ .string = @tagName(inj.place) });
    try o.put(a, "block_tokens", try json.intValue(a, inj.block_tokens));
    try o.put(a, "lookup_us", .{ .float = micro(inj.lookup_ns) });
    try o.put(a, "prompt_tokens", try json.intValue(a, inj.prompt));
    try o.put(a, "cached_tokens", if (inj.cached) |c| try json.intValue(a, c) else .null);
    try o.put(a, "pause_ms", if (inj.pause_ns) |p| .{ .float = micro(p) / 1e3 } else .null);
    try o.put(a, "build_us", .{ .float = micro(inj.build_ns) });
    try o.put(a, "drain_ms", .{ .float = micro(inj.drain_ns) / 1e3 });
    try o.put(a, "wait_ms", .{ .float = micro(inj.wait_ns) / 1e3 });
    return .{ .object = o };
}

/// The server's line for one injection: which notes, after which reply token, and the pause it made.
pub fn logLine(id: []const u8, inj: *const Injection) void {
    var buf: [200]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    for (inj.ids, 0..) |n, k| w.print("{s}{d}", .{ if (k > 0) "," else "", n }) catch break;
    const cached = inj.cached orelse 0;
    log.line("notes decode {s}: [{s}] put in after reply token {d} ({s}), pause {d:.1} ms (lookup {d:.1} us, render {d:.1} us, cancel {d:.1} ms, prefill and first token {d:.1} ms); prompt {d} tokens, {d} cached, {d} prefilled; notes {d} tokens", .{
        id,                        w.buffered(),             inj.at,     @tagName(inj.place), micro(inj.pause_ns orelse 0) / 1e3, micro(inj.lookup_ns), micro(inj.build_ns),
        micro(inj.drain_ns) / 1e3, micro(inj.wait_ns) / 1e3, inj.prompt, cached,              inj.prompt -| cached,               inj.block_tokens,
    });
}
