//! Request field checks shared by every route: numbers, stops, thinking switches, efforts, probabilities.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const efforts = [_][]const u8{ "none", "minimal", "low", "medium", "high", "xhigh", "max" };
const effort_order = [_][]const u8{ "max", "xhigh", "high", "medium", "low", "minimal" }; // highest first

/// Python's type name for ``'<type>' object has no attribute 'get'``.
pub fn typeName(v: Value) []const u8 {
    return switch (v) {
        .null => "NoneType",
        .bool => "bool",
        .int => "int",
        .float => "float",
        .string => "str",
        .array => "list",
        .object => "dict",
    };
}

/// ``fields.get`` on a body that is not an object fails as Python's AttributeError does.
pub fn requireObject(cx: *Cx, v: Value) errors.Refused!void {
    if (v != .object) return cx.fail(.other, "'{s}' object has no attribute 'get'", .{typeName(v)});
}

pub const Stops = struct { ignore_eos: bool = false, strings: []const []const u8 = &.{} };

/// ``stop_options``: ignore_eos must be a bool when present; stop a string, a list of them, or null.
pub fn stopOptions(cx: *Cx, fields: Value) errors.Refused!Stops {
    var out: Stops = .{};
    if (fields.get("ignore_eos")) |ignore| {
        if (ignore != .bool) return cx.refuse("ignore_eos must be a boolean");
        out.ignore_eos = ignore.bool;
    }
    const stops = fields.field("stop") orelse return out;
    if (stops == .string) {
        if (stops.string.len == 0) return cx.refuse("stop must be a nonempty string, a list of nonempty strings, or null");
        const one = try cx.a.alloc([]const u8, 1);
        one[0] = stops.string;
        out.strings = one;
        return out;
    }
    if (stops != .array) return cx.refuse("stop must be a nonempty string, a list of nonempty strings, or null");
    const list = try cx.a.alloc([]const u8, stops.array.len);
    for (stops.array, list) |s, *slot| {
        if (s != .string or s.string.len == 0) return cx.refuse("stop must be a nonempty string, a list of nonempty strings, or null");
        slot.* = s.string;
    }
    out.strings = list;
    return out;
}

const integer_fields = [_][]const u8{ "max_completion_tokens", "max_tokens", "seed", "thinking_budget", "top_k" };

/// ``parse_numbers``: the body with its numeric fields as numbers, or Python's refusal.
pub fn parseNumbers(cx: *Cx, fields: Value) errors.Refused!Value {
    try requireObject(cx, fields);
    _ = try stopOptions(cx, fields);
    const parsed = try json.copyObject(cx.a, fields.object);
    for (integer_fields ++ [_][]const u8{ "temperature", "top_p", "min_p" }) |name| {
        const value = fields.field(name) orelse continue;
        const integer = for (integer_fields) |f| {
            if (std.mem.eql(u8, f, name)) break true;
        } else false;
        const number: ?Value = if (integer) toInt(cx.a, value) catch null else toFloat(value);
        const n = number orelse return cx.fail(.request, "{s} must be {s} or null", .{ name, if (integer) "an integer" else "a finite number" });
        if (std.mem.eql(u8, name, "min_p") and !(n.float >= 0 and n.float <= 1)) return cx.refuse("min_p must be between 0 and 1, or null");
        const stored = if (std.mem.eql(u8, name, "top_k") and n.int[0] == '-') Value{ .int = "0" } else n;
        try parsed.put(cx.a, name, stored);
    }
    return .{ .object = parsed };
}

/// ``int(value)`` for an int, an integral float or a numeric string; null where Python raises.
fn toInt(a: Allocator, v: Value) Allocator.Error!?Value {
    switch (v) {
        .int => return v,
        .float => |f| {
            if (!std.math.isFinite(f) or f != @trunc(f)) return null;
            if (@abs(f) < 1e38) return .{ .int = try std.fmt.allocPrint(a, "{d}", .{@as(i128, @intFromFloat(f))}) };
            return null;
        },
        .string => |s| return if (try pyIntText(a, s)) |t| Value{ .int = t } else null,
        else => return null,
    }
}

/// ``float(value)`` kept only when finite.
fn toFloat(v: Value) ?Value {
    const f: f64 = switch (v) {
        .int => |t| std.fmt.parseFloat(f64, t) catch return null,
        .float => |f| f,
        .string => |s| pyFloat(s) orelse return null,
        else => return null,
    };
    return if (std.math.isFinite(f)) Value{ .float = f } else null;
}

fn pyStrip(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f");
}

/// Digits with single underscores between them, as Python's int() and float() accept.
fn digitRun(s: []const u8, out: *std.ArrayList(u8), a: Allocator) Allocator.Error!bool {
    if (s.len == 0 or !std.ascii.isDigit(s[0]) or !std.ascii.isDigit(s[s.len - 1])) return false;
    for (s, 0..) |ch, i| {
        if (ch == '_') {
            if (s[i - 1] == '_') return false;
            continue;
        }
        if (!std.ascii.isDigit(ch)) return false;
        try out.append(a, ch);
    }
    return true;
}

/// ``int(text)`` as canonical decimal text, or null.
pub fn pyIntText(a: Allocator, text: []const u8) Allocator.Error!?[]const u8 {
    var s = pyStrip(text);
    var negative = false;
    if (s.len > 0 and (s[0] == '-' or s[0] == '+')) {
        negative = s[0] == '-';
        s = s[1..];
    }
    var digits: std.ArrayList(u8) = .empty;
    if (!try digitRun(s, &digits, a)) return null;
    const trimmed = std.mem.trimStart(u8, digits.items, "0");
    if (trimmed.len == 0) return "0";
    return if (negative) try std.mem.concat(a, u8, &.{ "-", trimmed }) else trimmed;
}

/// ``float(text)``, or null.
pub fn pyFloat(text: []const u8) ?f64 {
    var s = pyStrip(text);
    var sign: f64 = 1;
    if (s.len > 0 and (s[0] == '-' or s[0] == '+')) {
        if (s[0] == '-') sign = -1;
        s = s[1..];
    }
    if (std.ascii.eqlIgnoreCase(s, "inf") or std.ascii.eqlIgnoreCase(s, "infinity")) return sign * std.math.inf(f64);
    if (std.ascii.eqlIgnoreCase(s, "nan")) return std.math.nan(f64);
    if (s.len > 100) return null;
    var buf: [256]u8 = undefined;
    var list: std.ArrayList(u8) = .initBuffer(&buf);
    const e_at = std.mem.indexOfAny(u8, s, "eE") orelse s.len;
    const mantissa = s[0..e_at];
    const dot = std.mem.indexOfScalar(u8, mantissa, '.');
    const whole = if (dot) |d| mantissa[0..d] else mantissa;
    const frac = if (dot) |d| mantissa[d + 1 ..] else "";
    if (whole.len == 0 and frac.len == 0) return null;
    if (whole.len > 0 and !digitsInto(whole, &list)) return null;
    if (list.items.len == 0) list.appendAssumeCapacity('0');
    list.appendAssumeCapacity('.');
    if (frac.len > 0 and !digitsInto(frac, &list)) return null;
    list.appendAssumeCapacity('0');
    if (e_at < s.len) {
        var e = s[e_at + 1 ..];
        list.appendAssumeCapacity('e');
        if (e.len > 0 and (e[0] == '-' or e[0] == '+')) {
            list.appendAssumeCapacity(e[0]);
            e = e[1..];
        }
        if (!digitsInto(e, &list)) return null;
    }
    return sign * (std.fmt.parseFloat(f64, list.items) catch return null);
}

fn digitsInto(s: []const u8, out: *std.ArrayList(u8)) bool {
    if (s.len == 0 or !std.ascii.isDigit(s[0]) or !std.ascii.isDigit(s[s.len - 1])) return false;
    for (s, 0..) |ch, i| {
        if (ch == '_') {
            if (s[i - 1] == '_') return false;
            continue;
        }
        if (!std.ascii.isDigit(ch)) return false;
        out.appendAssumeCapacity(ch);
    }
    return true;
}

/// ``chat_template_kwargs.thinking`` as a switch: a bool, or DeepSeek's ``{"type": "enabled" | "disabled"}``.
fn thinkingSwitch(v: ?Value) ?bool {
    const value = v orelse return null;
    if (value == .bool) return value.bool;
    if (value == .object) if (value.get("type")) |t| if (t == .string) {
        if (std.mem.eql(u8, t.string, "enabled")) return true;
        if (std.mem.eql(u8, t.string, "disabled")) return false;
    };
    return null;
}

pub const Thinking = struct { effort: ?[]const u8 = null, enable: ?bool = null };

/// ``thinking_fields``: a request's effort and thinking switch where it sets them.
pub fn thinkingFields(cx: *Cx, body: Value, levels: []const []const u8) errors.Refused!Thinking {
    var out: Thinking = .{};
    const raw = body.get("chat_template_kwargs");
    const kwargs: ?Value = if (raw != null and raw.?.truthy()) raw.? else null;
    const kw_object = kwargs == null or kwargs.? == .object;
    var enable: ?Value = if (kwargs) |k| k.get("enable_thinking") else null;
    const has_enable = kwargs != null and kwargs.?.has("enable_thinking");
    if (kw_object and !has_enable) if (thinkingSwitch(if (kwargs) |k| k.get("thinking") else null)) |sw| {
        enable = .{ .bool = sw };
    };
    var effort = body.field("reasoning_effort");
    if (effort == null and kw_object) if (kwargs) |k| {
        effort = k.field("reasoning_effort");
    };
    if (effort) |e| {
        if (e != .string or !isEffort(e.string)) return cx.refuse("reasoning_effort must be none, minimal, low, medium, high, xhigh or max");
        out.effort = coerceEffort(e.string, levels);
        out.enable = !std.mem.eql(u8, e.string, "none");
    }
    if (kw_object) if (enable) |on| {
        out.enable = on.truthy();
        if (out.enable.? and out.effort != null and std.mem.eql(u8, out.effort.?, "none")) out.effort = null;
    };
    return out;
}

fn isEffort(s: []const u8) bool {
    for (efforts) |e| if (std.mem.eql(u8, e, s)) return true;
    return false;
}

fn order(name: []const u8) ?usize {
    for (effort_order, 0..) |e, i| if (std.mem.eql(u8, e, name)) return i;
    return null;
}

/// The nearest level the template names, ties going higher; xhigh and none stay.
pub fn coerceEffort(effort: []const u8, levels: []const []const u8) []const u8 {
    if (levels.len == 0) {
        if (std.mem.eql(u8, effort, "high") or std.mem.eql(u8, effort, "max")) return "xhigh";
        if (std.mem.eql(u8, effort, "minimal")) return "low";
        return effort;
    }
    for (levels) |l| if (std.mem.eql(u8, l, effort)) return effort;
    if (std.mem.eql(u8, effort, "xhigh") or std.mem.eql(u8, effort, "none")) return effort;
    const want = order(effort) orelse return effort;
    var best: []const u8 = effort;
    var best_key: [2]usize = .{ std.math.maxInt(usize), std.math.maxInt(usize) };
    for (levels) |l| {
        const at = order(l) orelse continue;
        const key: [2]usize = .{ if (at > want) at - want else want - at, at };
        if (key[0] < best_key[0] or (key[0] == best_key[0] and key[1] < best_key[1])) {
            best = l;
            best_key = key;
        }
    }
    return best;
}

/// The efforts a chat template names in quotes (``effort_levels``).
pub fn effortLevels(a: Allocator, template: []const u8) Allocator.Error![]const []const u8 {
    var found: std.ArrayList([]const u8) = .empty;
    for (effort_order) |name| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, template, at, name)) |i| : (at = i + 1) {
            if (i == 0 or i + name.len >= template.len) continue;
            const open = template[i - 1];
            const close = template[i + name.len];
            if ((open == '\'' or open == '"') and (close == '\'' or close == '"')) {
                try found.append(a, name);
                break;
            }
        }
    }
    return found.items;
}

/// ``probability_options``: n must be 1; the ``top_logprobs`` count when logprobs are asked and ``supported``.
pub fn probabilityOptions(cx: *Cx, body: Value, supported: bool) errors.Refused!?u8 {
    if (body.field("n")) |n| if (n != .int or !std.mem.eql(u8, n.int, "1")) return cx.refuse("n must be 1; multiple choices are not supported");
    const enabled = body.field("logprobs");
    if (enabled) |e| if (e != .bool) return cx.refuse("logprobs must be a boolean or null");
    var count: u8 = 0;
    if (body.field("top_logprobs")) |top| {
        const n = top.int64();
        if (n == null or n.? < 0 or n.? > 20) return cx.refuse("top_logprobs must be an integer between 0 and 20, or null");
        if (enabled == null or !enabled.?.bool) return cx.refuse("top_logprobs requires logprobs: true");
        count = @intCast(n.?);
    }
    if (enabled == null or !enabled.?.bool) return null;
    if (!supported) return cx.refuse("logprobs are not supported by this model or backend");
    return count;
}

const media = [_][]const u8{ "image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url" };

/// ``validate_modalities``: text in, text out.
pub fn validateModalities(cx: *Cx, body: Value) errors.Refused!void {
    for (media) |k| if (json.truthyField(body, k)) return cx.refuse("this server accepts and produces text only; image, audio and video are unsupported");
    const modalities = body.field("modalities") orelse return;
    if (modalities == .array and modalities.array.len == 1 and modalities.array[0] == .string and std.mem.eql(u8, modalities.array[0].string, "text")) return;
    return cx.refuse("this server accepts and produces text only; image, audio and video are unsupported");
}

pub fn hasMedia(message: Value) bool {
    for (media) |k| if (json.truthyField(message, k)) return true;
    return false;
}

test "efforts and numbers" {
    const levels = [_][]const u8{ "low", "medium", "xhigh" };
    try std.testing.expectEqualStrings("xhigh", coerceEffort("high", &levels));
    try std.testing.expectEqualStrings("low", coerceEffort("minimal", &levels));
    try std.testing.expectEqualStrings("xhigh", coerceEffort("max", &.{}));
    try std.testing.expectEqual(@as(?f64, 1000.5), pyFloat(" 1_000.5 "));
    try std.testing.expect(pyFloat("1__0") == null and pyFloat(".") == null);
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("-12", (try pyIntText(arena.allocator(), " -0_12 ")).?);
    try std.testing.expect(try pyIntText(arena.allocator(), "8.0") == null);
}
