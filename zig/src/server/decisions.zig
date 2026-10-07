//! /v1/decisions checks a body the way 0.6.6 checks it, then scores through the engine hook when a family sets one.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const pyrepr = @import("pyrepr.zig");
const reply_text = @import("reply_text.zig");
const model_text = @import("model_text.zig");
const http_body = @import("http_body.zig");
const openai = @import("openai.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

const top_fields = [_][]const u8{ "input", "questions", "temperature", "chat_template_kwargs", "prompt_format_version", "return_prompt_token_ids", "model" };
const choice_fields = [_][]const u8{ "id", "type", "question", "options" };
const score_fields = [_][]const u8{ "id", "type", "question", "levels" };
const yes_no_fields = [_][]const u8{ "id", "type", "question", "yes", "no" };
const option_fields = [_][]const u8{ "name", "description" };
const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const Kind = enum { choice, score, yes_no };

const Prepared = struct {
    id: []const u8,
    kind: Kind,
    names: []const []const u8,
    prompt_ids: []const u32,
    label_ids: []const u32,
};

const Scored = struct { logits: []f64, logsumexp: f64 };

pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    var cx: Cx = .{ .a = a };
    const raw = http_body.readJson(conn, &cx) catch return sendErr(conn, a, &cx);
    const payload = reply(srv, &cx, raw) catch |e| {
        if (e == error.OutOfMemory) {
            cx.kind = .server;
            cx.message = "out of memory";
        }
        return sendErr(conn, a, &cx);
    };
    routes.sendValue(conn, a, 200, payload);
}

fn sendErr(conn: *Conn, a: Allocator, cx: *Cx) void {
    // This route's client errors all name the type, including a body that is not an object.
    if (cx.kind != .server) cx.kind = .request;
    const err = openai.errorBody(a, cx, null) catch return;
    routes.sendValue(conn, a, cx.status(), openai.wrapError(a, err) catch return);
}

/// The decisions object, or the refusal 0.6.6 would send for this body.
pub fn reply(srv: *Server, cx: *Cx, raw: Value) errors.Refused!Value {
    const body = try fields.parseNumbers(cx, raw);
    try validate(cx, body);
    const input = try renderText(cx, body.get("input"));
    if (reply_text.pyStrip(input).len == 0) return cx.refuse("input must not be blank");
    const questions = body.get("questions").?.array;
    var prepared: std.ArrayList(Prepared) = .empty;
    for (questions, 0..) |question, index| {
        const item = prepareOne(srv, cx, input, question) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => {
                const inner = cx.message;
                return cx.fail(.request, "question {s}: {s}", .{ try questionWhere(cx, question, index), inner });
            },
        };
        try prepared.append(cx.a, item);
    }
    return build(cx, body, prepared.items, try scoreAll(srv, cx, prepared.items));
}

fn validate(cx: *Cx, body: Value) errors.Refused!void {
    try rejectUnknown(cx, body.object, &top_fields);
    if (body.get("prompt_format_version")) |version| if (version != .null and !json.equal(version, .{ .int = "1" })) {
        return cx.fail(.request, "prompt_format_version {s} is not served, this server uses version 1", .{try shown(cx.a, version)});
    };
    try checkTemperature(cx, body);
    const kwargs = body.get("chat_template_kwargs") orelse .null;
    if (kwargs != .null) {
        if (kwargs != .object) return cx.refuse("chat_template_kwargs must be an object");
        try rejectUnknown(cx, kwargs.object, &.{"enable_thinking"});
        if (kwargs.get("enable_thinking")) |flag| if (!(flag == .bool and !flag.bool)) return cx.refuse("decisions need enable_thinking false or unset");
    }
    const questions = body.get("questions") orelse return cx.refuse("questions must contain at least one question");
    if (questions != .array or questions.array.len == 0) return cx.refuse("questions must contain at least one question");
    var seen: std.ArrayList([]const u8) = .empty;
    for (questions.array) |question| {
        if (question != .object) return cx.refuse("each question must be an object");
        const ident = question.get("id") orelse return cx.refuse("a question id must not be blank");
        if (ident != .string or reply_text.pyStrip(ident.string).len == 0) return cx.refuse("a question id must not be blank");
        for (seen.items) |old| if (std.mem.eql(u8, old, ident.string)) return cx.fail(.request, "question id {s} is repeated", .{try pyrepr.repr(cx.a, ident)});
        try seen.append(cx.a, ident.string);
    }
}

// Numeric strings are rewritten before this check. A non-positive number still uses the decisions sentence.
fn checkTemperature(cx: *Cx, body: Value) errors.Refused!void {
    const v = body.get("temperature") orelse return;
    if (v == .bool or v.float64() == null) return cx.refuse("temperature must be a number above 0");
    const n = v.float64().?;
    if (!std.math.isFinite(n) or n <= 0) return cx.refuse("temperature must be a number above 0");
}

fn rejectUnknown(cx: *Cx, obj: *json.Object, allowed: []const []const u8) errors.Refused!void {
    var bad: std.ArrayList([]const u8) = .empty;
    for (obj.keys()) |key| {
        const known = for (allowed) |name| {
            if (std.mem.eql(u8, name, key)) break true;
        } else false;
        if (!known) try bad.append(cx.a, key);
    }
    if (bad.items.len == 0) return;
    std.mem.sort([]const u8, bad.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.less);
    return cx.fail(.request, "unknown field {s}", .{try pyrepr.repr(cx.a, .{ .string = bad.items[0] })});
}

fn shown(a: Allocator, v: Value) Allocator.Error![]const u8 {
    return switch (v) {
        .null => "None",
        .bool => |b| if (b) "True" else "False",
        .int => |t| t,
        .string => |s| s,
        .float => |f| blk: {
            var buf: [64]u8 = undefined;
            break :blk try a.dupe(u8, json.floatRepr(&buf, f));
        },
        else => pyrepr.repr(a, v),
    };
}

fn renderText(cx: *Cx, v: ?Value) errors.Refused![]const u8 {
    const value = v orelse return "";
    return switch (value) {
        .null => "",
        .string => |s| s,
        .object, .array => json.stringify(cx.a, value, .{ .ascii = false, .compact = true }),
        else => cx.refuse("text must be a string, object, or array"),
    };
}

fn questionWhere(cx: *Cx, question: Value, index: usize) Allocator.Error![]const u8 {
    if (question == .object) if (question.get("id")) |id| if (id == .string and id.string.len > 0) return pyrepr.repr(cx.a, id);
    return std.fmt.allocPrint(cx.a, "at position {d}", .{index});
}

const Wording = struct {
    kind: Kind,
    names: []const []const u8,
    labels: []const []const u8,
    content: []const u8,
};

fn wording(cx: *Cx, input: []const u8, question: Value) errors.Refused!Wording {
    const kind_value = question.get("type") orelse .null;
    const kind: Kind = if (kind_value == .string and std.mem.eql(u8, kind_value.string, "choice"))
        .choice
    else if (kind_value == .string and std.mem.eql(u8, kind_value.string, "score"))
        .score
    else if (kind_value == .string and std.mem.eql(u8, kind_value.string, "yes_no"))
        .yes_no
    else
        return cx.fail(.request, "unknown question type {s}", .{try pyrepr.repr(cx.a, kind_value)});
    var lines: std.ArrayList([]const u8) = .empty;
    const names, const labels = switch (kind) {
        .choice => blk: {
            try rejectUnknown(cx, question.object, &choice_fields);
            const listed = try choiceList(cx, question);
            try lines.append(cx.a, try questionLine(cx, question));
            for (listed.labels, listed.names, listed.details) |label, name, detail| {
                const line = if (detail.len > 0)
                    try std.fmt.allocPrint(cx.a, "{s}: {s} - {s}", .{ label, name, detail })
                else
                    try std.fmt.allocPrint(cx.a, "{s}: {s}", .{ label, name });
                try lines.append(cx.a, line);
            }
            break :blk .{ listed.names, listed.labels };
        },
        .score => blk: {
            try rejectUnknown(cx, question.object, &score_fields);
            const listed = try scoreList(cx, question);
            try lines.append(cx.a, try questionLine(cx, question));
            for (listed.labels, listed.details) |label, detail| try lines.append(cx.a, try std.fmt.allocPrint(cx.a, "{s}: {s}", .{ label, detail }));
            break :blk .{ listed.names, listed.labels };
        },
        .yes_no => blk: {
            try rejectUnknown(cx, question.object, &yes_no_fields);
            const yes = try renderText(cx, question.get("yes"));
            const no = try renderText(cx, question.get("no"));
            const lead = try renderText(cx, question.get("question"));
            if (reply_text.pyStrip(lead).len == 0) return cx.refuse("a question must not be blank");
            try lines.append(cx.a, try std.fmt.allocPrint(cx.a, "Is the following true? {s}", .{lead}));
            if (yes.len > 0) try lines.append(cx.a, try std.fmt.allocPrint(cx.a, "yes: {s}", .{yes}));
            if (no.len > 0) try lines.append(cx.a, try std.fmt.allocPrint(cx.a, "no: {s}", .{no}));
            const names = [_][]const u8{ "yes", "no" };
            break :blk .{ try cx.a.dupe([]const u8, &names), try cx.a.dupe([]const u8, &names) };
        },
    };
    const closing = switch (kind) {
        .choice => "Answer with the letter of one option only.",
        .score => "Answer with the number of one level only.",
        .yes_no => "Answer with yes or no only.",
    };
    try lines.append(cx.a, closing);
    return .{ .kind = kind, .names = names, .labels = labels, .content = try contentOf(cx.a, input, lines.items) };
}

const Listed = struct {
    names: []const []const u8,
    details: []const []const u8,
    labels: []const []const u8,
};

fn choiceList(cx: *Cx, question: Value) errors.Refused!Listed {
    const options = question.get("options") orelse return cx.refuse("a choice needs 2 to 26 options");
    if (options != .array or options.array.len < 2 or options.array.len > 26) return cx.refuse("a choice needs 2 to 26 options");
    const n = options.array.len;
    const names = try cx.a.alloc([]const u8, n);
    const details = try cx.a.alloc([]const u8, n);
    const labels = try cx.a.alloc([]const u8, n);
    var seen: std.ArrayList([]const u8) = .empty;
    for (options.array, 0..) |option, i| {
        if (option != .object) return cx.refuse("each option must be an object");
        try rejectUnknown(cx, option.object, &option_fields);
        const name = option.get("name") orelse return cx.refuse("option names must not be blank or contain line breaks");
        if (name != .string or badName(name.string)) return cx.refuse("option names must not be blank or contain line breaks");
        const stripped = reply_text.pyStrip(name.string);
        if (try foldedSeen(cx, &seen, stripped)) return cx.fail(.request, "option name {s} repeats another name", .{try pyrepr.repr(cx.a, name)});
        names[i] = stripped;
        details[i] = try renderText(cx, option.get("description"));
        labels[i] = alphabet[i .. i + 1];
    }
    return .{ .names = names, .details = details, .labels = labels };
}

fn badName(name: []const u8) bool {
    const stripped = reply_text.pyStrip(name);
    if (stripped.len == 0) return true;
    for (stripped) |c| if (c < 32) return true;
    return false;
}

fn foldedSeen(cx: *Cx, seen: *std.ArrayList([]const u8), name: []const u8) errors.Refused!bool {
    const folded = try cx.a.alloc(u8, name.len);
    for (name, folded) |c, *d| d.* = std.ascii.toLower(c);
    for (seen.items) |old| if (std.mem.eql(u8, old, folded)) return true;
    try seen.append(cx.a, folded);
    return false;
}

fn scoreList(cx: *Cx, question: Value) errors.Refused!Listed {
    const levels = question.get("levels") orelse return cx.refuse("a score needs 2 to 10 levels");
    if (levels != .array or levels.array.len < 2 or levels.array.len > 10) return cx.refuse("a score needs 2 to 10 levels");
    const n = levels.array.len;
    const names = try cx.a.alloc([]const u8, n);
    const details = try cx.a.alloc([]const u8, n);
    for (levels.array, 0..) |level, i| {
        const text = try renderText(cx, level);
        if (reply_text.pyStrip(text).len == 0) return cx.refuse("a level must not be blank");
        details[i] = text;
        names[i] = try std.fmt.allocPrint(cx.a, "{d}", .{i});
    }
    return .{ .names = names, .details = details, .labels = names };
}

fn questionLine(cx: *Cx, question: Value) errors.Refused![]const u8 {
    const text = try renderText(cx, question.get("question"));
    if (reply_text.pyStrip(text).len == 0) return cx.refuse("a question must not be blank");
    return std.fmt.allocPrint(cx.a, "Question: {s}", .{text});
}

fn contentOf(a: Allocator, input: []const u8, lines: []const []const u8) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, input);
    try out.append(a, '\n');
    try out.append(a, '\n');
    for (lines, 0..) |line, i| {
        if (i > 0) try out.append(a, '\n');
        try out.appendSlice(a, line);
    }
    return out.toOwnedSlice(a);
}

fn prepareOne(srv: *Server, cx: *Cx, input: []const u8, question: Value) errors.Refused!Prepared {
    const word = try wording(cx, input, question);
    const prompt_text = try renderPrompt(srv, cx, word.content);
    const prompt_ids = try encodePrompt(srv, cx, prompt_text);
    try thinkOpen(cx, prompt_text);
    try fits(cx, srv.info.context_window, prompt_ids.len);
    return .{
        .id = question.strField("id") orelse "",
        .kind = word.kind,
        .names = word.names,
        .prompt_ids = prompt_ids,
        .label_ids = try labelIds(srv, cx, prompt_text, prompt_ids, word.labels),
    };
}

fn renderPrompt(srv: *Server, cx: *Cx, content: []const u8) errors.Refused![]const u8 {
    const user = try json.newObject(cx.a);
    try user.put(cx.a, "role", .{ .string = "user" });
    try user.put(cx.a, "content", .{ .string = content });
    const messages = try cx.a.alloc(Value, 1);
    messages[0] = .{ .object = user };
    var problem: []const u8 = "";
    return srv.text.render(cx.a, .{ .array = messages }, .{ .add_generation_prompt = true, .enable_thinking = false }, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.request, "the chat template failed: {s}", .{problem}),
    };
}

fn encodePrompt(srv: *Server, cx: *Cx, text: []const u8) errors.Refused![]u32 {
    return srv.text.encode(cx.a, text, false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.request, "the chat template failed: {s}", .{"the tokenizer cannot encode this prompt"}),
    };
}

fn thinkOpen(cx: *Cx, text: []const u8) errors.Refused!void {
    const open_at = std.mem.lastIndexOf(u8, text, "<think>") orelse return;
    if (std.mem.lastIndexOf(u8, text, "</think>")) |close_at| if (open_at <= close_at) return;
    return cx.refuse("the chat template leaves a reasoning block open at the answer position");
}

fn fits(cx: *Cx, window: u32, n: usize) errors.Refused!void {
    if (window > 0 and n >= window) return cx.fail(.request, "the prompt has {d} tokens, which does not fit the context length of {d} tokens", .{ n, window });
}

fn labelIds(srv: *Server, cx: *Cx, prompt: []const u8, prompt_ids: []const u32, labels: []const []const u8) errors.Refused![]u32 {
    var found: std.ArrayList(u32) = .empty;
    for (labels) |label| {
        const joined = try std.mem.concat(cx.a, u8, &.{ prompt, label });
        const ids = srv.text.encode(cx.a, joined, false) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return labelRefuse(cx, label),
        };
        const fresh = ids.len == prompt_ids.len + 1 and std.mem.eql(u32, ids[0..prompt_ids.len], prompt_ids) and std.mem.indexOfScalar(u32, found.items, ids[ids.len - 1]) == null;
        if (!fresh) return labelRefuse(cx, label);
        try found.append(cx.a, ids[ids.len - 1]);
    }
    return found.toOwnedSlice(cx.a);
}

fn labelRefuse(cx: *Cx, label: []const u8) errors.Refused {
    return cx.fail(.request, "the answer label {s} is not one distinct token after the chat prompt for this tokenizer, so this model is not supported", .{pyrepr.repr(cx.a, .{ .string = label }) catch label});
}

fn scoreAll(srv: *Server, cx: *Cx, prepared: []const Prepared) errors.Refused![]Scored {
    const scored = try cx.a.alloc(Scored, prepared.len);
    for (prepared, scored) |item, *row| {
        const logits = try cx.a.alloc(f64, item.label_ids.len);
        @memset(logits, 0);
        const logsumexp = srv.engine.score(item.prompt_ids, item.label_ids, logits) catch |e| switch (e) {
            error.Unsupported => return cx.refuse("decision scoring is not supported by this engine yet"),
            error.Failed => return cx.fail(.server, "decision scoring failed", .{}),
        };
        row.* = .{ .logits = logits, .logsumexp = logsumexp };
    }
    return scored;
}

fn build(cx: *Cx, body: Value, prepared: []const Prepared, scored: []const Scored) errors.Refused!Value {
    const a = cx.a;
    const temperature = if (body.get("temperature")) |v| v.float64().? else 1;
    const show_ids = if (body.get("return_prompt_token_ids")) |v| v.truthy() else false;
    const answers = try json.newObject(a);
    var prompt_tokens: usize = 0;
    for (prepared, scored) |item, row| {
        const probs = try a.alloc(f64, row.logits.len);
        softmax(row.logits, temperature, probs);
        const one = try json.newObject(a);
        try one.put(a, "type", .{ .string = @tagName(item.kind) });
        const map = try json.newObject(a);
        for (item.names, probs) |name, p| try map.put(a, name, .{ .float = p });
        try one.put(a, "probabilities", .{ .object = map });
        try one.put(a, "label_mass", .{ .float = labelMass(row.logits, row.logsumexp) });
        switch (item.kind) {
            .choice => {
                var best: usize = 0;
                for (probs, 0..) |p, i| if (p > probs[best]) {
                    best = i;
                };
                try one.put(a, "choice", .{ .string = item.names[best] });
            },
            .score => {
                var total: f64 = 0;
                for (probs, 0..) |p, i| total += @as(f64, @floatFromInt(i)) * p;
                try one.put(a, "score", .{ .float = total });
            },
            .yes_no => {},
        }
        if (show_ids) {
            try one.put(a, "prompt_token_ids", try idsValue(a, item.prompt_ids));
            try one.put(a, "label_token_ids", try idsValue(a, item.label_ids));
        }
        try answers.put(a, item.id, .{ .object = one });
        prompt_tokens += item.prompt_ids.len;
    }
    const named = body.get("model") orelse .null;
    const usage = try json.newObject(a);
    try usage.put(a, "prompt_tokens", try json.intValue(a, prompt_tokens));
    try usage.put(a, "completion_tokens", .{ .int = "0" });
    try usage.put(a, "total_tokens", try json.intValue(a, prompt_tokens));
    const out = try json.newObject(a);
    try out.put(a, "object", .{ .string = "decisions" });
    try out.put(a, "model", if (named.truthy()) named else .{ .string = "default" });
    try out.put(a, "prompt_format_version", .{ .int = "1" });
    try out.put(a, "answers", .{ .object = answers });
    try out.put(a, "usage", .{ .object = usage });
    return .{ .object = out };
}

fn softmax(logits: []const f64, temperature: f64, out: []f64) void {
    var peak = logits[0];
    for (logits[1..]) |x| peak = @max(peak, x);
    var total: f64 = 0;
    for (logits, out) |x, *p| {
        p.* = @exp((x - peak) / temperature);
        total += p.*;
    }
    for (out) |*p| p.* /= total;
}

fn labelMass(logits: []const f64, logsumexp: f64) f64 {
    var total: f64 = 0;
    for (logits) |logit| total += @exp(logit - logsumexp);
    return total;
}

fn idsValue(a: Allocator, ids: []const u32) Allocator.Error!Value {
    const items = try a.alloc(Value, ids.len);
    for (ids, items) |id, *slot| slot.* = try json.intValue(a, id);
    return .{ .array = items };
}
