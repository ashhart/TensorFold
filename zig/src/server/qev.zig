//! /v1/decisions and /v1/systemone in oMLX's shapes for a slot_lm decision model (elephant-1's readout.json).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const openai = @import("openai.zig");
const routes = @import("routes.zig");
const prompt = @import("qev_prompt.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

const max_questions = 200;
const max_choices = 255;
const max_levels = 10;
const max_branch_tokens = 32768;
const max_request_tokens = 65536;

/// Whether this server's model answers in the qev shapes: a deciding engine and a readout.json naming slot_lm.
pub fn serves(srv: *Server, a: Allocator) bool {
    if (srv.engine.vtable.decide == null or !srv.info.decides or srv.config.model_dir.len == 0) return false;
    const path = std.fs.path.join(a, &.{ srv.config.model_dir, "readout.json" }) catch return false;
    const text = std.Io.Dir.cwd().readFileAlloc(srv.io, path, a, .limited(1 << 16)) catch return false;
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return false;
    if (v != .object) return false;
    const r = v.object.get("readout") orelse return false;
    return r == .string and std.mem.eql(u8, r.string, "slot_lm");
}

/// One question of either shape, with what the answer needs to name its options.
const Asked = struct {
    name: []const u8,
    q: prompt.Question,
    values: []const Value = &.{}, // decisions choice: each option's value as sent (a string or a boolean)
    levels: []const []const u8 = &.{}, // decisions score: each level's label
    legend: []const Value = &.{}, // systemone score: each level's criterion as sent
};

pub fn post(srv: *Server, conn: *Conn, a: Allocator, systemone: bool) void {
    var cx: Cx = .{ .a = a };
    const payload = blk: {
        const raw = http_body.readJson(conn, &cx) catch break :blk null;
        if (raw != .object) {
            _ = cx.refuse("request body must be an object") catch {};
            break :blk null;
        }
        break :blk replyFor(srv, &cx, raw, systemone, conn) catch |e| {
            if (e == error.Cancelled) {
                conn.close = true; // the client left: nothing to send
                return;
            }
            if (e == error.OutOfMemory) {
                cx.kind = .server;
                cx.message = "out of memory";
            }
            break :blk null;
        };
    };
    if (payload) |p| return routes.sendValue(conn, a, 200, p);
    if (cx.kind != .server) cx.kind = .request;
    const err = openai.errorBody(a, &cx, null) catch return;
    routes.sendValue(conn, a, cx.status(), openai.wrapError(a, err) catch return);
}

/// How long a decision may wait and run before it is dropped with 503.
pub const timeout_ms: u32 = 60_000;

/// The answers to a decisions (oMLX's OpenAI Decisions) or System One body, or its refusal.
pub fn reply(srv: *Server, cx: *Cx, body: Value, systemone: bool) errors.Refused!Value {
    return replyFor(srv, cx, body, systemone, null) catch |e| switch (e) {
        error.Cancelled => unreachable, // only a connection's client can leave
        else => |x| x,
    };
}

/// `reply` for a connection's request: the decision is dropped when its client leaves (error.Cancelled).
pub fn replyFor(srv: *Server, cx: *Cx, body: Value, systemone: bool, conn: ?*Conn) (errors.Refused || error{Cancelled})!Value {
    const a = cx.a;
    const model = body.get("model") orelse return cx.refuse("a request needs a model");
    if (model != .string) return cx.refuse("model must be a string");
    var state_text: []const u8 = undefined;
    var asked: []const Asked = undefined;
    if (systemone) {
        const state = body.get("state") orelse return cx.refuse("a request needs a state");
        if (state == .null) return cx.refuse("state must not be null");
        if (body.get("images")) |v| if (v.truthy()) return cx.refuse("this decision model reads text only: send no images");
        if (body.get("videos")) |v| if (v.truthy()) return cx.refuse("videos are not supported");
        state_text = try prompt.state(a, state);
        asked = try systemQuestions(cx, body.get("questions") orelse .null);
    } else {
        state_text = try inputText(cx, body.get("input") orelse return cx.refuse("a request needs input"));
        asked = try decisionQuestions(cx, body.get("questions") orelse return cx.refuse("a request needs questions"));
    }
    var problem: []const u8 = "";
    const suffixes = try a.alloc([]const u8, asked.len);
    for (asked, suffixes) |q, *s| s.* = try prompt.suffix(a, q.q);
    const cut = prompt.split(a, srv.text, state_text, suffixes, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.server, "the chat template cannot render this decision: {s}", .{problem}),
    };
    var longest: usize = 0;
    var total: usize = cut.prefix.len;
    var labels: std.ArrayList(u32) = .empty;
    const counts = try a.alloc(usize, asked.len);
    for (asked, cut.suffixes, counts) |q, s, *n| {
        if (s.len == 0) return cx.fail(.server, "a question has an empty suffix: state block boundary mismatch", .{});
        longest = @max(longest, s.len);
        total += s.len;
        n.* = q.q.options.len;
        for (0..n.*) |i| {
            const id = prompt.slotToken(a, srv.text, i) catch return error.OutOfMemory;
            try labels.append(a, id orelse return cx.fail(.server, "answer slot {s} is not one token in this tokenizer", .{prompt.labels[i]}));
        }
    }
    if (cut.prefix.len + longest > max_branch_tokens) return cx.fail(.request, "state plus longest question is {d} tokens, limit is {d}", .{ cut.prefix.len + longest, max_branch_tokens });
    if (total > max_request_tokens) return cx.fail(.request, "request is {d} tokens, limit is {d}", .{ total, max_request_tokens });
    const logits = try a.alloc(f64, labels.items.len);
    const gone: ?api.Decision.Gone = if (conn) |c| .{ .ctx = c, .check = peerGone } else null;
    var d: api.Decision = .{ .trunk = cut.prefix, .branches = cut.suffixes, .counts = counts, .labels = labels.items, .logits = logits, .arena = a, .timeout_ms = timeout_ms, .gone = gone };
    srv.engine.decide(&d) catch |e| switch (e) {
        error.Refused => return cx.refuse(d.problem),
        else => switch (d.stopped) {
            .gone => return error.Cancelled,
            .timed_out => return cx.fail(.capacity, "the decision did not finish within {d} s", .{timeout_ms / 1000}),
            .no => return cx.fail(.server, "decision scoring failed", .{}),
        },
    };
    var at: usize = 0;
    const answers = try a.alloc(Value, asked.len);
    for (asked, answers) |q, *out| {
        const p = try softmax(a, logits[at..][0..q.q.options.len]);
        at += p.len;
        out.* = if (systemone) try systemAnswer(a, q, p) else try decisionAnswer(a, q, p);
    }
    const usage = try json.newObject(a);
    try usage.put(a, "input_tokens", try json.intValue(a, total));
    try usage.put(a, "output_tokens", .{ .int = "0" });
    const o = try json.newObject(a);
    if (!systemone) try o.put(a, "object", .{ .string = "decision" });
    try o.put(a, "model", model);
    if (systemone) {
        const keyed = try json.newObject(a);
        for (asked, answers) |q, v| try keyed.put(a, q.name, v);
        try o.put(a, "answers", .{ .object = keyed });
    } else try o.put(a, "answers", .{ .array = answers });
    try o.put(a, "usage", .{ .object = usage });
    return .{ .object = o };
}

/// ``"\n\n".join`` of the input's text parts (oMLX's to_systemone); pictures are refused.
fn inputText(cx: *Cx, input: Value) errors.Refused![]const u8 {
    if (input == .string) return input.string;
    if (input != .array or input.array.len == 0) return cx.refuse("input is a string or a non-empty array of user messages");
    var texts: std.ArrayList([]const u8) = .empty;
    for (input.array) |m| {
        const content = m.get("content") orelse return cx.refuse("an input message needs content");
        if (m.get("role")) |r| if (r != .string or !std.mem.eql(u8, r.string, "user")) return cx.refuse("input messages have role user");
        if (content == .string) {
            try texts.append(cx.a, content.string);
            continue;
        }
        if (content != .array) return cx.refuse("message content is a string or an array of parts");
        for (content.array) |part| {
            const kind = json.strOr(part.get("type"));
            if (std.mem.eql(u8, kind, "input_text")) {
                const t = part.get("text") orelse return cx.refuse("an input_text part needs text");
                if (t != .string) return cx.refuse("an input_text part's text is a string");
                try texts.append(cx.a, t.string);
            } else if (std.mem.eql(u8, kind, "input_image")) {
                return cx.refuse("this decision model reads text only: send no input_image parts");
            } else return cx.refuse("a content part is input_text or input_image");
        }
    }
    if (texts.items.len == 0) return cx.refuse("input has no text");
    return std.mem.join(cx.a, "\n\n", texts.items);
}

fn instructionsOf(cx: *Cx, q: Value) errors.Refused![]const u8 {
    const v = q.get("instructions") orelse return cx.refuse("a question needs instructions");
    if (v != .string or v.string.len == 0) return cx.refuse("a question's instructions are a non-empty string");
    return v.string;
}

fn decisionQuestions(cx: *Cx, raw: Value) errors.Refused![]const Asked {
    const a = cx.a;
    if (raw != .array or raw.array.len == 0 or raw.array.len > max_questions) return cx.fail(.request, "questions is a list of 1 to {d} questions", .{max_questions});
    const out = try a.alloc(Asked, raw.array.len);
    for (raw.array, out, 0..) |q, *x, i| {
        if (q != .object) return cx.refuse("a question is an object");
        const name = q.get("name") orelse return cx.refuse("a question needs a name");
        if (name != .string or name.string.len == 0) return cx.refuse("a question's name is a non-empty string");
        for (out[0..i]) |prior| if (std.mem.eql(u8, prior.name, name.string)) return cx.refuse("question names must be unique");
        const kind = json.strOr(q.get("type"));
        const instructions = try instructionsOf(cx, q);
        x.* = .{ .name = name.string, .q = undefined };
        if (std.mem.eql(u8, kind, "predicate")) {
            x.q = try noul(a, .{ .string = instructions }, null);
        } else if (std.mem.eql(u8, kind, "choice")) {
            const choices = q.get("choices") orelse .null;
            if (choices != .array or choices.array.len < 2 or choices.array.len > max_choices) return cx.fail(.request, "a choice question has 2 to {d} choices", .{max_choices});
            const options = try a.alloc(prompt.Option, choices.array.len);
            const values = try a.alloc(Value, choices.array.len);
            for (choices.array, options, values, 0..) |c, *o, *v, j| {
                const value = if (c == .object) c.get("value") orelse .null else c;
                if (value != .string and value != .bool) return cx.refuse("a choice value is a string or a boolean");
                const key = if (value == .bool) (if (value.bool) "true" else "false") else value.string;
                for (options[0..j]) |prior| if (std.mem.eql(u8, prior.name, key)) return cx.refuse("choice values must be unique");
                const desc = if (c == .object) c.get("description") else null;
                if (desc) |dv| if (dv != .null and dv != .string) return cx.refuse("a choice description is a string");
                o.* = .{ .name = key, .text = try prompt.entry(a, desc) };
                v.* = value;
            }
            x.q = .{ .kind = .choice, .instructions = try orDefault(a, instructions, "Which option applies to the state?"), .options = options };
            x.values = values;
        } else if (std.mem.eql(u8, kind, "score")) {
            const levels = q.get("levels") orelse .null;
            if (levels != .array or levels.array.len < 2 or levels.array.len > max_levels) return cx.fail(.request, "a score question has 2 to {d} levels", .{max_levels});
            const options = try a.alloc(prompt.Option, levels.array.len);
            const names = try a.alloc([]const u8, levels.array.len);
            for (levels.array, options, names, 0..) |l, *o, *label, j| {
                const lv = if (l == .object) l.get("label") orelse .null else l;
                if (lv != .string) return cx.refuse("a score level's label is a string");
                label.* = lv.string;
                const desc = json.strOr(if (l == .object) l.get("description") else null);
                const criterion = if (desc.len > 0) try std.fmt.allocPrint(a, "{s}: {s}", .{ lv.string, desc }) else lv.string;
                o.* = .{ .name = try std.fmt.allocPrint(a, "level {d}", .{j}), .text = try prompt.entry(a, .{ .string = criterion }) };
            }
            x.q = .{ .kind = .score, .instructions = try orDefault(a, instructions, "Which level describes the state?"), .options = options };
            x.levels = names;
        } else return cx.refuse("a question's type is predicate, choice or score");
    }
    return out;
}

fn systemQuestions(cx: *Cx, raw: Value) errors.Refused![]const Asked {
    const a = cx.a;
    if (raw != .object or raw.object.count() == 0) return cx.refuse("questions is a non-empty object of id: question");
    const out = try a.alloc(Asked, raw.object.count());
    for (raw.object.keys(), raw.object.values(), out) |id, q, *x| {
        if (q != .object) return cx.refuse("a question is an object with type, instructions and criteria");
        const kind = json.strOr(q.get("type"));
        const instructions = q.get("instructions");
        const crit = q.get("criteria") orelse .null;
        x.* = .{ .name = id, .q = undefined };
        if (std.mem.eql(u8, kind, "noul")) {
            if (crit != .null and crit != .object) return cx.refuse("noul criteria is an object with true and false");
            x.q = try noul(a, instructions orelse .null, if (crit == .object) crit else null);
        } else if (std.mem.eql(u8, kind, "choice")) {
            if (crit != .object or crit.object.count() == 0 or crit.object.count() > max_choices) return cx.fail(.request, "choice criteria is an object of 1 to {d} options", .{max_choices});
            const options = try a.alloc(prompt.Option, crit.object.count());
            for (crit.object.keys(), crit.object.values(), options) |k, v, *o| o.* = .{ .name = k, .text = try prompt.entry(a, v) };
            x.q = .{ .kind = .choice, .instructions = try orDefault(a, try prompt.entry(a, instructions), "Which option applies to the state?"), .options = options };
        } else if (std.mem.eql(u8, kind, "score")) {
            if (crit != .array or crit.array.len == 0 or crit.array.len > max_levels) return cx.fail(.request, "score criteria is a list of 1 to {d} levels", .{max_levels});
            const options = try a.alloc(prompt.Option, crit.array.len);
            for (crit.array, options, 0..) |v, *o, j| o.* = .{ .name = try std.fmt.allocPrint(a, "level {d}", .{j}), .text = try prompt.entry(a, v) };
            x.q = .{ .kind = .score, .instructions = try orDefault(a, try prompt.entry(a, instructions), "Which level describes the state?"), .options = options };
            x.legend = crit.array;
        } else return cx.refuse("a question's type is noul, choice or score");
    }
    return out;
}

/// A yes/no question: the criteria's true and false texts, else the reference's defaults.
fn noul(a: Allocator, instructions: Value, crit: ?Value) Allocator.Error!prompt.Question {
    const t = if (crit) |c| c.get("true") orelse .null else Value.null;
    const f = if (crit) |c| c.get("false") orelse .null else Value.null;
    const options = try a.dupe(prompt.Option, &.{
        .{ .name = "yes", .text = if (t != .null) try prompt.entry(a, t) else prompt.noul_true },
        .{ .name = "no", .text = if (f != .null) try prompt.entry(a, f) else prompt.noul_false },
    });
    return .{ .kind = .noul, .instructions = try orDefault(a, try prompt.entry(a, instructions), prompt.noul_instructions), .options = options };
}

fn orDefault(a: Allocator, text: []const u8, default: []const u8) Allocator.Error![]const u8 {
    const stripped = @import("reply_text.zig").pyStrip(text);
    return if (stripped.len > 0) try a.dupe(u8, stripped) else default;
}

/// qev's softmax at temperature 1, then its normalize: float64 throughout.
fn softmax(a: Allocator, logits: []const f64) Allocator.Error![]f64 {
    var peak = logits[0];
    for (logits[1..]) |x| peak = @max(peak, x);
    const p = try a.alloc(f64, logits.len);
    var s: f64 = 0;
    for (logits, p) |x, *y| {
        y.* = @exp(x - peak);
        s += y.*;
    }
    for (p) |*y| y.* /= s;
    var t: f64 = 0;
    for (p) |y| t += y;
    for (p) |*y| y.* /= t;
    return p;
}

fn best(p: []const f64) usize {
    var i: usize = 0;
    for (p, 0..) |x, j| if (x > p[i]) {
        i = j;
    };
    return i;
}

fn choiceConfidence(p: []const f64) f64 {
    if (p.len == 1) return 1.0;
    const uniform = 1.0 / @as(f64, @floatFromInt(p.len));
    return (p[best(p)] - uniform) / (1.0 - uniform);
}

fn scoreConfidence(p: []const f64) f64 {
    if (p.len == 1) return 1.0;
    const k: f64 = @floatFromInt(p.len);
    const mode: f64 = @floatFromInt(best(p));
    var distance: f64 = 0;
    for (p, 0..) |x, i| distance += x * @abs(@as(f64, @floatFromInt(i)) - mode);
    const center = (k - 1) / 2;
    var mad: f64 = 0;
    for (0..p.len) |i| mad += @abs(@as(f64, @floatFromInt(i)) - center);
    return @max(0.0, 1.0 - distance / (mad / k));
}

fn expected(p: []const f64) f64 {
    var s: f64 = 0;
    for (p, 0..) |x, i| s += @as(f64, @floatFromInt(i)) * x;
    return s;
}

fn num(x: f64) Value {
    return .{ .float = prompt.round6(x) };
}

fn systemAnswer(a: Allocator, q: Asked, p: []const f64) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = @tagName(q.q.kind) });
    switch (q.q.kind) {
        .noul => try o.put(a, "noul", num(std.math.clamp(p[0], 0.0, 1.0))),
        .choice => {
            try o.put(a, "choice", .{ .string = q.q.options[best(p)].name });
            const probs = try json.newObject(a);
            for (q.q.options, p) |opt, x| try probs.put(a, opt.name, num(x));
            try o.put(a, "probabilities", .{ .object = probs });
            try o.put(a, "confidence", num(choiceConfidence(p)));
        },
        .score => {
            try o.put(a, "score", num(expected(p)));
            const legend = try json.newObject(a);
            const probs = try json.newObject(a);
            for (q.legend, p, 0..) |l, x, i| {
                const key = try std.fmt.allocPrint(a, "{d}", .{i});
                try legend.put(a, key, l);
                try probs.put(a, key, num(x));
            }
            try o.put(a, "legend", .{ .object = legend });
            try o.put(a, "probabilities", .{ .object = probs });
            try o.put(a, "confidence", num(scoreConfidence(p)));
        },
    }
    return .{ .object = o };
}

fn decisionAnswer(a: Allocator, q: Asked, p: []const f64) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = switch (q.q.kind) {
        .noul => "predicate",
        .choice => "choice",
        .score => "score",
    } });
    try o.put(a, "name", .{ .string = q.name });
    switch (q.q.kind) {
        .noul => try o.put(a, "probability", num(std.math.clamp(p[0], 0.0, 1.0))),
        .choice => {
            try o.put(a, "choice", q.values[best(p)]);
            const list = try a.alloc(Value, p.len);
            for (q.values, p, list) |v, x, *item| {
                const e = try json.newObject(a);
                try e.put(a, "value", v);
                try e.put(a, "probability", num(x));
                item.* = .{ .object = e };
            }
            try o.put(a, "probabilities", .{ .array = list });
            try o.put(a, "confidence", num(choiceConfidence(p)));
        },
        .score => {
            try o.put(a, "score", num(expected(p)));
            const list = try a.alloc(Value, p.len);
            for (q.levels, p, list, 0..) |label, x, *item, i| {
                const e = try json.newObject(a);
                try e.put(a, "value", try json.intValue(a, i));
                try e.put(a, "label", .{ .string = label });
                try e.put(a, "probability", num(x));
                item.* = .{ .object = e };
            }
            try o.put(a, "probabilities", .{ .array = list });
            try o.put(a, "confidence", num(scoreConfidence(p)));
        },
    }
    return .{ .object = o };
}

fn peerGone(ctx: *anyopaque) bool {
    const c: *Conn = @ptrCast(@alignCast(ctx));
    return c.peerGone();
}
