//! /v1/systemone: llama-server's System One request, its options read at single-token forms through the engine.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const json_write = @import("json_write.zig");
const errors = @import("errors.zig");
const pyrepr = @import("pyrepr.zig");
const reply_text = @import("reply_text.zig");
const http_body = @import("http_body.zig");
const openai = @import("openai.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

const im_start = "<|im_start|>";
const im_end = "<|im_end|>";
const bos = "<|startoftext|>";
const yes_forms = [_][]const u8{ "yes", "Yes", "YES" };
const no_forms = [_][]const u8{ "no", "No", "NO" };
const media = [_][]const u8{ "input_audio", "audio", "video_url", "video" };
const max_pictures = 4; // TensorFold 0.6's ImageLimits.max_images

const Kind = enum { noul, choice, score };

const Question = struct {
    id: []const u8,
    kind: Kind,
    instructions: []const u8,
    keys: []const []const u8 = &.{}, // choice: option names; score: level texts; noul: none or true/false
    texts: []const ?Value = &.{}, // choice: descriptions; noul: the true and false descriptions
};

const Code = struct { text: []const u8, id: u32 };

pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    if (srv.engine.vtable.score == null) { // an engine that does not decide serves no such path
        _ = http_body.read(conn, a, http_body.limit) catch {};
        return routes.unknown(conn, a);
    }
    var cx: Cx = .{ .a = a };
    const payload = blk: {
        const raw = http_body.readJson(conn, &cx) catch break :blk null;
        if (raw != .object) {
            _ = cx.refuse("request body must be an object") catch {};
            break :blk null;
        }
        break :blk reply(srv, &cx, raw) catch |e| {
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

/// The answers to a System One body, or its refusal.
pub fn reply(srv: *Server, cx: *Cx, body: Value) errors.Refused!Value {
    const state = body.get("state") orelse return cx.refuse("a request needs a state (null when the question is the whole state)");
    const raw_questions = body.get("questions") orelse return cx.refuse("a request needs questions");
    const seen = try pictures(cx, body, state);
    if (raw_questions != .object or raw_questions.object.count() == 0) return cx.refuse("questions is a non-empty object of id: question");
    const questions = try cx.a.alloc(Question, raw_questions.object.count());
    for (raw_questions.object.keys(), raw_questions.object.values(), questions) |id, raw, *q| q.* = try parse(cx, id, raw);
    if (seen.urls.len > 0 and !srv.engine.info().pictures) return cx.refuse("this checkpoint has no vision tower");

    var f: Format = .{ .srv = srv, .cx = cx, .one = .init(cx.a) };
    // with pictures the head and the state are encoded apart, the pictures' tokens between them (the processor's ids)
    const head = if (seen.urls.len > 0) try f.encode(bos ++ im_start ++ "user\n") else &.{};
    const trunk = if (seen.urls.len > 0) try std.mem.concat(cx.a, u32, &.{ head, try f.encode(try prefix(cx.a, seen.state, false)) }) else try f.encode(try prefix(cx.a, seen.state, true));
    const codes = try cx.a.alloc([]const Code, questions.len);
    const branches = try cx.a.alloc([]const u32, questions.len);
    for (questions, codes, branches) |q, *c, *b| {
        c.* = if (q.kind == .choice) try f.codes(q.keys) else &.{};
        b.* = try f.encode(try suffix(cx.a, q, c.*));
    }
    const groups = try cx.a.alloc([]const []const u32, questions.len);
    for (questions, codes, groups) |q, c, *g| g.* = try f.readout(q, c);
    const window = srv.engine.info().context_window;
    var read: usize = trunk.len;
    for (branches) |b| {
        if (b.len == 0) return cx.refuse("each question needs a nonempty token sequence");
        // with pictures the engine checks the expanded trunk
        if (seen.urls.len == 0 and window > 0 and trunk.len + b.len > window) return cx.fail(.request, "a decision sequence exceeds the {d}-token context", .{window});
        read += b.len;
    }
    const scores = try scoreAll(srv, cx, .{ .trunk = trunk, .at = head.len, .pictures = seen.urls }, branches, groups, &read);
    const answers = try json.newObject(cx.a);
    for (questions, scores) |q, s| try answers.put(cx.a, q.id, try answer(cx.a, q, s));
    const usage = try json.newObject(cx.a);
    try usage.put(cx.a, "input_tokens", try json.intValue(cx.a, read));
    try usage.put(cx.a, "output_tokens", .{ .int = "0" });
    const out = try json.newObject(cx.a);
    try out.put(cx.a, "model", .{ .string = srv.config.served_name });
    try out.put(cx.a, "answers", .{ .object = answers });
    try out.put(cx.a, "usage", .{ .object = usage });
    return .{ .object = out };
}

const Seen = struct { urls: []const []const u8, state: Value };

/// The pictures (files, images, then the state's image parts) and the state without them; other media refused.
fn pictures(cx: *Cx, body: Value, state: Value) errors.Refused!Seen {
    var urls: std.ArrayList([]const u8) = .empty;
    for ([_][]const u8{ "files", "images" }) |key| if (body.get(key)) |v| switch (v) {
        .null => {},
        .array => |items| for (items) |item| try urls.append(cx.a, try url(cx, item)),
        else => return cx.fail(.request, "{s} must be an array of data URLs", .{key}),
    };
    if (body.get("videos")) |v| if (v.truthy()) return cx.refuse("videos are not supported");
    const messages = if (state == .object) state.get("messages") orelse .null else state;
    var kept_state = state;
    if (messages == .array and messages.array.len > 0) {
        const kept = try cx.a.alloc(Value, messages.array.len);
        for (messages.array, kept) |message, *out| {
            out.* = message;
            const content = if (message == .object) message.get("content") orelse .null else .null;
            if (content != .array) continue;
            var parts: std.ArrayList(Value) = .empty;
            for (content.array) |part| {
                const kind = if (part == .object) (part.get("type") orelse .null) else .null;
                if (kind == .string and std.mem.eql(u8, kind.string, "image_url")) {
                    const u = part.get("image_url") orelse .null;
                    try urls.append(cx.a, try url(cx, if (u == .object) u.get("url") orelse .null else u));
                    continue;
                }
                if (kind == .string) for (media) |m| if (std.mem.eql(u8, kind.string, m))
                    return cx.fail(.request, "{s} is not supported: a decision state takes images", .{m});
                try parts.append(cx.a, part);
            }
            const copy = try json.newObject(cx.a);
            for (message.object.keys(), message.object.values()) |k, v| try copy.put(cx.a, k, if (std.mem.eql(u8, k, "content")) .{ .array = parts.items } else v);
            out.* = .{ .object = copy };
        }
        if (state == .object) {
            const copy = try json.newObject(cx.a);
            for (state.object.keys(), state.object.values()) |k, v| try copy.put(cx.a, k, if (std.mem.eql(u8, k, "messages")) .{ .array = kept } else v);
            kept_state = .{ .object = copy };
        } else kept_state = .{ .array = kept };
    }
    if (urls.items.len > max_pictures) return cx.fail(.request, "at most {d} pictures a request", .{max_pictures});
    return .{ .urls = urls.items, .state = kept_state };
}

fn url(cx: *Cx, v: Value) errors.Refused![]const u8 {
    if (v != .string or v.string.len == 0) return cx.refuse("image_url must contain a non-empty url string");
    return v.string;
}

fn parse(cx: *Cx, id: []const u8, raw: Value) errors.Refused!Question {
    if (raw != .object) return cx.refuse("a question is an object with type, instructions and criteria");
    const kind_v = raw.get("type") orelse Value{ .string = "choice" };
    const instr_v = raw.get("instructions") orelse .null;
    if (instr_v == .null) return cx.refuse("a question needs instructions (llama-server's System One requires them)");
    const instructions = try text(cx.a, instr_v);
    const crit = raw.get("criteria") orelse .null;
    const kind: Kind = if (kind_v == .string) std.meta.stringToEnum(Kind, kind_v.string) orelse
        return cx.fail(.request, "unknown question type {s}: choose noul, choice or score", .{try pyrepr.repr(cx.a, kind_v)}) else return cx.fail(.request, "unknown question type {s}: choose noul, choice or score", .{try pyrepr.repr(cx.a, kind_v)});
    var q: Question = .{ .id = id, .kind = kind, .instructions = instructions };
    switch (kind) {
        .noul => {
            if (crit == .null) return q;
            if (crit != .object) return cx.refuse("noul criteria is an object with true and false");
            q.keys = &.{ "true", "false" };
            q.texts = try cx.a.dupe(?Value, &.{ crit.get("true"), crit.get("false") });
        },
        .score => {
            if (crit != .array or crit.array.len < 2 or crit.array.len > 10) return cx.refuse("score criteria is a list of 2 to 10 levels, lowest first");
            const keys = try cx.a.alloc([]const u8, crit.array.len);
            for (crit.array, keys) |c, *k| k.* = try text(cx.a, c);
            q.keys = keys;
        },
        .choice => {
            if (crit != .object or crit.object.count() == 0) return cx.refuse("choice criteria is a non-empty object of option: description");
            q.keys = crit.object.keys();
            const texts = try cx.a.alloc(?Value, crit.object.count());
            for (crit.object.values(), texts) |v, *t| t.* = v;
            q.texts = texts;
        },
    }
    return q;
}

/// Python's ``v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)``.
fn text(a: Allocator, v: Value) Allocator.Error![]const u8 {
    return if (v == .string) v.string else json.stringify(a, v, .{ .ascii = false });
}

/// Prompt pieces and readout tokens for the served tokenizer.
const Format = struct {
    srv: *Server,
    cx: *Cx,
    one: std.StringHashMap(?u32),

    fn encode(f: *Format, s: []const u8) errors.Refused![]const u32 {
        return f.srv.text.encode(f.cx.a, s, false) catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => f.cx.refuse("the tokenizer cannot encode this decision"),
        };
    }

    /// The token `s` is when it is exactly one.
    fn single(f: *Format, s: []const u8) errors.Refused!?u32 {
        if (f.one.get(s)) |hit| return hit;
        const ids = try f.encode(s);
        const id: ?u32 = if (ids.len == 1) ids[0] else null;
        try f.one.put(try f.cx.a.dupe(u8, s), id);
        return id;
    }

    /// A distinct single-token code per option: single letters as given, else A.. (00.. past 26), else the pool's next.
    fn codes(f: *Format, keys: []const []const u8) errors.Refused![]const Code {
        const a = f.cx.a;
        var letters = true;
        for (keys) |k| {
            const s = reply_text.pyStrip(k);
            if (!(s.len == 1 and std.ascii.isAlphabetic(s[0]))) letters = false;
        }
        const out = try a.alloc(Code, keys.len);
        var used: std.AutoHashMapUnmanaged(u32, void) = .empty;
        for (keys, out, 0..) |k, *code, i| {
            const wanted = if (letters) reply_text.pyStrip(k) else if (keys.len <= 26) try a.dupe(u8, &.{@as(u8, @intCast('A' + i))}) else try std.fmt.allocPrint(a, "{d:0>2}", .{i});
            var n: usize = 0;
            var candidate = wanted;
            while (true) : (n += 1) {
                if (n > 0) candidate = try pool(a, n - 1) orelse return f.cx.fail(.request, "no single-token code left for {d} options", .{keys.len});
                const id = try f.single(candidate) orelse continue;
                if (used.contains(id)) continue;
                try used.put(a, id, {});
                code.* = .{ .text = candidate, .id = id };
                break;
            }
        }
        return out;
    }

    /// Token ids an option is read at; an option scores its best form.
    fn readout(f: *Format, q: Question, codes_: []const Code) errors.Refused![]const []const u32 {
        const a = f.cx.a;
        switch (q.kind) {
            .noul => {
                const groups = try a.alloc([]const u32, 2);
                groups[0] = try f.forms(&yes_forms);
                groups[1] = try f.forms(&no_forms);
                if (groups[0].len == 0 or groups[1].len == 0) return f.cx.refuse("this tokenizer has no single-token yes/no, which a noul question is read at");
                return groups;
            },
            .score => {
                const groups = try a.alloc([]const u32, q.keys.len);
                for (groups, 0..) |*g, i| {
                    g.* = try f.forms(&.{try std.fmt.allocPrint(a, "{d}", .{i})});
                    if (g.len == 0) return f.cx.fail(.request, "a score with {d} levels needs single-token digits", .{q.keys.len});
                }
                return groups;
            },
            .choice => {
                const groups = try a.alloc([]const u32, codes_.len);
                for (codes_, groups) |c, *g| {
                    const extra = try f.single(try std.mem.concat(a, u8, &.{ " ", c.text }));
                    g.* = if (extra != null and extra.? != c.id) try a.dupe(u32, &.{ c.id, extra.? }) else try a.dupe(u32, &.{c.id});
                }
                return groups;
            },
        }
    }

    fn forms(f: *Format, texts: []const []const u8) errors.Refused![]const u32 {
        var seen: std.ArrayList(u32) = .empty;
        for (texts) |t| if (try f.single(t)) |id| if (std.mem.indexOfScalar(u32, seen.items, id) == null) try seen.append(f.cx.a, id);
        return seen.items;
    }
};

/// The pool's `i`th code: A-Z, 00-99, a-z, #0-#199, then AA-ZZ; null past its end.
fn pool(a: Allocator, i: usize) Allocator.Error!?[]const u8 {
    if (i < 26) return try a.dupe(u8, &.{@as(u8, @intCast('A' + i))});
    if (i < 126) return try std.fmt.allocPrint(a, "{d:0>2}", .{i - 26});
    if (i < 152) return try a.dupe(u8, &.{@as(u8, @intCast('a' + i - 126))});
    if (i < 352) return try std.fmt.allocPrint(a, "#{d}", .{i - 152});
    if (i < 352 + 676) return try a.dupe(u8, &.{ @as(u8, @intCast('A' + (i - 352) / 26)), @as(u8, @intCast('A' + (i - 352) % 26)) });
    return null;
}

fn prefix(a: Allocator, state: Value, head: bool) Allocator.Error![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    writePrefix(&out.writer, state, head) catch return error.OutOfMemory;
    return out.toOwnedSlice();
}

fn writePrefix(w: *std.Io.Writer, state: Value, head: bool) std.Io.Writer.Error!void {
    if (head) try w.writeAll(bos ++ im_start ++ "user\n");
    if (state == .null) return;
    if (state == .string) try w.writeAll(state.string) else try indented(w, state, 0);
    try w.writeAll("\n\n\nQUESTION:\n");
}

/// ``json.dumps(v, ensure_ascii=False, indent=2)``.
fn indented(w: *std.Io.Writer, v: Value, level: usize) std.Io.Writer.Error!void {
    const o: json_write.Options = .{ .ascii = false };
    switch (v) {
        .array => |items| {
            if (items.len == 0) return w.writeAll("[]");
            try w.writeByte('[');
            for (items, 0..) |item, i| {
                try w.writeAll(if (i > 0) ",\n" else "\n");
                try w.splatByteAll(' ', 2 * (level + 1));
                try indented(w, item, level + 1);
            }
            try w.writeByte('\n');
            try w.splatByteAll(' ', 2 * level);
            try w.writeByte(']');
        },
        .object => |obj| {
            if (obj.count() == 0) return w.writeAll("{}");
            try w.writeByte('{');
            for (obj.keys(), obj.values(), 0..) |k, item, i| {
                try w.writeAll(if (i > 0) ",\n" else "\n");
                try w.splatByteAll(' ', 2 * (level + 1));
                try json_write.writeString(w, k, o);
                try w.writeAll(": ");
                try indented(w, item, level + 1);
            }
            try w.writeByte('\n');
            try w.splatByteAll(' ', 2 * level);
            try w.writeByte('}');
        },
        else => try json_write.write(w, v, o),
    }
}

fn suffix(a: Allocator, q: Question, codes: []const Code) Allocator.Error![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    questionText(a, &out.writer, q, codes) catch return error.OutOfMemory;
    out.writer.writeAll(im_end ++ "\n" ++ im_start ++ "assistant\n") catch return error.OutOfMemory;
    return out.toOwnedSlice();
}

fn questionText(a: Allocator, w: *std.Io.Writer, q: Question, codes: []const Code) !void {
    try w.writeAll(q.instructions);
    switch (q.kind) {
        .choice => {
            try w.writeAll("\n\nOptions:\n");
            for (codes, q.keys, q.texts, 0..) |c, key, desc, i| {
                if (i > 0) try w.writeByte('\n');
                try w.print("{s} ", .{c.text});
                // a description as written (JSON when it is not text); without one, the name
                if (desc != null and desc.?.truthy()) try w.writeAll(try text(a, desc.?)) else for (key) |ch| try w.writeByte(if (ch == '_') ' ' else ch);
            }
            try w.writeAll("\n\nReply with the option code only.");
        },
        .noul => {
            // the template shows the descriptions only when one side has one, then both
            const any = q.texts.len == 2 and (q.texts[0] != null and q.texts[0].? != .null or q.texts[1] != null and q.texts[1].? != .null);
            if (any) for ([_][]const u8{ "Yes", "No" }, q.texts) |name, t| {
                const shown: Value = if (t != null and t.? != .null) t.? else .{ .string = "None" };
                try w.print("\n{s}: {s}", .{ name, try text(a, shown) });
            };
            try w.writeAll("\n\nReply with yes or no only.");
        },
        .score => {
            try w.writeAll("\n\n");
            for (q.keys, 0..) |name, i| try w.print("{s}{d} {s}", .{ if (i > 0) "\n" else "", i, name });
            try w.print("\n\nReply with a single digit 0-{d} only.", .{q.keys.len - 1});
        },
    }
}

/// Each question's option scores (its best form's logit): one packed pass when the engine decides, else a pass each.
fn scoreAll(srv: *Server, cx: *Cx, trunk: api.Decision, branches: []const []const u32, groups: []const []const []const u32, read: *usize) errors.Refused![]const []const f64 {
    var labels: std.ArrayList(u32) = .empty;
    const counts = try cx.a.alloc(usize, groups.len);
    for (groups, counts) |g, *n| {
        n.* = 0;
        for (g) |forms| {
            try labels.appendSlice(cx.a, forms);
            n.* += forms.len;
        }
    }
    const logits = try cx.a.alloc(f64, labels.items.len);
    @memset(logits, 0);
    if (srv.engine.vtable.decide != null) {
        var d = trunk;
        d.arena = cx.a;
        d.branches = branches;
        d.counts = counts;
        d.labels = labels.items;
        d.logits = logits;
        srv.engine.decide(&d) catch |e| switch (e) {
            error.Refused => return cx.refuse(d.problem),
            else => return cx.fail(.server, "decision scoring failed", .{}),
        };
        read.* = d.read;
    } else {
        var at: usize = 0;
        for (branches, counts) |b, n| {
            const prompt = try std.mem.concat(cx.a, u32, &.{ trunk.trunk, b });
            // a softmax over the options alone: the vocabulary's mass is not read
            _ = srv.engine.score(prompt, labels.items[at..][0..n], logits[at..][0..n], false) catch |e| switch (e) {
                error.Unsupported => return cx.refuse("decision scoring is not supported by this engine"),
                error.Failed => return cx.fail(.server, "decision scoring failed", .{}),
            };
            at += n;
        }
    }
    const out = try cx.a.alloc([]const f64, groups.len);
    var at: usize = 0;
    for (groups, out) |g, *row| {
        const scores = try cx.a.alloc(f64, g.len);
        for (g, scores) |forms, *s| {
            s.* = logits[at];
            for (logits[at..][0..forms.len]) |x| s.* = @max(s.*, x);
            at += forms.len;
        }
        row.* = scores;
    }
    return out;
}

fn answer(a: Allocator, q: Question, scores: []const f64) Allocator.Error!Value {
    var peak = scores[0];
    for (scores[1..]) |s| peak = @max(peak, s);
    const p = try a.alloc(f64, scores.len);
    var z: f64 = 0;
    for (scores, p) |s, *x| {
        x.* = @exp(s - peak);
        z += x.*;
    }
    for (p) |*x| x.* /= z;
    var best: usize = 0;
    for (p, 0..) |x, i| if (x > p[best]) {
        best = i;
    };
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = @tagName(q.kind) });
    switch (q.kind) {
        .noul => try o.put(a, "noul", .{ .float = p[0] }),
        .choice => {
            try o.put(a, "choice", .{ .string = q.keys[best] });
            try o.put(a, "confidence", .{ .float = p[best] });
            const probs = try json.newObject(a);
            for (q.keys, p) |k, x| try probs.put(a, k, .{ .float = x });
            try o.put(a, "probabilities", .{ .object = probs });
        },
        .score => {
            var expected: f64 = 0;
            for (p, 0..) |x, i| expected += @as(f64, @floatFromInt(i)) * x;
            try o.put(a, "score", .{ .float = expected });
            try o.put(a, "confidence", .{ .float = p[best] });
            const probs = try json.newObject(a);
            const legend = try json.newObject(a);
            for (q.keys, p, 0..) |k, x, i| {
                const name = try std.fmt.allocPrint(a, "{d}", .{i});
                try probs.put(a, name, .{ .float = x });
                try legend.put(a, name, .{ .string = k });
            }
            try o.put(a, "probabilities", .{ .object = probs });
            try o.put(a, "legend", .{ .object = legend });
        },
    }
    return .{ .object = o };
}
