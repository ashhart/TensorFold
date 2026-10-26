//! /v1/systemone through Cloudflare Clef (`--clef DIR`): the record encoded, the engine's final rows, the joint head.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const enc = @import("clef_encode.zig");
const head = @import("clef_head.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;

/// Clef's release default: a record's prompt is cut to this many tokens (its state truncated).
pub const max_length: u32 = 16384;

/// The head and the output-embedding table, mapped for the server's lifetime.
pub const Clef = struct {
    weights: head.Files,
    rows: head.Files,
    head: head.Head,
    lex: head.Lexical,

    /// DIR holds joint_head.safetensors, joint_head_config.json and lm_head_rows.safetensors.
    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8) !*Clef {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const c = try gpa.create(Clef);
        errdefer gpa.destroy(c);
        const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "joint_head_config.json" }), a, .limited(1 << 16));
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, cfg_text, .{});
        if (parsed != .object) return error.BadConfig;
        var cfg: head.Config = .{};
        inline for (.{ .{ "hidden_size", "hidden" }, .{ "width", "width" }, .{ "routing_layers", "routing" }, .{ "layers", "layers" }, .{ "heads", "heads" }, .{ "feedforward", "ff" } }) |f| {
            const v = parsed.object.get(f[0]) orelse return error.BadConfig;
            if (v != .integer or v.integer <= 0) return error.BadConfig;
            @field(cfg, f[1]) = @intCast(v.integer);
        }
        c.weights = try head.Files.open(gpa, io, try std.fs.path.join(a, &.{ dir, "joint_head.safetensors" }));
        errdefer c.weights.close(io);
        c.rows = try head.Files.open(gpa, io, try std.fs.path.join(a, &.{ dir, "lm_head_rows.safetensors" }));
        errdefer c.rows.close(io);
        c.head = try head.Head.load(gpa, &c.weights, cfg);
        const table = c.rows.get("lm_head.weight") orelse return error.MissingTensor;
        if (table.dtype != .bf16 or table.n % cfg.hidden != 0) return error.BadTensor;
        c.lex = .{ .table = table.bytes, .rows = table.n / cfg.hidden, .width = cfg.hidden };
        return c;
    }

    pub fn close(c: *Clef, gpa: Allocator, io: std.Io) void {
        c.head.deinit();
        c.rows.close(io);
        c.weights.close(io);
        gpa.destroy(c);
    }
};

/// The server's tokenizer as clef_encode asks: special-token text as its token, nothing prepended.
const Ids = struct {
    text: model_text.Text,
    pub fn ids(s: Ids, a: Allocator, text: []const u8) ![]u32 {
        return s.text.encode(a, text, false);
    }
};

/// Cloudflare's request checks (`systemone`), then the answers and the usage.
pub fn reply(srv: *Server, c: *const Clef, cx: *Cx, body: Value) errors.Refused!Value {
    const model = body.get("model") orelse return cx.refuse("model and state are required");
    if (model != .string or !body.has("state")) return cx.refuse("model and state are required");
    if (body.get("state").? == .null) return cx.refuse("state must not be null");
    for ([_][]const u8{ "images", "files", "videos" }) |k| if (body.get(k)) |v| if (v.truthy()) return cx.fail(.request, "{s}: this Clef server takes text records only", .{k});
    const qs = body.get("questions") orelse return cx.refuse("at least one question is required");
    if (qs != .object or qs.object.count() == 0) return cx.refuse("at least one question is required");
    for (qs.object.keys(), qs.object.values()) |id, q| {
        const t = if (q == .object) (q.get("type") orelse .null) else .null;
        const kind = if (t == .string) std.meta.stringToEnum(enc.Kind, t.string) else null;
        if (kind == null) return cx.fail(.request, "{s}: type must be noul, choice, or score", .{id});
        const crit = q.get("criteria") orelse .null;
        if (kind.? != .noul and !crit.truthy()) return cx.fail(.request, "{s}: criteria must not be empty", .{id});
        if (kind.? == .choice and crit != .object) return cx.fail(.request, "{s}: choice criteria map option ids to descriptions", .{id});
        if (kind.? == .score and crit != .array) return cx.fail(.request, "{s}: score criteria are a list of level descriptions", .{id});
    }
    const a = cx.a;
    const rec = enc.encode(a, Ids{ .text = srv.text }, body, max_length) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.SchemaTooLong => cx.fail(.request, "the schema needs more than {d} tokens before the state", .{max_length}),
        else => cx.fail(.request, "the record cannot be encoded ({s})", .{@errorName(e)}),
    };
    if (body.get("truncate")) |t| if (t == .bool and !t.bool) {
        const whole = enc.encode(a, Ids{ .text = srv.text }, body, std.math.maxInt(u32)) catch return cx.refuse("the record cannot be encoded");
        if (whole.ids.len > rec.ids.len) return cx.fail(.request, "the record needs {d} tokens; at most {d} fit", .{ whole.ids.len, max_length });
    };
    const width = c.head.cfg.hidden;
    const bits = try a.alloc(u16, rec.ids.len * width);
    srv.engine.hidden(rec.ids, bits) catch |e| return switch (e) {
        error.Unsupported => cx.refuse("this engine gives no final rows: Clef needs the qwen3_5 CUDA family's native engine"),
        error.Failed => cx.fail(.server, "the record's prompt pass failed", .{}),
    };
    const hidden = try a.alloc(f32, bits.len);
    for (hidden, bits) |*h, b| h.* = @bitCast(@as(u32, b) << 16);
    const logits = c.head.forward(a, hidden, rec, c.lex) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.fail(.server, "the head failed ({s})", .{@errorName(e)}),
    };
    const answers = try json.newObject(a);
    const raw = try json.newObject(a); // `"raw": true`: unrounded probabilities in encoder order (our extension)
    for (rec.questions, logits) |q, lg| {
        const probs = try a.alloc(f64, lg.len);
        softmax(lg, probs);
        try answers.put(a, q.id, try answer(a, qs.object.get(q.id).?, q, probs));
        const r = try json.newObject(a);
        try r.put(a, "type", .{ .string = @tagName(q.kind) });
        const names = try a.alloc(Value, q.option_ids.len);
        for (names, q.option_ids) |*n, o| n.* = .{ .string = o };
        try r.put(a, "options", .{ .array = names });
        const ps = try a.alloc(Value, probs.len);
        for (ps, probs) |*v, p| v.* = .{ .float = p };
        try r.put(a, "probs", .{ .array = ps });
        try raw.put(a, q.id, .{ .object = r });
    }
    const usage = try json.newObject(a);
    try usage.put(a, "input_tokens", try json.intValue(a, rec.ids.len));
    try usage.put(a, "output_tokens", .{ .int = "0" });
    const out = try json.newObject(a);
    try out.put(a, "model", model);
    try out.put(a, "answers", .{ .object = answers });
    try out.put(a, "usage", .{ .object = usage });
    if (body.get("raw")) |r| if (r == .bool and r.bool) try out.put(a, "raw", .{ .object = raw });
    return .{ .object = out };
}

fn softmax(lg: []const f32, out: []f64) void {
    var m: f64 = -std.math.inf(f64);
    for (lg) |v| m = @max(m, v);
    var s: f64 = 0;
    for (lg, out) |v, *o| {
        o.* = @exp(@as(f64, v) - m);
        s += o.*;
    }
    for (out) |*o| o.* /= s;
}

/// Python's round(x, 4) for these probabilities.
fn round4(x: f64) f64 {
    return @round(x * 1e4) / 1e4;
}

fn prob(probs: []const f64, q: enc.Question, id: []const u8) f64 {
    for (q.option_ids, probs) |o, p| if (std.mem.eql(u8, o, id)) return p;
    return 0;
}

/// `systemone_answer`: noul's P(true); choice's best option in criteria order; score's expected level.
fn answer(a: Allocator, raw: Value, q: enc.Question, probs: []const f64) !Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = @tagName(q.kind) });
    switch (q.kind) {
        .noul => try o.put(a, "noul", .{ .float = round4(prob(probs, q, "true")) }),
        .choice => {
            const crit = raw.get("criteria").?.object;
            var best: []const u8 = crit.keys()[0];
            for (crit.keys()) |k| if (prob(probs, q, k) > prob(probs, q, best)) {
                best = k;
            };
            try o.put(a, "choice", .{ .string = best });
            try o.put(a, "confidence", .{ .float = round4(prob(probs, q, best)) });
            const p = try json.newObject(a);
            for (crit.keys()) |k| try p.put(a, k, .{ .float = round4(prob(probs, q, k)) });
            try o.put(a, "probabilities", .{ .object = p });
        },
        .score => {
            const crit = raw.get("criteria").?.array;
            var expected: f64 = 0;
            var top: f64 = 0;
            const legend = try json.newObject(a);
            const p = try json.newObject(a);
            for (crit, 0..) |level, i| {
                const name = try std.fmt.allocPrint(a, "{d}", .{i});
                const v = prob(probs, q, name);
                expected += @as(f64, @floatFromInt(i)) * v;
                top = @max(top, v);
                try legend.put(a, name, level);
                try p.put(a, name, .{ .float = round4(v) });
            }
            try o.put(a, "score", .{ .float = round4(expected) });
            try o.put(a, "confidence", .{ .float = round4(top) });
            try o.put(a, "legend", .{ .object = legend });
            try o.put(a, "probabilities", .{ .object = p });
        },
    }
    return .{ .object = o };
}
