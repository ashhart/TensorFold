//! Clef's record encoder (Cloudflare `joint_schema_model.encode_record`): the prompt and each question's spans.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const system_prompt = "Read the complete state and schema. Decide every field jointly. " ++
    "Each answer must be exactly one of that field's allowed options.";
pub const prefix_text = "<|im_start|>system\n" ++ system_prompt ++ "<|im_end|>\n<|im_start|>user\nSTATE:\n";
pub const suffix_text = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";

pub const Kind = enum(u8) { noul = 0, choice = 1, score = 2 };

/// A question's criteria in the shape its type needs; the request's validation builds them (`fields`).
pub const Criteria = union(Kind) {
    noul: ?*const json.Object, // overrides of the true and false descriptions; null: none
    choice: *const json.Object, // option id to description, in request order, not empty
    score: []const Value, // level descriptions, not empty
};

/// One question of a validated request, in request order.
pub const Field = struct {
    id: []const u8,
    instructions: ?Value, // as given; missing, null and "" fall back to the id
    criteria: Criteria,

    pub fn kind(f: Field) Kind {
        return f.criteria;
    }
};

pub const Question = struct {
    id: []const u8,
    kind: Kind,
    span: [2]u32, // the rendered instruction's rows
    options: []const [2]u32, // each option's rendered semantics rows, in encoding order
    option_ids: []const []const u8, // encoding order: noul true then false, choice sorted, score by level
};

pub const Record = struct {
    ids: []const u32,
    questions: []const Question,
    state_rows: u32, // the state's rows kept after truncation
};

pub const Error = error{ OutOfMemory, Template, SchemaTooLong };

/// The fields of a request's `questions` object, or Invalid when a question lacks its type's shape.
pub fn fields(a: Allocator, questions: Value) error{ OutOfMemory, Invalid }![]Field {
    if (questions != .object or questions.object.count() == 0) return error.Invalid;
    const out = try a.alloc(Field, questions.object.count());
    for (questions.object.keys(), questions.object.values(), out) |id, q, *f| {
        if (q != .object) return error.Invalid;
        const kind = std.meta.stringToEnum(Kind, q.strField("type") orelse return error.Invalid) orelse return error.Invalid;
        const c = q.get("criteria") orelse Value.null;
        f.* = .{
            .id = id,
            .instructions = q.get("instructions"),
            .criteria = switch (kind) {
                // `criteria.update(question.get("criteria") or {})`: any falsy value overrides nothing
                .noul => .{ .noul = if (c == .object) c.object else if (c.truthy()) return error.Invalid else null },
                .choice => .{ .choice = if (c == .object and c.object.count() > 0) c.object else return error.Invalid },
                .score => .{ .score = if (c == .array and c.array.len > 0) c.array else return error.Invalid },
            },
        };
    }
    return out;
}

/// `json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`, a string as itself (Clef's `render`).
pub fn render(a: Allocator, v: Value) Allocator.Error![]const u8 {
    if (v == .string) return v.string;
    var out: std.Io.Writer.Allocating = .init(a);
    dump(a, &out.writer, v) catch return error.OutOfMemory;
    return out.written();
}

fn dump(a: Allocator, w: *std.Io.Writer, v: Value) (std.Io.Writer.Error || Allocator.Error)!void {
    switch (v) {
        .float => |f| {
            var buf: [40]u8 = undefined;
            // json.dumps spells the non-finite floats as JavaScript names, every other float as repr
            const nan = std.math.isNan(f);
            try w.writeAll(if (nan) "NaN" else if (std.math.isInf(f)) (if (f > 0) "Infinity" else "-Infinity") else json.floatRepr(&buf, f));
        },
        .array => |items| {
            try w.writeByte('[');
            for (items, 0..) |item, i| {
                if (i > 0) try w.writeByte(',');
                try dump(a, w, item);
            }
            try w.writeByte(']');
        },
        .object => |obj| {
            const idx = try a.alloc(usize, obj.count());
            defer a.free(idx);
            for (idx, 0..) |*x, i| x.* = i;
            // Python sorts str keys by code point, which is UTF-8 byte order
            std.mem.sort(usize, idx, obj.keys(), struct {
                fn lt(keys: [][]const u8, l: usize, r: usize) bool {
                    return std.mem.order(u8, keys[l], keys[r]) == .lt;
                }
            }.lt);
            try w.writeByte('{');
            for (idx, 0..) |k, i| {
                if (i > 0) try w.writeByte(',');
                try json.writeString(w, obj.keys()[k], .{ .ascii = false, .compact = true });
                try w.writeByte(':');
                try dump(a, w, obj.values()[k]);
            }
            try w.writeByte('}');
        },
        else => try json.write(w, v, .{ .ascii = false, .compact = true }),
    }
}

pub const Opt = struct { id: []const u8, desc: Value };

fn byId(_: void, l: Opt, r: Opt) bool {
    return std.mem.order(u8, l.id, r.id) == .lt;
}

/// `question_options`: noul's true then false (criteria may override them), choice's ids sorted, score's levels.
pub fn options(a: Allocator, c: Criteria) Allocator.Error![]Opt {
    switch (c) {
        .noul => |given| {
            const out = try a.alloc(Opt, 2);
            out[0] = .{ .id = "true", .desc = .{ .string = "The proposition is true or the answer is yes." } };
            out[1] = .{ .id = "false", .desc = .{ .string = "The proposition is false or the answer is no." } };
            if (given) |o| for (out) |*x| if (o.get(x.id)) |d| {
                x.desc = d;
            };
            return out;
        },
        .choice => |o| {
            const out = try a.alloc(Opt, o.count());
            for (o.keys(), o.values(), out) |k, d, *x| x.* = .{ .id = k, .desc = d };
            std.mem.sort(Opt, out, {}, byId);
            return out;
        },
        .score => |levels| {
            const out = try a.alloc(Opt, levels.len);
            for (levels, out, 0..) |d, *x, i| x.* = .{ .id = try std.fmt.allocPrint(a, "{d}", .{i}), .desc = d };
            return out;
        },
    }
}

fn add(a: Allocator, text: model_text.Text, ids: *std.ArrayList(u32), piece: []const u8) Error!void {
    try ids.appendSlice(a, try text.encode(a, piece, false));
}

/// The prompt and spans of a validated record: each piece tokenized alone, special-token text as its token.
pub fn encode(a: Allocator, text: model_text.Text, state: Value, record: []const Field, max_length: u32) Error!Record {
    var schema: std.ArrayList(u32) = .empty;
    try add(a, text, &schema, "\n\nSCHEMA FIELDS:\n");
    const questions = try a.alloc(Question, record.len);
    for (record, questions, 0..) |f, *out, qi| {
        const head = "\nFIELD {d}\nID: {s}\nTYPE: {s}\nINSTRUCTION: ";
        try add(a, text, &schema, try std.fmt.allocPrint(a, head, .{ qi + 1, f.id, @tagName(f.kind()) }));
        const q_start: u32 = @intCast(schema.items.len);
        const given = f.instructions orelse Value.null;
        const empty = given == .null or (given == .string and given.string.len == 0);
        try add(a, text, &schema, if (empty) f.id else try render(a, given));
        const q_end: u32 = @intCast(schema.items.len);
        try add(a, text, &schema, "\nALLOWED OPTIONS:\n");
        const opts = try options(a, f.criteria);
        const spans = try a.alloc([2]u32, opts.len);
        const names = try a.alloc([]const u8, opts.len);
        for (opts, spans, names, 0..) |o, *s, *n, oi| {
            try add(a, text, &schema, try std.fmt.allocPrint(a, "OPTION {d}: ", .{oi + 1}));
            const o_start: u32 = @intCast(schema.items.len);
            const sem = try json.newObject(a);
            try sem.put(a, "option_id", .{ .string = o.id });
            if (o.desc != .null) try sem.put(a, "description", o.desc);
            try add(a, text, &schema, try render(a, .{ .object = sem }));
            s.* = .{ o_start, @intCast(schema.items.len) };
            n.* = o.id;
            try add(a, text, &schema, "\n");
        }
        try add(a, text, &schema, "END FIELD\n");
        out.* = .{ .id = f.id, .kind = f.kind(), .span = .{ q_start, q_end }, .options = spans, .option_ids = names };
    }
    var prefix: std.ArrayList(u32) = .empty;
    try add(a, text, &prefix, prefix_text);
    var suffix: std.ArrayList(u32) = .empty;
    try add(a, text, &suffix, suffix_text);
    var state_ids: std.ArrayList(u32) = .empty;
    try add(a, text, &state_ids, try render(a, state));
    const fixed = prefix.items.len + schema.items.len + suffix.items.len;
    if (fixed > max_length) return error.SchemaTooLong;
    const keep = @min(state_ids.items.len, max_length - fixed);
    const offset: u32 = @intCast(prefix.items.len + keep);
    for (questions) |*q| {
        q.span = .{ q.span[0] + offset, q.span[1] + offset };
        const moved = try a.alloc([2]u32, q.options.len);
        for (q.options, moved) |s, *m| m.* = .{ s[0] + offset, s[1] + offset };
        q.options = moved;
    }
    const ids = try std.mem.concat(a, u32, &.{ prefix.items, state_ids.items[0..keep], schema.items, suffix.items });
    return .{ .ids = ids, .questions = questions, .state_rows = @intCast(keep) };
}
