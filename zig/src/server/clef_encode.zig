//! Clef's record encoder (Cloudflare `joint_schema_model.encode_record`): the prompt and each question's spans.
const std = @import("std");
const json = @import("json");
const tokenizer = @import("tokenizer");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const system_prompt = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";

pub const Kind = enum(u8) { noul = 0, choice = 1, score = 2 };

pub const Question = struct {
    id: []const u8,
    kind: Kind,
    span: [2]u32, // the rendered instruction's rows
    options: []const [2]u32, // each option's rendered semantics rows
    option_ids: []const []const u8,
};

pub const Record = struct {
    ids: []const u32,
    questions: []const Question,
    state_rows: u32, // the state's rows kept after truncation
};

pub const Error = error{ OutOfMemory, BadRecord, SchemaTooLong } || tokenizer.Error;

/// `json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`, a string as itself (Clef's `render`).
pub fn render(a: Allocator, v: Value) Allocator.Error![]const u8 {
    if (v == .string) return v.string;
    var out: std.Io.Writer.Allocating = .init(a);
    dump(&out.writer, v) catch return error.OutOfMemory;
    return out.written();
}

fn dump(w: *std.Io.Writer, v: Value) std.Io.Writer.Error!void {
    switch (v) {
        .float => |f| {
            var buf: [40]u8 = undefined;
            // json.dumps spells the non-finite floats as JavaScript names, every other float as repr
            try w.writeAll(if (std.math.isNan(f)) "NaN" else if (std.math.isInf(f)) (if (f > 0) "Infinity" else "-Infinity") else json.floatRepr(&buf, f));
        },
        .array => |items| {
            try w.writeByte('[');
            for (items, 0..) |item, i| {
                if (i > 0) try w.writeByte(',');
                try dump(w, item);
            }
            try w.writeByte(']');
        },
        .object => |obj| {
            var order: [64]usize = undefined;
            var heap: ?[]usize = null;
            defer if (heap) |h| std.heap.page_allocator.free(h);
            const idx = if (obj.count() <= order.len) order[0..obj.count()] else blk: {
                heap = std.heap.page_allocator.alloc(usize, obj.count()) catch return error.WriteFailed;
                break :blk heap.?;
            };
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
                try dump(w, obj.values()[k]);
            }
            try w.writeByte('}');
        },
        else => try json.write(w, v, .{ .ascii = false, .compact = true }),
    }
}

const Opt = struct { id: []const u8, desc: ?Value };

fn byId(_: void, l: Opt, r: Opt) bool {
    return std.mem.order(u8, l.id, r.id) == .lt;
}

/// `question_options`: noul's true/false (criteria may override them), choice's ids sorted, score's levels by index.
fn options(a: Allocator, kind: Kind, q: Value) Error![]Opt {
    const criteria = q.get("criteria");
    switch (kind) {
        .noul => {
            var t: Value = .{ .string = "The proposition is true or the answer is yes." };
            var f: Value = .{ .string = "The proposition is false or the answer is no." };
            if (criteria) |c| if (c == .object) {
                if (c.object.get("true")) |x| t = x;
                if (c.object.get("false")) |x| f = x;
            };
            const out = try a.alloc(Opt, 2);
            out[0] = .{ .id = "true", .desc = t };
            out[1] = .{ .id = "false", .desc = f };
            return out;
        },
        .choice => {
            const c = criteria orelse return error.BadRecord;
            if (c != .object) return error.BadRecord;
            const out = try a.alloc(Opt, c.object.count());
            for (c.object.keys(), c.object.values(), out) |k, d, *o| o.* = .{ .id = k, .desc = d };
            std.mem.sort(Opt, out, {}, byId);
            return out;
        },
        .score => {
            const c = criteria orelse return error.BadRecord;
            if (c != .array) return error.BadRecord;
            const out = try a.alloc(Opt, c.array.len);
            for (c.array, out, 0..) |d, *o, i| o.* = .{ .id = try std.fmt.allocPrint(a, "{d}", .{i}), .desc = d };
            return out;
        },
    }
}

fn add(a: Allocator, tok: *const tokenizer.Tokenizer, ids: *std.ArrayList(u32), text: []const u8) Error!void {
    const t = try tok.encode(a, text);
    try ids.appendSlice(a, t);
}

/// The record's prompt (system, STATE, the state cut to fit `max_length`, the schema, the answer opening) and spans.
pub fn encode(a: Allocator, tok: *const tokenizer.Tokenizer, record: Value, max_length: u32) Error!Record {
    const qs = record.get("questions") orelse return error.BadRecord;
    if (qs != .object or qs.object.count() == 0) return error.BadRecord;
    const state = record.get("state") orelse return error.BadRecord;
    var schema: std.ArrayList(u32) = .empty;
    try add(a, tok, &schema, "\n\nSCHEMA FIELDS:\n");
    const questions = try a.alloc(Question, qs.object.count());
    for (qs.object.keys(), qs.object.values(), questions, 0..) |qid, q, *out, qi| {
        const type_name = (q.get("type") orelse return error.BadRecord).str() orelse return error.BadRecord;
        const kind = std.meta.stringToEnum(Kind, type_name) orelse return error.BadRecord;
        try add(a, tok, &schema, try std.fmt.allocPrint(a, "\nFIELD {d}\nID: {s}\nTYPE: {s}\nINSTRUCTION: ", .{ qi + 1, qid, type_name }));
        const q_start: u32 = @intCast(schema.items.len);
        const given = q.get("instructions");
        const empty = given == null or given.? == .null or (given.? == .string and given.?.string.len == 0);
        try add(a, tok, &schema, if (empty) qid else try render(a, given.?));
        const q_end: u32 = @intCast(schema.items.len);
        try add(a, tok, &schema, "\nALLOWED OPTIONS:\n");
        const opts = try options(a, kind, q);
        const spans = try a.alloc([2]u32, opts.len);
        const names = try a.alloc([]const u8, opts.len);
        for (opts, spans, names, 0..) |o, *s, *n, oi| {
            try add(a, tok, &schema, try std.fmt.allocPrint(a, "OPTION {d}: ", .{oi + 1}));
            const o_start: u32 = @intCast(schema.items.len);
            const sem = try a.create(json.Object);
            sem.* = .empty;
            try sem.put(a, "option_id", .{ .string = o.id });
            if (o.desc) |d| if (d != .null) try sem.put(a, "description", d);
            try add(a, tok, &schema, try render(a, .{ .object = sem }));
            s.* = .{ o_start, @intCast(schema.items.len) };
            n.* = o.id;
            try add(a, tok, &schema, "\n");
        }
        try add(a, tok, &schema, "END FIELD\n");
        out.* = .{ .id = qid, .kind = kind, .span = .{ q_start, q_end }, .options = spans, .option_ids = names };
    }
    var prefix: std.ArrayList(u32) = .empty;
    try add(a, tok, &prefix, "<|im_start|>system\n" ++ system_prompt ++ "<|im_end|>\n<|im_start|>user\nSTATE:\n");
    var suffix: std.ArrayList(u32) = .empty;
    try add(a, tok, &suffix, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
    var state_ids: std.ArrayList(u32) = .empty;
    try add(a, tok, &state_ids, try render(a, state));
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
    var ids: std.ArrayList(u32) = .empty;
    try ids.appendSlice(a, prefix.items);
    try ids.appendSlice(a, state_ids.items[0..keep]);
    try ids.appendSlice(a, schema.items);
    try ids.appendSlice(a, suffix.items);
    return .{ .ids = ids.items, .questions = questions, .state_rows = @intCast(keep) };
}
