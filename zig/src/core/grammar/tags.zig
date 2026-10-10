//! Tool-call structural tags: xgrammar 0.2.8's ``get_model_structural_tag`` JSON for a tools spec, byte for byte.

const std = @import("std");

pub const Model = enum { deepseek_v4_1 };

pub const Error = error{ InvalidTools, OutOfMemory, WriteFailed };

/// Why a tools spec cannot be built (xgrammar's error text where it has one), set with error.InvalidTools.
pub const Problem = struct {
    message: []const u8 = "",
    buf: [512]u8 = undefined,
};

const blanks = 32;

/// The structural tag JSON for `spec_text` (gpa-owned).
pub fn build(gpa: std.mem.Allocator, model: Model, spec_text: []const u8, problem: *Problem) Error![]u8 {
    _ = model;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const spec = std.json.parseFromSliceLeaky(std.json.Value, a, spec_text, .{ .parse_numbers = false }) catch return fail(problem, "the tools spec is not valid JSON");
    if (spec != .object) return fail(problem, "the tools spec is not an object");
    const tools_v = spec.object.get("tools") orelse return fail(problem, "the tools spec has no tools");
    if (tools_v != .array) return fail(problem, "The 'tools' argument must be a list.");
    const parallel = if (spec.object.get("parallel_tool_calls")) |p| p == .bool and p.bool else true;

    var fns: std.ArrayList(std.json.ObjectMap) = .empty;
    for (tools_v.array.items) |t| {
        const f = (if (t == .object) t.object.get("function") else null) orelse continue;
        if (f != .object) continue;
        try fns.append(a, f.object);
    }
    const Choice = enum { auto, required, forced };
    var choice: Choice = .auto;
    const cv = spec.object.get("tool_choice") orelse std.json.Value.null;
    if (cv == .string) {
        if (std.mem.eql(u8, cv.string, "required")) choice = .required else if (std.mem.eql(u8, cv.string, "none")) fns.clearRetainingCapacity();
    } else if (cv == .object) {
        const name = nameOf(cv.object) orelse return fail(problem, "tool_choice names no function");
        var kept: std.ArrayList(std.json.ObjectMap) = .empty;
        for (fns.items) |f| if (std.mem.eql(u8, nameOf(f) orelse "", name)) try kept.append(a, f);
        if (kept.items.len == 0) return fail(problem, std.fmt.bufPrint(&problem.buf, "The tool with name '{s}' is not found in the tools list.", .{name}) catch "The tool is not found in the tools list.");
        if (kept.items.len != 1) return fail(problem, "Forced tool choice must resolve to exactly one tool.");
        fns = kept;
        choice = .forced;
    }
    if (choice == .required and fns.items.len == 0) return fail(problem, "The 'tools' list is empty, which is not allowed when 'tool_choice' is 'required'.");

    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"type\":\"structural_tag\",\"format\":");
    switch (choice) {
        .auto => if (fns.items.len == 0) {
            try w.writeAll("{\"type\":\"any_text\",\"excludes\":[\"<think>\",\"</think>\",\"<｜DSML｜ calls>\"]}");
        } else {
            try w.writeAll("{\"type\":\"triggered_tags\",\"triggers\":[\"<｜DSML｜ calls>\"],\"tags\":[{\"type\":\"tag\",\"begin\":\"<｜DSML｜ calls>\\n\",\"content\":");
            try invokes(w, fns.items, parallel);
            try w.print(",\"end\":\"</｜DSML｜ calls>\"}}],\"at_least_one\":false,\"stop_after_first\":{},\"excludes\":[\"<think>\",\"</think>\"]}}", .{!parallel});
        },
        .required => {
            try w.writeAll("{\"type\":\"sequence\",\"elements\":[{\"type\":\"const_string\",\"value\":\"\\n\\n<｜DSML｜ calls>\\n\"},");
            try invokes(w, fns.items, parallel);
            try w.writeAll(",{\"type\":\"const_string\",\"value\":\"</｜DSML｜ calls>\"}]}");
        },
        .forced => {
            try w.writeAll("{\"type\":\"sequence\",\"elements\":[{\"type\":\"const_string\",\"value\":\"\\n\\n<｜DSML｜ calls>\\n\"},");
            try invoke(w, fns.items[0]);
            try w.writeAll(",{\"type\":\"const_string\",\"value\":\"</｜DSML｜ calls>\"}]}");
        },
    }
    try w.writeByte('}');
    return out.toOwnedSlice();
}

fn fail(problem: *Problem, message: []const u8) Error {
    problem.message = message;
    return error.InvalidTools;
}

fn nameOf(o: std.json.ObjectMap) ?[]const u8 {
    if (o.get("name")) |n| if (n == .string) return n.string;
    if (o.get("function")) |f| if (f == .object) if (f.object.get("name")) |n| if (n == .string) return n.string;
    return null;
}

fn invokes(w: *std.Io.Writer, fns: []const std.json.ObjectMap, parallel: bool) !void {
    try w.writeAll("{\"type\":\"tags_with_separator\",\"tags\":[");
    for (fns, 0..) |f, i| {
        if (i > 0) try w.writeByte(',');
        try invoke(w, f);
    }
    try w.print("],\"separator\":\"\",\"at_least_one\":true,\"stop_after_first\":{}}}", .{!parallel});
}

/// One tool's invoke tag, its arguments held to its parameters (``strict: false`` or none: any JSON).
fn invoke(w: *std.Io.Writer, f: std.json.ObjectMap) !void {
    try w.writeAll("{\"type\":\"tag\",\"begin\":");
    var begin: [1024]u8 = undefined;
    const name = nameOf(f) orelse "";
    try string(w, std.fmt.bufPrint(&begin, "<｜DSML｜ invoke name=\"{s}\">\n", .{name}) catch return error.WriteFailed);
    try w.writeAll(",\"content\":{\"type\":\"json_schema\",\"json_schema\":");
    const strict_false = if (f.get("strict")) |s| s == .bool and !s.bool else false;
    const params = f.get("parameters");
    if (strict_false or params == null or params.? == .null) try w.writeAll("true") else try value(w, params.?);
    try w.print(",\"style\":\"deepseek_v4_1_xml\",\"any_order\":false,\"max_whitespace_cnt\":{d},\"excludes\":[]}},\"end\":\"</｜DSML｜ invoke>\\n\"}}", .{blanks});
}

/// A JSON value as pydantic writes it: compact, keys in order, numbers as given.
fn value(w: *std.Io.Writer, v: std.json.Value) !void {
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .float => |x| try w.print("{d}", .{x}),
        .number_string => |s| try w.writeAll(s),
        .string => |s| try string(w, s),
        .array => |arr| {
            try w.writeByte('[');
            for (arr.items, 0..) |x, i| {
                if (i > 0) try w.writeByte(',');
                try value(w, x);
            }
            try w.writeByte(']');
        },
        .object => |o| {
            try w.writeByte('{');
            var it = o.iterator();
            var i: usize = 0;
            while (it.next()) |e| : (i += 1) {
                if (i > 0) try w.writeByte(',');
                try string(w, e.key_ptr.*);
                try w.writeByte(':');
                try value(w, e.value_ptr.*);
            }
            try w.writeByte('}');
        },
    }
}

/// A JSON string as serde_json writes it: `"` `\` and control characters escaped, everything else raw.
pub fn string(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    var from: usize = 0;
    for (s, 0..) |ch, i| {
        const esc: ?[]const u8 = switch (ch) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            0x08 => "\\b",
            0x0c => "\\f",
            else => null,
        };
        if (esc == null and ch >= 0x20) continue;
        try w.writeAll(s[from..i]);
        if (esc) |e| try w.writeAll(e) else try w.print("\\u{x:0>4}", .{ch});
        from = i + 1;
    }
    try w.writeAll(s[from..]);
    try w.writeByte('"');
}
