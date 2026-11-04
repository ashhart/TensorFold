//! The body field that asked for a structure reaches the engine, and every other request's structure stays null.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const mt = @import("model_text.zig");
const Server = @import("server.zig").Server;
const a = std.testing.allocator;
const specs = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}},\"required\":[\"q\"]}}},{\"type\":\"function\",\"function\":{\"name\":\"now\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";

const Text = struct {
    fn text(t: *Text) mt.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = source } };
    }
    fn encode(_: *anyopaque, g: std.mem.Allocator, s: []const u8, _: bool) mt.Error![]u32 {
        const ids = try g.alloc(u32, s.len);
        for (s, ids) |ch, *id| id.* = if (ch == '\n') 95 else @as(u32, ch) - 32;
        return ids;
    }
    fn decode(_: *anyopaque, g: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 95) try out.append(g, @intCast(id + 32)) else if (id == 95) try out.append(g, '\n');
        return out.toOwnedSlice(g);
    }
    fn tokenId(_: *anyopaque, s: []const u8) ?u32 {
        return if (std.mem.eql(u8, s, "<tool_call>")) 28 else null;
    }
    fn tokenString(_: *anyopaque, g: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return decode(undefined, g, &.{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 97;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{96};
    }
    fn source(_: *anyopaque) []const u8 {
        return "";
    }
    fn render(_: *anyopaque, g: std.mem.Allocator, _: json.Value, _: mt.RenderOptions, _: *[]const u8) mt.Error!mt.Rendered {
        return .{ .text = try g.dupe(u8, "<tool_call>{\"name\":\"tfprobe_fn\",\"arguments\":{\"tfprobe_k\":\"v\"}}</tool_call>") };
    }
};

const Stub = struct {
    fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .name = "qwen27" };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        unreachable;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []const u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

// The compiler boundary records the kind it is asked for; no xgrammar library or GPU is loaded.
const Compile = struct {
    kind: c_int = -1,

    fn compiler(c: *Compile) api.grammar.Compiler {
        var out: api.grammar.Compiler = undefined;
        out.h = c;
        out.words = 4;
        out.mutex = .init;
        out.f.compile = compile;
        out.f.free = free;
        out.f.close = free;
        return out;
    }
    fn compile(ptr: *anyopaque, kind: c_int, _: [*]const u8, _: usize, _: [*]u8, _: usize) callconv(.c) ?*anyopaque {
        const c: *Compile = @ptrCast(@alignCast(ptr));
        c.kind = kind;
        return ptr;
    }
    fn free(_: *anyopaque) callconv(.c) void {}
};

fn value(g: std.mem.Allocator, s: []const u8) !json.Value {
    return (try json.parse(g, s)).ok;
}

test "structure field: each structured-output body field reaches the engine by its own name" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: Stub = .{};
    var compile: Compile = .{};
    var compiler = compile.compiler();
    var text: Text = .{};
    const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    srv.info.structures = true;
    srv.grammars = &compiler;
    const Case = struct { body: []const u8, kind: []const u8, field: []const u8 };
    const cases = [_]Case{
        .{ .body = "{\"response_format\":{\"type\":\"json_object\"}}", .kind = "json", .field = "response_format" },
        .{ .body = "{\"guided_json\":{\"type\":\"object\"}}", .kind = "json_schema", .field = "guided_json" },
        .{ .body = "{\"guided_regex\":\"a+\"}", .kind = "regex", .field = "guided_regex" },
        .{ .body = "{\"guided_choice\":[\"a\",\"b\"]}", .kind = "choice", .field = "guided_choice" },
        .{ .body = "{\"guided_grammar\":\"root ::= \\\"a\\\"\"}", .kind = "grammar", .field = "guided_grammar" },
        .{ .body = "{\"structured_outputs\":{\"json_object\":true}}", .kind = "json", .field = "structured_outputs" },
        .{ .body = "{\"structured_outputs\":{\"regex\":\"a+\"}}", .kind = "regex", .field = "structured_outputs" },
        .{ .body = "{\"structured_outputs\":{\"json\":{\"type\":\"object\"}}}", .kind = "json_schema", .field = "structured_outputs" },
        .{ .body = "{\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":{\"type\":\"object\"}}}}", .kind = "json_schema", .field = "response_format" },
    };
    for (cases) |c| {
        compile.kind = -1;
        var cx: errors.Cx = .{ .a = g };
        var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 64 };
        try srv.checkFeatures(&cx, try value(g, c.body), false, false, &.{1}, &.{}, &request);
        const structure = request.structure orelse return error.TestExpectedStructure;
        try std.testing.expectEqualStrings(c.kind, @tagName(structure.kind));
        try std.testing.expectEqualStrings(c.field, structure.field);
        try std.testing.expectEqual(@as(c_int, @intCast(@backingInt(structure.kind))), compile.kind);
    }
}

test "structure field: a plain request keeps no structure" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: Stub = .{};
    var compile: Compile = .{};
    var compiler = compile.compiler();
    var text: Text = .{};
    const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    srv.info.structures = true;
    srv.grammars = &compiler;
    var cx: errors.Cx = .{ .a = g };
    var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 64 };
    try srv.checkFeatures(&cx, try value(g, "{}"), false, false, &.{1}, &.{}, &request);
    try std.testing.expect(request.structure == null);
}

test "structure field: a required tool call names tool_choice" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: Stub = .{};
    var compile: Compile = .{};
    var compiler = compile.compiler();
    var text: Text = .{};
    const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    srv.info.structures = true;
    srv.grammars = &compiler;
    var cx: errors.Cx = .{ .a = g };
    var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 200 };
    const offered = (try value(g, specs)).array;
    try srv.checkFeatures(&cx, try value(g, "{\"tool_call_required\":true}"), true, false, &.{1}, offered, &request);
    const structure = request.structure orelse return error.TestExpectedStructure;
    try std.testing.expectEqualStrings("structural_tag", @tagName(structure.kind));
    try std.testing.expectEqualStrings("tool_choice", structure.field);
}

test "structure field: strict tools under auto name tool_choice" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: Stub = .{};
    var compile: Compile = .{};
    var compiler = compile.compiler();
    var text: Text = .{};
    const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    srv.info.structures = true;
    srv.grammars = &compiler;
    var cx: errors.Cx = .{ .a = g };
    var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 200 };
    const strict_tools = (try value(g, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"strict\":true,\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}},\"required\":[\"q\"]}}}]")).array;
    try srv.checkFeatures(&cx, try value(g, "{}"), true, false, &.{1}, strict_tools, &request);
    const structure = request.structure orelse return error.TestExpectedStructure;
    try std.testing.expectEqualStrings("structural_tag", @tagName(structure.kind));
    try std.testing.expectEqualStrings("tool_choice", structure.field);
}

test "structure field: the refusal still names the backend" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: Stub = .{};
    var text: Text = .{};
    const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    srv.info.structures = true;
    var cx: errors.Cx = .{ .a = g };
    var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 200 };
    try std.testing.expectError(error.Refused, srv.checkFeatures(&cx, try value(g, "{\"tool_call_required\":true}"), true, false, &.{1}, (try value(g, specs)).array, &request));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "tool_choice \"required\" or a named function on backend 'qwen27'") != null);
    try std.testing.expectEqual(@as(u16, 400), cx.status());
}
