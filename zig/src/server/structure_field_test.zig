//! The body field that asked for a structure reaches the engine, and every other request's structure stays null.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const mt = @import("model_text.zig");
const Server = @import("server.zig").Server;
const openai = @import("openai.zig");
const responses_translate = @import("responses_translate.zig");
const anthropic_translate = @import("anthropic_translate.zig");
const Conn = @import("http_conn.zig").Conn;
const nameStructure = @import("server.zig").nameStructure;
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

/// Records the structure the engine is handed, then answers two tokens, so a whole run stays on this thread.
const Record = struct {
    /// The name copied while the request is still valid: `api.Request` only promises its slices until `finished`.
    buf: [32]u8 = undefined,
    len: usize = 0,
    kind: []const u8 = "",

    fn field(r: *const Record) []const u8 {
        return r.buf[0..r.len];
    }

    fn engine(r: *Record) api.Engine {
        return .{ .ctx = r, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .name = "qwen27", .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const r: *Record = @ptrCast(@alignCast(ctx));
        if (request.structure) |st| {
            r.len = @min(st.field.len, r.buf.len);
            @memcpy(r.buf[0..r.len], st.field[0..r.len]);
            r.kind = @tagName(st.kind);
        }
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 65, 66 } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The reply sink a route writes to: the status of a refused reply, nothing else.
const Wire = struct {
    status: u16 = 0,

    fn out(w: *Wire) openai.Out {
        return .{ .ctx = w, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}
    fn reply(ctx: *anyopaque, status_code: u16, _: json.Value) void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.status = status_code;
    }
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

test "structure field: a Responses text.format reaches the engine as its own name" {
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
    const History = struct {
        pub fn conversation(_: @This(), c: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
            return c.refuse("no stored response");
        }
    };
    var cx: errors.Cx = .{ .a = g };
    const body = try value(g, "{\"model\":\"m\",\"input\":\"hi\",\"text\":{\"format\":{\"type\":\"json_object\"}}}");
    const r = try responses_translate.translate(&cx, body, History{});
    try std.testing.expect(std.mem.eql(u8, r.format.?, "text.format"));
    var input = try openai.chatInput(srv, &cx, r.chat);
    input.format = r.format;
    var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 64 };
    try srv.checkFeatures(&cx, input.fields, input.tools.len > 0, false, &.{1}, input.tools, &request);
    nameStructure(input.format, &request);
    const structure = request.structure orelse return error.TestExpectedStructure;
    try std.testing.expectEqualStrings("text.format", structure.field);
    try std.testing.expectEqualStrings("json", @tagName(structure.kind));
}

test "structure field: a Messages output_config.format reaches the engine as its own name" {
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
    const body = try value(g, "{\"model\":\"m\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"output_config\":{\"format\":{\"type\":\"json_schema\",\"schema\":{\"type\":\"object\"}}}}");
    const chat_body = try anthropic_translate.translate(&cx, body, false);
    const name = anthropic_translate.formatName(body);
    try std.testing.expectEqualStrings("output_config.format", name orelse "");
    var input = try openai.chatInput(srv, &cx, chat_body);
    input.format = name;
    var request: api.Request = .{ .prompt = &.{1}, .max_tokens = 64 };
    try srv.checkFeatures(&cx, input.fields, input.tools.len > 0, false, &.{1}, input.tools, &request);
    nameStructure(input.format, &request);
    const structure = request.structure orelse return error.TestExpectedStructure;
    try std.testing.expectEqualStrings("output_config.format", structure.field);
    try std.testing.expectEqualStrings("json_schema", @tagName(structure.kind));
}

test "structure field: the Responses translation hands the engine text.format" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var rec: Record = .{};
    var compile: Compile = .{};
    var compiler = compile.compiler();
    var text: Text = .{};
    const srv = try Server.init(a, std.testing.io, rec.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    srv.info.structures = true;
    srv.grammars = &compiler;
    const History = struct {
        pub fn conversation(_: @This(), c: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
            return c.refuse("no stored response");
        }
    };
    var cx: errors.Cx = .{ .a = g };
    const body = try value(g, "{\"model\":\"m\",\"input\":\"hi\",\"text\":{\"format\":{\"type\":\"json_object\"}}}");
    const r = try responses_translate.translate(&cx, body, History{});
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = g };
    var wire: Wire = .{};
    openai.runNamed(srv, g, wire.out(), .{ .conn = &conn }, true, r.chat, r.format);
    try std.testing.expectEqual(@as(u16, 200), wire.status);
    try std.testing.expectEqualStrings("text.format", rec.field());
    try std.testing.expectEqualStrings("json", rec.kind);
}

test "structure field: the chat completion hands the engine response_format" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var rec: Record = .{};
    var compile: Compile = .{};
    var compiler = compile.compiler();
    var text: Text = .{};
    const srv = try Server.init(a, std.testing.io, rec.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    srv.info.structures = true;
    srv.grammars = &compiler;
    const body = try value(g, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":{\"type\":\"json_object\"}}");
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = g };
    var wire: Wire = .{};
    openai.run(srv, g, wire.out(), .{ .conn = &conn }, true, body);
    try std.testing.expectEqual(@as(u16, 200), wire.status);
    try std.testing.expectEqualStrings("response_format", rec.field());
    try std.testing.expectEqualStrings("json", rec.kind);
}
