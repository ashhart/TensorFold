//! TensorFold's chat-template engine: Jinja2 rendered as Hugging Face's apply_chat_template renders it.
const std = @import("std");
const ast = @import("ast.zig");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const scope = @import("scope.zig");
const v = @import("value.zig");
const pyjson = @import("pyjson.zig");
const eval = @import("eval.zig");
const filters = @import("filters.zig");
const access = @import("access.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ TemplateFailed, OutOfMemory };

/// Why rendering failed; `raised` marks the template's own raise_exception message.
pub const Diag = struct { msg: []const u8 = "", raised: bool = false };

/// A parsed template; its arena owns the syntax tree.
pub const Compiled = struct {
    arena: std.heap.ArenaAllocator,
    tree: ast.Template,

    pub fn deinit(t: *Compiled) void {
        t.arena.deinit();
    }
};

fn checkNames(stmts: []const ast.Stmt, bad: *?[]const u8) void {
    const W = struct {
        fn expr(e: *const ast.Expr, out: *?[]const u8) void {
            switch (e.*) {
                .filter => |f| if (!access.among(f.name, &filters.implemented)) {
                    out.* = f.name;
                },
                .is => |t| if (!access.among(t.name, &filters.tests)) {
                    out.* = t.name;
                },
                else => {},
            }
            ast.eachChild(error{}, e, out, child) catch |err| switch (err) {};
        }
        fn child(out: *?[]const u8, e: *ast.Expr) error{}!void {
            expr(e, out);
        }
        fn body(b: []const ast.Stmt, out: *?[]const u8) void {
            for (b) |st| switch (st) {
                .output => |pieces| for (pieces) |pc| if (pc == .expr) expr(pc.expr, out),
                .for_ => |f| {
                    expr(f.iter, out);
                    if (f.filter) |t| expr(t, out);
                    body(f.body, out);
                    body(f.else_body, out);
                },
                .if_ => |n| ifs(n, out),
                .assign => |a| expr(a.value, out),
                .block_set => |b2| {
                    if (b2.filter) |f| expr(f, out);
                    body(b2.body, out);
                },
                .macro => |m| {
                    for (m.defaults) |d| expr(d, out);
                    body(m.body, out);
                },
                .scoped => |s| body(s.body, out),
                .break_, .continue_ => {},
            };
        }
        fn ifs(n: *const ast.If, out: *?[]const u8) void {
            expr(n.test_expr, out);
            body(n.body, out);
            for (n.elifs) |e| ifs(e, out);
            body(n.else_body, out);
        }
    };
    W.body(stmts, bad);
}

fn report(gpa: Allocator, diag: ?*Diag, comptime fmt: []const u8, args: anytype) Error {
    if (diag) |d| d.msg = std.fmt.allocPrint(gpa, fmt, args) catch return error.OutOfMemory;
    return error.TemplateFailed;
}

/// Lexes, parses and scopes `source`; unknown filters and tests fail here, as Jinja2's compiler does.
pub fn compile(gpa: Allocator, source: []const u8, diag: ?*Diag) Error!Compiled {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var lx = lexer.Lexer{ .a = a, .s = try lexer.normalize(a, source) };
    const tokens = lexer.tokenize(&lx) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else report(gpa, diag, "template syntax error: {s}", .{lx.msg});
    var p = parser.Parser{ .a = a, .toks = tokens };
    const body = p.parse() catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else report(gpa, diag, "template syntax error: {s}", .{p.msg});
    var z = scope.Analyzer{ .a = a };
    const frame = z.run(body) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else report(gpa, diag, "template syntax error: {s}", .{z.msg});
    var bad: ?[]const u8 = null;
    checkNames(body, &bad);
    if (bad) |name| return report(gpa, diag, "template uses unsupported filter or test '{s}'", .{name});
    return .{ .arena = arena, .tree = .{ .body = body, .frame = frame } };
}

/// Renders with Hugging Face's variables: messages, tools (none when absent), documents, add_generation_prompt, then `context`.
pub fn render(gpa: Allocator, t: *const Compiled, messages: std.json.Value, tools: std.json.Value, context: std.json.Value, add_generation_prompt: bool, diag: ?*Diag) Error![]u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var out: std.ArrayList(u8) = .empty;
    var vars: v.Map = .empty;
    var c = eval.Ctx{ .rt = .{ .a = a }, .vars = &vars, .out = &out };
    run(&c, &vars, t, messages, tools, context, add_generation_prompt) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        if (diag) |d| d.raised = c.rt.raised;
        return report(gpa, diag, "{s}", .{c.rt.msg});
    };
    return gpa.dupe(u8, out.items);
}

fn run(c: *eval.Ctx, vars: *v.Map, t: *const Compiled, messages: std.json.Value, tools: std.json.Value, context: std.json.Value, add_generation_prompt: bool) v.Error!void {
    const a = c.rt.a;
    try vars.put(a, "messages", try pyjson.fromJson(&c.rt, messages));
    try vars.put(a, "tools", try pyjson.fromJson(&c.rt, tools));
    try vars.put(a, "documents", .none);
    try vars.put(a, "add_generation_prompt", .{ .boolean = add_generation_prompt });
    if (context == .object) {
        var it = context.object.iterator();
        while (it.next()) |e| {
            const key = e.key_ptr.*;
            if (std.mem.eql(u8, key, "messages") or std.mem.eql(u8, key, "tools") or std.mem.eql(u8, key, "add_generation_prompt")) continue;
            try vars.put(a, key, try pyjson.fromJson(&c.rt, e.value_ptr.*));
        }
    }
    try eval.root(c, t.tree);
}

/// One-shot compile and render, the call the engine's chat prompts make.
pub fn renderChat(gpa: Allocator, source: []const u8, messages: std.json.Value, tools: std.json.Value, context: std.json.Value, add_generation_prompt: bool, diag: ?*Diag) Error![]u8 {
    var t = try compile(gpa, source, diag);
    defer t.deinit();
    return render(gpa, &t, messages, tools, context, add_generation_prompt, diag);
}

test {
    _ = @import("tests.zig");
    _ = @import("unicode.zig");
    _ = @import("value.zig");
}
