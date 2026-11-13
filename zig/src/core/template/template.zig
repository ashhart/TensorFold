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
pub const unicode = @import("unicode.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ TemplateFailed, OutOfMemory };
pub const Str = v.Str;
pub const Limits = eval.Limits;
pub const Span = v.Span;

/// Why rendering failed; `raised` marks the template's own raise_exception message.
pub const Diag = struct { msg: []const u8 = "", raised: bool = false };

/// A parsed template; its arena owns the syntax tree.
pub const Compiled = struct {
    /// Its child allocator also backs each render's scratch, so concurrent renders need it thread-safe.
    arena: std.heap.ArenaAllocator,
    tree: ast.Template,
    limits: eval.Limits = .{},

    pub fn deinit(t: *Compiled) void {
        t.arena.deinit();
    }
};

/// What a walk of the tree found: a filter or test the engine lacks, and an expression taller than the parser's depth.
const Check = struct { bad: ?[]const u8 = null, tall: bool = false, room: u32 = parser.max_depth };

fn checkNames(stmts: []const ast.Stmt, found: *Check) void {
    const W = struct {
        fn expr(e: *const ast.Expr, out: *Check) void {
            // a loop builds `a + a + a ...` without recursing: its height is checked here, before any whole walk
            if (out.room == 0) {
                out.tall = true;
                return;
            }
            out.room -= 1;
            defer out.room += 1;
            switch (e.*) {
                .filter => |f| if (!access.among(f.name, &filters.implemented)) {
                    out.bad = f.name;
                },
                .is => |t| if (!access.among(t.name, &filters.tests)) {
                    out.bad = t.name;
                },
                else => {},
            }
            ast.eachChild(error{}, e, out, child) catch |err| switch (err) {};
        }
        fn child(out: *Check, e: *ast.Expr) error{}!void {
            expr(e, out);
        }
        fn body(b: []const ast.Stmt, out: *Check) void {
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
        fn ifs(n: *const ast.If, out: *Check) void {
            expr(n.test_expr, out);
            body(n.body, out);
            for (n.elifs) |e| ifs(e, out);
            body(n.else_body, out);
        }
    };
    W.body(stmts, found);
}

/// The render arena's source of memory, which gives out `left` bytes and then refuses.
const Capped = struct {
    child: Allocator,
    left: usize,
    spent: bool = false,

    fn allocator(c: *Capped) Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn take(c: *Capped, n: usize) bool {
        if (n > c.left) c.spent = true else c.left -= n;
        return !c.spent;
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const c: *Capped = @ptrCast(@alignCast(ctx));
        if (!c.take(len)) return null;
        if (c.child.rawAlloc(len, alignment, ra)) |p| return p;
        c.left += len; // the child refused, so nothing was spent
        return null;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const c: *Capped = @ptrCast(@alignCast(ctx));
        const more = new_len -| memory.len;
        if (!c.take(more)) return false;
        if (c.child.rawResize(memory, alignment, new_len, ra)) return true;
        c.left += more; // the arena asks to grow a block in place before it takes a new one, and is often refused
        return false;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const c: *Capped = @ptrCast(@alignCast(ctx));
        const more = new_len -| memory.len;
        if (!c.take(more)) return null;
        if (c.child.rawRemap(memory, alignment, new_len, ra)) |p| return p;
        c.left += more;
        return null;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const c: *Capped = @ptrCast(@alignCast(ctx));
        c.child.rawFree(memory, alignment, ra);
    }
};

test "security: a render's byte cap counts what its allocator gave, not what it refused" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1, .resize_fail_index = 0 });
    var capped: Capped = .{ .child = failing.allocator(), .left = 4096 };
    const a = capped.allocator();
    const block = try a.alloc(u8, 1024);
    defer a.free(block);
    // the child grows nothing in place and gives no second block, so the cap has given nothing more
    try std.testing.expect(!a.resize(block, 3072));
    try std.testing.expect(a.remap(block, 3072) == null);
    try std.testing.expectError(error.OutOfMemory, a.alloc(u8, 2048));
    try std.testing.expectEqual(@as(usize, 3072), capped.left);
    try std.testing.expect(!capped.spent);
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
    var found: Check = .{};
    checkNames(body, &found);
    if (found.tall) return report(gpa, diag, "template syntax error: an expression is nested more than {d} levels deep", .{parser.max_depth});
    var z = scope.Analyzer{ .a = a };
    const frame = z.run(body) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else report(gpa, diag, "template syntax error: {s}", .{z.msg});
    if (found.bad) |name| return report(gpa, diag, "template uses unsupported filter or test '{s}'", .{name});
    return .{ .arena = arena, .tree = .{ .body = body, .frame = frame } };
}

/// Renders with HF's variables: messages, tools (none when absent), documents, add_generation_prompt, then `context`.
pub fn render(gpa: Allocator, t: *const Compiled, messages: std.json.Value, tools: std.json.Value, context: std.json.Value, add_generation_prompt: bool, diag: ?*Diag) Error![]u8 {
    const result = try renderProvenance(gpa, t, messages, tools, context, add_generation_prompt, diag);
    gpa.free(result.spans);
    return @constCast(result.s);
}

/// Renders the actual request text and its byte provenance; the caller owns both slices.
pub fn renderProvenance(gpa: Allocator, t: *const Compiled, messages: std.json.Value, tools: std.json.Value, context: std.json.Value, add_generation_prompt: bool, diag: ?*Diag) Error!Str {
    return renderProvenanceGlobals(gpa, t, messages, tools, context, .null, add_generation_prompt, diag);
}

/// Checkpoint-owned globals are template data; request context overrides them and retains its provenance.
pub fn renderProvenanceGlobals(gpa: Allocator, t: *const Compiled, messages: std.json.Value, tools: std.json.Value, context: std.json.Value, template_globals: std.json.Value, add_generation_prompt: bool, diag: ?*Diag) Error!Str {
    // not on `gpa`: an arena there, such as a server request's, would keep the freed scratch until it is reset
    var capped: Capped = .{ .child = t.arena.child_allocator, .left = t.limits.bytes };
    var arena = std.heap.ArenaAllocator.init(capped.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    var out: v.Buffer = .{};
    var vars: v.Map = .empty;
    var c = eval.Ctx{ .rt = .{ .a = a }, .vars = &vars, .out = &out, .limits = t.limits };
    run(&c, &vars, t, messages, tools, context, template_globals, add_generation_prompt) catch |err| {
        if (err == error.OutOfMemory and capped.spent) return report(gpa, diag, "the chat template needs more than {d} MiB to render this request", .{t.limits.bytes >> 20});
        if (err == error.OutOfMemory) return error.OutOfMemory;
        if (diag) |d| d.raised = c.rt.raised;
        return report(gpa, diag, "{s}", .{c.rt.msg});
    };
    const text = try gpa.dupe(u8, out.text.items);
    errdefer gpa.free(text);
    return .{ .s = text, .spans = try gpa.dupe(Span, out.spans.items) };
}

fn run(c: *eval.Ctx, vars: *v.Map, t: *const Compiled, messages: std.json.Value, tools: std.json.Value, context: std.json.Value, template_globals: std.json.Value, add_generation_prompt: bool) v.Error!void {
    const a = c.rt.a;
    if (template_globals == .object) {
        var it = template_globals.object.iterator();
        while (it.next()) |e| try vars.put(a, e.key_ptr.*, try pyjson.fromJsonDerived(&c.rt, e.value_ptr.*, false));
    }
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
    _ = @import("limits_test.zig");
    _ = @import("unicode.zig");
    _ = @import("value.zig");
    _ = @import("provenance_test.zig");
}
