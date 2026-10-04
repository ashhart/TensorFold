//! Jinja2's grammar and operator precedence for the statements chat templates use.
const std = @import("std");
const ast = @import("ast.zig");
const lex = @import("lexer.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ SyntaxError, OutOfMemory };

pub const Parser = struct {
    a: Allocator,
    toks: []const lex.Token,
    i: usize = 0,
    msg: []const u8 = "",

    fn fail(p: *Parser, comptime fmt: []const u8, args: anytype) Error {
        p.msg = std.fmt.allocPrint(p.a, "line {d}: " ++ fmt, .{p.cur().line} ++ args) catch return error.OutOfMemory;
        return error.SyntaxError;
    }

    fn cur(p: *const Parser) lex.Token {
        return p.toks[p.i];
    }

    fn look(p: *const Parser) lex.Token {
        return p.toks[@min(p.i + 1, p.toks.len - 1)];
    }

    fn advance(p: *Parser) lex.Token {
        const t = p.toks[p.i];
        if (p.i + 1 < p.toks.len) p.i += 1;
        return t;
    }

    fn isName(p: *const Parser, name: []const u8) bool {
        const t = p.cur();
        return t.tag == .name and std.mem.eql(u8, t.text, name);
    }

    fn isOp(p: *const Parser, op: lex.Op) bool {
        const t = p.cur();
        return t.tag == .op and t.op == op;
    }

    fn skipName(p: *Parser, name: []const u8) bool {
        if (!p.isName(name)) return false;
        _ = p.advance();
        return true;
    }

    fn skipOp(p: *Parser, op: lex.Op) bool {
        if (!p.isOp(op)) return false;
        _ = p.advance();
        return true;
    }

    fn expectOp(p: *Parser, op: lex.Op) !void {
        if (!p.skipOp(op)) return p.fail("expected '{s}'", .{@tagName(op)});
    }

    fn expectName(p: *Parser) ![]const u8 {
        if (p.cur().tag != .name) return p.fail("expected a name", .{});
        return p.advance().text;
    }

    fn expectTag(p: *Parser, tag: lex.Tag) !void {
        if (p.cur().tag != tag) return p.fail("expected {s}, got {s}", .{ @tagName(tag), @tagName(p.cur().tag) });
        _ = p.advance();
    }

    fn new(p: *Parser, v: anytype) !*@TypeOf(v) {
        const ptr = try p.a.create(@TypeOf(v));
        ptr.* = v;
        return ptr;
    }

    fn expr(p: *Parser, v: ast.Expr) !*ast.Expr {
        return p.new(v);
    }

    fn newName(p: *Parser, id: []const u8, ctx: ast.Ctx) !*ast.Name {
        return p.new(ast.Name{ .id = id, .ctx = ctx });
    }

    pub fn parse(p: *Parser) Error![]ast.Stmt {
        const body = try p.subparse(&.{});
        if (p.cur().tag != .eof) return p.fail("unexpected '{s}'", .{p.cur().text});
        return body;
    }

    fn subparse(p: *Parser, end: []const []const u8) Error![]ast.Stmt {
        var body: std.ArrayList(ast.Stmt) = .empty;
        var pieces: std.ArrayList(ast.Piece) = .empty;
        while (true) {
            switch (p.cur().tag) {
                .data => try pieces.append(p.a, .{ .text = p.advance().text }),
                .var_begin => {
                    _ = p.advance();
                    try pieces.append(p.a, .{ .expr = try p.tuple(.{}) });
                    try p.expectTag(.var_end);
                },
                .block_begin => {
                    if (pieces.items.len > 0) try body.append(p.a, .{ .output = try pieces.toOwnedSlice(p.a) });
                    _ = p.advance();
                    for (end) |e| if (p.isName(e)) return body.items;
                    try body.append(p.a, try p.statement());
                    try p.expectTag(.block_end);
                },
                .eof => {
                    if (pieces.items.len > 0) try body.append(p.a, .{ .output = try pieces.toOwnedSlice(p.a) });
                    return body.items;
                },
                else => return p.fail("unexpected {s}", .{@tagName(p.cur().tag)}),
            }
        }
    }

    /// Body up to one of `end`, leaving the end tag's name as the current token.
    fn statements(p: *Parser, end: []const []const u8) ![]ast.Stmt {
        _ = p.skipOp(.colon);
        try p.expectTag(.block_end);
        const body = try p.subparse(end);
        if (p.cur().tag == .eof) return p.fail("unexpected end of template, expected '{s}'", .{end[0]});
        return body;
    }

    fn statement(p: *Parser) !ast.Stmt {
        if (p.cur().tag != .name) return p.fail("tag name expected", .{});
        const tag = p.cur().text;
        if (std.mem.eql(u8, tag, "for")) return .{ .for_ = try p.forStmt() };
        if (std.mem.eql(u8, tag, "if")) return .{ .if_ = try p.ifStmt() };
        if (std.mem.eql(u8, tag, "set")) return p.setStmt();
        if (std.mem.eql(u8, tag, "macro")) return .{ .macro = try p.macroStmt() };
        if (std.mem.eql(u8, tag, "print")) {
            _ = p.advance();
            var pieces: std.ArrayList(ast.Piece) = .empty;
            while (p.cur().tag != .block_end) {
                if (pieces.items.len > 0) try p.expectOp(.comma);
                try pieces.append(p.a, .{ .expr = try p.expression(true) });
            }
            return .{ .output = pieces.items };
        }
        if (std.mem.eql(u8, tag, "break") or std.mem.eql(u8, tag, "continue")) {
            _ = p.advance();
            return if (tag[0] == 'b') .break_ else .continue_;
        }
        if (std.mem.eql(u8, tag, "generation")) {
            _ = p.advance();
            const body = try p.statements(&.{"endgeneration"});
            _ = p.advance();
            return .{ .scoped = try p.new(ast.Scoped{ .body = body }) };
        }
        return p.fail("unsupported tag '{s}'", .{tag});
    }

    fn forStmt(p: *Parser) !*ast.For {
        _ = p.advance();
        const target = try p.assignTarget(&.{"in"}, false);
        if (!p.skipName("in")) return p.fail("expected 'in'", .{});
        const iter = try p.tuple(.{ .condexpr = false, .end = &.{"recursive"} });
        const filter = if (p.skipName("if")) try p.expression(true) else null;
        if (p.isName("recursive")) return p.fail("recursive loops are not supported", .{});
        const body = try p.statements(&.{ "endfor", "else" });
        var else_body: []ast.Stmt = &.{};
        if (std.mem.eql(u8, p.advance().text, "else")) {
            else_body = try p.statements(&.{"endfor"});
            _ = p.advance();
        }
        return p.new(ast.For{ .target = target, .iter = iter, .body = body, .else_body = else_body, .filter = filter });
    }

    fn ifStmt(p: *Parser) !*ast.If {
        _ = p.advance();
        const result = try p.new(ast.If{ .test_expr = undefined, .body = &.{}, .elifs = &.{}, .else_body = &.{} });
        var node = result;
        var elifs: std.ArrayList(*ast.If) = .empty;
        while (true) {
            node.test_expr = try p.tuple(.{ .condexpr = false });
            node.body = try p.statements(&.{ "elif", "else", "endif" });
            const t = p.advance().text;
            if (std.mem.eql(u8, t, "elif")) {
                node = try p.new(ast.If{ .test_expr = undefined, .body = &.{}, .elifs = &.{}, .else_body = &.{} });
                try elifs.append(p.a, node);
                continue;
            }
            if (std.mem.eql(u8, t, "else")) {
                result.else_body = try p.statements(&.{"endif"});
                _ = p.advance();
            }
            break;
        }
        result.elifs = elifs.items;
        return result;
    }

    fn setStmt(p: *Parser) !ast.Stmt {
        _ = p.advance();
        const target = try p.assignTarget(&.{}, true);
        if (p.skipOp(.assign)) return .{ .assign = .{ .target = target, .value = try p.tuple(.{}) } };
        const filter = try p.filters(null);
        const body = try p.statements(&.{"endset"});
        _ = p.advance();
        return .{ .block_set = try p.new(ast.BlockSet{ .target = target, .filter = filter, .body = body }) };
    }

    fn macroStmt(p: *Parser) !*ast.Macro {
        _ = p.advance();
        const macro_name = try p.newName(try p.expectName(), .store);
        var params: std.ArrayList(*ast.Name) = .empty;
        var defaults: std.ArrayList(*ast.Expr) = .empty;
        try p.expectOp(.lparen);
        while (!p.isOp(.rparen)) {
            if (params.items.len > 0) try p.expectOp(.comma);
            try params.append(p.a, try p.newName(try p.expectName(), .param));
            if (p.skipOp(.assign)) {
                try defaults.append(p.a, try p.expression(true));
            } else if (defaults.items.len > 0) return p.fail("non-default argument follows default argument", .{});
        }
        try p.expectOp(.rparen);
        const body = try p.statements(&.{"endmacro"});
        _ = p.advance();
        return p.new(ast.Macro{ .name = macro_name, .params = params.items, .defaults = defaults.items, .body = body });
    }

    fn tupleEnd(p: *const Parser, end: []const []const u8) bool {
        switch (p.cur().tag) {
            .var_end, .block_end => return true,
            .op => if (p.cur().op == .rparen) return true,
            else => {},
        }
        for (end) |e| if (p.isName(e)) return true;
        return false;
    }

    /// parse_assign_target with tuples: names, nested parenthesized tuples and `ns.attr`.
    fn assignTarget(p: *Parser, end: []const []const u8, namespace: bool) !ast.Target {
        var items: std.ArrayList(ast.Target) = .empty;
        var is_tuple = false;
        while (true) {
            if (items.items.len > 0) try p.expectOp(.comma);
            if (p.tupleEnd(end)) break;
            try items.append(p.a, try p.targetItem(namespace));
            if (p.isOp(.comma)) is_tuple = true else break;
        }
        if (!is_tuple) {
            if (items.items.len == 0) return p.fail("expected an assignment target", .{});
            return items.items[0];
        }
        return .{ .tuple = items.items };
    }

    fn targetItem(p: *Parser, namespace: bool) Error!ast.Target {
        if (p.skipOp(.lparen)) {
            const inner = try p.assignTarget(&.{}, false);
            try p.expectOp(.rparen);
            return inner;
        }
        const id = try p.expectName();
        for ([_][]const u8{ "true", "false", "True", "False", "none", "None" }) |c| if (std.mem.eql(u8, id, c)) return p.fail("can't assign to const", .{});
        if (namespace and p.skipOp(.dot)) return .{ .ns = .{ .name = try p.newName(id, .load), .attr = try p.expectName() } };
        return .{ .name = try p.newName(id, .store) };
    }

    const TupleOpts = struct { condexpr: bool = true, end: []const []const u8 = &.{}, parens: bool = false };

    fn tuple(p: *Parser, o: TupleOpts) Error!*ast.Expr {
        var items: std.ArrayList(*ast.Expr) = .empty;
        var is_tuple = false;
        while (true) {
            if (items.items.len > 0) try p.expectOp(.comma);
            if (p.tupleEnd(o.end)) break;
            try items.append(p.a, try p.expression(o.condexpr));
            if (p.isOp(.comma)) is_tuple = true else break;
        }
        if (!is_tuple) {
            if (items.items.len > 0) return items.items[0];
            if (!o.parens) return p.fail("expected an expression", .{});
        }
        return p.expr(.{ .tuple = items.items });
    }

    fn expression(p: *Parser, condexpr: bool) Error!*ast.Expr {
        return if (condexpr) p.condExpr() else p.orExpr();
    }

    fn condExpr(p: *Parser) Error!*ast.Expr {
        var yes = try p.orExpr();
        while (p.skipName("if")) {
            const t = try p.orExpr();
            const no = if (p.skipName("else")) try p.condExpr() else null;
            yes = try p.expr(.{ .cond = .{ .test_expr = t, .yes = yes, .no = no } });
        }
        return yes;
    }

    fn orExpr(p: *Parser) Error!*ast.Expr {
        var left = try p.andExpr();
        while (p.skipName("or")) left = try p.expr(.{ .or_ = .{ left, try p.andExpr() } });
        return left;
    }

    fn andExpr(p: *Parser) Error!*ast.Expr {
        var left = try p.notExpr();
        while (p.skipName("and")) left = try p.expr(.{ .and_ = .{ left, try p.notExpr() } });
        return left;
    }

    fn notExpr(p: *Parser) Error!*ast.Expr {
        if (p.skipName("not")) return p.expr(.{ .not_ = try p.notExpr() });
        return p.compare();
    }

    fn compare(p: *Parser) Error!*ast.Expr {
        const first = try p.math1();
        var ops: std.ArrayList(ast.Operand) = .empty;
        while (true) {
            const t = p.cur();
            const op: ast.CmpOp = if (t.tag == .op) switch (t.op) {
                .eq => .eq,
                .ne => .ne,
                .lt => .lt,
                .lteq => .lteq,
                .gt => .gt,
                .gteq => .gteq,
                else => break,
            } else if (p.isName("in")) .in else if (p.isName("not") and p.look().tag == .name and std.mem.eql(u8, p.look().text, "in")) .notin else break;
            _ = p.advance();
            if (op == .notin) _ = p.advance();
            try ops.append(p.a, .{ .op = op, .expr = try p.math1() });
        }
        if (ops.items.len == 0) return first;
        return p.expr(.{ .compare = .{ .first = first, .ops = ops.items } });
    }

    fn math1(p: *Parser) Error!*ast.Expr {
        var left = try p.concat();
        while (p.isOp(.add) or p.isOp(.sub)) {
            const op: ast.BinOp = if (p.advance().op == .add) .add else .sub;
            left = try p.expr(.{ .binop = .{ .op = op, .l = left, .r = try p.concat() } });
        }
        return left;
    }

    fn concat(p: *Parser) Error!*ast.Expr {
        const first = try p.math2();
        if (!p.isOp(.tilde)) return first;
        var items: std.ArrayList(*ast.Expr) = .empty;
        try items.append(p.a, first);
        while (p.skipOp(.tilde)) try items.append(p.a, try p.math2());
        return p.expr(.{ .concat = items.items });
    }

    fn math2(p: *Parser) Error!*ast.Expr {
        var left = try p.pow();
        while (true) {
            const op: ast.BinOp = if (p.cur().tag == .op) switch (p.cur().op) {
                .mul => .mul,
                .div => .div,
                .floordiv => .floordiv,
                .mod => .mod,
                else => break,
            } else break;
            _ = p.advance();
            left = try p.expr(.{ .binop = .{ .op = op, .l = left, .r = try p.pow() } });
        }
        return left;
    }

    fn pow(p: *Parser) Error!*ast.Expr {
        var left = try p.unary(true);
        while (p.skipOp(.pow)) left = try p.expr(.{ .binop = .{ .op = .pow, .l = left, .r = try p.unary(true) } });
        return left;
    }

    fn unary(p: *Parser, with_filter: bool) Error!*ast.Expr {
        var node = if (p.skipOp(.sub)) try p.expr(.{ .neg = try p.unary(false) }) else if (p.skipOp(.add)) try p.expr(.{ .pos = try p.unary(false) }) else try p.primary();
        node = try p.postfix(node);
        if (with_filter) node = try p.filterExpr(node);
        return node;
    }

    fn primary(p: *Parser) Error!*ast.Expr {
        const t = p.cur();
        switch (t.tag) {
            .name => {
                _ = p.advance();
                if (std.mem.eql(u8, t.text, "true") or std.mem.eql(u8, t.text, "True")) return p.expr(.{ .constant = .{ .boolean = true } });
                if (std.mem.eql(u8, t.text, "false") or std.mem.eql(u8, t.text, "False")) return p.expr(.{ .constant = .{ .boolean = false } });
                if (std.mem.eql(u8, t.text, "none") or std.mem.eql(u8, t.text, "None")) return p.expr(.{ .constant = .none });
                return p.expr(.{ .name = try p.newName(t.text, .load) });
            },
            .string => {
                var buf: std.ArrayList(u8) = .empty;
                while (p.cur().tag == .string) try buf.appendSlice(p.a, p.advance().text);
                return p.expr(.{ .constant = .{ .str = buf.items } });
            },
            .integer => {
                const n = p.advance();
                return p.expr(.{ .constant = if (n.text.len > 0) .{ .big = n.text } else .{ .int = n.int } });
            },
            .float => return p.expr(.{ .constant = .{ .float = p.advance().float } }),
            .op => switch (t.op) {
                .lparen => {
                    _ = p.advance();
                    const inner = try p.tuple(.{ .parens = true });
                    try p.expectOp(.rparen);
                    return inner;
                },
                .lbracket => {
                    _ = p.advance();
                    var items: std.ArrayList(*ast.Expr) = .empty;
                    while (!p.isOp(.rbracket)) {
                        if (items.items.len > 0) try p.expectOp(.comma);
                        if (p.isOp(.rbracket)) break;
                        try items.append(p.a, try p.expression(true));
                    }
                    try p.expectOp(.rbracket);
                    return p.expr(.{ .list = items.items });
                },
                .lbrace => {
                    _ = p.advance();
                    var pairs: std.ArrayList(ast.Pair) = .empty;
                    while (!p.isOp(.rbrace)) {
                        if (pairs.items.len > 0) try p.expectOp(.comma);
                        if (p.isOp(.rbrace)) break;
                        const key = try p.expression(true);
                        try p.expectOp(.colon);
                        try pairs.append(p.a, .{ .key = key, .value = try p.expression(true) });
                    }
                    try p.expectOp(.rbrace);
                    return p.expr(.{ .dict = pairs.items });
                },
                else => {},
            },
            else => {},
        }
        return p.fail("unexpected {s}", .{if (t.tag == .op) @tagName(t.op) else @tagName(t.tag)});
    }

    fn postfix(p: *Parser, node_in: *ast.Expr) Error!*ast.Expr {
        var node = node_in;
        while (true) {
            if (p.isOp(.dot) or p.isOp(.lbracket)) {
                node = try p.subscript(node);
            } else if (p.isOp(.lparen)) {
                node = try p.expr(.{ .call = .{ .callee = node, .args = try p.callArgs() } });
            } else return node;
        }
    }

    fn filterExpr(p: *Parser, node_in: *ast.Expr) Error!*ast.Expr {
        var node = node_in;
        while (true) {
            if (p.isOp(.pipe)) {
                node = (try p.filters(node)).?;
            } else if (p.isName("is")) {
                node = try p.testExpr(node);
            } else if (p.isOp(.lparen)) {
                node = try p.expr(.{ .call = .{ .callee = node, .args = try p.callArgs() } });
            } else return node;
        }
    }

    fn subscript(p: *Parser, node: *ast.Expr) Error!*ast.Expr {
        if (p.skipOp(.dot)) {
            const t = p.advance();
            if (t.tag == .name) return p.expr(.{ .getattr = .{ .obj = node, .attr = t.text } });
            if (t.tag != .integer) return p.fail("expected name or number", .{});
            return p.expr(.{ .getitem = .{ .obj = node, .arg = try p.expr(.{ .constant = if (t.text.len > 0) .{ .big = t.text } else .{ .int = t.int } }) } });
        }
        try p.expectOp(.lbracket);
        var args: std.ArrayList(*ast.Expr) = .empty;
        while (!p.isOp(.rbracket)) {
            if (args.items.len > 0) try p.expectOp(.comma);
            try args.append(p.a, try p.subscribed());
        }
        try p.expectOp(.rbracket);
        const arg = if (args.items.len == 1) args.items[0] else try p.expr(.{ .tuple = args.items });
        return p.expr(.{ .getitem = .{ .obj = node, .arg = arg } });
    }

    fn subscribed(p: *Parser) Error!*ast.Expr {
        var start: ?*ast.Expr = null;
        if (!p.skipOp(.colon)) {
            const node = try p.expression(true);
            if (!p.skipOp(.colon)) return node;
            start = node;
        }
        var stop: ?*ast.Expr = null;
        if (!p.isOp(.colon) and !p.isOp(.rbracket) and !p.isOp(.comma)) stop = try p.expression(true);
        var step: ?*ast.Expr = null;
        if (p.skipOp(.colon) and !p.isOp(.rbracket) and !p.isOp(.comma)) step = try p.expression(true);
        return p.expr(.{ .slice = .{ .start = start, .stop = stop, .step = step } });
    }

    fn callArgs(p: *Parser) Error!ast.Args {
        try p.expectOp(.lparen);
        var pos: std.ArrayList(*ast.Expr) = .empty;
        var kw: std.ArrayList(ast.Kwarg) = .empty;
        var need_comma = false;
        while (!p.isOp(.rparen)) {
            if (need_comma) {
                try p.expectOp(.comma);
                if (p.isOp(.rparen)) break;
            }
            if (p.isOp(.mul) or p.isOp(.pow)) return p.fail("*args and **kwargs calls are not supported", .{});
            if (p.cur().tag == .name and p.look().tag == .op and p.look().op == .assign) {
                const key = p.advance().text;
                _ = p.advance();
                try kw.append(p.a, .{ .name = key, .value = try p.expression(true) });
            } else {
                if (kw.items.len > 0) return p.fail("invalid syntax for function call expression", .{});
                try pos.append(p.a, try p.expression(true));
            }
            need_comma = true;
        }
        try p.expectOp(.rparen);
        return .{ .pos = pos.items, .kw = kw.items };
    }

    fn dottedName(p: *Parser) ![]const u8 {
        const first = try p.expectName();
        if (!p.isOp(.dot)) return first;
        return p.fail("dotted filter and test names are not supported", .{});
    }

    /// A filter chain; with no value it is a `{% set x | f %}` block filter.
    fn filters(p: *Parser, node_in: ?*ast.Expr) Error!?*ast.Expr {
        var node = node_in;
        var inline_start = node_in == null and p.cur().tag == .op and p.cur().op == .pipe;
        while (p.isOp(.pipe) or inline_start) {
            _ = p.advance();
            inline_start = false;
            const filter_name = try p.dottedName();
            const args = if (p.isOp(.lparen)) try p.callArgs() else ast.Args{};
            node = try p.expr(.{ .filter = .{ .value = node, .name = filter_name, .args = args } });
        }
        return node;
    }

    fn testExpr(p: *Parser, node: *ast.Expr) Error!*ast.Expr {
        _ = p.advance();
        const negated = p.skipName("not");
        const test_name = try p.dottedName();
        var args = ast.Args{};
        if (p.isOp(.lparen)) {
            args = try p.callArgs();
        } else {
            const t = p.cur();
            const starts = switch (t.tag) {
                .name, .string, .integer, .float => true,
                .op => t.op == .lparen or t.op == .lbracket or t.op == .lbrace,
                else => false,
            };
            if (starts and !p.isName("else") and !p.isName("or") and !p.isName("and")) {
                if (p.isName("is")) return p.fail("You cannot chain multiple tests with is", .{});
                const arg = try p.postfix(try p.primary());
                const list = try p.a.alloc(*ast.Expr, 1);
                list[0] = arg;
                args.pos = list;
            }
        }
        const t = try p.expr(.{ .is = .{ .value = node, .name = test_name, .args = args } });
        return if (negated) p.expr(.{ .not_ = t }) else t;
    }
};
