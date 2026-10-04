//! Jinja2's frame analysis (idtracking): which frame owns each name and how a frame loads it on entry.
const std = @import("std");
const ast = @import("ast.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ SyntaxError, OutOfMemory };

const Map = std.StringArrayHashMapUnmanaged(ast.Init);
const Set = std.StringArrayHashMapUnmanaged(void);

const Symbols = struct {
    frame: *ast.Frame,
    parent: ?*const Symbols,
    refs: Map = .empty,
    stores: Set = .empty,

    fn findRef(s: *const Symbols, name: []const u8) ?*const Symbols {
        if (s.refs.contains(name)) return s;
        return if (s.parent) |p| p.findRef(name) else null;
    }
};

pub const Analyzer = struct {
    a: Allocator,
    msg: []const u8 = "",

    fn fail(z: *Analyzer, comptime fmt: []const u8, args: anytype) Error {
        z.msg = std.fmt.allocPrint(z.a, fmt, args) catch return error.OutOfMemory;
        return error.SyntaxError;
    }

    fn slot(z: *Analyzer, f: *ast.Frame, name: []const u8) !u32 {
        for (f.names.items, 0..) |n, i| if (std.mem.eql(u8, n, name)) return @intCast(i);
        try f.names.append(z.a, name);
        try f.inits.append(z.a, .undefined);
        return @intCast(f.names.items.len - 1);
    }

    fn load(z: *Analyzer, s: *Symbols, name: []const u8) !void {
        if (s.findRef(name) == null) try s.refs.put(z.a, name, .resolve);
    }

    fn store(z: *Analyzer, s: *Symbols, name: []const u8) !void {
        try s.stores.put(z.a, name, {});
        if (s.refs.contains(name)) return;
        if (s.parent) |p| if (p.findRef(name)) |owner| {
            return s.refs.put(z.a, name, .{ .alias = .{ .frame = owner.frame, .slot = try z.slot(owner.frame, name) } });
        };
        try s.refs.put(z.a, name, .undefined);
    }

    fn param(z: *Analyzer, s: *Symbols, name: []const u8) !void {
        try s.stores.put(z.a, name, {});
        try s.refs.put(z.a, name, .param);
    }

    fn copy(z: *Analyzer, s: *const Symbols) !Symbols {
        return .{ .frame = s.frame, .parent = s.parent, .refs = try s.refs.clone(z.a), .stores = try s.stores.clone(z.a) };
    }

    /// Names first stored inside an if branch load from the outer frame or the context on entry.
    fn branchUpdate(z: *Analyzer, s: *Symbols, branches: []const Symbols) !void {
        var fresh: Set = .empty;
        for (branches) |b| for (b.stores.keys()) |k| if (!s.stores.contains(k)) try fresh.put(z.a, k, {});
        for (branches) |b| {
            var it = b.refs.iterator();
            while (it.next()) |e| try s.refs.put(z.a, e.key_ptr.*, e.value_ptr.*);
            for (b.stores.keys()) |k| try s.stores.put(z.a, k, {});
        }
        for (fresh.keys()) |name| {
            if (s.parent) |p| if (p.findRef(name)) |owner| {
                try s.refs.put(z.a, name, .{ .alias = .{ .frame = owner.frame, .slot = try z.slot(owner.frame, name) } });
                continue;
            };
            try s.refs.put(z.a, name, .resolve);
        }
    }

    fn visitStmts(z: *Analyzer, s: *Symbols, body: []const ast.Stmt) Error!void {
        for (body) |st| switch (st) {
            .output => |pieces| for (pieces) |pc| if (pc == .expr) try z.visitExpr(s, pc.expr),
            .for_ => |f| try z.visitExpr(s, f.iter),
            .if_ => |n| try z.visitIf(s, n),
            .assign => |as| {
                try z.visitExpr(s, as.value);
                try z.visitTarget(s, as.target, false);
            },
            .block_set => |b| try z.visitTarget(s, b.target, false),
            .macro => |m| try z.store(s, m.name.id),
            .scoped, .break_, .continue_ => {},
        };
    }

    fn visitIf(z: *Analyzer, s: *Symbols, n: *const ast.If) Error!void {
        try z.visitExpr(s, n.test_expr);
        var branches: [3]Symbols = undefined;
        branches[0] = try z.copy(s);
        try z.visitStmts(&branches[0], n.body);
        branches[1] = try z.copy(s);
        for (n.elifs) |e| try z.visitIf(&branches[1], e);
        branches[2] = try z.copy(s);
        try z.visitStmts(&branches[2], n.else_body);
        try z.branchUpdate(s, &branches);
    }

    fn visitTarget(z: *Analyzer, s: *Symbols, t: ast.Target, as_param: bool) Error!void {
        switch (t) {
            .name => |n| if (as_param) try z.param(s, n.id) else try z.store(s, n.id),
            .tuple => |items| for (items) |it| try z.visitTarget(s, it, as_param),
            .ns => |ns| try z.load(s, ns.name.id),
        }
    }

    /// One analyzer and symbol table, carried through `ast.eachChild`.
    const Pass = struct {
        z: *Analyzer,
        s: *Symbols,

        fn visit(p: Pass, e: *ast.Expr) Error!void {
            return p.z.visitExpr(p.s, e);
        }

        fn bind(p: Pass, e: *ast.Expr) Error!void {
            return p.z.bindExpr(p.s, e);
        }
    };

    fn visitExpr(z: *Analyzer, s: *Symbols, e: *const ast.Expr) Error!void {
        if (e.* == .name) return z.load(s, e.name.id);
        return ast.eachChild(Error, e, Pass{ .z = z, .s = s }, Pass.visit);
    }

    fn finish(z: *Analyzer, s: *const Symbols) !void {
        var it = s.refs.iterator();
        while (it.next()) |e| s.frame.inits.items[try z.slot(s.frame, e.key_ptr.*)] = e.value_ptr.*;
    }

    fn child(z: *Analyzer, s: *const Symbols) !Symbols {
        const f = try z.a.create(ast.Frame);
        f.* = .{ .parent = s.frame };
        return .{ .frame = f, .parent = s };
    }

    fn bind(z: *Analyzer, s: *const Symbols, n: *ast.Name) !void {
        const owner = s.findRef(n.id) orelse return z.fail("name '{s}' has no frame", .{n.id});
        n.frame = owner.frame;
        n.slot = try z.slot(owner.frame, n.id);
    }

    fn bindTarget(z: *Analyzer, s: *const Symbols, t: ast.Target) Error!void {
        switch (t) {
            .name => |n| try z.bind(s, n),
            .tuple => |items| for (items) |it| try z.bindTarget(s, it),
            .ns => |ns| try z.bind(s, ns.name),
        }
    }

    /// Binds this frame's names, then analyzes each nested frame against the finished outer symbols.
    fn bindStmts(z: *Analyzer, s: *Symbols, body: []const ast.Stmt) Error!void {
        for (body) |st| switch (st) {
            .output => |pieces| for (pieces) |pc| if (pc == .expr) try z.bindExpr(s, pc.expr),
            .for_ => |f| try z.forFrames(s, f),
            .if_ => |n| try z.bindIf(s, n),
            .assign => |as| {
                try z.bindExpr(s, as.value);
                try z.bindTarget(s, as.target);
            },
            .block_set => |b| {
                try z.bindTarget(s, b.target);
                var c = try z.child(s);
                try z.visitStmts(&c, b.body);
                if (b.filter) |f| try z.visitExpr(&c, f);
                try z.finish(&c);
                b.frame = c.frame;
                if (b.filter) |f| try z.bindExpr(&c, f);
                try z.bindStmts(&c, b.body);
            },
            .macro => |m| {
                try z.bind(s, m.name);
                for (m.params) |p| for ([_][]const u8{ "caller", "varargs", "kwargs" }) |special| if (std.mem.eql(u8, p.id, special)) return z.fail("macro parameter '{s}' is not supported", .{special});
                for ([_][]const u8{ "caller", "varargs", "kwargs" }) |special| if (loadsName(m.body, special)) return z.fail("macros using '{s}' are not supported", .{special});
                var c = try z.child(s);
                for (m.params) |p| try z.param(&c, p.id);
                for (m.defaults) |d| try z.visitExpr(&c, d);
                try z.visitStmts(&c, m.body);
                try z.finish(&c);
                m.frame = c.frame;
                for (m.params) |p| try z.bind(&c, p);
                for (m.defaults) |d| try z.bindExpr(&c, d);
                try z.bindStmts(&c, m.body);
            },
            .scoped => |sc| {
                var c = try z.child(s);
                try z.visitStmts(&c, sc.body);
                try z.finish(&c);
                sc.frame = c.frame;
                try z.bindStmts(&c, sc.body);
            },
            .break_, .continue_ => {},
        };
    }

    fn bindIf(z: *Analyzer, s: *Symbols, n: *const ast.If) Error!void {
        try z.bindExpr(s, n.test_expr);
        try z.bindStmts(s, n.body);
        for (n.elifs) |e| try z.bindIf(s, e);
        try z.bindStmts(s, n.else_body);
    }

    fn forFrames(z: *Analyzer, s: *Symbols, f: *ast.For) Error!void {
        try z.bindExpr(s, f.iter);
        if (storesLoop(f)) return z.fail("Can't assign to special loop variable in for-loop target", .{});
        f.uses_loop = loadsName(f.body, "loop");
        var body = try z.child(s);
        if (f.uses_loop) try z.param(&body, "loop");
        try z.visitTarget(&body, f.target, true);
        try z.visitStmts(&body, f.body);
        try z.finish(&body);
        f.body_frame = body.frame;
        if (f.uses_loop) {
            f.loop_name = try z.a.create(ast.Name);
            f.loop_name.* = .{ .id = "loop", .ctx = .param, .frame = body.frame, .slot = try z.slot(body.frame, "loop") };
        }
        try z.bindTarget(&body, f.target);
        try z.bindStmts(&body, f.body);
        if (f.filter) |t| {
            var tf = try z.child(s);
            try z.visitTarget(&tf, f.target, true);
            try z.visitExpr(&tf, t);
            try z.finish(&tf);
            f.test_frame = tf.frame;
            try z.bindExpr(&tf, t);
        }
        var el = try z.child(s);
        try z.visitStmts(&el, f.else_body);
        try z.finish(&el);
        f.else_frame = el.frame;
        try z.bindStmts(&el, f.else_body);
    }

    fn bindExpr(z: *Analyzer, s: *Symbols, e: *ast.Expr) Error!void {
        if (e.* == .name) return z.bind(s, e.name);
        return ast.eachChild(Error, e, Pass{ .z = z, .s = s }, Pass.bind);
    }

    /// Python rejects break and continue outside a loop of the same function; macro and call bodies are functions.
    fn loopControl(z: *Analyzer, body: []const ast.Stmt, in_loop: bool) Error!void {
        for (body) |st| switch (st) {
            .break_, .continue_ => if (!in_loop) return z.fail("'{s}' outside loop", .{if (st == .break_) "break" else "continue"}),
            .for_ => |f| {
                try z.loopControl(f.body, true);
                try z.loopControl(f.else_body, in_loop);
            },
            .if_ => |n| try z.ifControl(n, in_loop),
            .block_set => |b| try z.loopControl(b.body, false),
            .macro => |m| {
                for (m.params, 0..) |p, i| for (m.params[0..i]) |q| if (std.mem.eql(u8, p.id, q.id)) return z.fail("duplicate argument '{s}' in macro '{s}'", .{ p.id, m.name.id });
                try z.loopControl(m.body, false);
            },
            .scoped => |sc| try z.loopControl(sc.body, false),
            .output, .assign => {},
        };
    }

    fn ifControl(z: *Analyzer, n: *const ast.If, in_loop: bool) Error!void {
        try z.loopControl(n.body, in_loop);
        for (n.elifs) |e| try z.ifControl(e, in_loop);
        try z.loopControl(n.else_body, in_loop);
    }

    pub fn run(z: *Analyzer, body: []const ast.Stmt) Error!*ast.Frame {
        try z.loopControl(body, false);
        const root = try z.a.create(ast.Frame);
        root.* = .{ .parent = null };
        var s = Symbols{ .frame = root, .parent = null };
        try z.visitStmts(&s, body);
        try z.finish(&s);
        try z.bindStmts(&s, body);
        return root;
    }
};

fn storesLoop(f: *const ast.For) bool {
    const Walk = struct {
        fn target(t: ast.Target) bool {
            return switch (t) {
                .name => |n| std.mem.eql(u8, n.id, "loop"),
                .tuple => |items| for (items) |it| {
                    if (target(it)) break true;
                } else false,
                .ns => false,
            };
        }
        fn stmts(body: []const ast.Stmt) bool {
            for (body) |st| if (switch (st) {
                .for_ => |x| target(x.target) or stmts(x.body) or stmts(x.else_body),
                .if_ => |x| ifs(x),
                .assign => |x| target(x.target),
                .block_set => |x| target(x.target) or stmts(x.body),
                .macro => |x| stmts(x.body),
                .scoped => |x| stmts(x.body),
                else => false,
            }) return true;
            return false;
        }
        fn ifs(n: *const ast.If) bool {
            if (stmts(n.body) or stmts(n.else_body)) return true;
            for (n.elifs) |e| if (ifs(e)) return true;
            return false;
        }
    };
    return Walk.target(f.target) or Walk.stmts(f.body) or Walk.stmts(f.else_body);
}

/// Whether any expression under `body` loads `id`, as Jinja2's find_undeclared sees it.
pub fn loadsName(body: []const ast.Stmt, id: []const u8) bool {
    const Walk = struct {
        id: []const u8,
        fn expr(w: @This(), e: *const ast.Expr) bool {
            if (e.* == .name) return e.name.ctx == .load and std.mem.eql(u8, e.name.id, w.id);
            ast.eachChild(error{Found}, e, w, found) catch return true;
            return false;
        }
        fn found(w: @This(), e: *ast.Expr) error{Found}!void {
            if (w.expr(e)) return error.Found;
        }
        fn target(w: @This(), t: ast.Target) bool {
            return switch (t) {
                .ns => |ns| std.mem.eql(u8, ns.name.id, w.id),
                .tuple => |items| for (items) |it| {
                    if (w.target(it)) break true;
                } else false,
                .name => false,
            };
        }
        fn stmts(w: @This(), b: []const ast.Stmt) bool {
            for (b) |st| if (switch (st) {
                .output => |pieces| for (pieces) |pc| {
                    if (pc == .expr and w.expr(pc.expr)) break true;
                } else false,
                .for_ => |f| w.expr(f.iter) or w.stmts(f.body) or w.stmts(f.else_body) or (if (f.filter) |t| w.expr(t) else false),
                .if_ => |n| w.ifs(n),
                .assign => |as| w.expr(as.value) or w.target(as.target),
                .block_set => |bs| w.stmts(bs.body) or w.target(bs.target) or (if (bs.filter) |f| w.expr(f) else false),
                .macro => |m| w.stmts(m.body) or for (m.defaults) |d| {
                    if (w.expr(d)) break true;
                } else false,
                .scoped => |sc| w.stmts(sc.body),
                .break_, .continue_ => false,
            }) return true;
            return false;
        }
        fn ifs(w: @This(), n: *const ast.If) bool {
            if (w.expr(n.test_expr) or w.stmts(n.body) or w.stmts(n.else_body)) return true;
            for (n.elifs) |e| if (w.ifs(e)) return true;
            return false;
        }
    };
    return (Walk{ .id = id }).stmts(body);
}
