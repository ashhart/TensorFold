//! Runs a parsed template the way Jinja2's compiled Python does: frames, statements, expressions, macro calls.
const std = @import("std");
const ast = @import("ast.zig");
const v = @import("value.zig");
const access = @import("access.zig");
const methods = @import("methods.zig");
const filters = @import("filters.zig");
const Value = v.Value;
const Error = v.Error;
const Kw = methods.Kw;

/// One entry into a frame; macros keep their defining activation as a closure.
pub const Activation = struct {
    frame: *const ast.Frame,
    parent: ?*Activation,
    slots: []Value,
};

/// Macro nesting limit; Python's own frame budget runs out earlier, so this only guards the stack.
const max_depth = 400;

pub const Ctx = struct {
    rt: v.Rt,
    vars: *const v.Map,
    out: *std.ArrayList(u8),
    depth: usize = 0,

    fn resolve(c: *Ctx, name: []const u8) Error!Value {
        if (c.vars.get(name)) |x| return x;
        if (std.meta.stringToEnum(v.Func, name)) |f| return .{ .func = f };
        return c.rt.undef("'{s}' is undefined", .{name});
    }
};

const Flow = enum { normal, brk, cont };

/// The slot of the nearest activation of `frame`; scope.zig binds names only to frames on the lexical chain.
fn slotOf(c: *Ctx, act: *Activation, frame: *const ast.Frame, slot: u32) Error!*Value {
    var cur: ?*Activation = act;
    while (cur) |x| : (cur = x.parent) if (x.frame == frame) return &x.slots[slot];
    return c.rt.fail("internal error: a name's frame is not active", .{});
}

/// Enters a frame: parameters stay missing for the caller, other names load as scope.zig decided.
fn enter(c: *Ctx, parent: ?*Activation, frame: *const ast.Frame) Error!*Activation {
    const act = try c.rt.a.create(Activation);
    act.* = .{ .frame = frame, .parent = parent, .slots = try c.rt.a.alloc(Value, frame.names.items.len) };
    for (act.slots, frame.inits.items, frame.names.items) |*slot, init, name| slot.* = switch (init) {
        .param, .undefined => .missing,
        .resolve => try c.resolve(name),
        .alias => |al| (try slotOf(c, parent.?, al.frame, al.slot)).*,
    };
    return act;
}

fn load(c: *Ctx, act: *Activation, n: *const ast.Name) Error!Value {
    const x = (try slotOf(c, act, n.frame, n.slot)).*;
    return if (x == .missing) c.rt.undef("'{s}' is undefined", .{n.id}) else x;
}

fn assign(c: *Ctx, act: *Activation, t: ast.Target, x: Value) Error!void {
    switch (t) {
        .name => |n| (try slotOf(c, act, n.frame, n.slot)).* = x,
        .tuple => |items| for (items, try access.unpack(&c.rt, x, items.len)) |it, part| try assign(c, act, it, part),
        .ns => |ns| {
            const target = (try slotOf(c, act, ns.name.frame, ns.name.slot)).*;
            if (target != .ns) return c.rt.fail("cannot assign attribute on non-namespace object", .{});
            try target.ns.map.put(c.rt.a, ns.attr, x);
        },
    }
}

/// Loop filters run in their own frame, whose slots are found by name.
fn assignByName(c: *Ctx, act: *Activation, t: ast.Target, x: Value) Error!void {
    switch (t) {
        .name => |n| for (act.frame.names.items, 0..) |name, i| {
            if (std.mem.eql(u8, name, n.id)) act.slots[i] = x;
        },
        .tuple => |items| for (items, try access.unpack(&c.rt, x, items.len)) |it, part| try assignByName(c, act, it, part),
        .ns => return c.rt.fail("a loop target cannot be a namespace attribute", .{}),
    }
}

fn exec(c: *Ctx, act: *Activation, body: []const ast.Stmt) Error!Flow {
    for (body) |st| switch (st) {
        .output => |pieces| for (pieces) |pc| switch (pc) {
            .text => |t| try c.out.appendSlice(c.rt.a, t),
            .expr => |e| try c.out.appendSlice(c.rt.a, try v.toStr(&c.rt, try eval(c, act, e))),
        },
        .for_ => |f| try execFor(c, act, f),
        .if_ => |n| {
            const flow = try execIf(c, act, n);
            if (flow != .normal) return flow;
        },
        .assign => |as| try assign(c, act, as.target, try eval(c, act, as.value)),
        .block_set => |b| {
            const inner = try enter(c, act, b.frame);
            var text = Value.string(try capture(c, inner, b.body));
            if (b.filter) |f| text = try applyFilters(c, inner, f, text);
            try assign(c, act, b.target, text);
        },
        .macro => |m| {
            const mv = try c.rt.a.create(v.Macro);
            mv.* = .{ .node = m, .closure = act, .name = m.name.id };
            (try slotOf(c, act, m.name.frame, m.name.slot)).* = .{ .macro = mv };
        },
        .scoped => |sc| try c.out.appendSlice(c.rt.a, try capture(c, try enter(c, act, sc.frame), sc.body)),
        .break_ => return .brk,
        .continue_ => return .cont,
    };
    return .normal;
}

fn capture(c: *Ctx, act: *Activation, body: []const ast.Stmt) Error![]const u8 {
    const saved = c.out;
    var buf: std.ArrayList(u8) = .empty;
    c.out = &buf;
    defer c.out = saved;
    _ = try exec(c, act, body);
    return buf.items;
}

fn execIf(c: *Ctx, act: *Activation, n: *const ast.If) Error!Flow {
    if (v.truthy(try eval(c, act, n.test_expr))) return exec(c, act, n.body);
    for (n.elifs) |e| if (v.truthy(try eval(c, act, e.test_expr))) return exec(c, act, e.body);
    return exec(c, act, n.else_body);
}

fn execFor(c: *Ctx, act: *Activation, f: *const ast.For) Error!void {
    var items = try access.iterate(&c.rt, try eval(c, act, f.iter));
    if (f.filter) |t| {
        const tf = try enter(c, act, f.test_frame);
        var kept: std.ArrayList(Value) = .empty;
        for (items) |item| {
            try assignByName(c, tf, f.target, item);
            if (v.truthy(try eval(c, tf, t))) try kept.append(c.rt.a, item);
        }
        items = kept.items;
    }
    const loop = try c.rt.a.create(v.Loop);
    loop.* = .{ .items = items };
    var completed = false;
    for (items, 0..) |item, i| {
        loop.index0 = i;
        const body = try enter(c, act, f.body_frame);
        if (f.uses_loop) body.slots[f.loop_name.slot] = .{ .loop = loop };
        try assign(c, body, f.target, item);
        switch (try exec(c, body, f.body)) {
            .brk => break,
            .cont => continue,
            .normal => completed = true,
        }
    }
    if (!completed and f.else_body.len > 0) _ = try exec(c, try enter(c, act, f.else_frame), f.else_body);
}

fn args(c: *Ctx, act: *Activation, a: ast.Args) Error!struct { []Value, []Kw } {
    const pos = try c.rt.a.alloc(Value, a.pos.len);
    for (a.pos, pos) |e, *x| x.* = try eval(c, act, e);
    const kw = try c.rt.a.alloc(Kw, a.kw.len);
    for (a.kw, kw) |k, *x| x.* = .{ .name = k.name, .value = try eval(c, act, k.value) };
    return .{ pos, kw };
}

fn applyFilters(c: *Ctx, act: *Activation, e: *const ast.Expr, hole: Value) Error!Value {
    const f = e.filter;
    const value = if (f.value) |inner| (if (inner.* == .filter) try applyFilters(c, act, inner, hole) else try eval(c, act, inner)) else hole;
    const a = try args(c, act, f.args);
    return filters.filter(c, f.name, value, a[0], a[1]);
}

pub fn call(c: *Ctx, callee: Value, pos: []const Value, kw: []const Kw) Error!Value {
    return switch (callee) {
        .macro => |m| callMacro(c, m, pos, kw),
        .method => |m| methods.callMethod(&c.rt, m, pos, kw),
        .func => |f| methods.callFunc(&c.rt, f, pos, kw),
        .missing, .undef => c.rt.fail("{s}", .{v.undefMsg(callee)}),
        .loop => c.rt.fail("Tried to call non recursive loop.  Maybe you forgot the 'recursive' modifier.", .{}),
        else => c.rt.fail("'{s}' object is not callable", .{v.typeName(callee)}),
    };
}

/// Jinja2's Macro.__call__: positions first, keywords only for the rest, then defaults in the macro's frame.
fn callMacro(c: *Ctx, m: *const v.Macro, pos: []const Value, kw: []const Kw) Error!Value {
    const node: *const ast.Macro = @ptrCast(@alignCast(m.node));
    const params = node.params;
    if (pos.len > params.len) return c.rt.fail("macro '{s}' takes not more than {d} argument(s)", .{ m.name, params.len });
    const act = try enter(c, @ptrCast(@alignCast(m.closure)), node.frame);
    var used = try c.rt.a.alloc(bool, kw.len);
    @memset(used, false);
    for (params, 0..) |p, i| {
        const slot = &act.slots[p.slot];
        if (i < pos.len) {
            slot.* = pos[i];
            continue;
        }
        for (kw, 0..) |k, j| if (!used[j] and std.mem.eql(u8, k.name, p.id)) {
            slot.* = k.value;
            used[j] = true;
        };
    }
    for (kw, used) |k, u| if (!u) return c.rt.fail("macro '{s}' takes no keyword argument '{s}'", .{ m.name, k.name });
    const first_default = params.len - node.defaults.len;
    for (params, 0..) |p, i| if (act.slots[p.slot] == .missing) {
        act.slots[p.slot] = if (i >= first_default) try eval(c, act, node.defaults[i - first_default]) else try c.rt.undef("parameter '{s}' was not provided", .{p.id});
    };
    if (c.depth >= max_depth) return c.rt.fail("maximum recursion depth exceeded", .{});
    c.depth += 1;
    defer c.depth -= 1;
    return Value.string(try capture(c, act, node.body));
}

fn number(c: *Ctx, x: Value, op: []const u8) Error!Value {
    return switch (x) {
        .int, .float => x,
        .boolean => |b| .{ .int = @intFromBool(b) },
        .missing, .undef => c.rt.fail("{s}", .{v.undefMsg(x)}),
        else => c.rt.fail("bad operand type for unary {s}: '{s}'", .{ op, v.typeName(x) }),
    };
}

fn toFloat(x: Value) f64 {
    return switch (x) {
        .int => |i| @floatFromInt(i),
        .float => |f| f,
        .boolean => |b| @floatFromInt(@intFromBool(b)),
        else => unreachable,
    };
}

fn isNum(x: Value) bool {
    return x == .int or x == .float or x == .boolean;
}

fn intOf(x: Value) i64 {
    return if (x == .boolean) @intFromBool(x.boolean) else x.int;
}

fn htmlEscape(c: *Ctx, s: []const u8) Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (s) |ch| switch (ch) {
        '&' => try out.appendSlice(c.rt.a, "&amp;"),
        '<' => try out.appendSlice(c.rt.a, "&lt;"),
        '>' => try out.appendSlice(c.rt.a, "&gt;"),
        '"' => try out.appendSlice(c.rt.a, "&#34;"),
        '\'' => try out.appendSlice(c.rt.a, "&#39;"),
        else => try out.append(c.rt.a, ch),
    };
    return out.items;
}

pub fn escape(c: *Ctx, x: Value) Error!Value {
    if (x == .str and x.str.safe) return x;
    return .{ .str = .{ .s = try htmlEscape(c, try v.toStr(&c.rt, x)), .safe = true } };
}

/// CPython's float floor division and modulo, signed like the divisor.
fn floatDivmod(a: f64, b: f64) [2]f64 {
    var mod = @rem(a, b);
    var div = (a - mod) / b;
    if (mod != 0) {
        if ((b < 0) != (mod < 0)) {
            mod += b;
            div -= 1.0;
        }
    } else mod = std.math.copysign(@as(f64, 0), b);
    var floor_div: f64 = undefined;
    if (div != 0) {
        floor_div = @floor(div);
        if (div - floor_div > 0.5) floor_div += 1.0;
    } else floor_div = std.math.copysign(@as(f64, 0), a / b);
    return .{ floor_div, mod };
}

fn arith(c: *Ctx, op: ast.BinOp, l: Value, r: Value) Error!Value {
    const rt = &c.rt;
    if (op == .div) {
        if (toFloat(r) == 0) return rt.fail("division by zero", .{});
        return .{ .float = toFloat(l) / toFloat(r) };
    }
    if (l != .float and r != .float) {
        const a = intOf(l);
        const b = intOf(r);
        if ((op == .floordiv or op == .mod) and b == 0) return rt.fail("integer division or modulo by zero", .{});
        if (op == .pow and b < 0) {
            if (a == 0) return rt.fail("zero to a negative power", .{});
            return .{ .float = std.math.pow(f64, @floatFromInt(a), @floatFromInt(b)) };
        }
        const res = switch (op) {
            .add => std.math.add(i64, a, b),
            .sub => std.math.sub(i64, a, b),
            .mul => std.math.mul(i64, a, b),
            .floordiv => if (a == std.math.minInt(i64) and b == -1) error.Overflow else @divFloor(a, b),
            .mod => @mod(a, b),
            .pow => std.math.powi(i64, a, b),
            .div => unreachable,
        } catch return rt.fail("integer overflow", .{});
        return .{ .int = res };
    }
    const a = toFloat(l);
    const b = toFloat(r);
    if ((op == .floordiv or op == .mod) and b == 0) return rt.fail("float divmod()", .{});
    if (op == .pow and a == 0 and b < 0) return rt.fail("zero to a negative power", .{});
    return .{ .float = switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .floordiv => floatDivmod(a, b)[0],
        .mod => floatDivmod(a, b)[1],
        .pow => std.math.pow(f64, a, b),
        .div => unreachable,
    } };
}

fn binop(c: *Ctx, op: ast.BinOp, l: Value, r: Value) Error!Value {
    const rt = &c.rt;
    if (v.isUndef(l) or v.isUndef(r)) return rt.fail("{s}", .{v.undefMsg(if (v.isUndef(l)) l else r)});
    const sym = switch (op) {
        .add => "+",
        .sub => "-",
        .mul => "*",
        .div => "/",
        .floordiv => "//",
        .mod => "%",
        .pow => "**",
    };
    if (isNum(l) and isNum(r)) return arith(c, op, l, r);
    if (op == .add and l == .str and r == .str) {
        const safe = l.str.safe or r.str.safe;
        const ls = if (safe and !l.str.safe) try htmlEscape(c, l.str.s) else l.str.s;
        const rs = if (safe and !r.str.safe) try htmlEscape(c, r.str.s) else r.str.s;
        return .{ .str = .{ .s = try std.mem.concat(rt.a, u8, &.{ ls, rs }), .safe = safe } };
    }
    if (op == .add and l == .list and r == .list and l.list.seq == r.list.seq and (l.list.seq == .list or l.list.seq == .tuple))
        return rt.list(try std.mem.concat(rt.a, Value, &.{ l.list.items, r.list.items }), l.list.seq);
    if (op == .mul and (l == .str or l == .list or r == .str or r == .list) and (isNum(l) or isNum(r)) and l != .float and r != .float) {
        const seq = if (isNum(l)) r else l;
        const n = @max(0, intOf(if (isNum(l)) l else r));
        if (seq == .str) {
            var out: std.ArrayList(u8) = .empty;
            for (0..@intCast(n)) |_| try out.appendSlice(rt.a, seq.str.s);
            return .{ .str = .{ .s = out.items, .safe = seq.str.safe } };
        }
        if (seq.list.seq == .list or seq.list.seq == .tuple) {
            var out: std.ArrayList(Value) = .empty;
            for (0..@intCast(n)) |_| try out.appendSlice(rt.a, seq.list.items);
            return rt.list(out.items, seq.list.seq);
        }
    }
    if (op == .add and l == .str) return rt.fail("can only concatenate str (not \"{s}\") to str", .{v.typeName(r)});
    if (op == .mod and l == .str) return rt.fail("printf-style string formatting is not supported", .{});
    return rt.fail("unsupported operand type(s) for {s}: '{s}' and '{s}'", .{ sym, v.typeName(l), v.typeName(r) });
}

pub fn compare(c: *Ctx, op: ast.CmpOp, l: Value, r: Value) Error!bool {
    const sym = switch (op) {
        .lt => "<",
        .lteq => "<=",
        .gt => ">",
        .gteq => ">=",
        else => "",
    };
    return switch (op) {
        .eq => v.eql(l, r),
        .ne => !v.eql(l, r),
        .in => access.contains(&c.rt, r, l),
        .notin => !try access.contains(&c.rt, r, l),
        else => {
            const ord = (try v.order(&c.rt, sym, l, r)) orelse return false;
            return switch (op) {
                .lt => ord == .lt,
                .lteq => ord != .gt,
                .gt => ord == .gt,
                .gteq => ord != .lt,
                else => unreachable,
            };
        },
    };
}

pub fn eval(c: *Ctx, act: *Activation, e: *const ast.Expr) Error!Value {
    const rt = &c.rt;
    return switch (e.*) {
        .constant => |k| switch (k) {
            .none => .none,
            .boolean => |b| .{ .boolean = b },
            .int => |i| .{ .int = i },
            .big => |b| .{ .big = b },
            .float => |f| .{ .float = f },
            .str => |s| Value.string(s),
        },
        .name => |n| load(c, act, n),
        .getattr => |g| access.getattr(rt, try eval(c, act, g.obj), g.attr),
        .getitem => |g| blk: {
            const obj = try eval(c, act, g.obj);
            if (g.arg.* == .slice) {
                const sl = g.arg.slice;
                const parts = [3]?*ast.Expr{ sl.start, sl.stop, sl.step };
                var vals: [3]Value = .{ .none, .none, .none };
                for (parts, &vals) |p, *x| if (p) |pe| {
                    x.* = try eval(c, act, pe);
                };
                break :blk access.slice(rt, obj, vals[0], vals[1], vals[2]);
            }
            break :blk access.getitem(rt, obj, try eval(c, act, g.arg));
        },
        .slice => rt.fail("slice outside a subscript", .{}),
        .call => |cl| blk: {
            const callee = try eval(c, act, cl.callee);
            const a = try args(c, act, cl.args);
            break :blk call(c, callee, a[0], a[1]);
        },
        .filter => applyFilters(c, act, e, .none),
        .is => |t| blk: {
            const value = try eval(c, act, t.value);
            const a = try args(c, act, t.args);
            break :blk .{ .boolean = try filters.testValue(c, t.name, value, a[0], a[1]) };
        },
        .cond => |cd| if (v.truthy(try eval(c, act, cd.test_expr))) eval(c, act, cd.yes) else if (cd.no) |no| eval(c, act, no) else rt.undef("the inline if-expression evaluated to false and no else section was defined.", .{}),
        .and_ => |p| blk: {
            const l = try eval(c, act, p[0]);
            break :blk if (!v.truthy(l)) l else eval(c, act, p[1]);
        },
        .or_ => |p| blk: {
            const l = try eval(c, act, p[0]);
            break :blk if (v.truthy(l)) l else eval(c, act, p[1]);
        },
        .not_ => |x| .{ .boolean = !v.truthy(try eval(c, act, x)) },
        .neg => |x| blk: {
            const n = try number(c, try eval(c, act, x), "-");
            break :blk if (n == .float) Value{ .float = -n.float } else if (n.int == std.math.minInt(i64)) rt.fail("integer overflow", .{}) else Value{ .int = -n.int };
        },
        .pos => |x| number(c, try eval(c, act, x), "+"),
        .compare => |cmp| blk: {
            var left = try eval(c, act, cmp.first);
            for (cmp.ops) |o| {
                const right = try eval(c, act, o.expr);
                if (!try compare(c, o.op, left, right)) break :blk .{ .boolean = false };
                left = right;
            }
            break :blk .{ .boolean = true };
        },
        .binop => |b| binop(c, b.op, try eval(c, act, b.l), try eval(c, act, b.r)),
        .concat => |items| blk: {
            var out: std.ArrayList(u8) = .empty;
            for (items) |x| try out.appendSlice(rt.a, try v.toStr(rt, try eval(c, act, x)));
            break :blk Value.string(out.items);
        },
        .list, .tuple => |items| blk: {
            const vals = try rt.a.alloc(Value, items.len);
            for (items, vals) |x, *y| y.* = try eval(c, act, x);
            break :blk rt.list(vals, if (e.* == .list) .list else .tuple);
        },
        .dict => |pairs| blk: {
            const d = try rt.a.create(v.Dict);
            d.* = .{ .map = .empty };
            for (pairs) |p| {
                const key = try eval(c, act, p.key);
                if (key != .str) break :blk rt.fail("dict literals with non-string keys are not supported", .{});
                try d.map.put(rt.a, key.str.s, try eval(c, act, p.value));
            }
            break :blk .{ .dict = d };
        },
    };
}

pub fn root(c: *Ctx, t: ast.Template) Error!void {
    _ = try exec(c, try enter(c, null, t.frame), t.body);
}
