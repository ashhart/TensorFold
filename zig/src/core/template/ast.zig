//! Template syntax tree; scope.zig fills in each name's frame slot and each frame's entry loads.
const std = @import("std");

pub const Const = union(enum) { none, boolean: bool, int: i64, big: []const u8, float: f64, str: []const u8 };

pub const Ctx = enum { load, store, param };

pub const Name = struct {
    id: []const u8,
    ctx: Ctx,
    frame: *const Frame = undefined,
    slot: u32 = 0,
};

pub const Kwarg = struct { name: []const u8, value: *Expr };

pub const Args = struct { pos: []*Expr = &.{}, kw: []Kwarg = &.{} };

pub const CmpOp = enum { eq, ne, lt, lteq, gt, gteq, in, notin };

pub const BinOp = enum { add, sub, mul, div, floordiv, mod, pow };

pub const Operand = struct { op: CmpOp, expr: *Expr };

pub const Pair = struct { key: *Expr, value: *Expr };

pub const Expr = union(enum) {
    constant: Const,
    name: *Name,
    getattr: struct { obj: *Expr, attr: []const u8 },
    getitem: struct { obj: *Expr, arg: *Expr },
    slice: struct { start: ?*Expr, stop: ?*Expr, step: ?*Expr },
    call: struct { callee: *Expr, args: Args },
    filter: struct { value: ?*Expr, name: []const u8, args: Args },
    is: struct { value: *Expr, name: []const u8, args: Args },
    cond: struct { test_expr: *Expr, yes: *Expr, no: ?*Expr },
    and_: [2]*Expr,
    or_: [2]*Expr,
    not_: *Expr,
    neg: *Expr,
    pos: *Expr,
    compare: struct { first: *Expr, ops: []Operand },
    binop: struct { op: BinOp, l: *Expr, r: *Expr },
    concat: []*Expr,
    list: []*Expr,
    tuple: []*Expr,
    dict: []Pair,
};

pub const Target = union(enum) {
    name: *Name,
    tuple: []Target,
    ns: struct { name: *Name, attr: []const u8 },
};

pub const Piece = union(enum) { text: []const u8, expr: *Expr };

pub const Stmt = union(enum) {
    output: []Piece,
    for_: *For,
    if_: *If,
    assign: struct { target: Target, value: *Expr },
    block_set: *BlockSet,
    macro: *Macro,
    scoped: *Scoped,
    break_,
    continue_,
};

pub const For = struct {
    target: Target,
    iter: *Expr,
    body: []Stmt,
    else_body: []Stmt,
    filter: ?*Expr,
    uses_loop: bool = false,
    loop_name: *Name = undefined,
    body_frame: *Frame = undefined,
    else_frame: *Frame = undefined,
    test_frame: *Frame = undefined,
};

pub const If = struct { test_expr: *Expr, body: []Stmt, elifs: []*If, else_body: []Stmt };

pub const BlockSet = struct { target: Target, filter: ?*Expr, body: []Stmt, frame: *Frame = undefined };

pub const Macro = struct {
    name: *Name,
    params: []*Name,
    defaults: []*Expr,
    body: []Stmt,
    frame: *Frame = undefined,
};

/// A body rendered in its own frame, like Hugging Face's `{% generation %}` block.
pub const Scoped = struct { body: []Stmt, frame: *Frame = undefined };

pub const Init = union(enum) { param, resolve, alias: struct { frame: *const Frame, slot: u32 }, undefined };

/// A Jinja2 frame: its variable slots and how each is set on entry.
pub const Frame = struct {
    parent: ?*const Frame,
    names: std.ArrayList([]const u8) = .empty,
    inits: std.ArrayList(Init) = .empty,
};

pub const Template = struct { body: []Stmt, frame: *Frame };

/// Calls `visit(ctx, child)` on each direct subexpression of `e` in source order, the order every pass walks.
pub fn eachChild(comptime E: type, e: *const Expr, ctx: anytype, comptime visit: fn (@TypeOf(ctx), *Expr) E!void) E!void {
    switch (e.*) {
        .constant, .name => {},
        .getattr => |g| try visit(ctx, g.obj),
        .getitem => |g| {
            try visit(ctx, g.obj);
            try visit(ctx, g.arg);
        },
        .slice => |s| for ([_]?*Expr{ s.start, s.stop, s.step }) |x| if (x) |y| try visit(ctx, y),
        .call => |c| {
            try visit(ctx, c.callee);
            try eachArg(E, c.args, ctx, visit);
        },
        .filter => |f| {
            if (f.value) |x| try visit(ctx, x);
            try eachArg(E, f.args, ctx, visit);
        },
        .is => |t| {
            try visit(ctx, t.value);
            try eachArg(E, t.args, ctx, visit);
        },
        .cond => |c| {
            try visit(ctx, c.test_expr);
            try visit(ctx, c.yes);
            if (c.no) |n| try visit(ctx, n);
        },
        .and_, .or_ => |pair| for (pair) |x| try visit(ctx, x),
        .not_, .neg, .pos => |x| try visit(ctx, x),
        .compare => |c| {
            try visit(ctx, c.first);
            for (c.ops) |o| try visit(ctx, o.expr);
        },
        .binop => |b| {
            try visit(ctx, b.l);
            try visit(ctx, b.r);
        },
        .concat, .list, .tuple => |items| for (items) |x| try visit(ctx, x),
        .dict => |pairs| for (pairs) |p| {
            try visit(ctx, p.key);
            try visit(ctx, p.value);
        },
    }
}

fn eachArg(comptime E: type, args: Args, ctx: anytype, comptime visit: fn (@TypeOf(ctx), *Expr) E!void) E!void {
    for (args.pos) |x| try visit(ctx, x);
    for (args.kw) |k| try visit(ctx, k.value);
}
