//! A block's launches as the forward issues them, tensors named by role, and the check of each against a captured op.

const std = @import("std");
const Value = std.json.Value;

pub const Dt = enum { bf16, f16, f32, f64, i8, u8, i16, i32, i64, bool };

/// Where a tensor's bytes live: one of our loader's weights, a named buffer of the forward, or nothing (numel 0).
pub const Role = union(enum) {
    weight: []const u8,
    /// scope by prefix: "s." across windows (slot state, tables), "L<i>." its layer, anything else the window
    buf: []const u8,
    empty,
};

pub const Tensor = struct {
    role: Role,
    dt: Dt,
    shape: []const i64,
    stride: []const i64,
    /// bytes from the role's first byte
    offset: i64 = 0,
};

pub const Arg = union(enum) {
    t: Tensor,
    i: i64,
    f: f64,
    b: bool,
    none,
    list: []const Arg,
    /// the capture's table contents are not compared (an L2 prefetch list of addresses)
    opaque_table: Tensor,
};

pub const Named = struct { name: []const u8 = "", arg: Arg };

/// Where a call starts a scope: the window's or a layer's roles begin fresh there.
pub const Begin = enum { none, window, layer };

pub const Call = struct {
    triton: bool,
    glue: bool = false, // the forward's own step (positions, exchanges, Engram rows): no captured launch
    begin: Begin = .none,
    /// the compiled kernel's name (Triton) or "<extension>.<function>"
    name: []const u8,
    grid: [3]i64 = .{ 1, 1, 1 },
    args: []const Named,
    side: bool = false, // TF_DSV41_BRANCHES: issued on the side stream (same launch; checks compare launches)
    fork: bool = false, // the side stream first waits for every main-stream launch so far
    join: bool = false, // the main stream first waits for the side stream
    mark: u8 = 0, // TF_DSV41_PF_OVERLAP_SITE: on a side call, record side mark m after it
    wait: u8 = 0, // on a main call, wait for mark m only (a `join` still waits for all)
    defer_side: bool = false, // TF_DSV41_MHC_DEFER: an mHC coefficient launch on the deferred stream
    defer_join: bool = false, // the main stream waits for it first (the coefficients' reader)
};

/// TF_DSV41_MHC_DEFER_AT=exchange: move each deferred launch to just before the next exchange or its join, in-region.
pub fn holdDeferred(cs: []Call) void {
    var i: usize = 0;
    while (i < cs.len) : (i += 1) {
        const c = cs[i];
        if (!c.defer_side or c.defer_join or c.side or c.fork or c.join or c.begin != .none) continue;
        var j = i + 1;
        while (j < cs.len and !cs[j].defer_join and !cs[j].defer_side and !isExchange(cs[j])) : (j += 1) {}
        if (j == cs.len or cs[j].defer_side) continue; // no exchange or join after it: left where it was
        std.mem.copyForwards(Call, cs[i .. j - 1], cs[i + 1 .. j]);
        cs[j - 1] = c;
        i = j - 1;
    }
}

/// A collective exchange step of the forward (glue.exchange / glue.exchange_f32).
pub fn isExchange(c: Call) bool {
    return c.glue and std.mem.startsWith(u8, c.name, "glue.exchange");
}

/// The calls a capture records (the glue steps left out), in order; a dropped step's scope start moves to the next.
pub fn launches(a: std.mem.Allocator, cs: []const Call) ![]const Call {
    var out: std.ArrayList(Call) = .empty;
    var pending: Begin = .none;
    for (cs) |c| {
        if (c.glue) {
            if (@backingInt(c.begin) > @backingInt(pending)) pending = c.begin;
            continue;
        }
        var x = c;
        if (pending == .window or (pending == .layer and x.begin == .none)) x.begin = pending;
        pending = .none;
        try out.append(a, x);
    }
    return out.items;
}

/// Contiguous strides of `shape` (elements).
pub fn contiguous(a: std.mem.Allocator, shape: []const i64) ![]const i64 {
    const s = try a.alloc(i64, shape.len);
    var acc: i64 = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        s[i] = acc;
        acc *= @max(shape[i], 1);
    }
    return s;
}

fn dtName(dt: Dt) []const u8 {
    return switch (dt) {
        .bf16 => "bfloat16",
        .f16 => "float16",
        .f32 => "float32",
        .f64 => "float64",
        .i8 => "int8",
        .u8 => "uint8",
        .i16 => "int16",
        .i32 => "int32",
        .i64 => "int64",
        .bool => "bool",
    };
}

/// Role -> captured storage key within its scope, learned as the check goes; another key for a role is a mismatch.
pub const Roles = struct {
    a: std.mem.Allocator,
    keys: std.StringHashMapUnmanaged([]const u8) = .empty,
    /// storage size a role maps to (the GPU run sizes its buffer from it)
    sizes: std.StringHashMapUnmanaged(u64) = .empty,

    /// Forgets the roles whose scope ends at `b`: "L." at a layer, "L." and "w." at a window.
    pub fn begin(r: *Roles, b: Begin) void {
        switch (b) {
            .none => {},
            .layer => r.endScope("w."),
            .window => r.endScope(""),
        }
    }

    /// Forgets every role but the persistent ("s.") ones and those starting with `prefix_keep`.
    pub fn endScope(r: *Roles, prefix_keep: []const u8) void {
        var it = r.keys.iterator();
        var drop: std.ArrayList([]const u8) = .empty;
        defer drop.deinit(r.a);
        while (it.next()) |e| {
            const k = e.key_ptr.*;
            if (std.mem.startsWith(u8, k, "s.")) continue;
            if (prefix_keep.len > 0 and std.mem.startsWith(u8, k, prefix_keep)) continue;
            drop.append(r.a, k) catch {};
        }
        for (drop.items) |k| _ = r.keys.remove(k);
    }
};

pub const Mismatch = struct { what: []const u8 };

/// The first difference between `c` and the captured op `op` (null: they agree); learns the roles' storage keys.
pub fn check(a: std.mem.Allocator, c: *const Call, op: std.json.ObjectMap, roles: *Roles) !?[]const u8 {
    const name = op.get("name").?.string;
    if (!std.mem.eql(u8, name, c.name)) return try std.fmt.allocPrint(a, "name {s} vs captured {s}", .{ c.name, name });
    if (c.triton) {
        const g = op.get("grid").?.array.items;
        for (c.grid, 0..) |x, i| if (g[i].integer != x) return try std.fmt.allocPrint(a, "grid {any} vs captured [{d},{d},{d}]", .{ c.grid, g[0].integer, g[1].integer, g[2].integer });
    }
    const got = op.get("args").?.array.items;
    if (got.len != c.args.len) return try std.fmt.allocPrint(a, "{d} args vs captured {d}", .{ c.args.len, got.len });
    var keys: std.StringHashMapUnmanaged(i64) = .empty; // this op's role -> buffer id (aliasing within the op)
    const bufs = op.get("buffers").?.array.items;
    for (c.args, got, 0..) |want, cap, i| {
        if (c.triton) {
            const cn = cap.object.get("name").?.string;
            if (!std.mem.eql(u8, cn, want.name)) return try std.fmt.allocPrint(a, "arg {d} named {s} vs captured {s}", .{ i, want.name, cn });
        }
        if (try argCheck(a, want.arg, cap, roles, &keys, bufs)) |why| return try std.fmt.allocPrint(a, "arg {d} ({s}): {s}", .{ i, if (want.name.len > 0) want.name else "-", why });
    }
    return null;
}

fn argCheck(a: std.mem.Allocator, want: Arg, cap: Value, roles: *Roles, keys: *std.StringHashMapUnmanaged(i64), bufs: []const Value) !?[]const u8 {
    const o = cap.object;
    const t = o.get("t").?.string;
    switch (want) {
        .none => return if (std.mem.eql(u8, t, "none")) null else try std.fmt.allocPrint(a, "none vs {s}", .{t}),
        .b => |v| {
            if (!std.mem.eql(u8, t, "bool")) return try std.fmt.allocPrint(a, "bool vs {s}", .{t});
            return if (o.get("v").?.bool == v) null else try std.fmt.allocPrint(a, "{} vs {}", .{ v, o.get("v").?.bool });
        },
        .i => |v| {
            if (!std.mem.eql(u8, t, "int")) return try std.fmt.allocPrint(a, "int {d} vs {s}", .{ v, t });
            return if (o.get("v").?.integer == v) null else try std.fmt.allocPrint(a, "{d} vs {d}", .{ v, o.get("v").?.integer });
        },
        .f => |v| {
            if (!std.mem.eql(u8, t, "float")) return try std.fmt.allocPrint(a, "float vs {s}", .{t});
            const bits = try std.fmt.parseInt(u64, o.get("f64").?.string[2..], 16);
            return if (bits == @as(u64, @bitCast(v))) null else try std.fmt.allocPrint(a, "{e} vs {e}", .{ v, @as(f64, @bitCast(bits)) });
        },
        .list => |items| {
            if (!std.mem.eql(u8, t, "list")) return try std.fmt.allocPrint(a, "list vs {s}", .{t});
            const ci = o.get("items").?.array.items;
            if (ci.len != items.len) return try std.fmt.allocPrint(a, "list of {d} vs {d}", .{ items.len, ci.len });
            for (items, ci, 0..) |x, y, j| if (try argCheck(a, x, y, roles, keys, bufs)) |why| return try std.fmt.allocPrint(a, "[{d}] {s}", .{ j, why });
            return null;
        },
        .t, .opaque_table => |w| {
            if (!std.mem.eql(u8, t, "tensor")) return try std.fmt.allocPrint(a, "tensor vs {s}", .{t});
            if (!std.mem.eql(u8, o.get("dtype").?.string, dtName(w.dt))) return try std.fmt.allocPrint(a, "dtype {s} vs {s}", .{ dtName(w.dt), o.get("dtype").?.string });
            const shape = o.get("shape").?.array.items;
            if (!intsEq(w.shape, shape)) return try std.fmt.allocPrint(a, "shape {any} vs captured", .{w.shape});
            var numel: i64 = 1;
            for (w.shape) |d| numel *= d;
            if (numel == 0) return null; // an empty tensor: only its dtype and shape count
            if (w.shape.len > 0 and !intsEq(w.stride, o.get("stride").?.array.items)) return try std.fmt.allocPrint(a, "stride {any} vs captured", .{w.stride});
            switch (w.role) {
                .empty => return try std.fmt.allocPrint(a, "empty role for a tensor of {d} elements", .{numel}),
                .weight => |name| {
                    const cw = o.get("weight") orelse return try std.fmt.allocPrint(a, "weight {s} vs a captured buffer", .{name});
                    if (!std.mem.eql(u8, cw.string, name)) return try std.fmt.allocPrint(a, "weight {s} vs {s}", .{ name, cw.string });
                    return if (o.get("offset").?.integer == w.offset) null else try std.fmt.allocPrint(a, "offset {d} vs {d}", .{ w.offset, o.get("offset").?.integer });
                },
                .buf => |role| {
                    if (o.get("weight")) |cw| return try std.fmt.allocPrint(a, "buffer {s} vs captured weight {s}", .{ role, cw.string });
                    if (o.get("offset").?.integer != w.offset) return try std.fmt.allocPrint(a, "{s} offset {d} vs {d}", .{ role, w.offset, o.get("offset").?.integer });
                    const id = o.get("buf").?.integer;
                    // within the op: the same role is the same buffer and two roles are two buffers
                    if (keys.get(role)) |prev| {
                        if (prev != id) return try std.fmt.allocPrint(a, "{s} is two captured buffers", .{role});
                    } else {
                        var it = keys.iterator();
                        while (it.next()) |e| if (e.value_ptr.* == id) return try std.fmt.allocPrint(a, "{s} and {s} are one captured buffer", .{ role, e.key_ptr.* });
                        try keys.put(a, role, id);
                    }
                    const key = for (bufs) |b| {
                        if (b.object.get("id").?.integer == id) break b.object.get("key").?.string;
                    } else return "no buffer entry";
                    const gop = try roles.keys.getOrPut(roles.a, role);
                    if (gop.found_existing) {
                        if (!std.mem.eql(u8, gop.value_ptr.*, key)) return try std.fmt.allocPrint(a, "{s} moved to another captured storage", .{role});
                    } else {
                        gop.key_ptr.* = try roles.a.dupe(u8, role);
                        gop.value_ptr.* = try roles.a.dupe(u8, key);
                    }
                    const size: u64 = @intCast(for (bufs) |b| {
                        if (b.object.get("id").?.integer == id) break b.object.get("nbytes").?.integer;
                    } else 0);
                    const sg = try roles.sizes.getOrPut(roles.a, gop.key_ptr.*);
                    if (!sg.found_existing or sg.value_ptr.* < size) sg.value_ptr.* = size;
                    return null;
                },
            }
        },
    }
}

fn intsEq(w: []const i64, c: []const Value) bool {
    if (w.len != c.len) return false;
    for (w, c) |x, y| if (y.integer != x) return false;
    return true;
}
