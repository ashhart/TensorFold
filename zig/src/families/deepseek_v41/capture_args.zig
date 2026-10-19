//! Checked capture ABI and operation packing, independent of module loading and GPU execution.
const std = @import("std");
const cuda = @import("cuda");
const Value = std.json.Value;
pub const Type = enum { pointer, i1, i8, i16, i32, i64, u8, u16, u32, u64, fp32, fp64 };
pub const Param = struct { name: []const u8, ty: Type };
pub const Resolve = *const fn (*anyopaque, Value) anyerror!?u64;
pub const Packed = struct { args: cuda.Args, grid: cuda.Dim3 };
pub fn typeOf(text: []const u8) !Type {
    if (text.len == 0) return error.UnsupportedScalar;
    if (text[0] == '*') {
        const base = text[1..];
        if (std.mem.eql(u8, base, "fp16") or std.mem.eql(u8, base, "bf16")) return .pointer;
        const t = std.meta.stringToEnum(Type, base) orelse return error.UnsupportedScalar;
        if (t == .pointer) return error.UnsupportedScalar;
        return .pointer;
    }
    const t = std.meta.stringToEnum(Type, text) orelse return error.UnsupportedScalar;
    if (t == .pointer) return error.UnsupportedScalar;
    return t;
}
pub fn identifier(text: []const u8) bool {
    if (text.len == 0 or text.len > 256 or (!std.ascii.isAlphabetic(text[0]) and text[0] != '_')) return false;
    for (text) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '_') return false;
    return true;
}
fn field(o: std.json.ObjectMap, key: []const u8) !Value {
    return o.get(key) orelse error.BadOperation;
}
fn string(v: Value) ![]const u8 {
    return if (v == .string) v.string else error.BadOperation;
}
fn integer(comptime T: type, v: Value) !T {
    return switch (v) {
        .integer => |n| std.math.cast(T, n) orelse error.IntRange,
        .number_string => |s| std.fmt.parseInt(T, s, 10) catch error.IntRange,
        .bool => |b| @intFromBool(b),
        else => error.BadOperation,
    };
}
pub fn scalar(args: *cuda.Args, ty: Type, v: Value) !void {
    if (v != .object or args.count >= cuda.Args.max_args - 2) return error.BadOperation;
    const o = v.object;
    const kind = try string(try field(o, "t"));
    if (ty == .fp32 or ty == .fp64) {
        if (!std.mem.eql(u8, kind, "float")) return error.WrongScalar;
        const bits = try string(try field(o, if (ty == .fp32) "f32" else "f64"));
        if (bits.len != (if (ty == .fp32) @as(usize, 10) else 18) or !std.mem.startsWith(u8, bits, "0x")) return error.BadOperation;
        for (bits[2..]) |ch| if (!std.ascii.isHex(ch)) return error.BadOperation;
        if (ty == .fp32) args.add(try std.fmt.parseInt(u32, bits[2..], 16)) else args.add(try std.fmt.parseInt(u64, bits[2..], 16));
        return;
    }
    const value = try field(o, "v");
    if (std.mem.eql(u8, kind, "bool")) {
        if (value != .bool) return error.WrongScalar;
    } else if (!std.mem.eql(u8, kind, "int") or (value != .integer and value != .number_string)) return error.WrongScalar;
    switch (ty) {
        .i1 => {
            const x = try integer(u8, value);
            if (x > 1) return error.IntRange;
            args.add(@as(i8, @intCast(x)));
        },
        .i8 => args.add(try integer(i8, value)),
        .i16 => args.add(try integer(i16, value)),
        .i32 => args.add(try integer(i32, value)),
        .i64 => args.add(try integer(i64, value)),
        .u8 => args.add(try integer(u8, value)),
        .u16 => args.add(try integer(u16, value)),
        .u32 => args.add(try integer(u32, value)),
        .u64 => args.add(try integer(u64, value)),
        else => return error.UnsupportedScalar,
    }
}
pub fn grid(op: std.json.ObjectMap) !cuda.Dim3 {
    const v = try field(op, "grid");
    if (v != .array or v.array.items.len != 3) return error.BadGrid;
    var out: [3]u32 = undefined;
    for (v.array.items, &out, 0..) |x, *d, i| {
        if (x != .integer) return error.BadGrid;
        d.* = std.math.cast(u32, x.integer) orelse return error.BadGrid;
        if (d.* == 0 or d.* > (if (i == 0) @as(u32, 2147483647) else 65535)) return error.BadGrid;
    }
    return .{ .x = out[0], .y = out[1], .z = out[2] };
}
fn pointer(v: Value) !void {
    const o = v.object;
    const kind = try string(try field(o, "t"));
    if (std.mem.eql(u8, kind, "none")) return;
    if (std.mem.eql(u8, kind, "int")) {
        _ = try integer(i64, try field(o, "v"));
        return;
    }
    if (!std.mem.eql(u8, kind, "tensor")) return error.WrongArgument;
    const shape = try field(o, "shape");
    if (shape != .array or shape.array.items.len > 8) return error.BadOperation;
    var count: i64 = 1;
    for (shape.array.items) |d| {
        if (d != .integer or d.integer < 0) return error.BadOperation;
        count = std.math.mul(i64, count, d.integer) catch return error.BadOperation;
    }
    _ = try integer(i64, try field(o, "offset"));
    if (o.get("weight")) |w| {
        if ((try string(w)).len == 0) return error.BadOperation;
    } else {
        const b = try field(o, "buf");
        if (b != .integer or b.integer < 0) return error.BadOperation;
    }
}
pub fn pack(params: []const Param, op: std.json.ObjectMap, ctx: *anyopaque, resolve: Resolve) !Packed {
    const launch_grid = try grid(op);
    const args = try field(op, "args");
    if (params.len > cuda.Args.max_args - 2 or args != .array or args.array.items.len > 4096) return error.BadOperation;
    const given = args.array.items;
    for (given, 0..) |arg, i| {
        if (arg != .object) return error.BadOperation;
        const name = try string(try field(arg.object, "name"));
        if (!identifier(name)) return error.BadOperation;
        for (given[0..i]) |prior| if (std.mem.eql(u8, name, try string(try field(prior.object, "name")))) return error.DuplicateArgument;
    }
    var packed_args: cuda.Args = .{};
    for (params) |p| {
        const arg = for (given) |g| {
            if (std.mem.eql(u8, try string(try field(g.object, "name")), p.name)) break g;
        } else return error.MissingArgument;
        if (p.ty == .pointer) {
            try pointer(arg);
            packed_args.add(@as(u64, (try resolve(ctx, arg)) orelse 0));
        } else try scalar(&packed_args, p.ty, arg);
    }
    return .{ .args = packed_args, .grid = launch_grid };
}
