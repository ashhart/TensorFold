//! A block.zig Triton call as the AOT set takes it: runtime args vs constexprs, so `Set.find` picks Triton's variant.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const aot = cuda.aot;

/// Triton's pointer type of a dtype ("*bf16" ...).
pub fn ptrType(dt: calls.Dt) []const u8 {
    return switch (dt) {
        .bf16 => "*bf16",
        .f16 => "*fp16",
        .f32 => "*fp32",
        .f64 => "*fp64",
        .i8 => "*i8",
        .u8 => "*u8",
        .i16 => "*i16",
        .i32 => "*i32",
        .i64 => "*i64",
        .bool => "*i1",
    };
}

pub const Split = struct { args: []aot.Arg, consts: []aot.Const };

/// The device address of a tensor argument (the caller's buffers; 0 for an empty one).
pub const Address = struct {
    ctx: *anyopaque,
    of: *const fn (ctx: *anyopaque, t: calls.Tensor) anyerror!u64,
};

/// Splits `c`'s named arguments over `variants` (every compiled variant of `c.name`).
pub fn split(a: std.mem.Allocator, c: *const calls.Call, variants: []const aot.Spec, addr: Address) !Split {
    var args: std.ArrayList(aot.Arg) = .empty;
    var consts: std.ArrayList(aot.Const) = .empty;
    for (c.args) |x| {
        const param = for (variants) |v| {
            const p = for (v.params) |p| {
                if (std.mem.eql(u8, p.name, x.name)) break p;
            } else continue;
            break p;
        } else null;
        switch (x.arg) {
            .none => {},
            .t, .opaque_table => |t| try args.append(a, aot.ptr(x.name, ptrType(t.dt), try addr.of(addr.ctx, t))),
            .b => |v| try consts.append(a, aot.ci(x.name, @intFromBool(v))),
            .i => |v| if (param) |p| {
                if (std.mem.eql(u8, p.type, "i32")) {
                    try args.append(a, aot.int(x.name, std.math.cast(i32, v) orelse return error.IntRange));
                } else if (std.mem.eql(u8, p.type, "u64") or std.mem.eql(u8, p.type, "i64")) {
                    try args.append(a, aot.word(x.name, @bitCast(v)));
                } else return error.UnsupportedParam;
            } else try consts.append(a, aot.ci(x.name, v)),
            .f => |v| if (param != null) {
                try args.append(a, aot.float(x.name, @floatCast(v)));
            } else try consts.append(a, aot.cf(x.name, @floatCast(v))),
            .list => return error.UnsupportedArg,
        }
    }
    return .{ .args = args.items, .consts = consts.items };
}
