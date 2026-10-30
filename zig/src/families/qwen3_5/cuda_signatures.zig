//! The q35 attention kernels' parameter lists, as their .cu sources declare them.
const std = @import("std");

/// A parameter's kind: the geometry struct, a device pointer, a 32-bit count or a float scale.
pub const Kind = enum { geo, ptr, i32, f32 };

/// One kernel's parameter list, in the .cu source's order.
pub const Sig = struct {
    /// The name a launch site names this list by.
    name: []const u8,
    /// The engine's field path that holds the kernel's handle.
    handle: []const u8,
    /// The symbol the .cu source declares.
    symbol: []const u8,
    /// The parameters, in order.
    params: []const Kind,
};

/// The bf16 decode partial: a block a (row, chunk, key head) of the query heads' rows.
pub const attn_partial = Sig{ .name = "attn_partial", .handle = "e.fns.attn_partial", .symbol = "tf_q35_attn_partial", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .ptr, .ptr, .ptr } };

/// The e4m3 decode partial: the bf16 one's tiles in the e4m3 scales.
pub const attn_partial8 = Sig{ .name = "attn_partial8", .handle = "e.fp8.attn_partial", .symbol = "tf_q35_attn_partial8", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .ptr, .ptr, .ptr } };

/// The bf16 decode attention over a long stream's rows: a block a group of rows.
pub const attn_rows = Sig{ .name = "attn_rows", .handle = "e.fns.attn_rows", .symbol = "tf_q35_attn_rows", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .i32, .ptr, .ptr, .ptr } };

/// The e4m3 decode attention over a long stream's rows.
pub const attn_rows8 = Sig{ .name = "attn_rows8", .handle = "e.fp8.attn_rows", .symbol = "tf_q35_attn_rows8", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .i32, .ptr, .ptr, .ptr } };

/// Whole-window instances use the same ABI without consulting Seg.pt.
pub const attn_partial_whole = Sig{ .name = "attn_partial_whole", .handle = "e.fns.attn_partial_whole", .symbol = "tf_q35_attn_partial_whole", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .ptr, .ptr, .ptr } };
pub const attn_partial8_whole = Sig{ .name = "attn_partial8_whole", .handle = "e.fp8.attn_partial8_whole", .symbol = "tf_q35_attn_partial8_whole", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .ptr, .ptr, .ptr } };
pub const attn_rows_whole = Sig{ .name = "attn_rows_whole", .handle = "e.fns.attn_rows_whole", .symbol = "tf_q35_attn_rows_whole", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .i32, .ptr, .ptr, .ptr } };
pub const attn_rows8_whole = Sig{ .name = "attn_rows8_whole", .handle = "e.fp8.attn_rows8_whole", .symbol = "tf_q35_attn_rows8_whole", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .i32, .ptr, .ptr, .ptr } };

/// The merge: a row's chunks' partials into each (row, query head)'s output.
pub const attn_merge = Sig{ .name = "attn_merge", .handle = "e.fns.attn_merge", .symbol = "tf_q35_attn_merge", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr } };

/// The gate: the attention's outputs through the second half of each query head's q_proj rows.
pub const attn_gate = Sig{ .name = "attn_gate", .handle = "e.fns.attn_gate", .symbol = "tf_q35_attn_gate", .params = &.{ .geo, .ptr, .ptr, .ptr, .i32, .ptr, .ptr } };

/// The rope: the queries rotated in place, the keys rotated and the values copied into the cache.
pub const rope = Sig{ .name = "rope", .handle = "e.fns.rope", .symbol = "tf_q35_rope", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .ptr } };

/// The e4m3 rope: the bf16 one's tiles in the e4m3 scales.
pub const rope8 = Sig{ .name = "rope8", .handle = "e.fp8.rope", .symbol = "tf_q35_rope8", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .ptr } };

/// The draft frame's partial: the draft rows' own cache over the ring.
pub const draft_partial = Sig{ .name = "draft_partial", .handle = "d.fns.attn_partial", .symbol = "tf_q35d_attn_partial", .params = &.{ .ptr, .ptr, .ptr, .ptr, .i32, .i32, .i32, .i32, .i32, .i32, .ptr, .ptr, .ptr } };

/// The draft frame's merge: the draft rows' chunks' partials into each row's output.
pub const draft_merge = Sig{ .name = "draft_merge", .handle = "d.fns.attn_merge", .symbol = "tf_q35d_attn_merge", .params = &.{ .ptr, .ptr, .ptr, .i32, .ptr } };

/// Every q35 attention kernel's list: the decode partials, the rows, the merge, the gate, the rope and the draft pair.
pub const all = [_]Sig{ attn_partial, attn_partial8, attn_rows, attn_rows8, attn_merge, attn_gate, rope, rope8, draft_partial, draft_merge, attn_partial_whole, attn_partial8_whole, attn_rows_whole, attn_rows8_whole };

/// The list a launch site's handle names, if the table holds one.
pub fn byHandle(handle: []const u8) ?Sig {
    for (all) |s| if (std.mem.eql(u8, s.handle, handle)) return s;
    return null;
}

/// The list a launch site names, if the table holds one.
pub fn byName(name: []const u8) ?Sig {
    for (all) |s| if (std.mem.eql(u8, s.name, name)) return s;
    return null;
}

/// The kind a host value carries: the geometry struct, a device pointer, a count or a scale.
pub fn kindOf(comptime T: type) ?Kind {
    if (T == u64) return .ptr;
    if (T == c_int) return .i32;
    if (T == f32) return .f32;
    const info = @typeInfo(T);
    if (info == .@"struct" and info.@"struct".layout == .@"extern") return .geo;
    return null;
}

/// A launch site's arguments against a kernel's list: a wrong count or kind does not build.
pub fn check(comptime s: Sig, comptime Args: type) void {
    comptime {
        const info = @typeInfo(Args);
        if (info != .@"struct") @compileError("q35 launch arguments are not a tuple: " ++ s.symbol);
        const types = info.@"struct".field_types;
        if (types.len != s.params.len) @compileError("q35 launch " ++ s.symbol ++ " takes " ++ std.fmt.comptimePrint("{d}", .{s.params.len}) ++ " arguments, the launch passes " ++ std.fmt.comptimePrint("{d}", .{types.len}));
        for (types, 0..) |T, i| {
            const got = kindOf(T) orelse @compileError("q35 launch argument kind is unknown: " ++ s.symbol);
            if (got != s.params[i]) @compileError("q35 launch argument " ++ std.fmt.comptimePrint("{d}", .{i}) ++ " of " ++ s.symbol ++ " has the wrong kind");
        }
    }
}
