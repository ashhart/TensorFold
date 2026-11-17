//! SenseNova two-stream mHC CUDA launch argument kinds in source order.
pub const Kind = enum { ptr, i32 };
pub const Sig = struct { symbol: []const u8, params: []const Kind };
pub const enter = Sig{ .symbol = "qwen35_mhc_enter", .params = &.{ .ptr, .ptr, .ptr, .i32, .ptr } };
pub const route = Sig{ .symbol = "qwen35_mhc_route", .params = &.{ .ptr, .ptr, .ptr, .ptr, .ptr, .ptr, .i32, .ptr } };
pub const merge = Sig{ .symbol = "qwen35_mhc_merge", .params = &.{ .ptr, .ptr, .ptr, .ptr, .i32, .ptr } };
pub const leave = Sig{ .symbol = "qwen35_mhc_leave", .params = &.{ .ptr, .ptr, .i32, .ptr } };
pub const all = [_]Sig{ enter, route, merge, leave };

/// Check a launch tuple's pointer and integer argument kinds at compile time.
pub fn check(comptime sig: Sig, comptime Args: type) void {
    const info = @typeInfo(Args);
    if (info != .@"struct") @compileError("mHC arguments must be a tuple");
    const fields = info.@"struct".field_types;
    if (fields.len != sig.params.len) @compileError("wrong mHC argument count: " ++ sig.symbol);
    inline for (fields, sig.params) |T, kind| {
        const actual: ?Kind = if (T == u64) .ptr else if (T == c_int) .i32 else null;
        if (actual == null or actual.? != kind) @compileError("wrong mHC argument kind: " ++ sig.symbol);
    }
}

test "mHC launch signatures" {
    check(enter, @TypeOf(.{ @as(u64, 0), @as(u64, 0), @as(u64, 0), @as(c_int, 0), @as(u64, 0) }));
    check(leave, @TypeOf(.{ @as(u64, 0), @as(u64, 0), @as(c_int, 0), @as(u64, 0) }));
}
