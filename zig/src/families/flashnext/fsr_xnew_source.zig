//! Slotted addresses in the resident six-bit row kernels; all sums, routing and rounding remain unchanged.
const std = @import("std");
fn once(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]const u8 {
    if (std.mem.count(u8, text, from) != 1) return error.FsrXnewPatch;
    return std.mem.replaceOwned(u8, a, text, from, to);
}

pub fn patch(a: std.mem.Allocator, original: []const u8) ![]const u8 {
    if (std.mem.indexOf(u8, original, "FSR_MAP") != null) return error.FsrXnewPatch;
    const end = std.mem.indexOf(u8, original, "[[kernel]] void fz_dense8") orelse return error.FsrXnewPatch;
    var text: []const u8 = original[0..end];
    text = try once(a, text, "constant uint4& OWN [[buffer(17)]],", "constant uint4& OWN [[buffer(17)]], const device int* FSR_MAP [[buffer(18)]], constant uint* FSR_STRIDES [[buffer(19)]],");
    text = try once(a, text, "constant uint4& OWN [[buffer(10)]],", "constant uint4& OWN [[buffer(10)]], const device int* FSR_MAP [[buffer(11)]], constant uint* FSR_STRIDES [[buffer(12)]],");
    text = try once(a, text, "  const int row = int(tg.y) * 8", "  const size_t slot_id = shared ? 0 : size_t(FSR_MAP[e]);\n  const int row = int(tg.y) * 8");
    text = try once(a, text, "(shared ? 0 : e * N * WPR) +", "");
    // The removed prefix leaves the row expression, without changing any floating-point operation.
    text = try once(a, text, "(shared ? 0 : e * N * KG) +", "");
    for ([_][]const u8{ "GW", "GS", "GB", "UW", "US", "UB" }, 0..) |name, i| {
        const from = try std.fmt.allocPrint(a, " : {s}) +", .{name});
        const to = try std.fmt.allocPrint(a, " : {s} + slot_id * FSR_STRIDES[{d}]) +", .{ name, i });
        text = try once(a, text, from, to);
    }
    text = try once(a, text, "  const int d = int(tg.y) * 32", "  const size_t slot_id = shared ? 0 : size_t(FSR_MAP[e]);\n  const int d = int(tg.y) * 32");
    text = try once(a, text, "DW + e * D * WPR", "DW + slot_id * FSR_STRIDES[6]");
    text = try once(a, text, "(shared ? 0 : e * D * KG) +", "");
    for ([_][]const u8{ "DS", "DB" }, 7..) |name, i| {
        const from = try std.fmt.allocPrint(a, " : {s}) + g0", .{name});
        const to = try std.fmt.allocPrint(a, " : {s} + slot_id * FSR_STRIDES[{d}]) + g0", .{ name, i });
        text = try once(a, text, from, to);
    }
    text = try once(a, text, "[[kernel]] void fz_xgu", "[[max_total_threads_per_threadgroup(128)]] [[kernel]] void fz_xgu");
    text = try once(a, text, "[[kernel]] void fz_xdown", "[[max_total_threads_per_threadgroup(128)]] [[kernel]] void fz_xdown");
    const tail = try once(a, original[end..], "[[kernel]] void fz_route(", "[[max_total_threads_per_threadgroup(512)]] [[kernel]] void fz_route(");
    return std.mem.concat(a, u8, &.{ text, tail });
}
