//! Decode only current affine-g64 host argument layouts; never inspect device storage.
const std = @import("std");
const linear = @import("qlinear.zig");
const model = @import("profile").attribution_model;
pub const Params = ?[*]?*anyopaque;
pub const Decoded = struct { projections: [4]model.Qmmf = undefined, count: usize = 0 };
fn value(comptime T: type, args: Params, index: usize) ?T {
    const ptr = (args orelse return null)[index] orelse return null;
    return @as(*align(1) const T, @ptrCast(ptr)).*;
}
fn same(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
pub fn recognizes(symbol: []const u8) bool {
    for ([_][]const u8{ "tf_qmm_prefill", "tf_qmm_group", "tf_lane_gemv" }) |part|
        if (std.mem.indexOf(u8, symbol, part) != null) return true;
    return false;
}
fn add(out: *Decoded, m: i32, n: i32, k: i32, npad: i32, sk: i32, cluster: [3]u32, reduction: []const u8) bool {
    if (out.count == 4 or m <= 0 or n <= 0 or k <= 0 or npad < n or @mod(npad, 64) != 0 or
        sk < 1 or sk > 8 or @mod(k, sk * 64) != 0) return false;
    out.projections[out.count] = .{ .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .sk = @intCast(sk), .cluster = cluster, .reduce = false, .mode = "mlx_affine4_g64", .reduction = if (sk == 1) "none" else reduction };
    out.count += 1;
    return true;
}
pub fn decode(symbol: []const u8, cluster: [3]u32, args: Params) Decoded {
    var out: Decoded = .{};
    const no: Decoded = .{};
    if (same(symbol, linear.symbols.prefill) or same(symbol, linear.symbols.prefill22)) {
        if (!add(&out, value(i32, args, 5) orelse return no, value(i32, args, 6) orelse return no, value(i32, args, 7) orelse return no, value(i32, args, 8) orelse return no, 1, cluster, "none")) return no;
        return out;
    }
    if (same(symbol, linear.symbols.prefill_many)) {
        const parts = value(linear.PrefillParts, args, 1) orelse return no;
        if (parts.count < 1 or parts.count > 4) return no;
        const m = value(i32, args, 2) orelse return no;
        const k = value(i32, args, 3) orelse return no;
        for (parts.p[0..@intCast(parts.count)]) |p|
            if (!add(&out, m, p.n, k, p.npad, 1, cluster, "none")) return no;
        return out;
    }
    const group = same(symbol, linear.symbols.group) or same(symbol, linear.symbols.verify32) or
        same(symbol, linear.symbols.verify64) or same(symbol, linear.symbols.verify128) or
        same(symbol, "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi128ELi1ELi8ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii");
    if (!group and !same(symbol, linear.symbols.gemv) and !same(symbol, linear.symbols.gemv_split) and
        !same(symbol, linear.symbols.gemv_seg) and !same(symbol, "tf_lane_gemv_many")) return no;
    const m = value(i32, args, 3) orelse return no;
    const k = value(i32, args, 4) orelse return no;
    if (group) {
        const parts = value(linear.Parts, args, 2) orelse return no;
        if (parts.count < 1 or parts.count > 4) return no;
        for (parts.p[0..@intCast(parts.count)]) |p|
            if (!add(&out, m, p.n, k, p.npad, p.sk, cluster, "cluster_k_order")) return no;
    } else if (same(symbol, linear.symbols.gemv) or same(symbol, linear.symbols.gemv_split)) {
        const p = value(linear.GemvPart, args, 2) orelse return no;
        if (!add(&out, m, p.n, k, p.npad, p.sk, cluster, if (same(symbol, linear.symbols.gemv)) "cta_k_order" else "ticket_k_order")) return no;
    } else if (same(symbol, "tf_lane_gemv_many")) {
        const parts = value(linear.GemvManyParts, args, 2) orelse return no;
        if (parts.count < 1 or parts.count > 4) return no;
        for (parts.p[0..@intCast(parts.count)]) |member| {
            const p = member.p;
            if (!add(&out, m, p.n, k, p.npad, p.sk, cluster, "cta_k_order")) return no;
        }
    } else if (same(symbol, linear.symbols.gemv_seg)) {
        const parts = value(linear.Segs, args, 2) orelse return no;
        if (parts.count < 1 or parts.count > 4) return no;
        for (parts.s[0..@intCast(parts.count)]) |member| {
            const p = member.p;
            if (!add(&out, m, p.n, k, p.npad, p.sk, cluster, "ticket_k_order")) return no;
        }
    }
    return out;
}
