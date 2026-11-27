//! Kernel descriptors read only for an explicitly instrumented CUDA driver.
const std = @import("std");
const model = @import("profile").attribution_model;
const group = @import("qmmf_group.zig");
const affine = @import("round_attribution_affine.zig");
pub const Params = ?[*]?*anyopaque;
pub const Meta = struct {
    symbol: []const u8,
    op: model.Op,
    bytes: u64 = 0,
    grid: [3]u32 = .{ 0, 0, 0 },
    block: [3]u32 = .{ 0, 0, 0 },
    projections: [4]model.Qmmf = undefined,
    count: usize = 0,
};
pub fn operation(s: []const u8) model.Op {
    if (affine.recognizes(s)) return .affine_g64;
    if (std.mem.indexOf(u8, s, "qmmf") != null) return .qmmf;
    for ([_][]const u8{ "attn", "attention" }) |part| if (std.mem.indexOf(u8, s, part) != null) return .attention;
    for ([_][]const u8{ "sample", "draw", "argmax", "topk", "logprob" }) |part| if (std.mem.indexOf(u8, s, part) != null) return .logits_sampling;
    return .other;
}
fn value(comptime T: type, args: Params, i: usize) ?T {
    const p = (args orelse return null)[i] orelse return null;
    return @as(*align(1) const T, @ptrCast(p)).*;
}
pub fn kernel(symbol: []const u8, grid: [3]u32, block: [3]u32, cluster: [3]u32, args: Params, previous: ?Meta) Meta {
    var out: Meta = .{ .symbol = symbol, .op = operation(symbol), .grid = grid, .block = block };
    if (out.op == .affine_g64) {
        const decoded = affine.decode(symbol, cluster, args);
        out.projections = decoded.projections;
        out.count = decoded.count;
        return out;
    }
    if (out.op != .qmmf or args == null) return out;
    const mode = modeOf(symbol);
    if (std.mem.indexOf(u8, symbol, "reduce_kernel") != null) {
        if (previous) |p| if (p.op == .qmmf and p.count == 1) {
            out.projections[0] = p.projections[0];
            out.projections[0].reduce = true;
            out.count = 1;
        };
        return out;
    }
    if (std.mem.startsWith(u8, symbol, "tf_qmmf_group_")) {
        const segs = value(group.Segs, args, 1) orelse return out;
        if (segs.count < 1 or segs.count > 4) return out;
        const m = value(u32, args, 2) orelse return out;
        const k = value(u32, args, 3) orelse return out;
        for (segs.s[0..@intCast(segs.count)]) |s| {
            if (s.n < 1 or s.sk < 1) {
                out.count = 0;
                return out;
            }
            out.projections[out.count] = .{ .m = m, .n = @intCast(s.n), .k = k, .sk = @intCast(s.sk), .cluster = cluster, .reduce = false, .mode = mode };
            out.count += 1;
        }
        return out;
    }
    if (std.mem.startsWith(u8, symbol, "tf_qmmf_shared")) {
        out.projections[0] = .{ .m = value(u32, args, 4) orelse return out, .n = value(u32, args, 5) orelse return out, .k = value(u32, args, 6) orelse return out, .sk = 1, .cluster = cluster, .reduce = false, .mode = "bf16" };
        out.count = 1;
        return out;
    }
    if (!std.mem.startsWith(u8, symbol, "_ZN7tf_qmmf11qmmf_kernel") and !std.mem.startsWith(u8, symbol, "_ZN7tf_qmmf22qmmf_prompt_k32_kernel") and !std.mem.startsWith(u8, symbol, "tf_qmmf_prompt")) return out;
    const m = value(u32, args, 6) orelse return out;
    const n = value(u32, args, 7) orelse return out;
    const k = value(u32, args, 8) orelse return out;
    const sk = value(u32, args, 9) orelse return out;
    if (m == 0 or n == 0 or k == 0 or sk == 0 or sk > 8) return out;
    out.projections[0] = .{ .m = m, .n = n, .k = k, .sk = sk, .cluster = cluster, .reduce = (value(u64, args, 5) orelse 0) != 0, .mode = mode };
    out.count = 1;
    return out;
}

fn modeOf(symbol: []const u8) []const u8 {
    if (std.mem.indexOf(u8, symbol, "fp4") != null) return "nvfp4";
    if (std.mem.indexOf(u8, symbol, "fp8") != null) return "fp8";
    if (std.mem.indexOf(u8, symbol, "bf16") != null or std.mem.indexOf(u8, symbol, "prompt_k32") != null) return "bf16";
    if (std.mem.indexOf(u8, symbol, "qmmf_kernelILi")) |at| {
        const pos = at + "qmmf_kernelILi".len;
        if (pos < symbol.len) return switch (symbol[pos]) {
            '0' => "nvfp4",
            '1' => "fp8",
            '2' => "mxfp8",
            '3' => "fp8g",
            '4' => "bf16",
            else => "unresolved",
        };
    }
    return "unresolved";
}
