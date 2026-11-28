//! Resolve each specialized Qwen27 stack record to the source its production loader compiles.
const std = @import("std");
const sources = @import("kernel_sources");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");

pub const Resolved = struct { here: []const u8, pre_m5: []const u8 };

pub fn resolve(a: std.mem.Allocator, device: mtl.Device, name: []const u8, offline: bool) !Resolved {
    if (std.mem.startsWith(u8, name, "qwen27 norm steer ")) {
        const q = tf.qwen27.quant_gpu;
        const here = try q.specializedSource(a, try q.steerKey(contract.steer.hidden, if (device.tensorUnits()) 64 else 0));
        return .{ .here = here, .pre_m5 = try q.specializedSource(a, try q.steerKey(contract.steer.hidden, 0)) };
    }
    if (std.mem.startsWith(u8, name, "q27 stack ")) {
        const body = try source(a, name);
        return .{ .here = body, .pre_m5 = if (offline) body else "" };
    }
    const qk = std.mem.startsWith(u8, name, "q27 qk ");
    const norm = std.mem.indexOf(u8, name, " norm ") != null;
    const paged = std.mem.indexOf(u8, name, " paged ") != null;
    const body = if (qk or norm) try a.dupe(u8, tf.qwen27.runtime_sources.glue) else try tf.qwen27.simd_attention.grouped(a, device.tensorUnits(), true, 6, 4, paged);
    return .{ .here = body, .pre_m5 = if (qk or norm) body else "" };
}
pub fn source(a: std.mem.Allocator, name: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, name, "q27 stack qk ")) {
        var words = std.mem.tokenizeScalar(u8, name[13..], ' ');
        const sk = try std.fmt.parseInt(u32, words.next().?[2..], 10);
        const rows = try std.fmt.parseInt(u32, words.next().?[4..], 10);
        return std.fmt.allocPrint(a, "#define TF_SK {d}\n#define TF_ROWS {d}\n{s}", .{ sk, rows, sources.core_qk });
    }
    const xs = std.mem.indexOf(u8, name, "xs1") != null;
    const body = try std.mem.replaceOwned(u8, a, sources.q27_mlp_xs, "Q27_SPECIALIZE", "constexpr int N=17408;");
    defer a.free(body);
    return std.fmt.allocPrint(a, "#define Q27_XS {d}\n{s}", .{ @intFromBool(xs), body });
}
