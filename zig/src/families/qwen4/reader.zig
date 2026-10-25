//! Owned read-only checkpoint maps are admitted completely before any GPU allocation or driver call.
const std = @import("std");
const checkpoint = @import("core").checkpoint;
const st = @import("core").safetensors;
const cfg = @import("config.zig");
const weights = @import("weights.zig");
const ngram = @import("hybrid_ngram");

pub const Reader = struct {
    a: std.mem.Allocator,
    checkpoint: checkpoint.Checkpoint,
    header: st.Header,
    namespace: weights.Namespace,
    stats: weights.Stats,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const cfg.Config, p: *cfg.Problem) !Reader {
        var ck = try checkpoint.Checkpoint.openModel(a, io, dir);
        var transferred = false;
        errdefer if (!transferred) ck.close();
        var header: st.Header = .empty;
        errdefer if (!transferred) header.deinit(a);
        for (ck.files.items) |file| {
            var it = file.names.iterator();
            while (it.next()) |e| {
                if ((try header.getOrPut(a, e.key_ptr.*)).found_existing) return p.reject(e.key_ptr.*);
                header.getPtr(e.key_ptr.*).?.* = e.value_ptr.*;
            }
        }
        const ns = try namespace(header, p);
        try normPolicy(ns, c, p);
        const stats = try weights.admit(a, c, ns, header, p);
        var out: Reader = .{ .a = a, .checkpoint = ck, .header = header, .namespace = ns, .stats = stats };
        transferred = true;
        errdefer out.close();
        try out.constants(c, p);
        return out;
    }
    pub fn close(r: *Reader) void {
        r.header.deinit(r.a);
        r.checkpoint.close();
        r.* = undefined;
    }
    pub fn get(r: *Reader, name: []const u8) !st.Tensor {
        return r.checkpoint.get(name);
    }
    fn constants(r: *Reader, c: *const cfg.Config, p: *cfg.Problem) !void {
        for (r.header.keys()) |name| {
            if (std.mem.endsWith(u8, name, ".input_scale") or std.mem.endsWith(u8, name, ".input_scale_2") or std.mem.endsWith(u8, name, ".weight_scale_2") or
                (std.mem.endsWith(u8, name, ".weight_scale") and r.header.get(name).?.dtype == .f32))
            {
                const t = try r.get(name);
                const value: f32 = @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
                if (!std.math.isFinite(value)) return p.reject(name);
            }
        }
        var ordinal: usize = 0;
        for (0..c.layers) |layer| if (c.ple[layer]) {
            var b: [256]u8 = undefined;
            const heads = (c.ngram - 1) * c.ngram_heads;
            const stem = try std.fmt.bufPrint(&b, "{s}layers.{d}.ple.ple_embedding", .{ r.namespace.prefix(), layer });
            var name: [320]u8 = undefined;
            const multiplier_name = try std.fmt.bufPrint(&name, "{s}.layer_multipliers", .{stem});
            const multipliers = try r.get(multiplier_name);
            for (0..c.ngram) |i| {
                if (integer(multipliers, i) != try ngram.multiplier(c.seed, ordinal, i, c.vocab)) return p.reject(multiplier_name);
            }
            const sizes = try r.get(try std.fmt.bufPrint(&name, "{s}.ngram_heads_vocab_sizes", .{stem}));
            const offsets = try r.get(try std.fmt.bufPrint(&name, "{s}.ngram_heads_offsets", .{stem}));
            var prime: u64 = c.ngram_vocab - 1;
            for (0..ordinal * heads) |_| prime = try ngram.nextPrime(prime);
            var offset: i64 = 0;
            for (0..heads) |i| {
                prime = try ngram.nextPrime(prime);
                if (integer(sizes, i) != prime or integer(offsets, i) != offset) return p.reject(stem);
                offset = try std.math.add(i64, offset, @intCast(prime));
            }
            ordinal += 1;
        };
    }
};
pub fn normPolicy(ns: weights.Namespace, c: *const cfg.Config, p: *cfg.Problem) !void {
    // The supplied HF model defines gamma. A converted MLX namespace does not establish its exporter's norm rule.
    if (ns == .mlx and !c.norm_explicit) return p.reject("norm_weight_mode");
}
fn integer(t: st.Tensor, i: usize) i64 {
    return std.mem.readInt(i64, t.bytes[i * 8 ..][0..8], .little);
}
pub fn namespace(header: st.Header, p: *cfg.Problem) !weights.Namespace {
    var result: ?weights.Namespace = null;
    var name: [128]u8 = undefined;
    for (std.enums.values(weights.Namespace)) |ns| {
        const key = try std.fmt.bufPrint(&name, "{s}embed_tokens.weight", .{ns.prefix()});
        if (!header.contains(key)) continue;
        if (result != null) return p.reject("embed_tokens.weight namespace");
        result = ns;
    }
    return result orelse p.reject("embed_tokens.weight");
}
