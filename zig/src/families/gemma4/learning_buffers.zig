//! A checked plan charges the complete tape, gradients and original-row weight staging before native allocation.
const std = @import("std");
const profile = @import("learning_profile.zig");
const hook = @import("learning_hook.zig");
pub const max_rows = 256;
pub const Slot = enum { dummy, grad, branch, norm, mixed, act, gate, up, context, q, k, v, raw, scores, dp, probability, ds, coeff, logits, capped, stats, targets, weights, delta, push, ids, counts, sparse, stage, frequency };
fn mul(xs: []const usize) !usize {
    var n: usize = 1;
    for (xs) |x| n = try std.math.mul(usize, n, x);
    return n;
}
pub fn width(s: profile.Shape, layer: ?usize, node: hook.Node) usize {
    if (layer) |l| return switch (node) {
        .q_raw, .query => s.heads * s.dim(l),
        .k_raw, .v_raw, .key, .value => s.kv(l) * s.dim(l),
        .gate, .up => s.intermediate,
        .attn_output, .mid, .mlp_input, .mlp_base, .mlp_output, .residual => s.hidden,
        else => 0,
    };
    return switch (node) {
        .input => s.hidden,
        .head_raw, .head => s.vocab,
        else => 0,
    };
}
pub fn Plan(comptime B: type) type {
    return struct {
        const Self = @This();
        pub const Ref = @import("learning_ref.zig").Ref(B);
        shape: profile.Shape,
        pack: usize,
        offsets: std.EnumArray(Slot, usize) = .initFill(0),
        lengths: std.EnumArray(Slot, usize) = .initFill(0),
        tape: [61]std.EnumArray(hook.Node, usize) = @splat(.initFill(0)),
        bytes: usize = 0,
        pub fn make(s: profile.Shape, rows: usize) !Self {
            if (s.layers < 2 or s.layers > 60 or rows < 2 or rows > max_rows or s.hidden == 0 or s.hidden > 5376 or
                s.hidden % 64 != 0 or s.intermediate == 0 or s.intermediate > 21504 or s.intermediate % 64 != 0 or
                s.heads == 0 or s.heads > 32 or s.sliding_kv == 0 or s.sliding_kv > 16 or s.global_kv == 0 or
                s.global_kv > 4 or s.heads % s.sliding_kv != 0 or s.heads % s.global_kv != 0 or s.vocab < 2 or s.vocab > 262144)
                return error.GemmaTrainingShape;
            var p: Self = .{ .shape = s, .pack = rows };
            const h = s.hidden;
            const q = s.heads * 512;
            const k = @max(s.sliding_kv * 256, s.global_kv * 512);
            const largest = @max(s.vocab * h, @max(s.intermediate * h, q * h));
            inline for (std.enums.values(Slot)) |slot| {
                const n = switch (slot) {
                    .dummy => 16,
                    .grad, .branch, .norm, .mixed => try mul(&.{ rows, h, 4 }),
                    .act, .gate, .up => try mul(&.{ rows, s.intermediate, 4 }),
                    .context, .q => try mul(&.{ rows, q, 4 }),
                    .k, .v, .raw => try mul(&.{ rows, @max(k, q), 4 }),
                    .scores, .dp, .probability, .ds => try mul(&.{ rows, s.heads, rows, 4 }),
                    .coeff => try mul(&.{ rows, 64, 2, 4 }),
                    .logits, .capped => try mul(&.{ rows, s.vocab, 4 }),
                    .stats => rows * 8,
                    .targets, .weights, .counts, .sparse => rows * 4,
                    .delta, .push => try mul(&.{ 3, s.layers, h, 4 }),
                    .ids => rows * rows * 4,
                    .stage => try std.math.add(usize, largest / 2 + largest / 16, 64),
                    .frequency => 256 * 4,
                };
                p.offsets.set(slot, try p.add(n));
                p.lengths.set(slot, n);
            }
            for (0..s.layers + 1) |l| inline for (std.enums.values(hook.Node)) |node| {
                const w = width(s, if (l == s.layers) null else l, node);
                if (w > 0) p.tape[l].set(node, try p.add(try mul(&.{ rows, w, 2 })));
            };
            _ = try B.allocationBytes(p.bytes);
            return p;
        }
        fn add(p: *Self, n: usize) !usize {
            const off = (try std.math.add(usize, p.bytes, 15)) & ~@as(usize, 15);
            p.bytes = try std.math.add(usize, off, n);
            return off;
        }
        pub fn buf(p: Self, b: B.Buffer, slot: Slot) Ref {
            return .{ .buffer = b, .offset = p.offsets.get(slot), .len = p.lengths.get(slot) };
        }
        pub fn taped(p: Self, b: B.Buffer, layer: ?usize, node: hook.Node, first: usize, rows: usize) !Ref {
            const w = width(p.shape, layer, node);
            if (w == 0 or first > p.pack or rows > p.pack - first) return error.GemmaTrainingTape;
            return .{ .buffer = b, .offset = p.tape[layer orelse p.shape.layers].get(node) + first * w * 2, .len = rows * w * 2 };
        }
        pub fn hostBytes(p: Self) !u64 {
            const s = p.shape;
            return try mul(&.{ s.layers, p.pack, s.hidden, 2 }) + try mul(&.{ 6, s.layers, s.hidden, 4 }) + p.pack * 20 + s.hidden * 2;
        }
    };
}
