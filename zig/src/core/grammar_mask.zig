//! Structured output on Metal: the lane core's bits for a row turn its disallowed logits to -inf before the draw.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Masks = struct {
    pipeline: mtl.Pipeline,
    bits: mtl.Buffer, // u32 [rows + 1][words]: a forward's rows by their place in it, then a first token's row
    rows: usize,
    words: usize,

    /// Bits for `rows` rows of a forward and one first token over `vocab` logits.
    pub fn init(device: mtl.Device, rows: usize, vocab: usize) !Masks {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_grammar_mask, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipeline = try mtl.Pipeline.init(device, lib, "tf_grammar_mask", false);
        errdefer pipeline.deinit();
        const words = (vocab + 31) / 32;
        const bits = try device.buffer((rows + 1) * words * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return .{ .pipeline = pipeline, .bits = bits, .rows = rows, .words = words };
    }

    pub fn deinit(m: *Masks) void {
        m.bits.deinit();
        m.pipeline.deinit();
    }

    /// The row a first token's bits go to.
    pub fn firstRow(m: *const Masks) usize {
        return m.rows;
    }

    /// Rows from `row` take `bits` (whole rows); the caller has no command in flight that reads them.
    pub fn put(m: *Masks, row: usize, bits: []const u32) !void {
        if (bits.len % m.words != 0 or row * m.words + bits.len > (m.rows + 1) * m.words) return error.InvalidGrammarMasks;
        @memcpy(m.bits.slice(u32, (m.rows + 1) * m.words)[row * m.words ..][0..bits.len], bits);
    }

    /// `rows` rows of bf16 logits from `logits_off` (`vocab` apart) masked by the bits from row `row`.
    pub fn encode(m: *const Masks, e: anytype, logits: mtl.Buffer, logits_off: usize, vocab: usize, row: usize, rows: usize) void {
        e.pipe(m.pipeline);
        e.buf(logits, logits_off, 0);
        e.buf(m.bits, row * m.words * 4, 1);
        e.bytes([4]u32{ @intCast(vocab), @intCast(m.words), @intCast(rows), 0 }, 2);
        e.run(.{ m.words, rows, 1 }, .{ 256, 1, 1 });
    }
};
