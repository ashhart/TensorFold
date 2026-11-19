//! Gather Hy4-flash's selected affine Engram rows from the mapped checkpoint.
const std = @import("std");
const core = @import("core");
const engram = @import("engram.zig");
const loader = @import("loader.zig");

const group = 32;

/// Borrows the loader's checkpoint mapping. The large embedding table can stay in file-backed memory.
fn willNeed(t: core.checkpoint_host.Tensor, spans: []const core.quant.Span) void {
    const os = @import("builtin").os.tag;
    if (os != .macos and os != .linux) return;
    if (t.shape[0] == 0 or t.bytes.len == 0) return;
    const row = t.bytes.len / t.shape[0];
    const page = std.heap.pageSize();
    for (spans) |span| {
        const lo = std.mem.alignBackward(usize, @intFromPtr(t.bytes.ptr) + span.from * row, page);
        const hi = std.mem.alignForward(usize, @intFromPtr(t.bytes.ptr) + span.to * row, page);
        _ = std.c.madvise(@ptrFromInt(lo), hi - lo, std.c.MADV.WILLNEED);
    }
}

pub const Table = struct {
    map: engram.TokenMap,
    hash: engram.Layer,
    affine: core.quant.mlx.Host,
    width: usize,

    pub fn open(model: *loader.Loaded, layer_number: u32) !Table {
        const text = model.config.value.text_config;
        if (std.mem.indexOfScalar(u32, text.engram_layer_ids, layer_number) == null) return error.UnknownEngramLayer;
        const hash = try engram.Layer.open(text, layer_number);
        const rows = try text.engramRows(layer_number);
        const width = text.engram_n_embed_per_ngram / engram.heads;
        if (width == 0 or width % group != 0) return error.BadEngramGeometry;

        var buffer: [160]u8 = undefined;
        var name = try std.fmt.bufPrint(&buffer, "model.layers.{d}.engram_layer.lookup_table", .{layer_number - 1});
        const map = try engram.TokenMap.open((try model.checkpoint.expect(name, .i32, &.{text.engram_lookup_table_size})).bytes, text);
        const prefix = "model.layers.{d}.engram_layer.multi_head_embedding.embedding";
        name = try std.fmt.bufPrint(&buffer, prefix ++ ".weight", .{layer_number - 1});
        const words = try model.checkpoint.expect(name, .u32, &.{ rows, @as(usize, width) * model.bits / 32 });
        try model.checkpoint.adviseRandom(name);
        name = try std.fmt.bufPrint(&buffer, prefix ++ ".scales", .{layer_number - 1});
        const scales = try model.checkpoint.expect(name, .bf16, &.{ rows, width / group });
        try model.checkpoint.adviseRandom(name);
        name = try std.fmt.bufPrint(&buffer, prefix ++ ".biases", .{layer_number - 1});
        const biases = try model.checkpoint.expect(name, .bf16, &.{ rows, width / group });
        try model.checkpoint.adviseRandom(name);
        return .{
            .map = map,
            .hash = hash,
            .affine = .{ .words = words, .scales = scales, .biases = biases, .bits = model.bits, .group = group },
            .width = width,
        };
    }

    /// The table rows a token window reads, one a column (the gather's spans).
    pub fn windowRows(self: Table, window: [engram.order]u32) [engram.columns]u32 {
        return self.hash.hash(self.map, window);
    }

    pub fn elementsPerToken(self: Table) usize {
        return engram.columns * self.width;
    }

    /// Each window is current token first, then older tokens, padded by the caller for a new stream.
    /// Output is raw BF16 embedding rows [token][hash head][width], before FP8 simulation or normalization.
    pub fn gather(self: Table, a: std.mem.Allocator, windows: []const [engram.order]u32, output: []u16) !void {
        const cells = try std.math.mul(usize, windows.len, self.elementsPerToken());
        if (output.len != cells) return error.BadEngramOutput;
        if (windows.len == 0) return;
        const spans = try a.alloc(core.quant.Span, windows.len * engram.columns);
        defer a.free(spans);
        for (windows, 0..) |window, token| {
            for (self.hash.hash(self.map, window), 0..) |row, head| {
                const index: usize = row;
                spans[token * engram.columns + head] = .{ .from = index, .to = index + 1 };
            }
        }

        // Ask for every row's pages at once (asynchronous read-ahead), so cold rows' faults overlap rather than queue
        // one by one in the copies below.
        for ([_]core.checkpoint_host.Tensor{ self.affine.words, self.affine.scales, self.affine.biases }) |t| willNeed(t, spans);
        const words = try core.quant.slice.takeRows(a, self.affine.words, spans);
        defer a.free(words.bytes);
        const scales = try core.quant.slice.takeRows(a, self.affine.scales, spans);
        defer a.free(scales.bytes);
        const biases = try core.quant.slice.takeRows(a, self.affine.biases, spans);
        defer a.free(biases.bytes);
        const values = try core.quant.mlx.dequant(a, words, scales, biases, group);
        defer a.free(values);
        for (output, values) |*word, v| word.* = core.quant.convert.bf16(v);
    }
};
