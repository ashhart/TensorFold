//! Forward-only causal reads: replace one row's layer output and read the correct-answer logits before softmax.
const train = @import("cuda_train.zig");
pub fn patch(t: anytype, f: anytype, w: anytype, l: usize) !void {
    if (t.locate.layer == null or l != t.locate.layer.? + 1) return;
    const c = t.c;
    const d = c.hidden;
    const b = f.b;
    const block = t.e.w.blocks[l];
    const o = t.e.ops();
    const offset = t.locate.row * d * 2;
    // chunkPre already summed the previous layer and normalized every row. Change only the intervened row.
    try o.copy(w.x + offset, t.savedAt(l - 1) + offset, d * 2);
    try f.glue.addRmsnorm(w.x + offset, null, block.norm, w.x + offset, b.y + offset, b.xs + t.locate.row * (d / 64) * 4, 1, d, c.eps);
    // Reproject the same batch shape, preserving all untouched rows' arithmetic and launch selection.
    switch (block.kind) {
        .mamba => try o.prefillDense(b.y, block.mamba.in_proj, b.proj, w.rows),
        .attention => try o.prefillDense(b.y, block.attn.qkv, b.qkv, w.rows),
        .moe => {},
    }
}
pub fn finish(t: *train.Trainer, sq: train.Seq, out: []train.Result) !void {
    // One BF16 logit per answer token fits in the mapped stats region, which this read never otherwise uses.
    const stats_at = train.pack_rows * 8;
    for (sq.ids[sq.start..], 0..) |target, r| {
        try t.e.ops().copy(t.host.dev + stats_at + r * 2, t.logits + (r * t.c.vocab + target) * 2, 2);
    }
    try t.e.stream.synchronize();
    const words: [*]const u16 = @ptrCast(@alignCast(t.host.host.bytes.ptr + stats_at));
    var mean: f32 = 0;
    const n = sq.ids.len - sq.start;
    for (words[0..n]) |word| mean += @as(f32, @bitCast(@as(u32, word) << 16));
    out[0] = .{ .loss = -mean / @as(f32, @floatFromInt(n)), .recalled = false };
}

test {
    _ = @import("cuda_train_locate_test.zig");
}
