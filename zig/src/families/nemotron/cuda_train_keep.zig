//! Preserve each layer's backward inputs in the trainer's own scratch.

/// Layer l's activations into its slot from the forward's scratch, for its backward.
pub fn keep(t: anytype, l: usize, rows: usize) !void {
    const c = t.c;
    const o = t.e.ops();
    const b = &t.buf;
    const kp = t.kept[l];
    switch (c.kinds[l]) {
        .mamba => {
            try o.copy(kp.proj, b.proj, rows * c.projDim() * 2);
            try o.copy(kp.ya, b.g, rows * c.inner() * 2);
        },
        .attention => try o.copy(kp.ya, b.att, rows * c.heads * c.head_dim * 2),
        .moe => {
            const S = c.slots();
            const W = c.expert_width;
            try o.copy(kp.act, b.act, rows * S * W * 2);
            try t.ops_().copyRows(b.act + @as(u64, c.top_k) * W * 2, S * W, 0, kp.upr, c.shared_width, c.shared_width, rows);
            try o.copy(kp.ys, b.ymoe, rows * S * c.hidden * 2);
            try o.copy(kp.ids, b.pick, rows * S * 4);
            try o.copy(kp.wts, b.wts, rows * S * 4);
        },
    }
}
