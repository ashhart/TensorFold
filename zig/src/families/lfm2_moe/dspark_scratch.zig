//! DSpark working buffers; the drafter releases owned buffers before their allocation list on failure.
const mtl = @import("metal");
const c = @import("config.zig");
const st = @import("state.zig");
const ds = @import("dspark.zig");
const block = @import("dspark_block.zig");
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const kw = c.kv_heads * c.head_dim;

/// The passes' working buffers, one pass at a time on the target's queue.
pub const Scratch = struct {
    ids: mtl.Buffer,
    h: mtl.Buffer,
    x: mtl.Buffer,
    r: mtl.Buffer,
    y: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    queries: mtl.Buffer, // projected Q; normalized Q reuses the padded x after all projections finish
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    po: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    logits: mtl.Buffer,
    sums: mtl.Buffer,
    fcx: mtl.Buffer, // an absorb pass's context rows: fc's output, normed in place
    best: mtl.Buffer, // the Markov scan's best score and token of each threadgroup, by stream
    at: mtl.Buffer,
    picks: mtl.Buffer,
    conf: mtl.Buffer, // the confidence logits, step by step, a pass's blocks each
    gather: mtl.Buffer, // a hold's small staged pieces, side by side, for one fc call
};

fn buffer(d: *ds.Drafter, bytes: usize) !mtl.Buffer {
    const b = try d.target.device.buffer(@max(bytes, 16), shared);
    errdefer b.deinit();
    try d.owned.append(d.gpa, b);
    return b;
}

pub fn prepare(d: *ds.Drafter) !void {
    const r = ds.pass_rows;
    const wide = @max(ds.pass_rows, ds.absorb_rows);
    const i = d.config.intermediate;
    const partials = d.partRows() * block.part_row;
    const sums = if (d.options.bits == 0) r * (c.hidden / 64) else @max(ds.absorb_rows * (d.config.tap_count * c.hidden / 64), r * (i / 64));
    d.s = .{
        .ids = try buffer(d, r * 4),
        .h = try buffer(d, r * c.hidden * 2),
        .x = try buffer(d, (r + st.attn_rows) * c.hidden * 2),
        .r = try buffer(d, r * c.hidden * 2),
        .y = try buffer(d, r * c.hidden * 2),
        .k = try buffer(d, wide * kw * 2),
        .v = try buffer(d, wide * kw * 2),
        .queries = try buffer(d, r * c.hidden * 2),
        .pm = try buffer(d, partials * 4),
        .pl = try buffer(d, partials * 4),
        .po = try buffer(d, partials * c.head_dim * 4),
        .gate = try buffer(d, r * i * 2),
        .up = try buffer(d, r * i * 2),
        .act = try buffer(d, r * i * 2),
        .logits = try buffer(d, r * c.vocab * 2),
        .sums = try buffer(d, sums * 4),
        .fcx = try buffer(d, ds.absorb_rows * c.hidden * 2),
        .best = try buffer(d, r * block.scan_groups * 4),
        .at = try buffer(d, r * block.scan_groups * 4),
        .picks = try buffer(d, d.config.block * r * 8),
        .conf = try buffer(d, d.config.block * r * 4),
        .gather = try buffer(d, ds.gather_rows * d.tapBytes()),
    };
}
