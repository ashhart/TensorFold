//! The loss adjoint traverses each intervening dense block, hyper-connection and causal mixer to every requested row.
const core = @import("core");
const hc = @import("train_hc.zig");
pub fn run(t: anytype) !void {
    const c = &t.e.c;
    const o = t.ops();
    var bits: u64 = 0;
    for (t.seq.nudge.layers) |mask| bits |= mask;
    try o.zero(t.buf(.push));
    if (bits == 0) return;
    const low: usize = @ctz(bits);
    try o.matrix(t.e.model.head, t.buf(.logits), t.buf(.mixed), t.rows, 0, 0, false);
    try hc.back(t, null, "model.hyper_connection_mixer", t.e.model.readout, try t.tape.indexed(c.layers - 1, "mlp_hyper_connection.residual", t.rows), false);
    var l = c.layers;
    while (l > low) {
        l -= 1;
        const layer = t.e.model.layers[l];
        try hc.branch(t, l, "mlp_hyper_connection", "mlp.pocket_output");
        for (t.seq.nudge.layers, t.seq.nudge.rows, 0..) |mask, row, key| {
            if (mask & (@as(u64, 1) << @intCast(l)) == 0) continue;
            try o.copy(try t.buf(.branch).plus(row * c.hidden * 4, c.hidden * 4), try t.buf(.push).plus((key * c.layers + l) * c.hidden * 4, c.hidden * 4));
        }
        if (l == low) break;
        try o.matrix(layer.dense.down, t.buf(.branch), t.buf(.act), t.rows, 0, 0, false);
        var p: core.living_rows.Params = .{};
        p.u[0] = @intCast(t.rows);
        p.u[1] = @intCast(c.intermediate);
        try o.row(.swiglu, p, &.{ .{ 0, try t.tape.indexed(l, "mlp.gate", t.rows) }, .{ 1, try t.tape.indexed(l, "mlp.up", t.rows) }, .{ 2, t.buf(.act) }, .{ 8, t.buf(.gate) }, .{ 9, t.buf(.up) } });
        try o.matrix(layer.dense.gate, t.buf(.gate), t.buf(.mixed), t.rows, 0, 0, false);
        try o.matrix(layer.dense.up, t.buf(.up), t.buf(.mixed), t.rows, 0, 0, true);
        if (t.runtime.pockets.layers[l]) |pocket| {
            if (pocket.used != 0) {
                p.u[1] = @intCast(c.hidden);
                p.u[2] = @intCast(pocket.used);
                p.u[3] = t.from[l];
                p.u[4] = 64;
                p.f[0] = t.seq.scale;
                const stored = o.raw(pocket.buffer);
                try o.row(.pocket_coeff, p, &.{ .{ 0, try t.tape.indexed(l, "mlp_hyper_connection.mix", t.rows) }, .{ 1, stored }, .{ 2, t.buf(.branch) }, .{ 8, t.buf(.pocket_coeff) } });
                try o.row(.pocket_back, p, &.{ .{ 0, t.buf(.pocket_coeff) }, .{ 1, stored }, .{ 8, t.buf(.mixed) } });
            }
        }
        try hc.back(t, l, "mlp_hyper_connection", layer.mlp, try t.tape.indexed(l, "attn_hyper_connection.residual", t.rows), true);
        try hc.branch(t, l, "attn_hyper_connection", if (c.kinds[l] == .linear_attention) "linear_attn.output" else "self_attn.output");
        switch (layer.mixer) {
            .linear => |w| try @import("train_mixers.zig").gdn(t, l, w),
            .indexed => |w| try @import("train_mixers.zig").qsa(t, l, w),
        }
        const raw = if (l == 0) try t.tape.get("embed", t.rows) else try t.tape.indexed(l - 1, "mlp_hyper_connection.residual", t.rows);
        try hc.back(t, l, "attn_hyper_connection", layer.attention, raw, true);
    }
}
