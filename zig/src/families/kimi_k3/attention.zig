//! Contiguous row spans keep dense and indexed MLA choices independent of other streams.
const mtl = @import("metal");
const Kn = @import("kernels.zig");
const index = @import("index.zig");
const Config = @import("config.zig").Config;
const Scratch = @import("round.zig").Scratch;

pub fn encode(k: *const Kn.Kernels, e: mtl.ComputeEncoder, c: *const Config, sc: *const Scratch, ix: ?*index.Work, args: Kn.MlaArgs) void {
    const entries = sc.mrows.slice(Kn.MlaRow, sc.rows);
    var first: usize = 0;
    while (first < entries.len) {
        const run = index.span(entries, first, ix != null);
        var a = args;
        a.rows = run.count;
        const q = sc.ref("q").at(first * a.q_stride * 2);
        const qlat = sc.ref("qlat").at(first * a.heads * c.kv_lora * 4);
        const partial = sc.ref("partial").at(first * a.heads * Kn.mla_splits * (c.kv_lora + 2) * 4);
        const olat = sc.ref("olat").at(first * a.heads * c.kv_lora * 4);
        if (run.indexed) {
            const work = ix.?;
            const p = k.indexView();
            p.attention(e, q, qlat, work.ref("rows").at(first * @sizeOf(index.Row)), work.ref("list"), partial, olat, a);
        } else {
            const rows = sc.ref("mrows").at(first * @sizeOf(Kn.MlaRow));
            k.mlaAttend(e, q, qlat, rows, partial, a);
            k.mlaMerge(e, partial, rows, olat, a);
        }
        first += run.count;
    }
}
