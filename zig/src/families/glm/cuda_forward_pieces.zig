//! The existing pieced prompt exchanges, isolated so their diagnostic spans stay beside each stream operation.
const relay = @import("cuda_relay.zig");
const fw = @import("cuda_forward.zig");
const prof = @import("cuda_prof.zig");
const Forward = fw.Forward;
const Pieces = fw.Pieces;
const Site = fw.Forward.Site;
const room = fw.Forward.room;
const share = fw.Forward.share;

/// overlap.Overlap.partials: each piece filled on the main stream, exchanged on the second (gathered or split).
pub fn exchangePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
    defer prof.context(f.prof, j, null, if (site == .mixer) .mixer else .ffn);
    const D = f.ranks[0].w.config.hidden;
    const world = f.ranks[0].w.share.world;
    for (cut, 0..) |piece, q| {
        prof.context(f.prof, j, q, if (site == .mixer) .mixer else .ffn);
        const lo = piece[0];
        const n = piece[1] - lo;
        const split = f.splits(n);
        for (f.ranks) |r| try f.fillRows(r, r.w.layers[j], site, lo, piece[1], if (split) r.b.part + lo * D * 4 else room(r, lo) + r.w.share.rank * n * D * 4);
        try p.filled[q].record(f.stream);
        var stamp = prof.span(f.prof, p.side, .wait_filled);
        try p.side.wait(p.filled[q]);
        stamp.end();
        stamp = prof.span(f.prof, p.side, .partials);
        const bytes = n * D * 4;
        if (!split) switch (f.exchange) {
            .local => for (f.ranks) |to| for (f.ranks) |from| {
                if (to.w.share.rank == from.w.share.rank) continue;
                const off = from.w.share.rank * bytes;
                try f.copy(p.side, room(to, lo) + off, room(from, lo) + off, bytes);
            },
            .nccl => |x| {
                const r = f.ranks[0];
                try x.lib.check(x.lib.api.ncclAllGather(room(r, lo) + r.w.share.rank * bytes, room(r, lo), n * D, .f32, x.comm, p.side.handle), "ncclAllGather");
            },
        } else switch (f.exchange) {
            .local => for (f.ranks) |to| {
                const mine = share(n, world, to.w.share.rank);
                const own = mine[1] - mine[0];
                for (f.ranks) |from| try f.copy(p.side, room(to, lo) + from.w.share.rank * own * D * 4, from.b.part + (lo + mine[0]) * D * 4, own * D * 4);
            },
            .nccl => |x| {
                const r = f.ranks[0];
                const me = r.w.share.rank;
                const mine = share(n, world, me);
                const own = mine[1] - mine[0];
                const plan: relay.Relay = .{ .world = world, .rank = me, .n = n };
                if (f.ring) try relay.partials(x.lib, x.comm, p.side, plan, room(r, lo), r.b.part, lo, D) else try relay.splitPartials(x.lib, x.comm, p.side, plan, room(r, lo), r.b.part, lo, D);
                try f.copy(p.side, room(r, lo) + me * own * D * 4, r.b.part + (lo + mine[0]) * D * 4, own * D * 4);
            },
        }
        stamp.end();
        try p.gathered[q].record(p.side);
    }
}

/// overlap.Overlap.glue: each piece's glue once its exchange is in, its rows then shared on the second stream.
pub fn gluePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
    defer prof.context(f.prof, j, null, if (site == .mixer) .mixer else .ffn);
    const c = &f.ranks[0].w.config;
    const D: usize = c.hidden;
    const world = f.ranks[0].w.share.world;
    const last = site == .ffn and j + 1 == f.ranks[0].w.layers.len;
    var sent: usize = 0;
    for (cut, 0..) |piece, q| {
        prof.context(f.prof, j, q, if (site == .mixer) .mixer else .ffn);
        const lo = piece[0];
        const n = piece[1] - lo;
        var stamp = prof.span(f.prof, f.stream, .wait_gathered);
        try f.stream.wait(p.gathered[q]);
        stamp.end();
        if (!f.splits(n)) {
            for (f.ranks) |r| try f.glueRows(r, j, site, lo, piece[1], room(r, lo));
            continue;
        }
        for (f.ranks) |r| {
            const mine = share(n, world, r.w.share.rank);
            try f.glueRows(r, j, site, lo + mine[0], lo + mine[1], room(r, lo));
        }
        try p.glues[q].record(f.stream);
        stamp = prof.span(f.prof, p.side, .wait_glues);
        try p.side.wait(p.glues[q]);
        stamp.end();
        stamp = prof.span(f.prof, p.side, .glued);
        switch (f.exchange) {
            .local => for (f.ranks) |from| {
                const mine = share(n, world, from.w.share.rank);
                const bytes = (mine[1] - mine[0]) * D * 2;
                for (f.ranks) |to| if (to.w.share.rank != from.w.share.rank) {
                    const at = (lo + mine[0]) * D * 2;
                    try f.copy(p.side, (if (last) to.b.hidden else to.b.normed) + at, (if (last) from.b.hidden else from.b.normed) + at, bytes);
                };
            },
            .nccl => |x| {
                const r = f.ranks[0];
                const me = r.w.share.rank;
                const rows_of = if (last) r.b.hidden else r.b.normed;
                const plan: relay.Relay = .{ .world = world, .rank = me, .n = n };
                if (f.ring) try relay.glued(x.lib, x.comm, p.side, plan, rows_of, lo, D) else try relay.splitGlued(x.lib, x.comm, p.side, plan, rows_of, lo, D);
            },
        }
        stamp.end();
        try p.shared[q].record(p.side);
        sent |= @as(usize, 1) << @intCast(q);
    }
    for (0..cut.len) |q| if (sent >> @intCast(q) & 1 == 1) {
        prof.context(f.prof, j, q, if (site == .mixer) .mixer else .ffn);
        const stamp = prof.span(f.prof, f.stream, .wait_shared);
        try f.stream.wait(p.shared[q]);
        stamp.end();
    };
}
