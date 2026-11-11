//! The CUDA engine's side of the hand-off export: slot state and pool pages read back to the host, plus the refusals.
const std = @import("std");
const model = @import("cuda_model.zig");
const sg = @import("sessions_gpu.zig");
const pf = @import("forward_prefill.zig");
const cx = @import("cuda_export.zig");
const ex = @import("handoff/export.zig");
const ced = @import("ced.zig");
const kvs = @import("cuda_kv.zig");
const plan = @import("handoff/plan.zig");

pub const Error = error{ BadWindow, UnsupportedKvFormat };

/// Why this engine cannot export hand-offs as configured, or null; checked once at open.
pub fn refusal(m: *const model.Model) ?[]const u8 {
    const ss = m.sessions orelse return "--mcdma-produce needs the session store over the paged pool: set TF_DSV41_POOL_TOKENS (and TF_DSV41_SESSIONS=1; with TF_DSV41_SLOTS above 1 also TF_DSV41_SESSION_SLOTS=1)";
    if (!m.gt.own_prefill or m.gt.prompt_tail != .verify) return "--mcdma-produce needs the served prompt shape: TF_DSV41_OWN_PREFILL=1 and TF_DSV41_PROMPT_TAIL=verify (the defaults), so the prompt's last token stays pending at the cut";
    const kx = ss.kx;
    if (kx.layout.opts.split and kx.world > plan.max_world) return "--mcdma-produce exports a split KV pool over at most 8 ranks (each rank exports its own pages)";
    if (kx.layout.opts.kv != .fp8) return "--mcdma-produce exports CUDA's fp8 comp rows (584 bytes); this pool holds another format";
    return null;
}

/// One captured state: the snapshot, its pieces over the snapshot's bytes, and the pages read for each KV source.
/// A rank of a split pool other than the first holds no snapshot: it exports its pool pages only.
pub const Capture = struct {
    gpa: std.mem.Allocator,
    snap: ?*sg.Sessions.Snap,
    pieces: []cx.Piece = &.{},
    pages: []cx.Pages = &.{},
    bufs: []const []u8 = &.{},
    st: cx.State = undefined,

    fn deinit(c: *Capture) void {
        for (c.bufs) |b| c.gpa.free(b);
        c.gpa.free(c.pages);
        c.gpa.free(c.pieces);
        if (c.snap) |sn| sg.Sessions.Snap.release(sn);
        c.gpa.destroy(c);
    }
};

/// This rank's pool pages of every KV source for the first `at` tokens, each family's pages back to back in logical
/// order. A split pool's comp pages are the ones the rank stores (page % world == rank; the index keys are replicated,
/// so only the same logical pages are read): the rank's shard of a KV_SPLIT export. Buffers go to `bufs`.
fn readPages(gpa: std.mem.Allocator, kx: *kvs.Kv, at: u64, bufs: *std.ArrayList([]u8)) ![]cx.Pages {
    const page = kx.pool.page;
    const npages: u32 = @intCast((at + page - 1) / page);
    if (kx.slot.len < npages) return error.BadWindow;
    const split = kx.layout.opts.split and kx.world > 1;
    const logical = kx.slot.pages[0..npages];
    // the logical pages this rank exports, and where its comp rows live in the family's tensor
    const own = try gpa.alloc(u32, if (split) (npages + kx.world - 1 - kx.rank) / kx.world else npages);
    defer gpa.free(own);
    const local = try gpa.alloc(u32, own.len);
    defer gpa.free(local);
    if (split) {
        for (own, local, 0..) |*o, *l, j| {
            const k: u32 = @intCast(j * kx.world + kx.rank);
            o.* = logical[k];
            l.* = kx.pool.localPage(logical[k]);
        }
    } else {
        @memcpy(own, logical);
        @memcpy(local, logical);
    }
    const store = kx.dev.store();
    const sources = kx.layout.sources[0..kx.layout.nsources];
    const out = try gpa.alloc(cx.Pages, sources.len);
    errdefer gpa.free(out);
    const fams = kx.layout.families();
    for (sources, out) |layer, *pg| {
        const comp = kx.layout.compOf(layer).?;
        const index = kx.layout.indexOf(layer).?;
        const comp_bytes = try gpa.alloc(u8, own.len * fams[comp].pageBytes(page));
        try bufs.append(gpa, comp_bytes);
        const index_bytes = try gpa.alloc(u8, own.len * fams[index].pageBytes(page));
        try bufs.append(gpa, index_bytes);
        try store.read(comp, local, comp_bytes);
        try store.read(index, own, index_bytes);
        pg.* = .{ .layer = layer, .comp = comp_bytes, .index = index_bytes };
    }
    return out;
}

/// The active slot's state at `at` tokens, on the engine thread with the slot standing there.
pub fn capture(gpa: std.mem.Allocator, ss: *sg.Sessions, at: u64) !*Capture {
    const f = ss.f;
    const kx = ss.kx;
    if (f.slot.pos != at or f.slot.pending != null or ss.hist[ss.cur()].items.len != at) return error.BadWindow;
    // replay: the cut stands after the decoder replay (target.zig), so the live slot holds the decoder layers' rings
    const replay = pf.state(f).mode == .replay;
    if (replay and f.slot.ced.hi != at) return error.BadWindow;
    if (kx.layout.opts.kv != .fp8) return error.UnsupportedKvFormat;
    const shards: u32 = if (kx.layout.opts.split and kx.world > 1) kx.world else 1;
    const shard: u32 = if (shards > 1) kx.rank else 0;
    // shard 0 carries the snapshot (rings, carries, drafter rows); the others only their pages
    const snap: ?*sg.Sessions.Snap = if (shard == 0) try ss.snapshot() else null;
    const c = gpa.create(Capture) catch |e| {
        if (snap) |sn| sg.Sessions.Snap.release(sn);
        return e;
    };
    c.* = .{ .gpa = gpa, .snap = snap };
    var bufs: std.ArrayList([]u8) = .empty;
    errdefer {
        for (bufs.items) |b| gpa.free(b);
        bufs.deinit(gpa);
        c.deinit();
    }

    if (snap) |sn| {
        // replay: the drafter's rings stand outside the snapshot; the decoder replay's hand-off filled them
        const rings = if (replay) ss.drafter_rings else &[0]sg.Piece{};
        c.pieces = try gpa.alloc(cx.Piece, ss.pieces.len + rings.len);
        var at_byte: usize = 0;
        for (ss.pieces, c.pieces[0..ss.pieces.len]) |p, *out| {
            out.* = .{ .role = p.role, .bytes = sn.bytes[at_byte..][0..p.len] };
            at_byte += p.len;
        }
        for (rings, c.pieces[ss.pieces.len..]) |p, *out| {
            const bytes = try gpa.alloc(u8, p.len);
            try bufs.append(gpa, bytes);
            try ss.readDrafterRing(p, bytes);
            out.* = .{ .role = p.role, .bytes = bytes };
        }
    }
    c.pages = try readPages(gpa, kx, at, &bufs);
    c.bufs = try bufs.toOwnedSlice(gpa);
    c.st = .{
        .pos = at,
        .ds_rows = if (snap) |sn| sn.ds_rows else 0,
        .index_format = if (kx.layout.opts.index == .fp8) ex.IndexFormat.fp8 else ex.IndexFormat.bf16,
        .pieces = c.pieces,
        .pages = c.pages,
        .decoder_from = if (replay) try ced.decoderStart(f.cfg) else cx.State.none,
        .replay_rows = if (replay) ced.keep else 0,
        .shard = shard,
        .shards = shards,
    };
    return c;
}

/// The `cuda_export.Source` over the engine's session store.
pub const Source = struct {
    ss: *sg.Sessions,

    pub fn source(s: *Source) cx.Source {
        return .{ .ptr = s, .take = take, .release = release };
    }

    fn take(ptr: *anyopaque, gpa: std.mem.Allocator, at: u64) anyerror!cx.Captured {
        const s: *Source = @ptrCast(@alignCast(ptr));
        const c = try capture(gpa, s.ss, at);
        return .{ .ctx = c, .state = c.st };
    }

    fn release(ctx: *anyopaque) void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.deinit();
    }
};
