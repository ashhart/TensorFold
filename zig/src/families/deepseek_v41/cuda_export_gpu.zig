//! The CUDA engine's side of the hand-off export: slot state and pool pages read back to the host, plus the refusals.
const std = @import("std");
const model = @import("cuda_model.zig");
const sg = @import("sessions_gpu.zig");
const pf = @import("forward_prefill.zig");
const cx = @import("cuda_export.zig");
const ex = @import("handoff/export.zig");
const ced = @import("ced.zig");

pub const Error = error{ BadWindow, SplitKvIsNotExported, UnsupportedKvFormat };

/// Why this engine cannot export hand-offs as configured, or null; checked once at open.
pub fn refusal(m: *const model.Model) ?[]const u8 {
    const ss = m.sessions orelse return "--mcdma-produce needs the session store over the paged pool: set TF_DSV41_POOL_TOKENS (and TF_DSV41_SESSIONS=1; with TF_DSV41_SLOTS above 1 also TF_DSV41_SESSION_SLOTS=1)";
    if (!m.gt.own_prefill or m.gt.prompt_tail != .verify) return "--mcdma-produce needs the served prompt shape: TF_DSV41_OWN_PREFILL=1 and TF_DSV41_PROMPT_TAIL=verify (the defaults), so the prompt's last token stays pending at the cut";
    const kx = ss.kx;
    if (kx.layout.opts.split and kx.world > 1) return "--mcdma-produce exports a replicated KV pool: unset TF_DSV41_KV_SPLIT (a split pool holds half the pages on the other rank)";
    if (kx.layout.opts.kv != .fp8) return "--mcdma-produce exports CUDA's fp8 comp rows (584 bytes); this pool holds another format";
    return null;
}

/// One captured state: the snapshot, its pieces over the snapshot's bytes, and the pages read for each KV source.
pub const Capture = struct {
    gpa: std.mem.Allocator,
    snap: *sg.Sessions.Snap,
    pieces: []cx.Piece = &.{},
    pages: []cx.Pages = &.{},
    bufs: []const []u8 = &.{},
    st: cx.State = undefined,

    fn deinit(c: *Capture) void {
        for (c.bufs) |b| c.gpa.free(b);
        c.gpa.free(c.pages);
        c.gpa.free(c.pieces);
        sg.Sessions.Snap.release(c.snap);
        c.gpa.destroy(c);
    }
};

/// The active slot's state at `at` tokens, on the engine thread with the slot standing there.
pub fn capture(gpa: std.mem.Allocator, ss: *sg.Sessions, at: u64) !*Capture {
    const f = ss.f;
    const kx = ss.kx;
    if (f.slot.pos != at or f.slot.pending != null or ss.hist[ss.cur()].items.len != at) return error.BadWindow;
    // replay: the cut stands after the decoder replay (target.zig), so the live slot holds the decoder layers' rings
    const replay = pf.state(f).mode == .replay;
    if (replay and f.slot.ced.hi != at) return error.BadWindow;
    if (kx.layout.opts.split and kx.world > 1) return error.SplitKvIsNotExported;
    if (kx.layout.opts.kv != .fp8) return error.UnsupportedKvFormat;
    const snap = try ss.snapshot();
    const c = gpa.create(Capture) catch |e| {
        sg.Sessions.Snap.release(snap);
        return e;
    };
    c.* = .{ .gpa = gpa, .snap = snap };
    var bufs: std.ArrayList([]u8) = .empty;
    errdefer {
        for (bufs.items) |b| gpa.free(b);
        bufs.deinit(gpa);
        c.deinit();
    }

    // replay: the drafter's rings stand outside the snapshot; the decoder replay's hand-off filled them
    const rings = if (replay) ss.drafter_rings else &[0]sg.Piece{};
    c.pieces = try gpa.alloc(cx.Piece, ss.pieces.len + rings.len);
    var at_byte: usize = 0;
    for (ss.pieces, c.pieces[0..ss.pieces.len]) |p, *out| {
        out.* = .{ .role = p.role, .bytes = snap.bytes[at_byte..][0..p.len] };
        at_byte += p.len;
    }
    for (rings, c.pieces[ss.pieces.len..]) |p, *out| {
        const bytes = try gpa.alloc(u8, p.len);
        try bufs.append(gpa, bytes);
        try ss.readDrafterRing(p, bytes);
        out.* = .{ .role = p.role, .bytes = bytes };
    }

    const page = kx.pool.page;
    const npages: u32 = @intCast((at + page - 1) / page);
    if (kx.slot.len < npages) return error.BadWindow;
    const phys = kx.slot.pages[0..npages];
    const store = kx.dev.store();
    const sources = kx.layout.sources[0..kx.layout.nsources];
    c.pages = try gpa.alloc(cx.Pages, sources.len);
    const fams = kx.layout.families();
    for (sources, c.pages) |layer, *pg| {
        const comp = kx.layout.compOf(layer).?;
        const index = kx.layout.indexOf(layer).?;
        const comp_bytes = try gpa.alloc(u8, npages * fams[comp].pageBytes(page));
        try bufs.append(gpa, comp_bytes);
        const index_bytes = try gpa.alloc(u8, npages * fams[index].pageBytes(page));
        try bufs.append(gpa, index_bytes);
        try store.read(comp, phys, comp_bytes);
        try store.read(index, phys, index_bytes);
        pg.* = .{ .layer = layer, .comp = comp_bytes, .index = index_bytes };
    }
    c.bufs = try bufs.toOwnedSlice(gpa);
    c.st = .{
        .pos = at,
        .ds_rows = snap.ds_rows,
        .index_format = if (kx.layout.opts.index == .fp8) ex.IndexFormat.fp8 else ex.IndexFormat.bf16,
        .pieces = c.pieces,
        .pages = c.pages,
        .decoder_from = if (replay) try ced.decoderStart(f.cfg) else cx.State.none,
        .replay_rows = if (replay) ced.keep else 0,
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
