//! Nemotron's KV pages on CUDA: the core pool's pages in one device arena, and the stream sequences that map them.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const Engine = @import("cuda_engine.zig").Engine;
const state = @import("cuda_state.zig");
const reuse = @import("cuda_reuse.zig");
const mtp = @import("cuda_mtp.zig");
const lesson_route = @import("lesson_route.zig");

/// One paged sequence, made at load, and the lease of the stream that holds it.
const Lane = struct { seq: *state.Seq, lease: kv.Lease = .{}, used: bool = false };

/// A request's need: its prompt, its reply and one last verify window.
pub fn need(l: kv.Layout, prompt: usize, max_tokens: usize) usize {
    return l.need(prompt, max_tokens, state.max_rows);
}

/// The head's buffers a paged sequence holds itself: its drafts' confidences (its KV rows are a page's last layer).
fn headSizes() [mtp.seq_fields.len]usize {
    return .{ 0, 0, mtp.max_chain * 4 };
}

/// What a stream of this engine holds: its fixed planes and table, its KV a token, a snapshot's slot.
pub fn layout(e: *const Engine) kv.Layout {
    const head = headSizes();
    const fixed = state.Seq.pagedBytes(e.c, if (e.head != null) &head else &.{}, e.max_len / state.chunk_keys);
    return .{ .fixed_bytes = fixed, .token_bytes = e.pageBytes() / state.chunk_keys, .page_tokens = state.chunk_keys, .snap_bytes = reuse.mambaBytes(e.c) };
}

pub const Pages = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    layout: kv.Layout,
    pool: kv.Pool,
    arena: ?cuda.DeviceBuffer,
    lanes: []Lane,
    poison: bool, // TF_KV_POISON=1: a stream's pages start as NaN, so a key read before its write shows in the reply

    /// `streams` paged sequences beside the engine's own and `pages` pages, all made now.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, streams: usize, pages: usize) !*Pages {
        if (state.chunk_keys != 512 or e.k.triton != null) return error.PagesNeedOwnGlue; // the kernels page 512 rows
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .e = e, .layout = layout(e), .pool = try kv.Pool.init(gpa, pages, e.pageBytes()), .arena = null, .lanes = &.{}, .poison = poisoned() };
        errdefer p.pool.deinit();
        if (pages > 0) p.arena = try cuda.DeviceBuffer.alloc(e.ctx.d, pages * e.pageBytes());
        errdefer if (p.arena) |*a| a.free();
        p.lanes = try gpa.alloc(Lane, streams);
        errdefer gpa.free(p.lanes);
        var made: usize = 0;
        errdefer for (p.lanes[0..made]) |l| p.freeSeq(l.seq);
        const head = headSizes();
        for (p.lanes) |*l| {
            const seq = try gpa.create(state.Seq);
            errdefer gpa.destroy(seq);
            seq.* = try state.Seq.initPaged(e.ctx.d, e.c, if (e.head != null) &head else &.{}, e.stream, if (p.arena) |a| a.ptr else 0, e.max_len / state.chunk_keys);
            errdefer seq.deinit();
            const free = @ctz(~e.slots);
            if (free >= lesson_route.max_slots) return error.TooManySequences;
            e.slots |= @as(u64, 1) << @intCast(free);
            seq.slot = @intCast(free);
            l.* = .{ .seq = seq };
            made += 1;
        }
        try e.stream.synchronize();
        return p;
    }

    fn poisoned() bool {
        const v = std.c.getenv("TF_KV_POISON") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    fn freeSeq(p: *Pages, seq: *state.Seq) void {
        p.e.freeSeq(seq); // its graphs, its lesson slot, its arena
    }

    pub fn deinit(p: *Pages) void {
        for (p.lanes) |*l| {
            if (l.used) p.pool.finish(&l.lease);
            p.freeSeq(l.seq);
        }
        p.gpa.free(p.lanes);
        if (p.arena) |*a| a.free();
        p.pool.deinit();
        p.gpa.destroy(p);
    }

    /// Whether a request's pages fit now and a paged sequence is free for it.
    pub fn fits(p: *const Pages, prompt: usize, max_tokens: usize) bool {
        for (p.lanes) |l| if (!l.used) return p.pool.fits(need(p.layout, prompt, max_tokens));
        return false;
    }

    /// A free paged sequence with the pages of `prompt` + `max_tokens` rows and a last window; LaneMemory: none yet.
    pub fn open(p: *Pages, prompt: usize, max_tokens: usize, untaught: bool) !*state.Seq {
        const l = for (p.lanes) |*l| {
            if (!l.used) break l;
        } else return error.LaneMemory;
        const n = need(p.layout, prompt, max_tokens);
        if (n * state.chunk_keys > p.e.max_len) return error.PromptTooLong;
        try p.pool.promise(&l.lease, n);
        errdefer p.pool.finish(&l.lease);
        var up: Upload = .{ .p = p, .seq = l.seq };
        try p.pool.cover(&l.lease, n, .{ .ctx = &up, .write = Upload.write });
        if (p.poison) for (l.lease.ids.items) |id| try p.e.ops().fill32(p.arena.?.ptr + id * p.e.pageBytes(), 0xFFFF_FFFF, p.e.pageBytes() / 4);
        if (l.seq.untaught != untaught) @import("cuda_seq_graphs.zig").free(p.gpa, &l.seq.graphs); // its graphs carry it
        l.seq.pages = l.lease.ids.items;
        l.seq.cap = n * state.chunk_keys;
        l.used = true;
        return l.seq;
    }

    /// The stream on `seq` ended: its promise and pages go back. False: `seq` is not one of these.
    pub fn close(p: *Pages, seq: *state.Seq) bool {
        const l = for (p.lanes) |*l| {
            if (l.seq == seq) break l;
        } else return false;
        if (!l.used) return true;
        p.pool.finish(&l.lease);
        seq.pages = &.{};
        seq.cap = state.max_rows; // no window past its own pages until it opens again
        l.used = false;
        for (p.lanes) |x| if (x.used) return true;
        p.report();
        return true;
    }

    /// The idle server's line: nothing promised, every page free (the copy store holds none), the slots accounted.
    fn report(p: *const Pages) void {
        if (@import("builtin").is_test) return;
        const c = p.pool.census();
        const sn = &p.e.snaps;
        const ok = p.pool.idle() and c.free == c.total;
        std.log.info("kv pool idle{s}: promised {d}, free {d} + store {d} == total {d}; slots free {d} + held {d} == {d}", .{ if (ok) "" else " (UNEVEN)", c.promised, c.free, c.kept, c.total, sn.n, sn.count - sn.n, sn.count });
    }

    /// A stream's page table entries written to its sequence's device table.
    const Upload = struct {
        p: *Pages,
        seq: *state.Seq,

        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const u: *Upload = @ptrCast(@alignCast(ctx));
            try u.p.e.ops().upload(u.seq.table + from * 4, std.mem.sliceAsBytes(ids[from..]));
        }
    };
};
