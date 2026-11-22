//! TP over processes until #552: rank 0's Leader sends each step (header, ids, slot) first; other ranks `follow` it.

const std = @import("std");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const engine = @import("cuda_engine.zig");
const prompt_solo = @import("cuda_prompt_solo.zig");
const prompt_cuts = @import("cuda_prompt_cuts.zig");
const images_mod = @import("cuda_images.zig");
const vcfg = @import("vision_config.zig");
const sampler = @import("cuda_sampler.zig");
const prompt_wire = @import("cuda_prompt_wire.zig");
const dflash = @import("dflash.zig");
const pages_mod = @import("cuda_pages.zig");
const sparse = @import("cuda_sparse.zig");
const prompt_pair = @import("cuda_prompt_pair.zig");
const Engine = engine.Engine;

pub const Kind = enum(u32) { begin = 1, prefill, verify, keep, draft, end, drop, release, chunk, shared, image, block, many, redraw, idle, snapshots, pool, form, cuts };

/// One step as it crosses: 64 bytes, then `n` u32 ids.
pub const Header = extern struct {
    kind: u32,
    n: u32 = 0, // ids that follow
    depth: u32 = 0, // draft: drafts to chain; chunk: where it starts; shared: its parts
    flags: u32 = 0, // begin: sampled; draft: from the prompt (1), chances (2); prefill, chunk: prompt_logprobs (1); chunk: indexer rows split (2); verify, shared: whole rows; snapshots: pages shared
    a: u64 = 0, // begin: the prompt cache's pages first in its table; keep: rows; prefill, chunk: the kept state it resumes (0: none); redraw: the prompt's length; form: a bit a window's rows; verify/draft/block with TF_GLM_KEEP_FOLD: k+1, or 0 for no keep
    seed: u64 = 0, // begin: the sampling seed; prefill, chunk: the kept state's position
    temperature: f64 = 0,
    top_p: f64 = 0,
    min_p: f64 = 0,
    top_k: u32 = 0, // begin: top_k; chunk: the prompt's length
    slot: u32 = 0, // the stream's sequence the step runs on (Engine.bind)

    comptime {
        std.debug.assert(@sizeOf(Header) == 64);
    }
};

/// tf_glm_put_words' argument: a step's header (16 words) and up to `put_ids` ids.
pub const Words = extern struct { w: [96]u32 };
/// Ids the first broadcast of a step carries beside its header: a window's, or a block's reqs, in one message.
const inline_ids = 48;
const first_bytes = 64 + inline_ids * 4;
pub const put_ids = 96 - 16;

/// The most ids one step carries (a prompt at the engine's capacity).
pub const Link = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    buf: cuda.DeviceBuffer, // header, then ids
    ids: []u32, // host copy of the last step's ids
    gpa: std.mem.Allocator,
    wait: bool, // TF_GLM_LINK_WAIT=1: every step uploaded and waited for, as before the put
    asleep: bool = false, // rank 0: the other ranks sleep on their lines (cuda.tp_watch) until its next step
    fold_keep: bool = false, // TF_GLM_KEEP_FOLD=1: fold the next keep into the following verify/draft/block header
    commit_keep: ?usize = null, // rank 0: a keep already executed locally but not yet sent to the other ranks
    commit_slot: u32 = 0, // rank 0: the slot on which commit_keep ran

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, asked_ids: usize, fold_keep: bool) !Link {
        const max_ids = @max(asked_ids, prompt_wire.words);
        const buf = try cuda.DeviceBuffer.alloc(d, 64 + max_ids * 4);
        errdefer {
            var owned = buf;
            owned.free();
        }
        const wait = try @import("sys").environment.read(bool, "TF_GLM_LINK_WAIT", false);
        return .{ .d = d, .lib = lib, .comm = comm, .stream = stream, .buf = buf, .ids = try gpa.alloc(u32, max_ids), .gpa = gpa, .wait = wait, .fold_keep = fold_keep };
    }

    pub fn deinit(l: *Link) void {
        l.gpa.free(l.ids);
        l.buf.free();
    }

    fn bcast(l: *Link, bytes: usize) !void {
        try l.post(bytes);
        try l.stream.synchronize();
    }

    fn post(l: *Link, bytes: usize) !void {
        try l.lib.check(l.lib.api.ncclBroadcast(l.buf.ptr, l.buf.ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast");
    }

    /// Rank 0: the step and ids to every rank; up to `put_ids` ids go by `put` unwaited, longer ones upload and wait.
    pub fn send(l: *Link, put: cuda.Function, h: Header, ids: []const u32) !void {
        if (ids.len > l.ids.len) return error.TooManyIds;
        var hh = h;
        hh.n = @intCast(ids.len);
        if (!l.wait and ids.len <= put_ids) {
            var w: Words = undefined;
            @memcpy(w.w[0..16], std.mem.bytesAsSlice(u32, std.mem.asBytes(&hh)));
            @memcpy(w.w[16..][0..ids.len], ids);
            var a: cuda.Args = .{};
            a.add(l.buf.ptr);
            a.add(@as(u32, @intCast(16 + ids.len)));
            a.add(w);
            try cuda_pdl.launch(put, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, l.stream, &a);
            try l.post(first_bytes);
            if (ids.len > inline_ids) try l.post(64 + ids.len * 4);
            return;
        }
        try l.buf.uploadAsync(0, std.mem.asBytes(&hh), l.stream.handle);
        if (ids.len > 0 and ids.len <= inline_ids) try l.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), l.stream.handle);
        try l.bcast(first_bytes);
        if (ids.len > inline_ids) {
            try l.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), l.stream.handle);
            try l.bcast(64 + ids.len * 4);
        }
    }

    /// `bytes` at `ptr` from rank 0 into every rank's `ptr` (image rows: the follower's buffer is the same size).
    pub fn rows(l: *Link, ptr: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try l.lib.check(l.lib.api.ncclBroadcast(ptr, ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast");
    }

    /// Other ranks: the next step and its ids (valid until the next receive).
    pub fn receive(l: *Link) !struct { h: Header, ids: []const u32 } {
        try l.bcast(first_bytes);
        var first: [first_bytes]u8 align(8) = undefined;
        try l.buf.download(0, &first);
        const h = std.mem.bytesAsValue(Header, first[0..@sizeOf(Header)]).*;
        if (h.n > l.ids.len) return error.TooManyIds;
        if (h.n > inline_ids) {
            try l.bcast(64 + h.n * 4);
            try l.buf.download(64, std.mem.sliceAsBytes(l.ids[0..h.n]));
        } else if (h.n > 0) {
            @memcpy(l.ids[0..h.n], std.mem.bytesAsSlice(u32, first[64..][0 .. h.n * 4]));
        }
        return .{ .h = h, .ids = l.ids[0..h.n] };
    }
};

pub fn keepHeader(n: usize) Header {
    return .{ .kind = @backingInt(Kind.keep), .a = n };
}

fn foldsKeep(kind: Kind) bool {
    return switch (kind) {
        .verify, .draft, .block => true,
        else => false,
    };
}

/// A foldable step's deferred keep, with zero encoding the historical no-keep form.
fn foldedKeep(h: Header) ?usize {
    if (h.a == 0 or !foldsKeep(@fromBackingInt(@intCast(h.kind)))) return null;
    return @intCast(h.a - 1);
}

pub const Outgoing = struct {
    before: ?Header = null, // a non-foldable command needs the historical standalone keep first
    before_slot: u32 = 0, // the slot the standalone keep ran on (it may precede a bind)
    h: Header,
    folded: bool = false,
};

pub fn outgoing(l: *const Link, h: Header, slot: u32) Outgoing {
    const n = l.commit_keep orelse return .{ .h = h };
    const kind: Kind = @fromBackingInt(@intCast(h.kind));
    if (!foldsKeep(kind) or l.commit_slot != slot) return .{ .before = keepHeader(n), .before_slot = l.commit_slot, .h = h };
    var folded = h;
    folded.a = n + 1;
    return .{ .h = folded, .folded = true };
}

/// The follower side of verify/draft/block/keep. A folded keep runs before its step, exactly as a standalone one did.
pub fn foldableStep(e: anytype, h: Header, ids: []const u32, sampled: *[engine.shared_rows]u32, drafts: *[engine.max_drafts]u32, chances: *[engine.max_drafts]f64) !bool {
    const kind: Kind = @fromBackingInt(@intCast(h.kind));
    if (!foldsKeep(kind) and kind != .keep) return false;
    if (foldedKeep(h)) |n| try e.keep(n);
    switch (kind) {
        .verify => try e.verify(ids, sampled[0..ids.len], 0),
        .draft => {
            const owned = try e.gpa.dupe(u32, ids);
            defer e.gpa.free(owned);
            try e.draft(h.flags & 1 != 0, owned, h.depth, drafts, if (h.flags & 2 != 0) chances else null);
        },
        .block => {
            const k = h.depth;
            if (k == 0 or k > dflash.MAX_S or ids.len != 3 * k) return error.BadStep;
            var reqs: [dflash.MAX_S]dflash.Req = undefined;
            for (reqs[0..k], 0..) |*q, i| q.* = .{ .slot = ids[3 * i], .pos = 0, .n = ids[3 * i + 1], .anchor = ids[3 * i + 2], .propose = @import("dflash_proposals.zig").enabled(h.flags, i) };
            try e.draftBlock(reqs[0..k]);
        },
        .keep => try e.keep(@intCast(h.a)),
        else => unreachable,
    }
    return true;
}

/// A prompt step's outcome (Leader.chunk): the first token once it ends the prompt, and the rows it took.
pub const Chunk = struct { first: ?u32, rows: usize };

/// Rank 0's engine: every step goes to the other ranks first.
pub const Leader = struct {
    e: *Engine,
    link: *Link,

    /// The step on the bound stream's sequence (its slot) to every rank.
    fn send(x: Leader, h: Header, ids: []const u32) !void {
        const out = outgoing(x.link, h, x.e.slot);
        if (out.before) |first| {
            try x.transmitAt(first, &.{}, out.before_slot);
            x.link.commit_keep = null;
        }
        try x.transmit(out.h, ids);
        if (out.folded) x.link.commit_keep = null;
    }

    fn transmit(x: Leader, h: Header, ids: []const u32) !void {
        return x.transmitAt(h, ids, x.e.slot);
    }

    fn transmitAt(x: Leader, h: Header, ids: []const u32, slot: u32) !void {
        var hh = h;
        hh.slot = slot;
        if (x.link.asleep) { // the other ranks wait on their lines, not in NCCL: wake them for this step
            cuda.tp_watch.wake();
            x.link.asleep = false;
        }
        try x.link.send(x.e.m.glue.put_words, hh, ids);
    }

    /// A prompt step's flags: 1 when its rows go through the head (the other ranks join each pass's broadcasts).
    fn rows(x: Leader) u32 {
        return @intFromBool(x.e.prompt_logprobs != null);
    }

    /// No stream is left, or rank 0's tower is about to run: the other ranks sleep on their lines until the next step.
    pub fn idle(x: Leader) !void {
        if (x.link.asleep or !cuda.tp_watch.open()) return;
        try x.send(.{ .kind = @backingInt(Kind.idle) }, &.{});
        x.link.asleep = true;
    }

    /// The stream's sequence for the steps that follow (each step names it, so the other ranks bind it too).
    pub fn bind(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
    }

    /// Frees a stream's sequence on every rank.
    pub fn release(x: Leader, slot: u32) !void {
        try x.flush();
        try x.e.bind(slot);
        try x.send(.{ .kind = @backingInt(Kind.release) }, &.{});
        x.e.release(slot);
    }

    /// States rank 0 dropped since the last step, so the other ranks free theirs (ids that fit one step at a time).
    fn flush(x: Leader) !void {
        const d = &x.e.dropped;
        while (d.items.len > 0) {
            const n = @min(d.items.len, x.link.ids.len);
            try x.send(.{ .kind = @backingInt(Kind.drop) }, d.items[0..n]);
            d.replaceRangeAssumeCapacity(0, n, &.{});
        }
    }

    fn flushKeep(x: Leader) !void {
        const n = x.link.commit_keep orelse return;
        try x.transmitAt(keepHeader(n), &.{}, x.link.commit_slot);
        x.link.commit_keep = null;
    }
    pub fn begin(x: Leader, s: ?sampler.Sampling) !void {
        return x.beginOn(s, &.{}, 0);
    }

    /// `begin` on a paged sequence: its page ids (rank 0's lease, the cache's `kept` first) go to every rank's table.
    pub fn beginOn(x: Leader, s: ?sampler.Sampling, pages: []const u32, kept: usize) !void {
        try x.flush();
        const c = sampler.check(s);
        var h: Header = .{ .kind = @backingInt(Kind.begin), .a = kept };
        if (c) |v| {
            h.flags = 1;
            h.seed = v.seed;
            h.temperature = v.temperature;
            h.top_p = v.top_p;
            h.min_p = v.min_p;
            h.top_k = v.top_k;
        }
        try x.send(h, pages);
        try x.e.begin(c);
    }

    /// Every rank's prompt cache at rank 0's budget (snapshots.agree), its states past the pool's pages as rank 0's.
    pub fn snapshots(x: Leader, budget: u64) !void {
        try x.send(.{ .kind = @backingInt(Kind.snapshots), .a = budget, .flags = @intFromBool(x.e.share_pages) }, &.{});
        try @import("cuda_snapshots.zig").agree(x.e, budget);
    }

    /// The other ranks make rank 0's pool: `streams` paged sequences and `pages` pages, each page's bytes checked.
    pub fn pool(x: Leader, streams: u32, pages: usize, page_bytes: u64) !void {
        try x.send(.{ .kind = @backingInt(Kind.pool), .depth = streams, .a = pages, .seed = page_bytes }, &.{});
    }

    /// Rank 0's tower encodes the bound slot's images past `from` with the other ranks asleep (#3614), then their rows.
    pub fn encode(x: Leader, v: *images_mod.Images, list: []const @import("lanes").stream.Image, from: usize) !void {
        try x.idle(); // a net plugin may time out an NCCL wait as long as the pass: they wait on their lines instead
        try v.begin(x.e.slot, list, from); // a refusal leaves every rank as it was, asleep until rank 0's next step
        try x.images(v);
    }

    /// The prompt pass's image rows (rank 0 encoded them for the bound slot) to every rank: spans, then the rows.
    pub fn images(x: Leader, v: *const images_mod.Images) !void {
        try x.flush();
        const words = try spanWords(v.gpa, v.spans.items, x.link.ids.len);
        defer v.gpa.free(words);
        try x.send(.{ .kind = @backingInt(Kind.image) }, words);
        try x.link.rows(v.rows.ptr, v.rowCount() * vcfg.out_hidden * 2);
    }

    /// `cancel` (rank 0's), asked before each chunk, decides for every rank; `from` every rank holds; `marks` rank 0's.
    pub fn prefill(x: Leader, ids: []const u32, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks) !u32 {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.prefill), .flags = x.rows(), .a = if (from) |r| r.id else 0, .seed = if (from) |r| r.at else 0 }, ids);
        return x.e.prefill(ids, from, cancel, marks);
    }

    /// The bound sequence's request chunk starts (TF_GLM_REQUEST_STARTS) to every rank, before its first prompt step.
    pub fn cuts(x: Leader, starts: []const u32) !void {
        try prompt_cuts.set(x.e.gpa, &x.e.cur.cuts, starts); // a refusal here leaves every rank as it was
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.cuts) }, starts);
    }

    /// One prompt step on every rank (Engine.prefillChunk): to the next cut, or two if `alone` and solo or pair allow.
    pub fn chunk(x: Leader, ids: []const u32, start: usize, alone: bool, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks) !Chunk {
        if (start >= ids.len) return error.BadChunk;
        try x.flush();
        const wide = alone and (prompt_solo.at(x.e, start, ids.len, from, marks) or prompt_pair.pairedAt(x.e, start, ids.len, from, marks));
        const n = prompt_cuts.leaderRows(ids.len, x.e.cur.cuts.items, start, wide);
        const part = ids[start..@min(ids.len, start + n + 1)];
        const split = sparse.splitAt(start, x.e.settings.index_split, x.e.share.world); // a paired step's chunks share it
        try x.send(.{ .kind = @backingInt(Kind.chunk), .flags = x.rows() | @as(u32, @intFromBool(split)) << 1, .a = if (from) |r| r.id else 0, .seed = if (from) |r| r.at else 0, .depth = @intCast(start), .top_k = @intCast(ids.len) }, part);
        x.e.index_split = split;
        const first = try x.e.prefillChunk(part, start, ids.len, from, cancel, marks);
        return .{ .first = first, .rows = @min(ids.len - start, n) };
    }

    /// `bits`: the rows' grammar words (structured output, with Engine.structured), else 0.
    pub fn verify(x: Leader, ids: []const u32, out: []u32, bits: u64) !void {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.verify), .flags = @intFromBool(x.e.structured) }, ids);
        try x.e.verify(ids, out, bits);
    }

    /// Fold the next keep into a compatible step only when TF_GLM_KEEP_FOLD=1.
    pub fn keep(x: Leader, n: usize) !void {
        try x.flushKeep();
        try x.flush();
        if (x.link.fold_keep) {
            try x.e.keep(n);
            x.link.commit_keep = n;
            x.link.commit_slot = x.e.slot;
        } else {
            try x.send(keepHeader(n), &.{});
            try x.e.keep(n);
        }
    }

    /// Several streams' windows in one forward on every rank (Engine.verifyShared): parts' slots and rows, then ids.
    pub fn verifyShared(x: Leader, parts: []const engine.Part) !void {
        if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
        try x.flush();
        var buf: [2 * engine.max_shared + engine.shared_rows]u32 = undefined;
        var n: usize = 2 * parts.len;
        for (parts, 0..) |p, i| {
            if (n + p.ids.len > buf.len) return error.WindowTooWide;
            buf[2 * i] = p.slot;
            buf[2 * i + 1] = @intCast(p.ids.len);
            @memcpy(buf[n..][0..p.ids.len], p.ids);
            n += p.ids.len;
        }
        try x.send(.{ .kind = @backingInt(Kind.shared), .flags = @intFromBool(x.e.structured), .depth = @intCast(parts.len) }, buf[0..n]);
        try x.e.verifyShared(parts);
    }

    pub fn prefillMany(x: Leader, parts: []const engine.Prompt, firsts: []u32) !void {
        var buf: [prompt_wire.words]u32 = undefined;
        const ids = try prompt_wire.encode(parts, x.e.capacity, &buf);
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.many), .depth = @intCast(parts.len) }, ids);
        try x.e.prefillMany(parts, firsts);
    }

    /// Structured output's first token: every rank gathers the prompt's last row again, rank 0 draws it among `bits`.
    pub fn redraw(x: Leader, len: usize, bits: u64) !u32 {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.redraw), .a = len }, &.{});
        return x.e.redrawFirst(len, bits);
    }

    /// With `chances` every rank picks with its softmax sum, so rank 0 reads each draft's chance (--mtp-confidence).
    pub fn draft(x: Leader, from_prompt: bool, follow_ids: []const u32, depth: usize, out: []u32, chances: ?[]f64) !void {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.draft), .depth = @intCast(depth), .flags = @intFromBool(from_prompt) | @as(u32, @intFromBool(chances != null)) << 1 }, follow_ids);
        try x.e.draft(from_prompt, follow_ids, depth, out, chances);
    }

    /// DFlash2's round on every rank (Engine.draftBlock): each stream's slot, kept rows and anchor.
    pub fn block(x: Leader, reqs: []dflash.Req) !void {
        if (reqs.len == 0 or reqs.len > dflash.MAX_S) return error.TooManyStreams;
        try x.flush();
        var buf: [3 * dflash.MAX_S]u32 = undefined;
        for (reqs, 0..) |q, i| buf[3 * i ..][0..3].* = .{ q.slot, @intCast(q.n), q.anchor };
        try x.send(.{ .kind = @backingInt(Kind.block), .depth = @intCast(reqs.len), .flags = @import("dflash_proposals.zig").mask(reqs) }, buf[0 .. 3 * reqs.len]);
        try x.e.draftBlock(reqs);
    }

    /// Every rank's expert forms, bit r - 1 for a window of r rows (the same bits; load timing picks them by width).
    pub fn expertForm(x: Leader, forms: u64) !void {
        try x.send(.{ .kind = @backingInt(Kind.form), .a = forms }, &.{});
        x.e.m.experts.fact = forms;
    }

    /// The other ranks leave their loops.
    pub fn end(x: Leader) !void {
        try x.send(.{ .kind = @backingInt(Kind.end) }, &.{});
    }
};

/// A prompt step whose rows go through the head (Leader.rows): this rank's columns join each pass's broadcasts.
pub fn promptRows(h: Header) bool {
    const kind: Kind = @fromBackingInt(@intCast(h.kind));
    return (kind == .prefill or kind == .chunk) and h.flags & 1 != 0;
}

/// A prompt chunk whose indexer rows split across the ranks (Leader.chunk): this rank picks for its block of rows.
pub fn indexSplit(h: Header) bool {
    const kind: Kind = @fromBackingInt(@intCast(h.kind));
    return kind == .chunk and h.flags & 2 != 0;
}

/// A verify step under a grammar (Engine.structured): this rank's columns join the gather of each draw's whole rows.
pub fn structuredStep(h: Header) bool {
    const kind: Kind = @fromBackingInt(@intCast(h.kind));
    return (kind == .verify or kind == .shared) and h.flags & 1 != 0;
}

/// A folded block binds its keep's slot; an ordinary block binds each request itself.
pub fn bindStep(e: anytype, h: Header) !void {
    const kind: Kind = @fromBackingInt(@intCast(h.kind));
    const unbound = switch (kind) {
        .end, .drop, .shared, .many, .idle, .snapshots, .pool, .form => true,
        .block => h.a == 0,
        else => false,
    };
    if (!unbound) try e.bind(h.slot);
}

/// A rank other than 0 runs rank 0's steps until it ends; a cancelled prefill is error.Cancelled and the loop goes on.
pub fn follow(e: *Engine, link: *Link) !void {
    var drafts: [engine.max_drafts]u32 = undefined;
    var chances: [engine.max_drafts]f64 = undefined;
    var sampled: [engine.shared_rows]u32 = undefined;
    cuda.tp_watch.sleep(e.io); // rank 0 starts with no stream: its first step wakes this rank
    while (true) {
        const step = try link.receive();
        const h = step.h;
        const kind: Kind = @fromBackingInt(@intCast(h.kind));
        try bindStep(e, h);
        e.prompt_logprobs = if (promptRows(h)) .{} else null;
        e.structured = structuredStep(h); // rank 0 alone masks them
        if (try foldableStep(e, h, step.ids, &sampled, &drafts, &chances)) continue;
        switch (kind) {
            .verify, .keep, .draft, .block => unreachable, // foldableStep handled these above
            .begin => {
                if (step.ids.len > 0) try (e.pages orelse return error.BadStep).map(h.slot, step.ids, std.math.cast(usize, h.a) orelse return error.BadStep);
                try e.beginFollowing(if (h.flags & 1 != 0) .{ .seed = h.seed, .temperature = h.temperature, .top_p = h.top_p, .min_p = h.min_p, .top_k = h.top_k } else null);
            },
            .pool => {
                if (e.pages != null) return error.BadStep;
                e.pages = try pages_mod.Pages.follow(e.gpa, e, h.depth, @intCast(h.a), h.seed);
            },
            .cuts => try prompt_cuts.set(e.gpa, &e.cur.cuts, step.ids), // the bound sequence's, before its first prompt step
            .prefill => {
                const ids = try e.gpa.dupe(u32, step.ids);
                defer e.gpa.free(ids);
                const from: ?engine.Resume = if (h.a != 0) .{ .id = h.a, .at = h.seed } else null;
                _ = e.prefill(ids, from, null, null) catch |err| switch (err) {
                    error.Cancelled => {},
                    else => return err,
                };
            },
            .chunk => {
                const ids = try e.gpa.dupe(u32, step.ids);
                defer e.gpa.free(ids);
                const from: ?engine.Resume = if (h.a != 0) .{ .id = h.a, .at = h.seed } else null;
                e.index_split = indexSplit(h);
                _ = e.prefillChunk(ids, h.depth, h.top_k, from, null, null) catch |err| switch (err) {
                    error.Cancelled => {},
                    else => return err,
                };
            },
            .redraw => _ = try e.redrawFirst(@intCast(h.a), 0),
            .shared => {
                const k = h.depth;
                if (k == 0 or k > engine.max_shared or step.ids.len < 2 * k) return error.BadStep;
                var parts: [engine.max_shared]engine.Part = undefined;
                var outs: [engine.shared_rows]u32 = undefined;
                var at: usize = 2 * k;
                var row: usize = 0;
                for (parts[0..k], 0..) |*p, i| {
                    const n = step.ids[2 * i + 1];
                    if (at + n > step.ids.len or row + n > outs.len) return error.BadStep;
                    p.* = .{ .slot = step.ids[2 * i], .ids = step.ids[at..][0..n], .out = outs[row..][0..n] };
                    at += n;
                    row += n;
                }
                try e.verifyShared(parts[0..k]);
            },
            .many => {
                var parts: [engine.max_together]engine.Prompt = undefined;
                const decoded = try prompt_wire.decode(step.ids, h.depth, e.capacity, &parts);
                var firsts: [engine.max_together]u32 = undefined;
                try e.prefillMany(decoded, &firsts);
            },
            .form => e.m.experts.fact = h.a,
            .image => {
                const v = e.images orelse return error.NoVision; // the ranks agreed on --vision (cuda_ranks)
                try v.take(h.slot, step.ids);
                try link.rows(v.rows.ptr, v.rowCount() * vcfg.out_hidden * 2);
            },
            .snapshots => {
                e.share_pages = h.flags & 1 != 0; // TF_KV_SHARE on rank 0: a kept state leaves the shared store's pages out
                try @import("cuda_snapshots.zig").agree(e, h.a);
            },
            .drop => for (step.ids) |id| e.dropState(id),
            .release => e.release(h.slot),
            .idle => cuda.tp_watch.sleep(e.io),
            .end => return cuda.tp_watch.ended(), // rank 0 ends on purpose: its line closing from here is expected
        }
    }
}

// The production image-step serializer, isolated so the host can test the offered video limits.
pub fn spanWords(a: std.mem.Allocator, spans: []const images_mod.Span, max_words: usize) ![]u32 {
    const n = std.math.mul(usize, spans.len, 3) catch return error.TooManyIds;
    if (n > max_words) return error.TooManyIds;
    const words = try a.alloc(u32, n);
    for (spans, 0..) |sp, i| words[3 * i ..][0..3].* = .{ sp.at, sp.n, sp.row };
    return words;
}

test {
    _ = @import("cuda_follow_test.zig");
}
