//! Ling-3.1-flash's TP=1 CUDA engine: a sequence per slot, prompt chunks cut anywhere, verify windows, kept states.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const config = @import("config.zig");
const weights = @import("weights.zig");
const fw = @import("forward.zig");
const head = @import("mtp.zig");

const sampler = glm.sampler;
pub const Sampling = sampler.Sampling;

/// A prompt pass's chunk rows by default (Options.prompt_rows); marks cut chunks anywhere besides.
pub const prompt_rows = 1024;
/// A verify window's most rows: the pending token and up to 15 drafts (forward.window_rows).
pub const window_rows = fw.window_rows;
/// A shared verify's most rows and streams.
pub const shared_rows = 64;
pub const max_shared = 8;
/// The MTP layer's chain at most.
pub const max_drafts = head.max_drafts;

pub const Options = struct {
    capacity: usize, // tokens a sequence holds
    prompt_rows: usize = prompt_rows, // rows a prompt chunk takes at most
    mtp: bool = false, // load the MTP layer, if the checkpoint has one, and draft with it
    layers: ?[]const u32 = null, // a thin model's backbone layers, ascending (checks on a real checkpoint); null: all
};

/// Rank 0's word on its request (cancelled or not), asked before each chunk.
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };
/// One stream's window of a shared verify: its slot, its ids, and its drawn tokens.
pub const Part = struct { slot: u32, ids: []const u32, out: []u32 };
/// A kept prompt state as the prompt cache holds it: its id and position.
pub const Kept = struct { id: u64, at: u32 };
/// Where a prompt pass resumes: kept state `id` at `at`.
pub const Resume = struct { id: u64, at: usize };
/// The pass's marks: it cuts a chunk at each and calls `call` standing there.
pub const Marks = struct { at: []const u32, ctx: *anyopaque, call: *const fn (ctx: *anyopaque, at: u32) void };
/// One chunk's outcome: where the pass stands, and the first token once the prompt is done.
pub const Step = struct { at: usize, first: ?u32 };

/// A stream's sequence: the layers' caches, the MTP layer's, its sampling rule, the rows its next draft reads.
const Seq = struct {
    st: fw.State,
    hd: ?fw.State = null, // the MTP layer's MLA cache at its own position
    hold: ?cuda.DeviceBuffer = null, // the prompt's last final-norm row: the first draft's hidden row
    sampling: ?Sampling = null, // null: greedy
    in_head: usize = 0, // chained draft rows the MTP cache holds past the absorbed ones
    rows_at: u64 = 0, // the last verify's final-norm rows (bf16 [rows][hidden]): the next draft's hidden rows
};

/// A kept state: the sequence's arena copied whole, and the host fields beside it.
const Snap = struct { buf: cuda.DeviceBuffer, hd: ?cuda.DeviceBuffer, pos: usize, hd_pos: usize, cur: []u1 };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    c: config.Config,
    m: fw.Modules,
    stream: cuda.Stream,
    w: weights.Weights, // every loaded layer
    main: weights.Weights, // a view of w: the backbone layers alone
    g: fw.Graph,
    mtp: ?head.Head = null,
    bp: fw.Buffers, // prompt chunks
    bd: fw.Buffers, // one stream's verify window
    bs: fw.Buffers, // shared verify windows
    smp: sampler.Sampler,
    pairs: cuda.DeviceBuffer, // (value, id) a row of greedy draws
    seqs: std.ArrayList(?*Seq) = .empty, // by slot; slot 0 made at init
    cur: *Seq = undefined,
    slot: u32 = 0,
    capacity: usize,
    chunk: usize,
    snaps: std.AutoHashMapUnmanaged(u64, Snap) = .empty,
    next_snap: u64 = 1,

    /// The checkpoint at `dir` in its own format.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, dir: []const u8, o: Options) !*Engine {
        if (o.prompt_rows == 0 or o.capacity < window_rows + 2) return error.BadOptions;
        var why: config.Why = .{};
        var opened = weights.open(gpa, io, dir, &why) catch |err| {
            std.log.err("{s}: {s}", .{ dir, why.text() });
            return err;
        };
        defer opened.ck.close();
        const c = opened.c;
        const mtp = o.mtp and c.mtp > 0;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .d = d, .c = c, .m = undefined, .stream = undefined, .w = undefined, .main = undefined, .g = undefined, .bp = undefined, .bd = undefined, .bs = undefined, .smp = undefined, .pairs = undefined, .capacity = o.capacity, .chunk = o.prompt_rows };
        e.m = try fw.Modules.load(d, ctx);
        errdefer e.m.deinit();
        e.stream = try cuda.Stream.init(d, false);
        errdefer e.stream.deinit();
        const layers = try backbone(gpa, &c, o.layers);
        defer gpa.free(layers);
        {
            const all = try std.mem.concat(gpa, u32, &.{ layers, if (mtp) &.{c.layers} else &.{} });
            defer gpa.free(all);
            var pack = try cuda.modelopt.Packer.load(d);
            defer pack.unload();
            e.w = try weights.load(gpa, d, e.stream, &pack, &opened.ck, c, all, .{});
        }
        errdefer e.w.drop(weights.drains(e.stream)); // leaked on a failed drain: a queued launch may read them
        e.main = e.w;
        e.main.layers = e.w.layers[0..layers.len];
        e.g = try fw.Graph.init(gpa, d, &e.m, e.stream, &e.main);
        errdefer e.g.drop(weights.drains(e.stream)); // as e.w
        if (mtp) {
            e.mtp = @as(head.Head, undefined);
            e.mtp.?.init(gpa, d, &e.m, e.stream, &e.w, o.prompt_rows, o.capacity) catch |err| {
                e.mtp = null;
                return err;
            };
        }
        errdefer if (e.mtp) |*h| h.deinit();
        e.bp = try fw.Buffers.init(d, &e.m, &e.w, o.prompt_rows, true, o.capacity);
        errdefer e.bp.deinit();
        e.bd = try fw.Buffers.init(d, &e.m, &e.w, window_rows, false, o.capacity);
        errdefer e.bd.deinit();
        e.bs = try fw.Buffers.init(d, &e.m, &e.w, shared_rows, false, o.capacity);
        errdefer e.bs.deinit();
        e.smp = try sampler.Sampler.init(d, c.vocab, window_rows);
        errdefer e.smp.deinit();
        e.pairs = try cuda.DeviceBuffer.alloc(d, 8 * window_rows);
        errdefer e.pairs.free();
        errdefer {
            for (0..e.seqs.items.len) |i| e.free(i);
            e.seqs.deinit(gpa);
        }
        try e.bind(0);
        try e.stream.synchronize();
        return e;
    }

    /// The backbone layers to load: `thin` (ascending, each below c.layers), or all of them.
    fn backbone(gpa: std.mem.Allocator, c: *const config.Config, thin: ?[]const u32) ![]u32 {
        if (thin) |t| {
            for (t, 0..) |i, k| if (i >= c.layers or (k > 0 and i <= t[k - 1])) return error.BadLayers;
            return gpa.dupe(u32, t);
        }
        const all = try gpa.alloc(u32, c.layers);
        for (all, 0..) |*l, i| l.* = @intCast(i);
        return all;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        var it = e.snaps.valueIterator();
        while (it.next()) |s| e.freeSnap(s);
        e.snaps.deinit(e.gpa);
        for (0..e.seqs.items.len) |i| e.free(i);
        e.seqs.deinit(e.gpa);
        e.pairs.free();
        e.smp.deinit();
        e.bs.deinit();
        e.bd.deinit();
        e.bp.deinit();
        if (e.mtp) |*h| h.deinit();
        e.g.deinit();
        e.w.deinit();
        e.stream.deinit();
        e.m.deinit();
        const gpa = e.gpa;
        gpa.destroy(e);
    }

    pub fn forward(e: *const Engine, b: *const fw.Buffers) fw.Forward {
        return .{ .m = &e.m, .g = &e.g, .s = e.stream, .b = b };
    }

    /// Binds slot `slot`'s sequence for the steps that follow, making it (zeroed) on first use.
    pub fn bind(e: *Engine, slot: u32) !void {
        if (slot >= e.seqs.items.len) try e.seqs.appendNTimes(e.gpa, null, slot + 1 - e.seqs.items.len);
        if (e.seqs.items[slot] == null) {
            const q = try e.gpa.create(Seq);
            errdefer e.gpa.destroy(q);
            q.* = .{ .st = try fw.State.init(e.gpa, e.d, &e.g, e.capacity) };
            errdefer e.freeSeq(q);
            if (e.mtp) |*h| {
                q.hd = try fw.State.init(e.gpa, e.d, &h.g, e.capacity);
                q.hold = try cuda.DeviceBuffer.alloc(e.d, @as(usize, e.c.hidden) * 2);
            }
            e.seqs.items[slot] = q;
        }
        e.cur = e.seqs.items[slot].?;
        e.slot = slot;
    }

    /// Frees slot `slot`'s sequence (slot 0 stays) and binds slot 0.
    pub fn release(e: *Engine, slot: u32) void {
        if (slot != 0) e.free(slot);
        e.cur = e.seqs.items[0].?;
        e.slot = 0;
    }

    fn free(e: *Engine, slot: usize) void {
        const q = e.seqs.items[slot] orelse return;
        e.stream.synchronize() catch {};
        e.freeSeq(q);
        e.gpa.destroy(q);
        e.seqs.items[slot] = null;
    }

    fn freeSeq(e: *Engine, q: *Seq) void {
        _ = e;
        q.st.deinit();
        if (q.hd) |*h| h.deinit();
        if (q.hold) |*b| b.free();
        q.hd = null;
        q.hold = null;
    }

    /// Bytes one more stream's sequence takes: its arenas.
    pub fn seqBytes(e: *const Engine) u64 {
        const q = e.seqs.items[0].?;
        return q.st.arena.buf.len + if (q.hd) |h| h.arena.buf.len + @as(usize, e.c.hidden) * 2 else 0;
    }

    /// The committed length: the pending token sits here, a window's row r draws position pos + 1 + r.
    pub fn pos(e: *const Engine) usize {
        return e.cur.st.pos;
    }

    /// A fresh sequence with `sampling` (null or temperature 0: greedy).
    pub fn begin(e: *Engine, sampling: ?Sampling) !void {
        e.cur.sampling = sampler.check(sampling);
        e.cur.in_head = 0;
        try e.cur.st.reset(e.stream);
        if (e.cur.hd) |*h| try h.reset(e.stream);
    }

    /// The whole prompt pass: chunks, then the first token from the last row; `from` restores a kept state.
    pub fn prefill(e: *Engine, ids: []const u32, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !u32 {
        var at = try e.prefillStart(ids.len, from);
        while (true) {
            const s = try e.prefillChunk(ids, at, cancel, marks);
            if (s.first) |t| return t;
            at = s.at;
        }
    }

    /// Where a pass over `len` ids starts: 0, or `from`'s position with its kept state restored.
    pub fn prefillStart(e: *Engine, len: usize, from: ?Resume) !usize {
        if (len == 0 or len + window_rows >= e.capacity) return error.PromptTooLong;
        const r = from orelse return 0;
        if (r.at == 0 or r.at >= len) return error.BadResume;
        try e.restoreState(r.id);
        if (e.cur.st.pos != r.at) return error.BadResume;
        return r.at;
    }

    /// One chunk of `ids` from `at` (the sequence's position): up to the chunk rows, cut at the next mark.
    pub fn prefillChunk(e: *Engine, ids: []const u32, at: usize, cancel: ?Cancel, marks: ?Marks) !Step {
        const len = ids.len;
        if (at >= len or at != e.cur.st.pos) return error.BadChunk;
        if (marks) |m| if (at > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(at)) != null) m.call(m.ctx, @intCast(at));
        if (cancel) |c| if (c.check(c.ptr)) return error.Cancelled;
        var end = @min(len, at + e.chunk);
        if (marks) |m| for (m.at) |x| if (x > at and x < end) {
            end = x;
            break;
        };
        const rows = end - at;
        const f = e.forward(&e.bp);
        const seg = [_]fw.Seg{.{ .st = &e.cur.st, .row0 = 0, .rows = rows }};
        try f.embed(@ptrCast(ids[at..end]));
        try f.stack(&seg);
        try f.final(rows);
        var first: ?u32 = null;
        if (end == len) {
            try f.logitsOf(rows - 1, 1);
            var t: [1]u32 = undefined;
            try e.draw(e.bp.logits, 1, len, &t);
            first = t[0];
        }
        if (e.mtp) |*h| { // each row whose next token is known through the MTP layer (it overwrites b.normed)
            const q = e.cur;
            if (end == len) try e.copy(q.hold.?.ptr, e.bp.normed + (rows - 1) * e.c.hidden * 2, e.c.hidden * 2);
            const known = @min(rows, len - 1 - at);
            if (known > 0) try h.absorb(&e.m, e.stream, &e.bp, &q.hd.?, @ptrCast(ids[at + 1 ..][0..known]), e.bp.normed);
        }
        try f.commit(&e.cur.st, rows);
        return .{ .at = end, .first = first };
    }

    /// A verify window: `ids` (pending token, then drafts) at the committed position; `out[r]` drawn at pos + 1 + r.
    pub fn verify(e: *Engine, ids: []const u32, out: []u32) !void {
        if (ids.len == 0 or ids.len > window_rows) return error.WindowTooWide;
        const f = e.forward(&e.bd);
        const at = e.cur.st.pos;
        try f.run(&.{.{ .st = &e.cur.st, .row0 = 0, .rows = ids.len }}, @ptrCast(ids));
        e.cur.rows_at = e.bd.normed;
        try e.draw(e.bd.logits, ids.len, at + 1, out);
    }

    /// Several streams' windows in one forward, each stream's rows on its own sequence.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var segs: [max_shared]fw.Seg = undefined;
        var ids: [shared_rows]u32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > window_rows or rows + p.ids.len > shared_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            try e.bind(p.slot);
            segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        try e.forward(&e.bs).run(segs[0..parts.len], @ptrCast(ids[0..rows]));
        for (parts, segs[0..parts.len]) |p, sg| {
            try e.bind(p.slot);
            e.cur.rows_at = e.bs.normed + sg.row0 * e.c.hidden * 2;
            try e.draw(e.bs.logits + sg.row0 * e.c.vocab * 2, sg.rows, sg.st.pos + 1, p.out);
        }
    }

    /// The last window's first `n` rows committed (the KDA states replayed when fewer than it ran).
    pub fn keep(e: *Engine, n: usize) !void {
        try e.forward(&e.bd).commit(&e.cur.st, n);
    }

    /// The MTP layer absorbs the kept rows with `follow` (the token after each), then chains `depth` drafts.
    pub fn draft(e: *Engine, from_prompt: bool, follow: []const u32, depth: usize, out: []u32) !void {
        const h = if (e.mtp) |*x| x else return error.NoDraftHead;
        if (depth > max_drafts or out.len < depth) return error.TooDeep;
        if (follow.len == 0) return error.NoFollow;
        const q = e.cur;
        const hd = &q.hd.?;
        const D: usize = e.c.hidden;
        try hd.setPos(e.stream, hd.pos - q.in_head); // the last chain's rows go
        q.in_head = 0;
        const hidden = if (from_prompt) q.hold.?.ptr else q.rows_at;
        var s0: usize = 0;
        while (s0 < follow.len) : (s0 += window_rows) {
            const part = follow[s0..@min(follow.len, s0 + window_rows)];
            try h.absorb(&e.m, e.stream, &h.bm, hd, @ptrCast(part), hidden + s0 * D * 2);
        }
        if (hd.pos != q.st.pos) return error.MtpOutOfStep;
        const f = h.forward(&e.m, e.stream, &h.bm);
        var last = (follow.len - 1) % window_rows; // h.bm's row of the newest absorbed token
        for (0..depth) |j| {
            try f.final(last + 1);
            try f.logitsOf(last, 1);
            try e.draw(h.bm.logits, 1, q.st.pos + 1 + j, out[j..][0..1]);
            if (j + 1 == depth) break;
            try h.absorb(&e.m, e.stream, &h.bm, hd, @ptrCast(out[j..][0..1]), h.bm.normed + last * D * 2);
            q.in_head += 1;
            last = 0;
        }
    }

    /// `rows` logits rows -> the tokens at positions first, first + 1, ...
    pub fn draw(e: *Engine, logits: u64, rows: usize, first: u64, out: []u32) !void {
        if (e.cur.sampling) |s| {
            const span = [_]glm.shard.Span{.{ .lo = 0, .hi = e.c.vocab }};
            return e.smp.draw(e.stream, .local, &span, &.{.{ .rank = 0, .logits = logits }}, rows, first, sampler.rule(s), out);
        }
        const V: usize = e.c.vocab;
        for (0..rows) |r| {
            var a: cuda.Args = .{};
            inline for (.{ logits + r * V * 2, @as(i64, @intCast(V)), @as(u32, 0), e.pairs.ptr + r * 8 }) |v| a.add(v);
            try cuda.launch.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
        }
        try e.stream.synchronize();
        var got: [window_rows][2]u32 = undefined;
        try e.pairs.download(0, std.mem.sliceAsBytes(got[0..rows]));
        for (out[0..rows], got[0..rows]) |*o, p| o.* = p[1];
    }

    /// Bytes one kept state takes: the sequence's arenas.
    pub fn stateBytes(e: *const Engine) u64 {
        return e.seqBytes();
    }

    /// A copy of the bound sequence's state under a new id (the pass standing at a mark).
    pub fn keepState(e: *Engine) !u64 {
        const q = e.cur;
        if (q.st.pending != 0) return error.NotCommitted;
        var buf = try cuda.DeviceBuffer.alloc(e.d, q.st.arena.buf.len);
        errdefer if (weights.drains(e.stream)) buf.free(); // else leaked: the copy into it may be queued
        try e.copy(buf.ptr, q.st.arena.buf.ptr, buf.len);
        var hd: ?cuda.DeviceBuffer = null;
        errdefer if (hd) |*x| if (weights.drains(e.stream)) x.free();
        if (q.hd) |h| {
            if (q.in_head != 0) return error.NotCommitted;
            hd = try cuda.DeviceBuffer.alloc(e.d, h.arena.buf.len);
            try e.copy(hd.?.ptr, h.arena.buf.ptr, h.arena.buf.len);
        }
        try e.stream.synchronize();
        const cur = try e.gpa.dupe(u1, q.st.cur);
        errdefer e.gpa.free(cur);
        const id = e.next_snap;
        try e.snaps.put(e.gpa, id, .{ .buf = buf, .hd = hd, .pos = q.st.pos, .hd_pos = if (q.hd) |h| h.pos else 0, .cur = cur });
        e.next_snap += 1;
        return id;
    }

    /// The bound sequence made kept state `id`'s.
    fn restoreState(e: *Engine, id: u64) !void {
        const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
        const q = e.cur;
        try e.copy(q.st.arena.buf.ptr, s.buf.ptr, s.buf.len);
        @memcpy(q.st.cur, s.cur);
        q.st.pending = 0;
        q.st.prompt = false;
        try q.st.setPos(e.stream, s.pos);
        if (q.hd) |*h| {
            try e.copy(h.arena.buf.ptr, (s.hd orelse return error.NoKeptState).ptr, h.arena.buf.len);
            h.pending = 0;
            try h.setPos(e.stream, s.hd_pos);
        }
        q.in_head = 0;
        try e.stream.synchronize();
    }

    pub fn holds(e: *const Engine, id: u64) bool {
        return e.snaps.contains(id);
    }

    pub fn dropState(e: *Engine, id: u64) void {
        var kv = e.snaps.fetchRemove(id) orelse return;
        e.freeSnap(&kv.value);
    }

    fn freeSnap(e: *Engine, s: *Snap) void {
        s.buf.free();
        if (s.hd) |*x| x.free();
        e.gpa.free(s.cur);
    }

    pub fn copy(e: *const Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
    }
};
