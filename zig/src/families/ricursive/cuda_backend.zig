//! Ripple behind lanes.Backend on CUDA: a slot a stream, a chunk a step, a round all windows; cuda.sample draws.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const fwd = @import("cuda_model.zig");
/// The prompt cache's states (cuda_snap.zig), reached through the backend.
pub const snap = @import("cuda_snap.zig");

const be = lanes.backend;
const Stream = lanes.Stream;
const LogRow = lanes.LogprobRow;
const row_words: usize = lanes.logprob.words;

pub const prompt_rows: u32 = fwd.prompt_rows;
/// Rows a round takes over every stream: the model's scratch for rounds is opened at this many.
pub const round_rows: u32 = 128;
/// The widest window a stream verifies (B1: a window's rows are the one-row steps' rows).
pub const max_window: u32 = 16;
/// Positions a slot keeps past a reply for a verify window's rows.
pub const reserve: usize = max_window;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;
/// Rows of logprob words at once: a round's, or a prompt pass's head rows.
const word_rows = @max(round_rows, fwd.head_rows);

/// The positions opened for an asked context: it and the verify reserve, within the maximum, at least a round's rows.
pub fn physical(context: usize) usize {
    return std.math.clamp(context +| reserve, @as(usize, round_rows), cfg.max_context);
}

/// A prompt and its reply fit `positions` with the verify reserve left over.
pub fn fits(positions: usize, prompt: usize, reply: usize) bool {
    return prompt <= positions and reply <= positions - prompt and reserve <= positions - prompt - reply;
}

const Lane = struct {
    slot: usize,
    len: u64 = 0, // the cache length: positions [0, len) hold this stream's keys
    base: u64 = 0, // the cache length the last verify started from
    hi: usize = 0, // positions written (rejected rows too): release zeroes [0, hi)
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    last: u64, // the prompt's last row's bf16 logits [V] (the first draw reads them)
    rule: u64, // sample.cu's Rule, when the stream samples
    sampled: bool = false,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    first: ?LogRow = null, // the first token's logprob row, with `logprobs`
};

pub const Ripple = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *const fwd.Model,
    st: fwd.Seq = .{}, // slot 0's positions when the CLI drives the forward itself (cli/cuda_ricursive.zig)
    tok: cuda.DeviceBuffer, // int32 drawn tokens: the handle ring, then a round's rows
    meta: cuda.DeviceBuffer, // a draw's first position minus one, a window each
    words: cuda.DeviceBuffer, // logprob words, word_rows rows of row_words
    picks: cuda.DeviceBuffer, // a prompt pass's next tokens (head_rows)
    slot_bufs: cuda.DeviceBuffer, // a slot's last-row logits, then its rule, a slot each
    sample: cuda.Module,
    draws: cuda.sample.Draws,
    lanes: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: []bool, // slot i is free
    next: u64 = 0,

    /// A slot's share of slot_bufs: its last row's logits, then its rule.
    fn slotStride(vocab: usize) usize {
        return std.mem.alignForward(usize, vocab * 2, 256) + 256;
    }

    /// The device bytes `init` allocates for `slots` slots (the caches are the model's, fwd.Model.openBytes).
    pub fn openBytes(vocab: usize, slots: usize) usize {
        return (ring + round_rows) * 4 + round_rows * 16 + word_rows * row_words * 4 + fwd.head_rows * 4 + slots * slotStride(vocab);
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *const fwd.Model) !Ripple {
        const slots = m.arena.slots;
        const free = try gpa.alloc(bool, slots);
        errdefer gpa.free(free);
        @memset(free, true);
        var tok = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * 4);
        errdefer tok.free();
        var meta = try cuda.DeviceBuffer.alloc(d, round_rows * 16);
        errdefer meta.free();
        var words = try cuda.DeviceBuffer.alloc(d, word_rows * row_words * 4);
        errdefer words.free();
        var picks = try cuda.DeviceBuffer.alloc(d, fwd.head_rows * 4);
        errdefer picks.free();
        var slot_bufs = try cuda.DeviceBuffer.alloc(d, slots * slotStride(m.c.vocab));
        errdefer slot_bufs.free();
        var sample = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer sample.unload();
        const draws = try cuda.sample.Draws.resolve(sample);
        return .{ .gpa = gpa, .d = d, .m = m, .tok = tok, .meta = meta, .words = words, .picks = picks, .slot_bufs = slot_bufs, .sample = sample, .draws = draws, .free = free };
    }

    pub fn deinit(x: *Ripple) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| l.rows.deinit(x.gpa);
        x.lanes.deinit(x.gpa);
        x.gpa.free(x.free);
        x.sample.unload();
        x.slot_bufs.free();
        x.picks.free();
        x.words.free();
        x.meta.free();
        x.tok.free();
        x.* = undefined;
    }

    pub fn backend(x: *Ripple) be.Backend {
        return .{ .ptr = x, .vtable = &vtable, .chunk_invariant = true }; // M1: chunk cuts give the same bits
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .cancel_prefill = cancelPrefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .first_row = firstRow, .prompt_rows = true, .reports_fill = true };

    /// The round loop's facts: no draft head; host copies verified in windows up to 16 rows, streams sharing a round.
    pub fn facts(x: *const Ripple) lanes.Model {
        const rows: u32 = @intCast(x.m.round_rows);
        return .{ .copy_match = 8, .exact_width = @min(max_window, rows), .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = rows, .max_streams = @intCast(@min(x.m.arena.slots, rows)) };
    }

    fn self(ptr: *anyopaque) *Ripple {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(x: *Ripple, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(x.d, dst, bytes, x.m.s);
    }

    fn tokAt(x: *const Ripple, i: usize) u64 {
        return x.tok.ptr + i * 4;
    }

    /// `rows` bf16 logits rows drawn into `out`, row r at `position0 + r` (argmax when greedy).
    fn drawRows(x: *Ripple, l: *const Lane, logits: u64, rows: usize, position0: u64, out: u64, meta_at: usize) !void {
        const v = x.m.c.vocab;
        if (!l.sampled) return x.m.k.argmaxRows(x.m.s, logits, rows, v, v, out);
        const meta = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        const mp = x.meta.ptr + meta_at * 16;
        try x.upload(mp, std.mem.asBytes(&meta));
        try x.draws.draw(x.m.s, logits, v, l.rule, mp, 0, out, rows, null, null);
    }

    /// The stream's caches at `at` for the prompt cache, copied behind the chunk that ends there.
    pub fn saveState(x: *Ripple, owner: ?*anyopaque, at: u32) !*snap.Saved {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if ((l.fill orelse l.len) != at) return error.NotAtMark; // a state is copied only where the pass stands
        return snap.save(x.gpa, x.m, l.slot, at);
    }

    /// The stream's slot made to hold `k`'s state (a restore outside its prompt pass).
    pub fn restoreState(x: *Ripple, owner: ?*anyopaque, k: *const snap.Saved) !void {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        try snap.restore(x.m, l.slot, k);
        l.len = k.at;
        l.hi = @max(l.hi, k.at);
    }

    pub fn dropState(x: *Ripple, k: *snap.Saved) void {
        snap.drop(x.gpa, x.m.s, k);
    }

    fn freeSlot(x: *const Ripple) ?usize {
        return std.mem.indexOfScalar(bool, x.free, true);
    }

    /// Whether a request could start now: a free slot; one that never fits passes (its prefill refuses it by name).
    pub fn admits(x: *const Ripple, prompt: usize, max_new: usize) bool {
        if (!fits(x.m.context, prompt, max_new)) return true;
        return x.freeSlot() != null;
    }

    /// A prompt pass begins: a slot, the stream's rule, a kept state restored; where its first chunk starts.
    fn opening(x: *Ripple, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (!fits(x.m.context, p.len, @as(usize, s.max_new))) return error.PastContext;
        if (s.grammar != null) return error.StructuresUnsupported;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (!got.found_existing) {
            const slot = x.freeSlot() orelse {
                x.lanes.removeByPtr(got.key_ptr);
                return error.NoFreeSlot;
            };
            x.free[slot] = false;
            const at = x.slot_bufs.ptr + slot * slotStride(x.m.c.vocab);
            got.value_ptr.* = .{ .slot = slot, .last = at, .rule = at + slotStride(x.m.c.vocab) - 256 };
        }
        const l = got.value_ptr;
        l.sampled = false;
        l.first = null;
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const r = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
            try x.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        };
        s.cached = 0;
        s.prompt_rows.clearRetainingCapacity();
        // a pass with prompt logprobs starts at 0: every prompt row is scored
        l.fill = if (s.prompt_logprobs != null) 0 else x.resumeAt(s, l, p.len);
        s.fill_at.store(@intCast(l.fill.?), .monotonic);
        return l;
    }

    /// The kept state's position, restored into the slot (0: none, not this prompt's, or another shape's).
    fn resumeAt(x: *Ripple, s: *Stream, l: *Lane, len: usize) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const snap.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or !snap.sameKey(k.key, snap.keyOf(x.m.c))) {
            s.reuse_failed = true;
            return 0;
        }
        snap.restore(x.m, l.slot, k) catch {
            s.reuse_failed = true;
            return 0;
        };
        l.hi = @max(l.hi, k.at);
        s.cached = k.at;
        return k.at;
    }

    /// The chunk's rows [at, end) that predict a prompt token, `head_rows` at a time: words and rank to prompt rows.
    fn promptRows(x: *Ripple, s: *Stream, at: usize, end: usize, k: u8) !void {
        const p = s.prompt();
        const v = x.m.c.vocab;
        const scored = @min(end, p.len - 1) -| at;
        var r0: usize = 0;
        while (r0 < scored) : (r0 += fwd.head_rows) {
            const n = @min(fwd.head_rows, scored - r0);
            const next = p[at + r0 + 1 ..][0..n];
            try x.m.headRows(r0, n);
            try x.upload(x.picks.ptr, std.mem.sliceAsBytes(next));
            try x.draws.logprobRows(x.m.s, x.m.b.rows, v, x.picks.ptr, k, true, x.words.ptr, n);
            var words: [fwd.head_rows * row_words]u32 = undefined;
            try x.m.download(x.words.ptr, std.mem.sliceAsBytes(words[0 .. n * row_words]));
            for (next, 0..) |t, i| try s.prompt_rows.append(x.gpa, LogRow.fromWords(words[i * row_words ..][0..row_words], t, k));
        }
    }

    /// One chunk of `l.fill`, at most `max_rows` (0: full), cut at the next mark; true once the last prompt row is in.
    fn chunk(x: *Ripple, s: *Stream, l: *Lane, max_rows: u32) !bool {
        const p = s.prompt();
        const at = l.fill orelse return true;
        if (s.isCancelled()) {
            s.keepInterruptedPrefill(@intCast(at));
            return error.Cancelled;
        }
        const rows: usize = if (max_rows == 0) prompt_rows else @min(max_rows, prompt_rows);
        var end = @min(p.len, at + rows);
        for (s.reuse.marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        const done = end == p.len;
        try x.m.prompt(l.slot, at, p[at..end]);
        if (s.prompt_logprobs) |k| try x.promptRows(s, at, end, k);
        if (done) {
            try x.m.head(end - at - 1);
            try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(l.last, x.m.b.logits, x.m.c.vocab * 2, x.m.s.handle), "cuMemcpyDtoDAsync");
        }
        s.fill_at.store(@intCast(end), .monotonic);
        l.hi = @max(l.hi, end);
        l.fill = if (done) null else end;
        if (done) l.len = p.len;
        if (s.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) hk.at(hk.ptr, s, @intCast(end));
        return done;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        const l = try x.opening(s);
        errdefer x.dropLane(s);
        while (!try x.chunk(s, l, 0)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream, max_rows: u32) anyerror!bool {
        const x = self(ptr);
        const going = if (x.lanes.getPtr(s)) |l| l.fill != null else false;
        const l = if (going) x.lanes.getPtr(s).? else try x.opening(s);
        errdefer x.dropLane(s);
        return x.chunk(s, l, max_rows);
    }

    fn cancelPrefill(ptr: *anyopaque, s: *Stream) !void {
        const l = self(ptr).lanes.getPtr(s) orelse return;
        if (l.fill) |at| s.keepInterruptedPrefill(@intCast(at));
    }

    fn laneOf(x: *Ripple, s: *const Stream) !*Lane {
        return x.lanes.getPtr(s) orelse error.UnknownStream;
    }

    fn handle(x: *Ripple) u64 {
        const h = x.next;
        x.next += 1;
        return h;
    }

    /// The prompt's last row drawn at `position`; with logprobs, its row kept for first_row.
    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = try x.laneOf(s);
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const h = x.handle();
        try x.drawRows(l, l.last, 1, position, x.tokAt(h % ring), 0);
        if (s.logprobs) |k| {
            try x.draws.logprobRows(x.m.s, l.last, x.m.c.vocab, x.tokAt(h % ring), k, false, x.words.ptr, 1);
            var w: [row_words]u32 = undefined;
            try x.m.download(x.words.ptr, std.mem.sliceAsBytes(&w));
            l.first = LogRow.fromWords(&w, try read(x, h), k);
        }
        return h;
    }

    fn firstRow(ptr: *anyopaque, s: *Stream) anyerror!LogRow {
        const l = try self(ptr).laneOf(s);
        return l.first orelse error.NoFirstRow;
    }

    fn value(x: *Ripple, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| read(x, h),
        };
    }

    /// One token at the cache's end: a round of one row, its row through the head, drawn at `position`.
    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = try x.laneOf(s);
        const t = try x.value(feed);
        if (l.fill != null or position != l.len + 1) return error.PositionMismatch;
        try x.m.round(&.{.{ .slot = l.slot, .p0 = l.len, .tokens = 1 }}, &.{t});
        try x.m.rowLogits(1);
        l.len += 1;
        l.hi = @max(l.hi, l.len);
        const h = x.handle();
        try x.drawRows(l, x.m.b.rows, 1, position, x.tokAt(h % ring), 0);
        return h;
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try x.m.download(x.tokAt(h % ring), std.mem.sliceAsBytes(&out));
        return out[0];
    }

    /// Every window's rows in one round (a chain a stream, from its cache length), each window drawn at its positions.
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        const v = x.m.c.vocab;
        const chains = try x.gpa.alloc(fwd.Chain, windows.len);
        defer x.gpa.free(chains);
        var ids: [round_rows]u32 = undefined;
        var total: usize = 0;
        for (windows, chains) |w, *c| {
            if (w.held != 0) return error.NoHeldDrafts; // no draft head: every draft comes from the host
            if (w.parents != null) return error.ChainsOnly;
            if (w.masks.len != 0) return error.StructuresUnsupported;
            const l = try x.laneOf(w.stream);
            if (w.rows() > max_window) return error.WindowTooWide;
            if (total + w.rows() > @min(round_rows, x.m.round_rows)) return error.RoundTooWide;
            if (l.fill != null) return error.PositionMismatch;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(x.gpa, w.pending);
            try l.rows.appendSlice(x.gpa, w.tokens);
            for (w.positions, 0..) |pos, r| if (pos != l.base + r + 1) return error.PositionMismatch;
            c.* = .{ .slot = l.slot, .p0 = l.base, .tokens = l.rows.items.len };
            @memcpy(ids[total..][0..l.rows.items.len], l.rows.items);
            total += l.rows.items.len;
        }
        try x.m.round(chains, ids[0..total]);
        try x.m.rowLogits(total);
        var at: usize = 0;
        var scored = false;
        for (windows, out, 0..) |w, o, i| {
            const l = x.lanes.getPtr(w.stream).?;
            const n = l.rows.items.len;
            const logits = x.m.b.rows + at * v * 2;
            try x.drawRows(l, logits, n, w.positions[0], x.tokAt(ring + at), i);
            if (o.rows.len > 0) {
                try x.draws.logprobRows(x.m.s, logits, v, x.tokAt(ring + at), w.stream.logprobs.?, false, x.words.ptr + at * row_words * 4, n);
                scored = true;
            }
            at += n;
        }
        var got: [round_rows]u32 = undefined;
        try x.m.download(x.tokAt(ring), std.mem.sliceAsBytes(got[0..total]));
        var words: [round_rows * row_words]u32 = undefined;
        if (scored) try x.m.download(x.words.ptr, std.mem.sliceAsBytes(words[0 .. total * row_words]));
        at = 0;
        for (windows, out) |w, o| {
            const l = x.lanes.getPtr(w.stream).?;
            const n = l.rows.items.len;
            @memcpy(o.sampled[0..n], got[at..][0..n]);
            @memcpy(o.drafts[0 .. n - 1], l.rows.items[1..]);
            for (o.rows, o.sampled[0..o.rows.len], at..) |*r, pick, row| r.* = LogRow.fromWords(words[row * row_words ..][0..row_words], pick, w.stream.logprobs.?);
            l.len = l.base + n; // a chain keeps every row unless rolled back
            l.hi = @max(l.hi, l.len);
            at += n;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            const l = try x.laneOf(w.stream);
            if (path.len == 0 or path.len > l.rows.items.len) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.ChainsOnly; // a chain keeps a prefix of its rows
            l.len = l.base + path.len; // caches are positional: the next round overwrites what the rejected rows wrote
        }
    }

    /// No draft head: absorbing is a no-op, and a request for drafts is refused by name.
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    /// The stream's slot zeroed where it wrote (a fresh arena's bytes for the next stream) and freed.
    fn dropLane(x: *Ripple, s: *const Stream) void {
        if (x.lanes.fetchRemove(s)) |entry| {
            var l = entry.value;
            x.m.reset(l.slot, l.hi) catch |e| std.log.warn("ripple: slot {d} not zeroed on release ({s})", .{ l.slot, @errorName(e) });
            x.free[l.slot] = true;
            l.rows.deinit(x.gpa);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }
};

test "ricursive cuda backend: the verify reserve past a reply; positions opened for a context" {
    try std.testing.expect(fits(32784, 30000, 2768));
    try std.testing.expect(fits(32784, 32784 - reserve - 1, 1));
    try std.testing.expect(!fits(32784, 32784 - reserve, 1));
    try std.testing.expect(!fits(32784, 32785, 0));
    try std.testing.expectEqual(@as(usize, 32768 + reserve), physical(32768));
    try std.testing.expect(fits(physical(32768), 32000, 768)); // the asked context holds a prompt and its reply
    try std.testing.expectEqual(@as(usize, 262144), physical(262144)); // the model's maximum caps it
    try std.testing.expectEqual(@as(usize, round_rows), physical(1));
}

test "ricursive cuda backend: windows of 16 rows, a stream a slot, rows a round" {
    var m: fwd.Model = undefined;
    m.round_rows = round_rows;
    m.arena.slots = 6;
    const x: Ripple = .{ .gpa = std.testing.allocator, .d = undefined, .m = &m, .tok = undefined, .meta = undefined, .words = undefined, .picks = undefined, .slot_bufs = undefined, .sample = undefined, .draws = undefined, .free = &.{} };
    const f = x.facts();
    try std.testing.expectEqual(@as(u32, 16), f.exact_width);
    try std.testing.expectEqual(@as(u32, 6), f.max_streams);
    try std.testing.expectEqual(@as(u32, 128), f.batch_rows);
    try std.testing.expect(f.hidden_rows and f.copy_drafts and !f.mtp and !f.gpu_tokens);
    try std.testing.expect(Ripple.slotStride(100352) >= 100352 * 2 + @sizeOf(cuda.sample.Rule));
    std.testing.refAllDecls(Ripple);
}

test {
    _ = snap;
}
