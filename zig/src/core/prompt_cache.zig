//! Exact prompt reuse: states are keyed by tokens and canonical decoded intervals through their position.
const std = @import("std");
const Allocator = std.mem.Allocator;
const imprint = @import("prompt_imprint.zig");
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");
const learned = @import("prompt_learn.zig");
const lifetime = @import("prompt_cache_lifetime.zig");
const policy = @import("prompt_cache_policy.zig");
const cuts = @import("prompt_cuts.zig");
const interior = @import("prompt_interior.zig");
const index = @import("prompt_cache_index.zig");
const forks = @import("prompt_cache_fork.zig");
pub const rows = @import("rows_store.zig");
/// A family's copy of one state.
pub const Saved = *anyopaque;
/// A prefill chunk's kernel class by rows: the decode path (<= 8), the shared body (9..63), the tensor tile (>= 64).
pub fn chunkClass(n: usize) u8 {
    return if (n <= 8) 0 else if (n < 64) 1 else 2;
}

/// The rows a fresh pass's last chunk holds at `len` for a zero-anchored grid `grid`: `len % grid`, or one whole
/// chunk when it divides. ROUND-502: the class of this is what a resumed suffix's last chunk must keep.
pub fn freshLast(grid: u32, len: usize) usize {
    const g: usize = @max(1, grid);
    return if (len % g != 0) len % g else @min(len, g);
}
pub const singleReclaim = imprint.singleReclaim; // the learned-state file helpers, beside the Imprint that owns them
pub const singleForgetChecked = imprint.singleForgetChecked;
/// What a family gives the cache: copies of its state while the prompt pass stands at a chunk end.
pub const Snapshots = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        /// The most new storage `save` takes at `at` tokens (checked against the budget before any copy).
        bytes: *const fn (ptr: *anyopaque, at: u32) u64,
        /// Copy the live state after `at` prompt tokens into new storage; `owner` is the lane core's stream (null: one stream).
        save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved,
        /// Make the live state `saved`'s: the next prompt chunk starts at its position.
        restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: Saved) anyerror!void,
        drop: *const fn (ptr: *anyopaque, saved: Saved) void,
        /// The storage a kept state holds: new storage up to `bytes`, or the spare storage it took (null: `bytes`).
        charged: ?*const fn (ptr: *anyopaque, saved: Saved) u64 = null,
        /// Free storage the family keeps for later saves, inside the budget (null: none).
        spare: ?*const fn (ptr: *anyopaque) u64 = null,
        /// Free spare storage until at most `room` bytes stay spare.
        trim: ?*const fn (ptr: *anyopaque, room: u64) void = null,
        /// Whether a save at `at` would take spare storage (no new storage).
        reuses: ?*const fn (ptr: *anyopaque, at: u32) bool = null,
        /// Write `saved` as learned state `key`, in files of the family's own under `dir` (null: nothing kept on disk).
        write: ?*const fn (ptr: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64) anyerror!void = null,
        /// Learned state `key` of `at` tokens, read back from its files under `dir` into new storage.
        read: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!Saved = null,
        /// Peer disk demand and reservation lifetime; null functions mean a single local rank.
        reclaim: ?*const fn (*anyopaque, [:0]const u8, u64) anyerror!u64 = null,
        peer_other_used: ?*const fn (*anyopaque, u64) anyerror!u64 = null,
        peer_need: ?*const fn (ptr: *anyopaque, saved: Saved, extra: u64) anyerror!u64 = null,
        peer_reclaim: ?*const fn (ptr: *anyopaque, key: u64) anyerror!u64 = null,
        peer_reserve: ?*const fn (ptr: *anyopaque, saved: Saved, key: u64, extra: u64) anyerror!void = null,
        peer_finish: ?*const fn (ptr: *anyopaque, key: u64, success: bool) anyerror!void = null,
        peer_other_next: ?*const fn (*anyopaque, ?u64) anyerror!?u64 = null,
        peer_other_bytes: ?*const fn (*anyopaque, u64) anyerror!u64 = null,
        peer_other_remove: ?*const fn (*anyopaque, u64) anyerror!void = null,
        forget_checked: ?*const fn (*anyopaque, [:0]const u8, u64) anyerror!void = null,
        /// Learned state `key`'s files under `dir` removed (the cap needs room, or they no longer read back).
        forget: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64) void = null,
        /// --learn-conversations: the state's image as rows and parts, and copies out of and into kept states.
        rows: ?*const rows.Image = null,
    };
};
/// What a paged store asks of the backend's pages: who holds each, and how many are free.
pub const Pages = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    /// Bytes of one page, over every layer.
    bytes: u64,
    pub const VTable = struct {
        retain: *const fn (ptr: *anyopaque, id: u32) void,
        release: *const fn (ptr: *anyopaque, id: u32) void,
        /// How many hold the page: the store counts as one.
        holders: *const fn (ptr: *anyopaque, id: u32) u32,
        /// Pages free and not promised to a stream.
        available: *const fn (ptr: *anyopaque) usize,
    };
};
/// What a family's states depend on beyond their position.
pub const Rules = @import("prompt_rules.zig").Rules;

pub const Entry = struct {
    image_frame: u32 = 0, // the family frame can affect text rows before the first image
    images: []images.Identity = &.{}, // exact image identities through the keyed prefix, including lookahead
    tokens: []u32, // the state's tokens and `lookahead` more
    decode_spans: []modes.Span = &.{}, // canonical row arithmetic through at, excluding lookahead
    at: u32,
    saved: Saved,
    bytes: u64,
    born: u32, // the prompt length of the request that kept it (a reply's end and its warm pass: that reply's): a later turn's is longer
    used: u64, // the store's clock at its last keep or resume
    last: []u32, // the last prompt that kept or resumed it
    shared: bool = false, // kept at a shared system block's cut: other conversations resume it, so its own never supersedes it
    branched: bool = false, // resumed again after a later prompt kept a longer state extending it: never superseded
    expired: bool = false, // hidden from lookups immediately, reclaimed only at safe host boundaries
    borrowers: u32 = 0, // outstanding lookups: released by resumed or release, never evicted while borrowed
    ext: std.ArrayList(*Entry) = .empty, // entries extending its tokens, born later (policy.link): `extended` reads them
    own: []u32 = &.{}, // the kept prompt's one copy, which `tokens` and at first `last` slice (empty: they own theirs)
};
pub const Counts = struct { hits: u64 = 0, misses: u64 = 0, kept: u64 = 0, evicted: u64 = 0, refused: u64 = 0, failed: u64 = 0 };
pub const RestoreDetails = @import("prompt_cache_restore.zig").Details;
pub const RestoreLog = @import("prompt_cache_restore.zig").Log;
/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };
/// The borrowed entry a backend restores; the caller must report resumed or release it, even on cancellation.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{}, snap_at: u32 = 0 };
pub const Store = struct {
    image_frame: u32 = 0,
    image_query: ?[]const @import("lanes").stream.Image = null, // scoped to one serial lookup or keep
    restore_log: ?RestoreLog = null,
    gpa: Allocator,
    family: Snapshots,
    rules: Rules,
    budget: u64,
    most: usize = std.math.maxInt(usize), // states kept at most: a family's snapshot slots, one a state
    held: u64 = 0,
    clock: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,
    by_position: std.ArrayList(*Entry) = .empty, // nonowning, longest first; equal positions retain insertion order
    counts: Counts = .{},
    shared_keys: std.ArrayList(u64) = .empty, // the tokens of recently planned shared cuts, hashed (at most SHARED_KEYS)
    imprint: ?*imprint.Imprint = null, // --learn: shared cuts' states on disk, read back by later sessions and servers
    disk: ?*rows.Rows = null, // --learn-conversations: every kept state written behind to disk, read back where it pays
    active_rewind_at: u32 = 0, // the last lookup's rewind state (0: none); hosts keep a pass's states before the next lookup
    links_lost: bool = false, // an allocation failure left the extension links incomplete: `extended` scans
    hold: bool = false, // memory pressure: lookups plan no marks and keeps refuse, so no new state is kept
    repeat_log: @import("prompt_cache_repeat.zig").Log = .{},
    const SHARED_KEYS = 64;
    pub const lookupUserImages = images.lookupUser;
    pub const keepBornImages = images.keepBorn;
    pub const lookupUserFrame = images.lookupUserFrame;
    pub const keepBornFrame = images.keepBornFrame;

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        return .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
    }

    pub fn deinit(s: *Store) void {
        if (s.disk) |r| rows.write.stop(r, rows.write.nowNs());
        for (s.entries.items) |e| s.free(e);
        s.entries.deinit(s.gpa);
        s.by_position.deinit(s.gpa);
        s.shared_keys.deinit(s.gpa);
        s.repeat_log.deinit(s.gpa);
        s.held = 0;
    }

    fn free(s: *Store, e: *Entry) void {
        if (s.disk) |r| rows.write.dropped(r, e);
        s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(if (e.own.len > 0) e.own else e.tokens);
        s.gpa.free(e.decode_spans);
        s.gpa.free(e.images);
        if (e.own.len == 0 or e.last.ptr != e.own.ptr) s.gpa.free(e.last);
        e.ext.deinit(s.gpa);
        s.gpa.destroy(e);
    }

    pub fn remove(s: *Store, i: usize) void {
        std.debug.assert(s.entries.items[i].borrowers == 0);
        index.remove(s, s.entries.items[i]);
        const e = s.entries.orderedRemove(i);
        policy.unlink(s, e);
        s.held -= e.bytes;
        if (s.disk) |r| if (!e.expired) rows.write.evicted(r, e); // --spill-gib: stage an owned copy before the entry goes
        s.free(e);
    }

    /// The family's free storage kept for later saves.
    pub fn spare(s: *const Store) u64 {
        return if (s.family.vtable.spare) |f| f(s.family.ptr) else 0;
    }

    /// A kept state takes `extra` more bytes (a family copied what it shared into it) if the budget has room; never evicts, a pass is under way.
    pub fn grow(s: *Store, saved: Saved, extra: u64) bool {
        for (s.entries.items) |e| if (!e.expired and e.tokens.len > 0 and e.saved == saved) {
            if (extra > s.room()) return false;
            e.bytes += extra;
            s.held += extra;
            return true;
        };
        return false;
    }

    /// Whether an outstanding lookup still borrows `saved`'s state: the family spares it while restoring.
    pub fn borrowed(s: *Store, saved: Saved) bool {
        for (s.entries.items) |e| if (e.saved == saved and !e.expired and e.borrowers > 0) return true;
        return false;
    }

    pub const expired = lifetime.expired;
    pub const collectExpired = lifetime.collect;
    pub const release = lifetime.release;

    /// What the budget leaves past the kept states and the spare storage: a family readies storage only inside it.
    pub fn room(s: *const Store) u64 {
        return s.budget -| s.held -| s.spare();
    }

    /// Kept states free their storage, least valuable first and never `restoring`, until `left.fits()` (false: never).
    pub fn giveBack(s: *Store, restoring: ?Saved, prompt: []const u32, left: anytype) bool {
        s.collectExpired();
        while (!left.fits()) {
            const skip = for (s.entries.items) |e| {
                if (restoring != null and e.saved == restoring.?) break e;
            } else null;
            s.remove(s.victimBut(skip, prompt, @intCast(prompt.len)) orelse return false);
            s.counts.evicted += 1;
        }
        return true;
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or a chunk start for planned families.
    /// ROUND-472: a `tail` family also keeps one off-grid state before a prompt's end, so a later turn resumes
    /// at its own boundary instead of at the last grid chunk.
    pub fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or (s.rules.tail_align > 1 and (s.rules.tail or s.rules.end_mark)) or s.startAt(starts, at) == at);
    }

    /// Whether `at` is a chunk start of this pass (every position for a family without a planned grid).
    pub fn aligned(s: *const Store, at: u32, starts: []const u32) bool {
        return !s.rules.planned or s.startAt(starts, at) == at;
    }

    /// The chunk start at or before `at`: the request's, else the family's grid (0 when none).
    fn startAt(s: *const Store, starts: []const u32, at: u32) u32 {
        if (starts.len == 0 and s.rules.grid > 0) return at - at % s.rules.grid;
        return cuts.floorStart(starts, at);
    }

    /// The latest kept state this prompt can resume at `at` or under min_gap before it (null: none).
    fn keptNear(s: *const Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, at: u32) ?u32 {
        var best: ?u32 = null;
        for (s.entries.items) |e| {
            if (e.expired or e.at > at or at - e.at >= s.rules.min_gap or e.tokens.len > prompt.len or !s.usable(e.at, starts)) continue;
            if (!std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens) or !modes.equal(e.decode_spans, spans, e.at) or !images.matches(s, e)) continue;
            if (best == null or e.at > best.?) best = e.at;
        }
        return best;
    }

    /// Whether the state another pass keeps at `at` of `kept` (`kept_spans`) would serve `prompt` as `find` does.
    pub fn wouldServe(s: *const Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, kept: []const u32, kept_spans: []const modes.Span, at: u32) bool {
        const n = @as(usize, at) + s.rules.lookahead;
        if (at >= prompt.len or n > prompt.len or n > kept.len or !s.usable(at, starts)) return false;
        return std.mem.eql(u32, prompt[0..n], kept[0..n]) and modes.equal(kept_spans, spans, at);
    }

    /// The longest kept token prefix with matching canonical spans through its position, leaving at least one row.
    pub const find = index.find;

    /// Restore the longest prefix matching tokens and canonical spans, then plan prompt-cache marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        return s.beginRewind(a, prompt, history_len, 0, shared, starts, owner, spans);
    }

    /// `begin`, also keeping a state before the latest user message (`rewind_len`, 0: none) that its edits resume.
    pub fn beginRewind(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, rewind_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        return s.beginUser(a, prompt, history_len, rewind_len, 0, shared, starts, owner, spans);
    }

    /// Begin with a known end of the latest user message, for its optional interior state (0: unknown).
    pub fn beginUser(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, rewind_len: u32, user_end: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        var l = try s.lookupUser(a, prompt, @intCast(prompt.len), history_len, rewind_len, user_end, shared, starts, spans);
        // the first lookup's marks, kept for the pass from the start if no shorter kept state resumes
        var fresh_marks: ?[]const u32 = null;
        while (l.entry) |e| {
            if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| {
                const at = e.at; // read before the reserve may evict the entry
                s.resumedWithReason(e, prompt, null);
                if (fresh_marks) |m| if (m.ptr != l.marks.ptr) a.free(m);
                s.reserve(prompt, @intCast(prompt.len), null, l.marks, spans); // restored: the state it resumed may go too (a peer restores before it drops)
                return .{ .from = at, .marks = l.marks };
            } else |err| {
                // a refused state is dropped, then the next shorter RAM state is tried before the pass from the start
                if (fresh_marks == null) fresh_marks = l.marks else a.free(l.marks);
                s.resumedWithReason(e, prompt, err);
                l = try s.lookupUserAt(a, prompt, @intCast(prompt.len), history_len, rewind_len, user_end, shared, starts, spans, false);
            }
        }
        const marks_ = fresh_marks orelse l.marks;
        if (fresh_marks != null and l.marks.ptr != marks_.ptr) a.free(l.marks);
        s.reserve(prompt, @intCast(prompt.len), null, marks_, spans);
        return .{ .marks = marks_ };
    }

    /// Look up canonical-span state and marks without restoring; the caller later reports resumed.
    pub fn lookup(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        return s.lookupRewind(a, prompt, @intCast(prompt.len), history_len, 0, shared, starts, spans);
    }

    /// `lookup` plus a rewind state on the starts or grid (or one under min_gap before it); `born` as in keepBorn.
    pub fn lookupRewind(s: *Store, a: Allocator, prompt: []const u32, born: u32, history_len: u32, rewind_len: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        return s.lookupUser(a, prompt, born, history_len, rewind_len, 0, shared, starts, spans);
    }

    /// `lookupRewind` with an explicit last-user end: history may also include later tool or assistant messages.
    pub fn lookupUser(s: *Store, a: Allocator, prompt: []const u32, born: u32, history_len: u32, rewind_len: u32, user_end: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        return s.lookupUserAt(a, prompt, born, history_len, rewind_len, user_end, shared, starts, spans, true);
    }

    /// `lookupUser`; with `disk` false only RAM states (a retry after a refusal must not recall the refused one).
    fn lookupUserAt(s: *Store, a: Allocator, prompt: []const u32, born: u32, history_len: u32, rewind_len: u32, user_end: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span, disk: bool) !Lookup {
        s.collectExpired();
        const endpoint = if (s.rules.planned) s.startAt(starts, history_len) else history_len;
        const rewind = if (s.rules.planned) s.startAt(starts, rewind_len -| s.rules.lookahead) else rewind_len -| s.rules.lookahead;
        const endpoint_bytes = if (endpoint > 0 and (endpoint < prompt.len or (s.rules.warm != .off and endpoint == prompt.len)) and endpoint + s.rules.lookahead <= prompt.len) s.family.vtable.bytes(s.family.ptr, endpoint) else 0;
        const rewind_bytes = if (rewind > 0 and rewind < prompt.len and rewind + s.rules.lookahead <= prompt.len and s.usable(rewind, starts)) s.family.vtable.bytes(s.family.ptr, rewind) else 0;
        const priority_bytes = if (endpoint_bytes <= s.budget) endpoint_bytes else 0;
        s.active_rewind_at = if (prompt.len >= s.rules.min_prompt and rewind_bytes > 0 and rewind != endpoint and rewind_bytes <= s.budget - priority_bytes) rewind else 0;
        if (s.active_rewind_at > 0) if (s.keptNear(prompt, starts, spans, s.active_rewind_at)) |at| {
            s.active_rewind_at = at; // a state kept under min_gap before it is the rewind state: no new copy
        };
        const e = if (disk)
            rows.read.recall(s, prompt, born, starts, spans, s.recall(prompt, born, starts, spans, s.find(prompt, starts, spans)))
        else
            s.find(prompt, starts, spans);
        if (e == null) s.counts.misses += 1;
        const from = if (e) |x| x.at else 0;
        var tail_at: u32 = 0;
        const ordinary = if (s.active_rewind_at == 0)
            try s.fitting(a, try s.marksRewind(a, prompt, from, history_len, 0, shared, starts, spans, if (e) |x| x.last else &.{}, &tail_at))
        else
            try s.fittingRewind(a, try s.fitting(a, try s.marksRewind(a, prompt, from, history_len, s.active_rewind_at, shared, starts, spans, if (e) |x| x.last else &.{}, &tail_at)), endpoint, s.active_rewind_at, endpoint_bytes);
        const known_end = if (user_end <= history_len) user_end else 0;
        const marks_ = interior.add(s, a, prompt, from, known_end, rewind_len, starts, spans, ordinary, endpoint, endpoint_bytes, e) catch |err| {
            a.free(ordinary);
            return err;
        };
        for (shared) |w| { // the shared cuts this pass keeps, and a short prompt's mark inside a shared block: other conversations resume them
            const at = if (s.rules.planned) s.startAt(starts, w) else w;
            for (marks_) |m| if (m == at or (m < at and !s.rules.planned and prompt.len < s.rules.min_prompt)) s.noteShared(prompt[0 .. m + s.rules.lookahead]);
        }
        s.reserve(prompt, born, e, marks_, spans);
        if (e) |entry| entry.borrowers += 1;
        // ROUND-504: the tail mark this lookup planned, where the backend may snapshot it inside the chunk that
        // runs past it. Only a tensor-tile tail (>= 64 rows) and only with a bulk piece behind it: the merged
        // chunk then keeps the kernel class the cut's own tail chunk would have run.
        const snap_at: u32 = if (s.rules.tail_snap and tail_at > from and tail_at < prompt.len and
            prompt.len - tail_at >= 64 and std.mem.indexOfScalar(u32, marks_, tail_at) != null) tail_at else 0;
        return .{ .entry = e, .marks = marks_, .snap_at = snap_at };
    }

    /// Room for every state the pass keeps, made before it: a peer that mirrors the keeps hears of these evictions with the request.
    fn reserve(s: *Store, prompt: []const u32, born: u32, from: ?*Entry, marks_: []const u32, spans: []const modes.Span) void {
        var need: u64 = 0;
        var new: usize = 0; // states it keeps that the store lacks: a slot each
        for (marks_) |at| {
            const n = @as(usize, at) + s.rules.lookahead;
            const exists = for (s.entries.items) |e| {
                if (!e.expired and e.at == at and e.tokens.len == n and n <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at) and images.matches(s, e)) break true;
            } else false;
            if (!exists) need += s.family.vtable.bytes(s.family.ptr, at);
            new += @intFromBool(!exists);
        }
        while (s.held + need > s.budget or s.entries.items.len + @min(new, s.most) > s.most) {
            const i = s.victimBut(from, prompt, born) orelse return;
            s.remove(i);
            s.counts.evicted += 1;
        }
    }

    /// Every kept state goes after a weight change, and learned states on disk stop being read: old arithmetic.
    pub fn clear(s: *Store) void {
        if (s.disk) |r| rows.write.stop(r, rows.write.nowNs());
        for (s.entries.items) |e| e.expired = true;
        s.collectExpired();
        s.shared_keys.clearRetainingCapacity();
        s.imprint = null;
        s.disk = null;
        s.active_rewind_at = 0;
    }

    /// Forget a failed peer state matching the token prefix and canonical spans through at.
    pub fn forget(s: *Store, prompt: []const u32, at: u32, spans: []const modes.Span) void {
        for (s.entries.items) |e| if (!e.expired and e.at == at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]) and modes.equal(e.decode_spans, spans, at) and images.matches(s, e)) {
            s.counts.failed += 1;
            e.expired = true;
            return s.collectExpired();
        };
    }

    fn sharedKey(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
    }

    fn noteShared(s: *Store, tokens: []const u32) void {
        const k = sharedKey(tokens);
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, k) != null) return;
        if (s.shared_keys.items.len == SHARED_KEYS) _ = s.shared_keys.orderedRemove(0);
        s.shared_keys.append(s.gpa, k) catch {};
    }

    const fitting = @import("prompt_cache_fit.zig").fitting;
    const fittingRewind = @import("prompt_cache_fit.zig").fittingRewind;

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (dropped: the pass ran from 0).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        return s.resumedWithReason(e, prompt, if (ok) null else error.RestoreRejected);
    }

    pub fn resumedWithReason(s: *Store, e: *Entry, prompt: []const u32, failure: ?anyerror) void {
        if (std.mem.indexOfScalar(*Entry, s.entries.items, e) == null) return;
        defer s.release(e);
        if (failure) |err| {
            @import("prompt_cache_restore.zig").report(s.restore_log, .{ .kept = e.at, .prompt = prompt.len, .expired = e.expired, .err = err });
            s.counts.failed += 1;
            e.expired = true;
            return;
        }
        s.clock += 1;
        e.used = s.clock;
        e.branched = e.branched or s.extended(e); // a turn several continuations share (a re-sent, edited or forked one)
        s.counts.hits += 1;
        const last = s.gpa.dupe(u32, prompt) catch return;
        if (e.own.len == 0 or e.last.ptr != e.own.ptr) s.gpa.free(e.last); // the first `last` is the tokens' copy
        e.last = last;
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        var tail_at: u32 = 0;
        return s.marksRewind(a, prompt, from, history_len, 0, shared, starts, &.{}, previous, &tail_at);
    }

    /// `marks` with the active rewind second: kept however near the other marks, never at or before the resume point.
    /// `tail_at_out` is the tail mark it planned (0: none), which a backend may snapshot inside a chunk (tail_snap).
    fn marksRewind(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, rewind: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span, previous: []const u32, tail_at_out: *u32) ![]const u32 {
        tail_at_out.* = 0;
        const short = !s.rules.planned and prompt.len < s.rules.min_prompt; // the history's mark only, where the pass cuts a chunk anyway
        if (short and s.rules.chunk == 0) return &.{};
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        // a warm pass (its history the whole prompt) keeps where a state can be keyed, its lookahead before the end
        const warm_pass = s.rules.warm != .off and history_len == prompt.len and s.rules.lookahead > 0;
        const repeat_at = s.repeat_log.mark(s.rules.repeat_min, prompt, from, s.rules.lookahead, s.rules.planned);
        const history: u32 = if (warm_pass) @intCast(prompt.len -| s.rules.lookahead) else if (history_len == 0) repeat_at else history_len;
        try want.append(a, if (short) cuts.chunkEnd(s.rules.chunk, from, history, prompt.len, starts, spans) else history);
        // ROUND-472: ONE off-grid state per conversation, at the pass's last whole tile before the end, so the
        // next turn resumes at its own boundary and prefills only what it added. Tile-aligned (a resumed suffix
        // must sit in the lanes a fresh pass gives it) and used only while its suffix is one chunk.
        // ROUND-475: IT ADVANCES ON A RESUMED PASS. With the mark planned only on a pass from 0, a growing
        // conversation's suffix grew by every turn's new rows until it passed the grid, and the store fell back to
        // the oldest grid state (a 21k chat at 30k tokens re-prefilled 8k rows, and past 37.8k the 15,872 one).
        // The mark is planned whenever the bulk piece from `from` is a tensor-tile chunk (>= 64 rows), so the
        // pass splits where its own grid split would put a chunk end anyway and the next turn resumes near its end.
        // ROUND-502: WITH THE END STATE KEPT, A RESUMED PASS NEEDS NO TAIL CUT. The cut exists so a later turn
        // can start on a tile mark; the end state is the exact state a fresh pass stands on (receipt: on/off/
        // fresh equal at 22,705) and a lookup may also fall back to an OLDER end state whose suffix keeps the
        // class. What the cut costs is the second chunk itself: its all-expert read is ~0.5 s, whatever its row
        // count (the 25-33 % block-filter round and the 1.86-2.64 s cached turn are both that term). So the
        // tail is planned on a pass from 0, where no end state exists yet -- one extra chunk per
        // conversation, not per turn. ROUND-506: ON A RESUMED PASS IT IS PLANNED AS A SNAPSHOT, not a cut:
        // the R-504 stash captures it inside the chunk that runs past it, so the pass keeps both the tile
        // state and its own end in one forward. It is planned only when its suffix is a tensor tile (>= 64
        // rows), where `snap_at` can fire; a shorter suffix would make the mark a barrier again.
        const resumed_tail = if (std.c.getenv("TF_DSV41_RESUMED_TAIL")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        const tail_at: u32 = if (s.rules.tail and s.rules.tail_align > 1 and !short and prompt.len > s.rules.min_prompt and
            (!s.rules.end_mark or from == 0 or (s.rules.tail_snap and resumed_tail)))
        blk: {
            const tile: usize = @max(1, s.rules.tail_align);
            // ROUND-475: THE CUT'S LAST CHUNK KEEPS THE REFERENCE'S KERNEL CLASS. A pass with no marks ends on
            // `prompt.len % grid` rows (the grid's whole chunks, then the remainder); the cut ends on
            // `prompt.len % tile`. A chunk's class changes at 8 rows (decode) and 64 (shared body vs the tensor
            // tile), so a remainder in the other class runs different arithmetic and can move the reply --
            // ROUND-474's four-row failure (P_B = 184x128 + 4, the reference's last chunk 7,684 rows) is this
            // rule, and so is its >= 9-row fix. Back the cut off whole tiles until the class matches.
            const fresh_last: usize = if (s.rules.grid > 0 and prompt.len % s.rules.grid != 0)
                prompt.len % s.rules.grid
            else
                @min(prompt.len, @as(usize, @max(1, s.rules.grid)));
            const class_want = chunkClass(fresh_last);
            var f: usize = prompt.len % tile;
            if (f == 0) f = tile;
            var guard: u32 = 0;
            while (chunkClass(f) != class_want and guard < 4) : (guard += 1) f += tile;
            if (chunkClass(f) != class_want or f > prompt.len) break :blk 0;
            const at: u32 = @intCast(prompt.len - f);
            // A resumed pass advances only when its bulk piece is a tensor-tile chunk (the reference's own
            // intermediate chunks are) and the new state is past the point this pass resumed at.
            if (from > 0 and (at <= from or at - from < 64)) break :blk 0;
            if (from > 0 and prompt.len - at < 64) break :blk 0; // a resumed tail must be snapshottable, never a cut
            break :blk at;
        } else 0;
        tail_at_out.* = tail_at;
        if (tail_at != 0 and std.mem.indexOfScalar(u32, want.items, tail_at) == null) try want.append(a, tail_at);
        // ROUND-502: THE PASS'S OWN END IS A STATE TOO, CLASS-CHECKED. The tile tail serves a turn that adds only
        // a few rows; a turn that adds a tensor-tile chunk does not need the 128 rows behind it re-prefilled (one
        // whole chunk's all-expert read, the cache turn's floor). A state at the very end is the state a fresh
        // pass stands on when this pass's last chunk kept the fresh remainder's class: rows are boundary- and
        // lane-invariant inside a class (ROUND-472 form 4: 176 rows against the reference's 6,192, exact).
        // `find` re-checks the class against the next turn's own remainder, so a small turn still takes the tail.
        const end_at: u32 = if (s.rules.end_mark and !short and prompt.len > s.rules.min_prompt) @intCast(prompt.len) else 0;
        if (end_at != 0 and std.mem.indexOfScalar(u32, want.items, end_at) == null) try want.append(a, end_at);
        if (rewind > 0 and s.active_rewind_at == rewind) try want.append(a, rewind);
        // the prompt this one parts from: the one it resumed, or (fork rule) on a miss the kept one sharing the most
        const kept = if (previous.len > 0 or !s.rules.fork) previous else forks.longestShared(Entry, s.entries.items, prompt);
        const part: u32 = if (kept.len == 0 or short) 0 else @intCast(std.mem.indexOfDiff(u32, kept, prompt) orelse @min(kept.len, prompt.len));
        // the fork rule keeps one before the part, however near: a state at `at` is keyed with the token at `at`
        const stable: u32 = if (part == 0 or part >= history_len or part < history_len / 2) 0 else if (s.rules.fork) part -| s.rules.lookahead else part;
        if (stable > 0) try want.append(a, stable);
        if (!short) try want.appendSlice(a, shared);
        // tail: a prompt that names no history keeps its last chunk too (a chat keeps its history's mark alone)
        if (!short and s.rules.tail and history_len == 0) try want.append(a, @intCast(prompt.len -| @max(1, s.rules.lookahead)));
        const fork_at = if (s.rules.fork and s.rules.planned) s.startAt(starts, stable) else stable; // where the loop snaps it
        for (want.items, 0..) |w, k| {
            const tail = tail_at != 0 and w == tail_at;
            const end = end_at != 0 and w == end_at;
            const at = if (s.rules.planned and !tail and !end) s.startAt(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at > prompt.len or (at >= prompt.len and !tail and !end) or !s.usable(at, starts)) continue;
            const priority = tail or end or (rewind > 0 and k == 1 and at == rewind) or (s.rules.fork and at == fork_at);
            if (!priority and !(k == 0 and warm_pass) and at - from < s.rules.min_gap and (k > 0 or (from > 0 and s.rules.warm == .on))) continue; // near the resume point
            if (k > 0 and !priority) { // the history's mark always (warm families: away from the resume point); the others away from it and each other
                const near = for (out.items) |o| {
                    if (@max(o, at) - @min(o, at) < s.rules.min_gap) break true;
                } else false;
                if (near) continue;
            }
            if (std.mem.indexOfScalar(u32, out.items, at) == null) try out.append(a, at);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        if (end_at != 0) { // the last chunk this pass actually runs must keep the fresh remainder's class
            var last_from: u32 = from;
            for (out.items) |o| {
                if (o > last_from and o < end_at) last_from = o;
            }
            const span: usize = end_at - last_from;
            const fresh: usize = freshLast(s.rules.grid, end_at);
            if (s.rules.grid < 64 or span < 64 or chunkClass(span) != chunkClass(fresh)) {
                if (std.mem.indexOfScalar(u32, out.items, end_at)) |i| {
                    _ = out.orderedRemove(i);
                }
            }
        }
        return out.toOwnedSlice(a);
    }

    /// Keep the state at `at` with its span prefix, evicting within the budget; `starts` are a learned cut's chunks.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span) bool {
        return s.keepBorn(prompt, @intCast(prompt.len), at, owner, starts, spans);
    }

    /// `keep` as a prompt of `born` tokens: a reply's end and its warm pass supersede none of their request's states.
    pub fn keepBorn(s: *Store, prompt: []const u32, born: u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span) bool {
        s.collectExpired();
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len or s.hold) return false;
        s.clock += 1;
        const shared = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(prompt[0..n])) != null;
        for (s.entries.items) |e| if (!e.expired and e.at == at and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at) and images.matches(s, e)) {
            e.used = s.clock; // the same state again: no copy
            e.shared = e.shared or shared;
            if (shared) learned.learn(s, e, starts);
            return true;
        };
        const image_ids = images.prefix(s.gpa, s.image_query orelse &.{}, @intCast(n)) catch |err| return s.fail(at, err);
        var owns_images = true;
        defer if (owns_images) s.gpa.free(image_ids);
        const decoded = modes.prefix(s.gpa, spans, at) catch |err| return s.fail(at, err);
        var owns_decoded = true;
        defer if (owns_decoded) s.gpa.free(decoded);
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes > s.budget or s.most == 0) {
            s.counts.refused += 1;
            note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            return false;
        }
        while (s.entries.items.len > 0) { // room: spare storage this save can take, else new storage inside the budget
            if (s.family.vtable.reuses) |f| if (s.entries.items.len < s.most and f(s.family.ptr, at)) break;
            const free_ = s.spare();
            if (s.held + free_ + bytes <= s.budget and s.entries.items.len < s.most) break;
            if (free_ > 0) {
                if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| bytes); // spare that cannot take this state goes before any state
                if (s.spare() < free_) continue;
            }
            s.remove(s.victimBut(null, prompt, born) orelse return s.fail(at, error.CacheBorrowed)); // its storage may come back spare and take this state
            s.counts.evicted += 1;
        }
        if (s.entries.items.len == 0) if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| bytes);
        const e = s.gpa.create(Entry) catch return s.fail(at, error.OutOfMemory);
        const last = s.gpa.dupe(u32, prompt) catch { // one copy: the state's tokens are its prefix
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
            s.gpa.free(last);
            s.gpa.destroy(e);
            return s.fail(at, err);
        };
        const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
        e.* = .{ .tokens = last[0..n], .images = image_ids, .image_frame = s.image_frame, .decode_spans = decoded, .at = at, .saved = saved, .bytes = charged, .born = born, .used = s.clock, .last = last, .shared = shared, .own = last };
        owns_decoded = false;
        owns_images = false;
        index.add(s, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
        policy.link(s, e);
        s.held += charged;
        s.counts.kept += 1;
        if (shared) learned.learn(s, e, starts);
        if (s.disk) |r| rows.write.queued(r, e, starts);
        if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held); // spare storage only inside what the budget leaves
        return true;
    }

    /// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
    fn recall(s: *Store, prompt: []const u32, born: u32, starts: []const u32, spans: []const modes.Span, have: ?*Entry) ?*Entry {
        const im = s.imprint orelse return have;
        const read = s.family.vtable.read orelse return have;
        const m = im.best(prompt, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
        if (s.image_frame != 0 or !images.diskSafe(m.tokens) or !images.equal(&.{}, s.image_query orelse &.{}, @intCast(m.tokens.len))) return have;
        if (!modes.equal(&.{}, spans, m.at)) return have; // the request decodes rows the learned prompt pass prefilled
        const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
        while (bytes <= s.budget and (s.held + s.spare() + bytes > s.budget or s.entries.items.len >= s.most)) {
            s.remove(s.victimBut(have, prompt, born) orelse return have);
            s.counts.evicted += 1;
        }
        if (bytes > s.budget) return have;
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = born, .used = s.clock, .last = &.{}, .shared = true };
        e.tokens = s.gpa.dupe(u32, m.tokens) catch return s.drop(e, have, false);
        e.last = s.gpa.dupe(u32, prompt) catch return s.drop(e, have, false);
        const key = m.key;
        e.saved = read(s.family.ptr, im.dir, key, m.at) catch |err| {
            note("reading the learned {d} tokens failed ({s}); forgotten, a later pass learns them again", .{ e.at, @errorName(err) });
            learned.unlearn(s, im, key);
            return s.drop(e, have, false);
        };
        im.touch(key);
        index.add(s, e) catch return s.drop(e, have, true);
        policy.link(s, e);
        s.held += bytes;
        if (!@import("builtin").is_test) std.log.info("prompt cache: read {d} learned tokens from disk", .{m.at});
        return e;
    }

    /// A half-made recalled entry undone (`saved` once read); the lookup goes on with `have`.
    fn drop(s: *Store, e: *Entry, have: ?*Entry, saved: bool) ?*Entry {
        if (saved) s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
        return have;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the store holds.
    pub fn report(s: *const Store, prompt: usize, from: u32, kept: u64) void {
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed });
    }

    fn fail(s: *Store, at: u32, err: anyerror) bool {
        defer s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return false;
    }

    /// The entry to free first, never `skip`: a superseded one, no branch point, oldest; else the oldest, rewind last.
    pub const victimBut = policy.victimBut;
    const extended = policy.extended;
};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
pub fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

test {
    _ = @import("prompt_cache_index_test.zig");
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_cache_learn_test.zig");
    _ = @import("prompt_cache_rewind_test.zig");
    _ = @import("prompt_cache_end_test.zig");
    _ = @import("prompt_cache_interior_test.zig");
    _ = @import("prompt_cache_branch_test.zig");
    _ = @import("prompt_cache_fork_test.zig");
    _ = @import("prompt_cache_give_test.zig");
    _ = @import("prompt_cache_lifetime_test.zig");
    _ = @import("prompt_cache_scan_test.zig");
    _ = @import("learned_fault_test.zig");
    _ = modes;
    _ = rows;
}

test {
    _ = @import("prompt_cache_repeat_test.zig");
    _ = @import("prompt_cache_tail_test.zig");
}
