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
const intermediate = @import("prompt_intermediate.zig");
const index = @import("prompt_cache_index.zig");
const forks = @import("prompt_cache_fork.zig");
pub const others = @import("prompt_cache_others.zig");
pub const rows = @import("rows_store.zig");
/// A family's copy of one state.
pub const Saved = *anyopaque;
pub const singleReclaim = imprint.singleReclaim; // the learned-state file helpers, beside the Imprint that owns them
pub const singleForgetChecked = imprint.singleForgetChecked;
/// What a family gives the cache: copies of its state while the prompt pass stands at a chunk end.
pub const Snapshots = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        /// The most new storage `save` takes at `at` tokens (checked against the budget before any copy).
        bytes: *const fn (ptr: *anyopaque, at: u32) u64,
        /// Copy the live state after `at` prompt tokens to new storage; `owner`: lane core's stream (null: one stream).
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
    seen: Seen = .{}, // wall-clock created and last-use times and resumes, for the dashboard (never policy)
    last: []u32, // the last prompt that kept or resumed it
    shared: bool = false, // kept at a shared system block's cut: other conversations resume it, so its own never supersedes it
    rewind: bool = false, // selected as a rewind: a reply born with this request does not supersede it
    branched: bool = false, // resumed again after a later prompt kept a longer state extending it: never superseded
    intermediate: bool = false, // a ladder checkpoint under the history: kept across append turns, never superseded
    expired: bool = false, // hidden from lookups immediately, reclaimed only at safe host boundaries
    borrowers: u32 = 0, // outstanding lookups: released by resumed or release, never evicted while borrowed
    ext: std.ArrayList(*Entry) = .empty, // live extensions born no earlier
    own: []u32 = &.{}, // the kept prompt's one copy, which `tokens` and at first `last` slice (empty: they own theirs)
};
pub const Counts = struct { hits: u64 = 0, misses: u64 = 0, kept: u64 = 0, evicted: u64 = 0, refused: u64 = 0, failed: u64 = 0, trimmed: u64 = 0 };
pub const RestoreDetails = @import("prompt_cache_restore.zig").Details;
pub const RestoreLog = @import("prompt_cache_restore.zig").Log;
/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };
/// The borrowed entry a backend restores; the caller must report resumed or release it, even on cancellation.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{} };
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
    active_rewind_key: u64 = 0, // the selected rewind prefix, so interleaved keeps tag only its token path
    links_lost: bool = false, // an allocation failure left the extension links incomplete: `extended` scans
    reservation_guard: ?@import("prompt_cache_reserve.zig").Guard = null, // scoped to an admission lookup
    hold: bool = false, // memory pressure: lookups plan no marks and keeps refuse, so no new state is kept
    over: []const u8 = "", // what refusing a state past the whole budget adds: the setting that keeps it otherwise
    keep_others: bool = false, // a new state may evict only its own conversation's states (prompt_cache_others.zig)
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

    pub fn free(s: *Store, e: *Entry) void {
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

    /// A kept state takes `extra` bytes (a family's copied-in share) within budget; never evicts, a pass is under way.
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
    pub fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or s.startAt(starts, at) == at);
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
                // Restored: this state may go too; a peer restores before it drops.
                s.reserve(prompt, @intCast(prompt.len), null, l.marks, spans, @import("prompt_cache_reserve.zig").endpointFor(s, prompt, at, history_len, starts, spans));
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
        s.reserve(prompt, @intCast(prompt.len), null, marks_, spans, @import("prompt_cache_reserve.zig").endpointFor(s, prompt, 0, history_len, starts, spans));
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
        s.active_rewind_key = if (s.active_rewind_at > 0) sharedKey(prompt[0 .. s.active_rewind_at + s.rules.lookahead]) else 0;
        for (s.entries.items) |kept| if (s.active_rewind_at > 0 and kept.at == s.active_rewind_at and kept.tokens.len <= prompt.len and
            std.mem.eql(u32, kept.tokens, prompt[0..kept.tokens.len]) and modes.equal(kept.decode_spans, spans, kept.at) and images.matches(s, kept))
        {
            kept.rewind = true;
        };
        const e = if (disk)
            rows.read.recall(s, prompt, born, starts, spans, s.recall(prompt, born, starts, spans, s.find(prompt, starts, spans)))
        else
            s.find(prompt, starts, spans);
        @import("prompt_cache_divergence.zig").report(s, prompt, spans, e);
        if (e == null) s.counts.misses += 1;
        const from = if (e) |x| x.at else 0;
        const ordinary = if (s.active_rewind_at == 0)
            try s.fitting(a, try s.marksRewind(a, prompt, from, history_len, 0, shared, starts, spans, if (e) |x| x.last else &.{}))
        else
            try s.fittingRewind(a, try s.fitting(a, try s.marksRewind(a, prompt, from, history_len, s.active_rewind_at, shared, starts, spans, if (e) |x| x.last else &.{})), endpoint, s.active_rewind_at, endpoint_bytes);
        const known_end = if (user_end <= history_len) user_end else 0;
        const marks_ = interior.add(s, a, prompt, from, known_end, rewind_len, starts, spans, ordinary, endpoint, endpoint_bytes, e) catch |err| {
            a.free(ordinary);
            return err;
        };
        const marked = intermediate.add(s, a, prompt, from, history_len, starts, spans, marks_, endpoint, endpoint_bytes, e) catch |err| {
            a.free(marks_);
            return err;
        };
        for (shared) |w| { // the shared cuts this pass keeps, and a short prompt's mark inside a shared block: other conversations resume them
            const at = if (s.rules.planned) s.startAt(starts, w) else w;
            for (marked) |m| if (m == at or (m < at and !s.rules.planned and prompt.len < s.rules.min_prompt)) s.noteShared(prompt[0 .. m + s.rules.lookahead]);
        }
        var primary = @import("prompt_cache_reserve.zig").endpointFor(s, prompt, from, history_len, starts, spans);
        if (history_len == 0 and primary == 0) for (marked) |at| {
            if (at != s.active_rewind_at) primary = @max(primary, at);
        };
        s.reserve(prompt, born, e, marked, spans, primary);
        const fitted = try @import("prompt_cache_reserve.zig").fitting(s, a, prompt, marked, spans, e, born, primary);
        if (e) |entry| entry.borrowers += 1;
        return .{ .entry = e, .marks = fitted };
    }

    const reserve = @import("prompt_cache_reserve.zig").reserve;

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

    pub fn sharedKey(tokens: []const u32) u64 {
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

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (the refused state is dropped).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        return s.resumedWithReason(e, prompt, if (ok) null else error.RestoreRejected);
    }

    pub const resumedWithReason = lifetime.resumedWithReason;

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        return s.marksRewind(a, prompt, from, history_len, 0, shared, starts, &.{}, previous);
    }

    /// `marks` with the active rewind second: kept however near the other marks, never at or before the resume point.
    fn marksRewind(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, rewind: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span, previous: []const u32) ![]const u32 {
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
            const at = if (s.rules.planned) s.startAt(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at >= prompt.len or !s.usable(at, starts)) continue;
            const priority = (rewind > 0 and k == 1 and at == rewind) or (s.rules.fork and at == fork_at);
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
        return out.toOwnedSlice(a);
    }

    /// Keep the state at `at` with its span prefix, evicting within the budget; `starts` are a learned cut's chunks.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span) bool {
        return s.keepBorn(prompt, @intCast(prompt.len), at, owner, starts, spans);
    }

    /// `keep` as a prompt of `born` tokens: its longer reply supersedes older states on the same token path.
    pub const keepBorn = @import("prompt_cache_keep.zig").keepBorn;

    /// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
    fn recall(s: *Store, prompt: []const u32, born: u32, starts: []const u32, spans: []const modes.Span, have: ?*Entry) ?*Entry {
        const im = s.imprint orelse return have;
        const read = s.family.vtable.read orelse return have;
        const m = im.best(prompt, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
        if (s.image_frame != 0 or !images.diskSafe(m.tokens) or !images.equal(&.{}, s.image_query orelse &.{}, @intCast(m.tokens.len))) return have;
        if (!modes.equal(&.{}, spans, m.at)) return have; // the request decodes rows the learned prompt pass prefilled
        const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
        while (bytes <= s.budget and (s.held + s.spare() + bytes > s.budget or s.entries.items.len >= s.most)) {
            s.remove(others.victim(s, have, prompt, born, bytes, 1) orelse return have);
            s.counts.evicted += 1;
        }
        if (bytes > s.budget) return have;
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = born, .used = s.clock, .seen = .of(s, m.tokens, &.{}, m.starts, m.at), .last = &.{}, .shared = true };
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
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d}, trimmed {d})", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed, s.counts.trimmed });
    }

    pub fn fail(s: *Store, at: u32, err: anyerror) bool {
        defer s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return false;
    }

    /// The entry to free first, never `skip`: a superseded one, no branch point, oldest; else the oldest, rewind last.
    pub const victimBut = policy.victimBut;
    const extended = policy.extended;
};

pub const note = @import("prompt_cache_note.zig").note; // a line on the engine's log (tests stay quiet)
pub const Seen = @import("prompt_cache_status.zig").Seen;
pub const status = @import("prompt_cache_status.zig"); // read-only copies for /stats and the dashboard

test {
    _ = @import("prompt_cache_review_test.zig");
    _ = @import("prompt_cache_promised_test.zig");
    _ = @import("prompt_cache_queue_room_test.zig");
    _ = @import("prompt_cache_index_test.zig");
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_cache_learn_test.zig");
    _ = @import("prompt_cache_rewind_test.zig");
    _ = @import("prompt_cache_interior_test.zig");
    _ = @import("prompt_cache_intermediate_test.zig");
    _ = @import("prompt_cache_branch_test.zig");
    _ = @import("prompt_cache_newest_test.zig");
    _ = @import("prompt_cache_fork_test.zig");
    _ = @import("prompt_cache_give_test.zig");
    _ = @import("prompt_cache_lifetime_test.zig");
    _ = @import("prompt_cache_scan_test.zig");
    _ = @import("learned_fault_test.zig");
    _ = modes;
    _ = rows;
    _ = @import("prompt_cache_repeat_test.zig");
    _ = @import("prompt_cache_tail_test.zig");
    _ = @import("prompt_cache_others_test.zig");
}
