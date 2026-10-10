//! Exact prompt reuse: states are keyed by tokens and canonical decoded intervals through their position.
const std = @import("std");
const Allocator = std.mem.Allocator;
const imprint = @import("prompt_imprint.zig");
const modes = @import("cache_modes.zig");
const learned = @import("prompt_learn.zig");

/// A family's copy of one state.
pub const Saved = *anyopaque;

pub fn singleReclaim(_: *anyopaque, dir: [:0]const u8, key: u64) !u64 {
    var buf: [1100]u8 = undefined;
    return @import("lanes").learned_dirs.fileBytes(try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0));
}
pub fn singleForgetChecked(_: *anyopaque, dir: [:0]const u8, key: u64) !void {
    var buf: [1100]u8 = undefined;
    try @import("lanes").learned_dirs.unlink(try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0));
}
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
pub const Rules = struct {
    /// Tokens past the position a state read (Flash Next's MTP head keys row at-1 with token at: 1).
    lookahead: u32 = 0,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and keep only at the request's chunk starts.
    planned: bool = false,
    /// A planned family's own zero-anchored chunk rows, its starts when a request names none (0: none).
    grid: u32 = 0,
    /// A mark other than the history's is kept only this far from every other mark and the resume point.
    min_gap: u32 = 256,
    /// Shorter prompts keep nothing: below it a mark's extra prompt call costs more than a later turn's reuse saves.
    min_prompt: u32 = 4096,
    /// Replies are prefilled in background passes kept at their end: a pass resumed near its history keeps no mark.
    warm: bool = false,
    /// A paged store (prompt_radix.zig): the tokens a page holds, where its marks and resumes sit (0: not paged).
    page: u32 = 0,
    /// A paged store also keeps the prompt's last whole page, for a later turn whose history the request did not name.
    tail: bool = false,
};

pub const Entry = struct {
    tokens: []u32, // the state's tokens and `lookahead` more
    decode_spans: []modes.Span = &.{}, // canonical row arithmetic through at, excluding lookahead
    at: u32,
    saved: Saved,
    bytes: u64,
    born: u32, // the length of the prompt that kept it: a later turn's is longer
    used: u64, // the store's clock at its last keep or resume
    last: []u32, // the last prompt that kept or resumed it
    shared: bool = false, // kept at a shared system block's cut: other conversations resume it, so its own never supersedes it
};

pub const Counts = struct { hits: u64 = 0, misses: u64 = 0, kept: u64 = 0, evicted: u64 = 0, refused: u64 = 0, failed: u64 = 0 };

/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// A paged store's `keep`: whether the mark holds a state, and how many leading pages the stream swaps for the store's.
pub const Kept = struct { held: bool, shared: usize };

/// The entry a backend restores itself (null: the pass starts at 0) and the pass's marks.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{} };

pub const Store = struct {
    gpa: Allocator,
    family: Snapshots,
    rules: Rules,
    budget: u64,
    held: u64 = 0,
    clock: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,
    counts: Counts = .{},
    shared_keys: std.ArrayList(u64) = .empty, // the tokens of recently planned shared cuts, hashed (at most SHARED_KEYS)
    imprint: ?*imprint.Imprint = null, // --learn: shared cuts' states on disk, read back by later sessions and servers
    active_rewind_at: u32 = 0, // the last lookup's rewind state (0: none); hosts keep a pass's states before the next lookup

    const SHARED_KEYS = 64;

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        return .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
    }

    pub fn deinit(s: *Store) void {
        for (s.entries.items) |e| s.free(e);
        s.entries.deinit(s.gpa);
        s.shared_keys.deinit(s.gpa);
        s.held = 0;
    }

    fn free(s: *Store, e: *Entry) void {
        s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.decode_spans);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
    }

    fn remove(s: *Store, i: usize) void {
        const e = s.entries.orderedRemove(i);
        s.held -= e.bytes;
        s.free(e);
    }

    /// The family's free storage kept for later saves.
    pub fn spare(s: *const Store) u64 {
        return if (s.family.vtable.spare) |f| f(s.family.ptr) else 0;
    }

    /// A kept state now takes `extra` more bytes (a family copied what it shared into the state's own storage): charged
    /// when the budget leaves room for it, refused otherwise; nothing is evicted, since a prompt pass is under way.
    pub fn grow(s: *Store, saved: Saved, extra: u64) bool {
        for (s.entries.items) |e| if (e.tokens.len > 0 and e.saved == saved) {
            if (extra > s.room()) return false;
            e.bytes += extra;
            s.held += extra;
            return true;
        };
        return false;
    }

    /// What the budget leaves past the kept states and the spare storage: a family readies storage only inside it.
    pub fn room(s: *const Store) u64 {
        return s.budget -| s.held -| s.spare();
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or a chunk start for planned families.
    fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or s.startAt(starts, at) == at);
    }

    /// The chunk start at or before `at`: the request's, else the family's grid (0 when none).
    fn startAt(s: *const Store, starts: []const u32, at: u32) u32 {
        if (starts.len == 0 and s.rules.grid > 0) return at - at % s.rules.grid;
        return floorStart(starts, at);
    }

    /// The latest kept state this prompt can resume at `at` or under min_gap before it (null: none).
    fn keptNear(s: *const Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, at: u32) ?u32 {
        var best: ?u32 = null;
        for (s.entries.items) |e| {
            if (e.at > at or at - e.at >= s.rules.min_gap or e.tokens.len > prompt.len or !s.usable(e.at, starts)) continue;
            if (!std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens) or !modes.equal(e.decode_spans, spans, e.at)) continue;
            if (best == null or e.at > best.?) best = e.at;
        }
        return best;
    }

    /// Find a kept token prefix with matching canonical spans through its position, leaving at least one row.
    pub fn find(s: *Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) ?*Entry {
        var best: ?*Entry = null;
        for (s.entries.items) |e| {
            if (e.tokens.len > prompt.len or e.at >= prompt.len or !s.usable(e.at, starts)) continue;
            if (best != null and e.at <= best.?.at) continue;
            const n = e.tokens.len;
            if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
            if (!modes.equal(e.decode_spans, spans, e.at)) continue;
            best = e;
        }
        return best;
    }

    /// Restore the longest prefix matching tokens and canonical spans, then plan prompt-cache marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        return s.beginRewind(a, prompt, history_len, 0, shared, starts, owner, spans);
    }

    /// `begin`, also keeping a state before the latest user message (`rewind_len`, 0: none) that its edits resume.
    pub fn beginRewind(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, rewind_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        const l = try s.lookupRewind(a, prompt, history_len, rewind_len, shared, starts, spans);
        const e = l.entry orelse return .{ .marks = l.marks };
        const ok = if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| true else |err| blk: {
            note("restoring {d} tokens failed ({s}); prefilling from the start", .{ e.at, @errorName(err) });
            break :blk false;
        };
        const from = if (ok) e.at else 0;
        s.resumed(e, prompt, ok);
        s.reserve(prompt, null, l.marks, spans); // restored: the state it resumed may go too (a peer restores before it drops)
        return .{ .from = from, .marks = l.marks };
    }

    /// Look up canonical-span state and marks without restoring; the caller later reports resumed.
    pub fn lookup(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        return s.lookupRewind(a, prompt, history_len, 0, shared, starts, spans);
    }

    /// `lookup` plus a rewind state on the starts or grid; a state kept under min_gap before it serves instead.
    pub fn lookupRewind(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, rewind_len: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        const endpoint = if (s.rules.planned) s.startAt(starts, history_len) else history_len;
        const rewind = if (s.rules.planned) s.startAt(starts, rewind_len -| s.rules.lookahead) else rewind_len -| s.rules.lookahead;
        const endpoint_bytes = if (endpoint > 0 and (endpoint < prompt.len or (s.rules.warm and endpoint == prompt.len)) and endpoint + s.rules.lookahead <= prompt.len) s.family.vtable.bytes(s.family.ptr, endpoint) else 0;
        const rewind_bytes = if (rewind > 0 and rewind < prompt.len and rewind + s.rules.lookahead <= prompt.len and s.usable(rewind, starts)) s.family.vtable.bytes(s.family.ptr, rewind) else 0;
        const priority_bytes = if (endpoint_bytes <= s.budget) endpoint_bytes else 0;
        s.active_rewind_at = if (prompt.len >= s.rules.min_prompt and rewind_bytes > 0 and rewind != endpoint and rewind_bytes <= s.budget - priority_bytes) rewind else 0;
        if (s.active_rewind_at > 0) if (s.keptNear(prompt, starts, spans, s.active_rewind_at)) |at| {
            s.active_rewind_at = at; // a state kept under min_gap before it is the rewind state: no new copy
        };
        const e = s.recall(prompt, starts, spans, s.find(prompt, starts, spans));
        if (e == null) s.counts.misses += 1;
        const from = if (e) |x| x.at else 0;
        const marks_ = if (s.active_rewind_at == 0)
            try s.fitting(a, try s.marks(a, prompt, from, history_len, shared, starts, if (e) |x| x.last else &.{}))
        else
            try s.fittingRewind(a, try s.fitting(a, try s.marksRewind(a, prompt, from, history_len, s.active_rewind_at, shared, starts, if (e) |x| x.last else &.{})), endpoint, s.active_rewind_at, endpoint_bytes);
        for (shared) |w| { // the shared cuts this pass keeps: their states serve other conversations too
            const at = if (s.rules.planned) s.startAt(starts, w) else w;
            if (std.mem.indexOfScalar(u32, marks_, at) != null) s.noteShared(prompt[0 .. at + s.rules.lookahead]);
        }
        s.reserve(prompt, e, marks_, spans);
        return .{ .entry = e, .marks = marks_ };
    }

    /// Room for every state the pass keeps, made before it: a peer that mirrors the keeps hears of these evictions with the request.
    fn reserve(s: *Store, prompt: []const u32, from: ?*Entry, marks_: []const u32, spans: []const modes.Span) void {
        var need: u64 = 0;
        for (marks_) |at| {
            const n = @as(usize, at) + s.rules.lookahead;
            const exists = for (s.entries.items) |e| {
                if (e.at == at and e.tokens.len == n and n <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at)) break true;
            } else false;
            if (!exists) need += s.family.vtable.bytes(s.family.ptr, at);
        }
        while (s.held + need > s.budget) {
            const i = s.victimBut(from, prompt) orelse return;
            s.remove(i);
            s.counts.evicted += 1;
        }
    }

    /// Every kept state goes after a weight change, and learned states on disk stop being read: old arithmetic.
    pub fn clear(s: *Store) void {
        while (s.entries.items.len > 0) s.remove(s.entries.items.len - 1);
        s.shared_keys.clearRetainingCapacity();
        s.imprint = null;
        s.active_rewind_at = 0;
    }

    /// Forget a failed peer state matching the token prefix and canonical spans through at.
    pub fn forget(s: *Store, prompt: []const u32, at: u32, spans: []const modes.Span) void {
        for (s.entries.items, 0..) |e, i| if (e.at == at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]) and modes.equal(e.decode_spans, spans, at)) {
            s.counts.failed += 1;
            return s.remove(i);
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

    /// The marks whose state the budget can hold (a new slice in `a`); the rest are refused now, so no pass, nor a peer, copies them.
    fn fitting(s: *Store, a: Allocator, marks_: []const u32) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        for (marks_) |at| {
            const bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (bytes <= s.budget) try out.append(a, at) else {
                s.counts.refused += 1;
                note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            }
        }
        return out.toOwnedSlice(a);
    }

    /// Fitted marks that the budget holds together: the history's first, then the rewind, then the rest in order.
    fn fittingRewind(s: *Store, a: Allocator, marks_: []const u32, endpoint: u32, rewind: u32, endpoint_bytes: u64) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        var used: u64 = 0;
        if (endpoint_bytes <= s.budget and endpoint_bytes > 0) {
            used = endpoint_bytes;
            if (std.mem.indexOfScalar(u32, marks_, endpoint) != null) try out.append(a, endpoint);
        }
        if (s.active_rewind_at == rewind and std.mem.indexOfScalar(u32, marks_, rewind) != null) try out.append(a, rewind);
        if (s.active_rewind_at == rewind) used += s.family.vtable.bytes(s.family.ptr, rewind);
        for (marks_) |at| {
            if (at == endpoint or at == rewind) continue;
            const bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (bytes <= s.budget -| used) {
                try out.append(a, at);
                used += bytes;
            }
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (dropped: the pass ran from 0).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        const i = std.mem.indexOfScalar(*Entry, s.entries.items, e) orelse return;
        if (!ok) {
            s.counts.failed += 1;
            return s.remove(i);
        }
        s.clock += 1;
        e.used = s.clock;
        s.counts.hits += 1;
        const last = s.gpa.dupe(u32, prompt) catch return;
        s.gpa.free(e.last);
        e.last = last;
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        return s.marksRewind(a, prompt, from, history_len, 0, shared, starts, previous);
    }

    /// `marks` with the active rewind second: kept however near the other marks, never at or before the resume point.
    fn marksRewind(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, rewind: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        if (prompt.len < s.rules.min_prompt) return &.{};
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        try want.append(a, history_len);
        if (rewind > 0 and s.active_rewind_at == rewind) try want.append(a, rewind);
        if (previous.len > 0) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        try want.appendSlice(a, shared);
        for (want.items, 0..) |w, k| {
            const at = if (s.rules.planned) s.startAt(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at >= prompt.len or !s.usable(at, starts)) continue;
            const is_rewind = rewind > 0 and k == 1 and at == rewind;
            if (!is_rewind and at - from < s.rules.min_gap and (k > 0 or (from > 0 and s.rules.warm))) continue; // near the resume point
            if (k > 0 and !is_rewind) { // the history's mark always (warm families: away from the resume point); the others away from it and each other
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
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len) return false;
        s.clock += 1;
        const shared = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(prompt[0..n])) != null;
        for (s.entries.items) |e| if (e.at == at and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at)) {
            e.used = s.clock; // the same state again: no copy
            e.shared = e.shared or shared;
            if (shared) learned.learn(s, e, starts);
            return true;
        };
        const decoded = modes.prefix(s.gpa, spans, at) catch |err| return s.fail(at, err);
        var owns_decoded = true;
        defer if (owns_decoded) s.gpa.free(decoded);
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes > s.budget) {
            s.counts.refused += 1;
            note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            return false;
        }
        while (s.entries.items.len > 0) { // room: spare storage this save can take, else new storage inside the budget
            if (s.family.vtable.reuses) |f| if (f(s.family.ptr, at)) break;
            const free_ = s.spare();
            if (s.held + free_ + bytes <= s.budget) break;
            if (free_ > 0) {
                if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| bytes); // spare that cannot take this state goes before any state
                if (s.spare() < free_) continue;
            }
            s.remove(s.victimBut(null, prompt).?); // its storage may come back spare and take this state
            s.counts.evicted += 1;
        }
        if (s.entries.items.len == 0) if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| bytes);
        const e = s.gpa.create(Entry) catch return s.fail(at, error.OutOfMemory);
        const tokens = s.gpa.dupe(u32, prompt[0..n]) catch {
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const last = s.gpa.dupe(u32, prompt) catch {
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, err);
        };
        const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
        e.* = .{ .tokens = tokens, .decode_spans = decoded, .at = at, .saved = saved, .bytes = charged, .born = @intCast(prompt.len), .used = s.clock, .last = last, .shared = shared };
        owns_decoded = false;
        s.entries.append(s.gpa, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
        s.held += charged;
        s.counts.kept += 1;
        if (shared) learned.learn(s, e, starts);
        if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held); // spare storage only inside what the budget leaves
        return true;
    }

    /// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
    fn recall(s: *Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, have: ?*Entry) ?*Entry {
        const im = s.imprint orelse return have;
        const read = s.family.vtable.read orelse return have;
        const m = im.best(prompt, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
        if (!modes.equal(&.{}, spans, m.at)) return have; // the request decodes rows the learned prompt pass prefilled
        const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
        while (bytes <= s.budget and s.held + s.spare() + bytes > s.budget) {
            s.remove(s.victimBut(have, prompt) orelse return have);
            s.counts.evicted += 1;
        }
        if (bytes > s.budget) return have;
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = @intCast(prompt.len), .used = s.clock, .last = &.{}, .shared = true };
        e.tokens = s.gpa.dupe(u32, m.tokens) catch return s.drop(e, have, false);
        e.last = s.gpa.dupe(u32, prompt) catch return s.drop(e, have, false);
        const key = m.key;
        e.saved = read(s.family.ptr, im.dir, key, m.at) catch |err| {
            note("reading the learned {d} tokens failed ({s}); forgotten, a later pass learns them again", .{ e.at, @errorName(err) });
            learned.unlearn(s, im, key);
            return s.drop(e, have, false);
        };
        im.touch(key);
        s.entries.append(s.gpa, e) catch return s.drop(e, have, true);
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

    /// The entry to free first, never `skip`: a superseded one, oldest first; else the oldest, the turn's rewind last.
    fn victimBut(s: *const Store, skip: ?*Entry, prompt: []const u32) ?usize {
        var best: ?usize = null;
        for (s.entries.items, 0..) |e, i| {
            if (e.shared or e == skip or s.isActiveRewind(e, prompt)) continue; // a shared cut is evicted only as the least recently used
            const now = prompt.len > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens); // this prompt extends it
            const moved_on = now or for (s.entries.items) |o| {
                if (o != e and o.born > e.born and o.tokens.len > e.tokens.len and std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) break true;
            } else false;
            if (moved_on and (best == null or e.used < s.entries.items[best.?].used)) best = i;
        }
        if (best) |i| return i;
        var oldest: ?usize = null;
        for (s.entries.items, 0..) |e, i| if (e != skip and (oldest == null or s.before(e, s.entries.items[oldest.?], prompt))) {
            oldest = i;
        };
        return oldest;
    }

    /// Eviction order past the superseded states: least recently used first, the turn's rewind state last.
    fn before(s: *const Store, e: *const Entry, o: *const Entry, prompt: []const u32) bool {
        const e_rewind = s.isActiveRewind(e, prompt);
        const o_rewind = s.isActiveRewind(o, prompt);
        return if (e_rewind != o_rewind) o_rewind else e.used < o.used;
    }

    /// The last lookup's rewind state, while the prompt still extends it.
    fn isActiveRewind(s: *const Store, e: *const Entry, prompt: []const u32) bool {
        return e.at == s.active_rewind_at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
    }
};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
pub fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

/// The chunk start at or before `at` (0 when none).
fn floorStart(starts: []const u32, at: u32) u32 {
    var best: u32 = 0;
    for (starts) |p| if (p <= at and p > best) {
        best = p;
    };
    return best;
}

test {
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_cache_rewind_test.zig");
    _ = @import("learned_fault_test.zig");
    _ = modes;
}
