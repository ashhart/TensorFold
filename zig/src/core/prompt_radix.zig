//! Prompt reuse as a radix tree over token pages: a request takes its longest match and resumes its deepest snapshot.

const std = @import("std");
const pc = @import("prompt_cache.zig");
const tree_mod = @import("prompt_radix_tree.zig");
const Allocator = std.mem.Allocator;

pub const Saved = tree_mod.Saved;
pub const Snapshots = tree_mod.Snapshots;
pub const Pages = tree_mod.Pages;
pub const Counts = tree_mod.Counts;
pub const Node = tree_mod.Node;
pub const Tree = tree_mod.Tree;

/// The rules of prompt_cache.zig, paged: `page` set, `lookahead` and `planned` not (a row's bits ignore its chunk).
pub const Rules = pc.Rules;

/// What the tree may hold: pages, and snapshots.
pub const Limits = struct { pages: usize = 0, snaps: usize = 0 };

pub const Plan = pc.Plan;
pub const Kept = pc.Kept;

pub const Store = struct {
    tree: Tree,
    rules: Rules,
    limits: Limits = .{},
    /// The tokens of recently planned shared cuts, hashed (at most SHARED_KEYS).
    shared_keys: std.ArrayList(u64) = .empty,

    const SHARED_KEYS = 64;

    pub fn init(gpa: Allocator, family: Snapshots, pages: Pages, rules: Rules) !Store {
        std.debug.assert(rules.page > 0 and rules.lookahead == 0 and !rules.planned);
        return .{ .tree = try Tree.init(gpa, family, pages, rules.page), .rules = rules };
    }

    pub fn deinit(s: *Store) void {
        s.shared_keys.deinit(s.tree.gpa);
        s.tree.deinit();
    }

    /// The tree may hold `limits` from now on: what passes them goes first.
    pub fn limit(s: *Store, limits: Limits) void {
        s.limits = limits;
        while (s.tree.snaps > limits.snaps) s.tree.evict(s.victim(&.{}, null) orelse break);
        while (s.tree.held_pages > limits.pages) if (!s.tree.evictLeaf()) break;
    }

    /// Whether anything can be kept.
    pub fn on(s: *const Store) bool {
        return s.limits.pages > 0 and s.limits.snaps > 0;
    }

    /// The deepest node with a state `prompt` resumes exactly: its tokens a prefix, at least one token left.
    pub fn find(s: *const Store, prompt: []const u32) ?*Node {
        if (!s.on() or prompt.len < 2) return null;
        return s.tree.deepest(prompt, (prompt.len - 1) / s.rules.page * s.rules.page);
    }

    /// Restores the longest state `prompt` resumes into `owner`, its pages into `adopt` (held once more); plans marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, owner: ?*anyopaque, adopt: *std.ArrayList(u32)) !Plan {
        if (!s.on()) return .{ .marks = try a.alloc(u32, 0) };
        const found = s.find(prompt);
        if (found == null) s.tree.counts.misses += 1;
        const resume_at: u32 = if (found) |n| n.end else 0;
        const planned = try s.marks(a, prompt, resume_at, history_len, shared, if (found) |n| n.entry.?.last else &.{});
        const marks_ = try s.fitting(a, planned);
        errdefer a.free(marks_);
        for (shared) |w| { // the shared cuts this pass keeps: their states serve other conversations too
            const cut = w - w % s.rules.page;
            if (std.mem.indexOfScalar(u32, marks_, cut) != null) s.noteShared(prompt[0..cut]);
        }
        s.reserve(prompt, found, marks_);
        var from: u32 = 0;
        if (found) |n| {
            const family = s.tree.family;
            const ok = if (family.vtable.restore(family.ptr, owner, n.entry.?.saved)) |_| true else |err| blk: {
                note("restoring {d} tokens failed ({s}); prefilling from the start", .{ n.end, @errorName(err) });
                break :blk false;
            };
            if (ok) {
                try s.tree.take(n, adopt);
                from = n.end;
            }
            s.resumed(n, prompt, ok);
        }
        return .{ .from = from, .marks = marks_ };
    }

    /// A restored state: it is now the prompt's (a failed restore drops it).
    pub fn resumed(s: *Store, n: *Node, prompt: []const u32, ok: bool) void {
        if (!ok) {
            s.tree.counts.failed += 1;
            return s.tree.evict(n);
        }
        const e = &n.entry.?;
        s.tree.clock += 1;
        e.used = s.tree.clock;
        e.hit = true;
        s.tree.counts.hits += 1;
        const last = s.tree.gpa.dupe(u32, prompt) catch return;
        s.tree.gpa.free(e.last);
        e.last = last;
    }

    /// Room for every state the pass keeps, made before it: the snapshots and the new pages of its last mark.
    fn reserve(s: *Store, prompt: []const u32, from: ?*Node, marks_: []const u32) void {
        if (marks_.len == 0) return;
        const last = marks_[marks_.len - 1];
        while (true) {
            const new = last / s.rules.page - s.tree.have(prompt[0..last]);
            if (s.tree.held_pages + new <= s.limits.pages and s.tree.snaps + marks_.len <= s.limits.snaps) return;
            s.tree.evict(s.victim(prompt, from) orelse return);
        }
    }

    /// A kept state a peer could not resume goes, so later prompts do not ask for it again.
    pub fn forget(s: *Store, prompt: []const u32, at: u32) void {
        const n = s.find(prompt[0..@min(prompt.len, at + 1)]) orelse return;
        if (n.end != at) return;
        s.tree.counts.failed += 1;
        s.tree.evict(n);
    }

    fn sharedKey(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
    }

    fn noteShared(s: *Store, tokens: []const u32) void {
        const k = sharedKey(tokens);
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, k) != null) return;
        if (s.shared_keys.items.len == SHARED_KEYS) _ = s.shared_keys.orderedRemove(0);
        s.shared_keys.append(s.tree.gpa, k) catch {};
    }

    /// The marks whose state the limits can hold (a new slice in `a`); the rest are refused now.
    fn fitting(s: *Store, a: Allocator, marks_: []const u32) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        for (marks_) |m| {
            if (m / s.rules.page <= s.limits.pages and s.limits.snaps > 0) try out.append(a, m) else {
                s.tree.counts.refused += 1;
                note("kept nothing at {d} tokens: {d} pages pass the limit of {d}", .{ m, m / s.rules.page, s.limits.pages });
            }
        }
        return out.toOwnedSlice(a);
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix, blocks, last page.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, previous: []const u32) ![]const u32 {
        if (prompt.len < s.rules.min_prompt or prompt.len < 2) return a.alloc(u32, 0);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        try want.append(a, history_len);
        if (previous.len > 0) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        try want.appendSlice(a, shared);
        if (s.rules.tail) try want.append(a, @intCast(prompt.len - 1));
        for (want.items, 0..) |w, k| {
            const m = w - w % s.rules.page;
            if (m <= from or m >= prompt.len) continue;
            if (k > 0) { // the history's mark always; the others only away from it, each other and the resume point
                if (m - from < s.rules.min_gap) continue;
                const near = for (out.items) |o| {
                    if (@max(o, m) - @min(o, m) < s.rules.min_gap) break true;
                } else false;
                if (near) continue;
            }
            if (std.mem.indexOfScalar(u32, out.items, m) == null) try out.append(a, m);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// At `at` the tree takes the pages of `mine` it lacks and the snapshot, evicting to fit; `path` gets its pages.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, mine: []const u32, path: []u32) Kept {
        const t = &s.tree;
        if (!s.on() or at == 0 or at % s.rules.page != 0 or at > prompt.len or mine.len != at / s.rules.page or path.len != mine.len) return .{ .held = false, .shared = 0 };
        t.clock += 1;
        const tokens = prompt[0..at];
        const flagged = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(tokens)) != null;
        while (true) {
            var last: ?*Node = null;
            const got = t.heldPrefix(tokens, mine.len, path, &last);
            if (got == mine.len and last != null and last.?.end == at and last.?.entry != null) {
                const e = &last.?.entry.?;
                e.used = t.clock; // the same state again: no copy
                e.shared = e.shared or flagged;
                t.touch(last.?);
                return .{ .held = true, .shared = mine.len };
            }
            if (t.held_pages + (mine.len - got) <= s.limits.pages and t.snaps + 1 <= s.limits.snaps) break;
            const v = s.victim(prompt, null) orelse {
                t.counts.refused += 1;
                note("kept nothing at {d} tokens: no room", .{at});
                return .{ .held = false, .shared = got };
            };
            t.evict(v);
        }
        const node = t.insert(tokens, mine, path) catch |err| return s.fail(at, err, 0);
        const family = t.family;
        const saved = family.vtable.save(family.ptr, owner, at) catch |err| return s.fail(at, err, mine.len);
        const last = t.gpa.dupe(u32, prompt) catch {
            family.vtable.drop(family.ptr, saved);
            return s.fail(at, error.OutOfMemory, mine.len);
        };
        node.entry = .{ .saved = saved, .bytes = family.vtable.bytes(family.ptr, at), .born = @intCast(prompt.len), .used = t.clock, .last = last, .shared = flagged };
        t.snaps += 1;
        t.counts.kept += 1;
        return .{ .held = true, .shared = mine.len };
    }

    fn fail(s: *Store, at: u32, err: anyerror, shared: usize) Kept {
        s.tree.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return .{ .held = false, .shared = shared };
    }

    /// Whether `a` is a less valuable entry than `b`: one never resumed before one that was, then the older.
    fn worse(a: *const Node, b: *const Node) bool {
        const x = a.entry.?;
        const y = b.entry.?;
        if (x.hit != y.hit) return !x.hit;
        return x.used < y.used;
    }

    /// Whether `prompt` repeats every token up to the end of `n`.
    fn extends(n: *const Node, prompt: []const u32) bool {
        const parent = n.parent orelse return true;
        return prompt.len >= n.end and std.mem.eql(u32, n.tokens, prompt[parent.end..n.end]) and extends(parent, prompt);
    }

    /// Whether a later prompt kept a state below `n`.
    fn superseded(n: *const Node, born: u32) bool {
        for (n.kids.items) |k| {
            if ((k.entry != null and k.entry.?.born > born) or superseded(k, born)) return true;
        }
        return false;
    }

    fn pick(n: *Node, prompt: []const u32, skip: ?*Node, moved: *?*Node, oldest: *?*Node) void {
        if (n.entry) |e| if (n != skip) {
            // a state its conversation moved past goes first, a shared cut only as the least recently used
            const gone = !e.shared and ((prompt.len > e.born and prompt.len > n.end and extends(n, prompt)) or superseded(n, e.born));
            if (gone and (moved.* == null or e.used < moved.*.?.entry.?.used)) moved.* = n;
            if (oldest.* == null or worse(n, oldest.*.?)) oldest.* = n;
        };
        for (n.kids.items) |k| pick(k, prompt, skip, moved, oldest);
    }

    /// The entry to free first, never `skip`: one a later prompt extends, oldest first; else the least valuable.
    fn victim(s: *const Store, prompt: []const u32, skip: ?*Node) ?*Node {
        var moved: ?*Node = null;
        var oldest: ?*Node = null;
        pick(s.tree.root, prompt, skip, &moved, &oldest);
        return moved orelse oldest;
    }

    /// Evicts until `want` pool pages are free and not promised; false when the tree cannot give that many.
    pub fn reclaim(s: *Store, want: usize) bool {
        const pages = s.tree.pages;
        while (pages.vtable.available(pages.ptr) < want) if (!s.tree.evictLeaf()) return false;
        return true;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the tree holds.
    pub fn report(s: *const Store, prompt: usize, from: u32, kept: u64) void {
        if (@import("builtin").is_test) return;
        const t = &s.tree;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states and {d} pages ({d} MiB) of {d} states and {d} pages (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, t.snaps, t.held_pages, t.held() >> 20, s.limits.snaps, s.limits.pages, t.counts.hits, t.counts.misses, t.counts.evicted, t.counts.refused, t.counts.failed });
    }
};

/// A line on the engine's log; tests stay quiet.
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

test {
    _ = @import("prompt_radix_test.zig");
}
