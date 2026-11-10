//! Prompt reuse as a radix tree over token pages: a request takes its longest match and resumes its deepest snapshot.

const std = @import("std");
const pc = @import("prompt_cache.zig");
const tree_mod = @import("prompt_radix_tree.zig");
const parts = @import("prompt_radix_parts.zig");
const make_mod = @import("prompt_radix_make.zig");
const Allocator = std.mem.Allocator;

pub const Saved = tree_mod.Saved;
pub const Snapshots = tree_mod.Snapshots;
pub const Pages = tree_mod.Pages;
pub const Counts = tree_mod.Counts;
pub const Node = tree_mod.Node;
pub const Entry = tree_mod.Entry;
pub const Tree = tree_mod.Tree;
pub const Groups = parts.Groups;

/// The rules of prompt_cache.zig, paged: `page` set; a planned store cuts and resumes only on `grid`, a multiple of it.
pub const Rules = pc.Rules;

/// What the tree may hold: pages, and snapshots.
pub const Limits = struct { pages: usize = 0, snaps: usize = 0 };

/// An owner's pages as the store holds them: `T` has retain, release, holders, available and maybe hold (kv_pool.Pool).
pub fn pagesOf(comptime T: type, owner: *T, bytes: u64) Pages {
    const V = struct {
        fn of(p: *anyopaque) *T {
            return @ptrCast(@alignCast(p));
        }
        fn retain(p: *anyopaque, id: u32) void {
            of(p).retain(id);
        }
        fn release(p: *anyopaque, id: u32) void {
            of(p).release(id);
        }
        fn holders(p: *anyopaque, id: u32) u32 {
            return of(p).holders(id);
        }
        fn available(p: *anyopaque) usize {
            return of(p).available();
        }
        fn hold(p: *anyopaque, out: []u32) bool {
            return of(p).hold(out);
        }
    };
    const has_hold = @hasDecl(T, "hold");
    return .{ .ptr = owner, .bytes = bytes, .vtable = &.{ .retain = V.retain, .release = V.release, .holders = V.holders, .available = V.available, .hold = if (has_hold) V.hold else null } };
}

/// Where a pass resumes and the marks it keeps at (in the caller's allocator), and its retained parts to map (C3).
pub const Plan = struct {
    from: u32 = 0,
    marks: []const u32 = &.{},
    /// Each retained group's whole pages, held once for the stream: it adopts each at `base + i` (or releases it).
    parts: []parts.Retained = &.{},
    /// Partial pages to copy into pages of the stream's promise; `finish` ends them.
    copies: []parts.Copy = &.{},

    /// The plan's slices back to `a` (the ids' holds are the stream's to adopt or release).
    pub fn deinit(p: Plan, a: Allocator) void {
        a.free(p.marks);
        for (p.parts) |r| a.free(r.ids);
        a.free(p.parts);
        a.free(p.copies);
    }
};
pub const Parts = parts;
pub const PartHooks = parts.PartHooks;
pub const Retained = parts.Retained;
pub const Copy = parts.Copy;

/// What a pass asks the store about: the words the tree compares, and what reuse must also match (enforced from C6).
pub const Query = struct {
    /// The tree's words: the request's cache key, one word a model row; for text, the prompt.
    key: []const u32,
    /// The prompt's length; a key of another length reuses nothing.
    prompt_len: u32,
    /// The exact identities of the request's images.
    images: []const @import("cache_images.zig").Identity = &.{},
    /// The request's image frame.
    frame: u32 = 0,
    /// What the rows depend on beyond key, images and frame.
    namespace: u64 = 0,
    /// The request's own chunk starts; empty: the store's grid.
    starts: []const u32 = &.{},
    /// Rows this pass prefills; null: the prompt's length.
    prefilled: ?u32 = null,

    /// A query of `key` alone, as long as its prompt.
    pub fn of(key: []const u32) Query {
        return .{ .key = key, .prompt_len = @intCast(key.len) };
    }
};
/// A paged store's `keep`: whether the mark holds a state, and how many leading ids the stream swaps for the store's.
pub const Kept = struct { held: bool, shared: usize };

pub const Store = struct {
    tree: Tree,
    rules: Rules,
    limits: Limits = .{},
    /// The tokens of recently planned shared cuts, hashed (at most SHARED_KEYS).
    shared_keys: std.ArrayList(u64) = .empty,
    /// The checkpoint groups (C3) and the family's hooks for them; none for a store of legacy snapshots.
    decl: make_mod.Decl = .{},
    /// Whether an entry holds a legacy snapshot.
    legacy: bool = true,

    const SHARED_KEYS = 64;

    pub fn init(gpa: Allocator, family: Snapshots, pages: Pages, rules: Rules) !Store {
        return initGroups(gpa, family, pages, rules, .{}, null);
    }

    /// `init` with the family's cache groups (C3, #2444) and its part hooks; `groups`' slices outlive the store.
    pub fn initGroups(gpa: Allocator, family: Snapshots, pages: Pages, rules: Rules, groups: Groups, hooks: ?PartHooks) !Store {
        if (rules.page == 0 or (rules.planned and (rules.grid == 0 or rules.grid % rules.page != 0))) return error.BadPromptRadixRules;
        for (groups.prefix_rows) |rows| if (rows != rules.page) return error.UnequalPrefixRows; // one key a page for all
        if (groups.parts.len > 0 and hooks == null) return error.PartsNeedHooks;
        if (groups.parts.len > 0 and pages.vtable.hold == null) return error.PartsNeedHold; // copies the store holds
        for (groups.parts) |spec| if (spec.make == .retained and spec.rows == 0) return error.RetainedRows;
        if (groups.none()) return error.PageOnlyStore; // a store with no checkpoint at all: not yet served
        return .{
            .tree = try Tree.init(gpa, family, pages, rules.page, rules.lookahead, groups.prefixCount()),
            .rules = rules,
            .decl = .{ .specs = groups.parts, .hooks = hooks },
            .legacy = groups.legacy,
        };
    }

    /// Page ids a logical page: one per prefix group.
    fn per(s: *const Store) usize {
        return s.tree.per;
    }

    /// Where a pass may cut and resume: a page edge, or a planned store's grid.
    fn step(s: *const Store) u32 {
        return if (s.rules.planned) s.rules.grid else s.rules.page;
    }

    /// Tokens a resume leaves past its state: the lookahead it read, and at least one to prefill.
    fn past(s: *const Store) u32 {
        return @max(1, s.rules.lookahead);
    }

    pub fn deinit(s: *Store) void {
        s.shared_keys.deinit(s.tree.gpa);
        s.tree.deinit();
    }

    /// The tree may hold `limits` from now on: what passes them goes first.
    pub fn limit(s: *Store, limits: Limits) void {
        s.limits = limits;
        while (s.tree.snaps > limits.snaps) s.tree.evict(s.victim(&.{}, null) orelse break);
        while (s.tree.heldPages() > limits.pages) if (!s.tree.evictLeaf()) break;
    }

    /// Whether anything can be kept.
    pub fn on(s: *const Store) bool {
        return s.limits.pages > 0 and s.limits.snaps > 0;
    }

    /// The deepest node whose state `prompt` resumes exactly: its tokens and lookahead a prefix, a token left.
    pub fn find(s: *const Store, prompt: []const u32) ?*Node {
        if (!s.on() or prompt.len < s.past() + 1) return null;
        return s.tree.deepest(prompt, (prompt.len - s.past()) / s.rules.page * s.rules.page, s.step());
    }

    /// Restores the longest state `prompt` resumes into `owner`, its pages into `adopt` (held once more); plans marks.
    pub fn begin(s: *Store, a: Allocator, q: Query, history_len: u32, shared: []const u32, owner: ?*anyopaque, adopt: *std.ArrayList(u32)) !Plan {
        const prompt = q.key;
        if (!s.on()) return .{ .marks = try a.alloc(u32, 0) };
        const found = s.find(prompt);
        if (found == null) s.tree.counts.misses += 1;
        const resume_at: u32 = if (found) |n| n.end else 0;
        const planned = try s.marks(a, prompt, resume_at, history_len, shared, if (found) |n| n.entry.?.last else &.{});
        const marks_ = try s.fitting(a, planned);
        errdefer a.free(marks_);
        for (shared) |w| { // the shared cuts this pass keeps: their states serve other conversations too
            const cut = w - w % s.step();
            if (std.mem.indexOfScalar(u32, marks_, cut) != null) s.noteShared(prompt[0 .. cut + s.rules.lookahead]);
        }
        s.reserve(prompt, found, marks_);
        var from: u32 = 0;
        var out: Plan = .{ .marks = marks_ };
        if (found) |n| {
            const ok = if (s.restore(owner, n)) |_| true else |err| blk: {
                note("restoring {d} tokens failed ({s}); prefilling from the start", .{ n.end, @errorName(err) });
                break :blk false;
            };
            if (ok) {
                const had = adopt.items.len;
                try s.tree.take(n, adopt);
                from = n.end;
                if (n.entry.?.parts.len > 0) {
                    const handed = s.decl.handOut(&s.tree, a, n.end, n.entry.?.parts) catch |err| {
                        for (adopt.items[had..]) |id| s.tree.pages.vtable.release(s.tree.pages.ptr, id); // this begin's
                        adopt.shrinkRetainingCapacity(had);
                        return err;
                    };
                    out.parts = handed.parts;
                    out.copies = handed.copies;
                }
            }
            s.resumed(n, prompt, ok);
        }
        out.from = from;
        return out;
    }

    /// The entry at `n` into `owner`, snapshot then copied parts; pinned, so reclaim cannot evict its own source.
    fn restore(s: *Store, owner: ?*anyopaque, n: *Node) !void {
        const e = &n.entry.?;
        n.pins += 1;
        defer n.pins -= 1;
        const family = s.tree.family;
        if (e.saved) |saved| try family.vtable.restore(family.ptr, owner, saved);
        try s.decl.restore(owner, e.parts);
    }

    /// A resume's copies are done (the stream mapped its own pages from them): each source's hold goes back.
    pub fn finish(s: *Store, owner: ?*anyopaque, copies: []const Copy) void {
        if (copies.len == 0) return;
        const h = s.decl.hooks.?;
        h.vtable.fence(h.ptr, owner); // the stream's copies out of the sources complete first
        for (copies) |c| s.tree.pages.vtable.release(s.tree.pages.ptr, c.src);
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
        e.branched = e.branched or superseded(n, e.born); // a turn several continuations share (re-sent, edited, forked)
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
            const new = (last / s.rules.page - s.tree.have(prompt[0 .. last + s.rules.lookahead])) * s.per();
            if (s.tree.heldPages() + new <= s.limits.pages and s.tree.snaps + marks_.len <= s.limits.snaps) return;
            s.tree.evict(s.victim(prompt, from) orelse return);
        }
    }

    /// A kept state a peer could not resume goes, so later prompts do not ask for it again.
    pub fn forget(s: *Store, prompt: []const u32, at: u32) void {
        const n = s.find(prompt[0..@min(prompt.len, at + s.past())]) orelse return;
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
            if (m / s.rules.page * s.per() <= s.limits.pages and s.limits.snaps > 0) try out.append(a, m) else {
                s.tree.counts.refused += 1;
                note("kept nothing at {d} tokens: {d} pages pass the limit of {d}", .{ m, m / s.rules.page * s.per(), s.limits.pages });
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
        if (s.rules.tail and prompt.len > s.past()) try want.append(a, @intCast(prompt.len - s.past()));
        for (want.items, 0..) |w, k| {
            const m = w - w % s.step();
            if (m <= from or m >= prompt.len or m + s.rules.lookahead > prompt.len) continue;
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

    /// At `at` the tree takes the ids of `mine` (page-major, `per` a page) it lacks, and the snapshot; `path` its ids.
    pub fn keep(s: *Store, q: Query, at: u32, owner: ?*anyopaque, mine: []const u32, path: []u32) Kept {
        return s.keepParts(q, at, owner, mine, &.{}, path);
    }

    /// `keep` with the retained groups' pages through `at` (one entry a declared part; copied ones empty).
    pub fn keepParts(s: *Store, q: Query, at: u32, owner: ?*anyopaque, mine: []const u32, retained: []const Retained, path: []u32) Kept {
        const prompt = q.key;
        const t = &s.tree;
        const np = at / s.rules.page; // logical pages below `at`
        if (!s.on() or at == 0 or at % s.step() != 0 or at + s.rules.lookahead > prompt.len or mine.len != np * s.per() or path.len != mine.len) return .{ .held = false, .shared = 0 };
        if (!s.decl.fits(at, retained)) return .{ .held = false, .shared = 0 };
        t.clock += 1;
        const tokens = prompt[0 .. at + s.rules.lookahead]; // a page past which the lookahead differs is another page
        const flagged = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(tokens)) != null;
        while (true) {
            var last: ?*Node = null;
            const got = t.heldPrefix(tokens, np, path, &last);
            if (got == np and last != null and last.?.end == at and last.?.entry != null) {
                const e = &last.?.entry.?;
                e.used = t.clock; // the same state again: no copy
                e.shared = e.shared or flagged;
                t.touch(last.?);
                return .{ .held = true, .shared = mine.len };
            }
            const ids = (np - got) * s.per() + s.decl.ids(t, at, retained); // every id the keep adds to the holds
            if (t.heldPages() + ids <= s.limits.pages and t.snaps + 1 <= s.limits.snaps) break;
            const v = s.victim(prompt, null) orelse {
                t.counts.refused += 1;
                note("kept nothing at {d} tokens: no room", .{at});
                return .{ .held = false, .shared = got * s.per() };
            };
            t.evict(v);
        }
        const node = t.insert(tokens, mine, path) catch |err| return s.fail(at, err, 0);
        node.pins += 1; // the parts' holds may reclaim: not this node, nor its path
        defer node.pins -= 1;
        // the path stays whatever follows; the entry is published only whole
        const made = s.decl.make(t, owner, at, retained) catch |err| return s.fail(at, err, mine.len);
        const family = t.family;
        const saved: ?Saved = if (s.legacy) family.vtable.save(family.ptr, owner, at) catch |err| {
            t.dropParts(made);
            return s.fail(at, err, mine.len);
        } else null;
        const last = t.gpa.dupe(u32, prompt) catch {
            if (saved) |sv| family.vtable.drop(family.ptr, sv);
            t.dropParts(made);
            return s.fail(at, error.OutOfMemory, mine.len);
        };
        const bytes = if (s.legacy) family.vtable.bytes(family.ptr, at) else 0;
        node.entry = .{ .saved = saved, .parts = made, .bytes = bytes, .born = @intCast(prompt.len), .used = t.clock, .last = last, .shared = flagged };
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

    /// Whether `prompt` repeats every token `n` holds: up to its end, and its lookahead.
    fn extends(n: *const Node, prompt: []const u32) bool {
        const parent = n.parent orelse return true;
        return prompt.len >= parent.end + n.tokens.len and std.mem.eql(u32, n.tokens, prompt[parent.end..][0..n.tokens.len]) and extends(parent, prompt);
    }

    /// Whether a later prompt kept a state below `n`.
    fn superseded(n: *const Node, born: u32) bool {
        for (n.kids.items) |k| {
            if ((k.entry != null and k.entry.?.born > born) or superseded(k, born)) return true;
        }
        return false;
    }

    fn pick(n: *Node, prompt: []const u32, skip: ?*Node, moved: *?*Node, oldest: *?*Node) void {
        if (n.entry) |e| if (n != skip and n.pins == 0) {
            // a state its conversation moved past goes first; a shared cut or a branch point only as the least valuable
            const gone = !e.shared and !e.branched and ((prompt.len > e.born and prompt.len > n.end and extends(n, prompt)) or superseded(n, e.born));
            if (gone and (moved.* == null or e.used < moved.*.?.entry.?.used)) moved.* = n;
            if (oldest.* == null or worse(n, oldest.*.?)) oldest.* = n;
        };
        for (n.kids.items) |k| pick(k, prompt, skip, moved, oldest);
    }

    /// The entry to free first, never `skip`: one a later prompt extends (no branch point), oldest first; else worst.
    fn victim(s: *const Store, prompt: []const u32, skip: ?*Node) ?*Node {
        var moved: ?*Node = null;
        var oldest: ?*Node = null;
        pick(s.tree.root, prompt, skip, &moved, &oldest);
        return moved orelse oldest;
    }

    /// Evicts until `want` pool pages are free and not promised; false when the tree cannot give that many.
    pub fn reclaim(s: *Store, want: usize) bool {
        const pages = s.tree.pages;
        if (s.decl.specs.len > 0) while (pages.vtable.available(pages.ptr) < want) { // parts first: their path may be in use
            s.tree.evict(s.partVictim() orelse break);
        };
        while (pages.vtable.available(pages.ptr) < want) if (!s.tree.evictLeaf()) return false;
        return true;
    }

    /// The least valuable entry with a part page only the store holds, never a pinned one.
    fn partVictim(s: *const Store) ?*Node {
        var best: ?*Node = null;
        s.pickParts(s.tree.root, &best);
        return best;
    }

    fn pickParts(s: *const Store, n: *Node, best: *?*Node) void {
        if (n.entry) |e| if (n.pins == 0 and s.freesPart(e) and (best.* == null or worse(n, best.*.?))) {
            best.* = n;
        };
        for (n.kids.items) |k| s.pickParts(k, best);
    }

    fn freesPart(s: *const Store, e: Entry) bool {
        const pages = s.tree.pages;
        for (e.parts) |part| for (part.ids) |id| if (pages.vtable.holders(pages.ptr, id) == 1) return true;
        return false;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the tree holds.
    pub fn report(s: *const Store, prompt: usize, from: u32, kept: u64) void {
        if (@import("builtin").is_test) return;
        const t = &s.tree;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states and {d} pages ({d} MiB) of {d} states and {d} pages (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, t.snaps, t.heldPages(), t.held() >> 20, s.limits.snaps, s.limits.pages, t.counts.hits, t.counts.misses, t.counts.evicted, t.counts.refused, t.counts.failed });
    }
};

/// A line on the engine's log; tests stay quiet.
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

test {
    _ = @import("prompt_radix_test.zig");
    _ = @import("prompt_radix_parts.zig");
    _ = @import("prompt_radix_groups_test.zig");
}
