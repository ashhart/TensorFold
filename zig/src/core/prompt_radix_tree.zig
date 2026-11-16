//! The paged prompt cache's tree: nodes are runs of whole pages keyed by tokens, each may end in a family snapshot.

const std = @import("std");
const Allocator = std.mem.Allocator;

const pc = @import("prompt_cache.zig");
const parts = @import("prompt_radix_parts.zig");

pub const Saved = pc.Saved;
pub const Snapshots = pc.Snapshots;
pub const Pages = pc.Pages;
pub const Counts = pc.Counts;
pub const Identity = @import("cache_images.zig").Identity;

/// What rows depend on beyond their key and images: a tree never shares a page between two of these.
pub const Ns = struct { namespace: u64 = 0, frame: u32 = 0 };

/// What a walk matches: the words from the prompt's start, the images over them in prompt order, and their Ns.
pub const Key = struct {
    tokens: []const u32,
    images: []const Identity = &.{},
    ns: Ns = .{},
};

/// The images whose rows overlap [lo, hi), of a list in prompt order.
fn overlapping(ids: []const Identity, lo: usize, hi: usize) []const Identity {
    var a: usize = 0;
    while (a < ids.len and ids[a].at + ids[a].tokens <= lo) a += 1;
    var b = a;
    while (b < ids.len and ids[b].at < hi) b += 1;
    return ids[a..b];
}

/// Whether the same images, exactly, overlap [lo, hi) in both lists.
fn sameImages(x: []const Identity, y: []const Identity, lo: usize, hi: usize) bool {
    const a = overlapping(x, lo, hi);
    const b = overlapping(y, lo, hi);
    if (a.len != b.len) return false;
    for (a, b) |i, j| if (!std.meta.eql(i, j)) return false;
    return true;
}

/// The state kept at a node's end.
pub const Entry = struct {
    /// The family's legacy snapshot; null when it declares none (its state is in `parts`).
    saved: ?Saved,
    /// The checkpoint groups' pages (C3), each id named in the tree's `holds`.
    parts: []parts.Part = &.{},
    bytes: u64,
    /// The length of the prompt that kept it: a later turn's is longer.
    born: u32,
    /// The store's clock at its last keep or resume.
    used: u64,
    /// The last prompt that kept or resumed it.
    last: []u32,
    /// Kept at a shared system block's cut: other conversations resume it, so its own never supersedes it.
    shared: bool,
    /// A later prompt resumed from it: it outlives the entries nobody has asked for.
    hit: bool = false,
    /// Resumed again after a later prompt kept a state below it: a turn several continuations share, never superseded.
    branched: bool = false,
};

/// A matched position: `pages` logical pages into `node` (1 to its count).
pub const Hit = struct {
    node: *Node,
    pages: usize,

    /// The tokens below the position.
    pub fn at(h: Hit, t: *const Tree) u32 {
        return h.node.end - @as(u32, @intCast((t.npages(h.node) - h.pages) * t.page));
    }
};

pub const Node = struct {
    parent: ?*Node,
    /// Tokens up to the end of this node.
    end: u32,
    /// The tokens of this node's pages, a page's worth each, and the tree's `ahead` more past its last page.
    tokens: []u32,
    pages: []u32,
    /// The images overlapping its rows and `ahead` words, at prompt positions; empty for text.
    images: []Identity = &.{},
    /// The Ns its rows were computed under.
    ns: Ns = .{},
    kids: std.ArrayList(*Node) = .empty,
    entry: ?Entry = null,
    used: u64 = 0,
    /// A keep or a restore in progress here: no eviction or prune takes this node, its entry or its path.
    pins: u32 = 0,
};

pub const Tree = struct {
    gpa: Allocator,
    family: Snapshots,
    pages: Pages,
    /// Tokens a page holds.
    page: usize,
    /// Prefix groups: page ids a logical page, page-major on each node (`pages[p * per + g]`).
    per: usize = 1,
    /// Tokens past a page that its rows read (the family's lookahead): a page matches only when they match too.
    ahead: usize,
    root: *Node,
    clock: u64 = 0,
    /// The pages the tree holds, one pool reference each however many nodes name them.
    holds: parts.Holds = .{},
    /// Snapshots the tree holds.
    snaps: usize = 0,
    counts: Counts = .{},

    pub fn init(gpa: Allocator, family: Snapshots, pages: Pages, page: usize, ahead: usize, per: usize) !Tree {
        const root = try gpa.create(Node);
        root.* = .{ .parent = null, .end = 0, .tokens = &.{}, .pages = &.{} };
        return .{ .gpa = gpa, .family = family, .pages = pages, .page = page, .per = per, .ahead = ahead, .root = root };
    }

    /// Logical pages of `n`: its ids over the prefix groups.
    pub fn npages(t: *const Tree, n: *const Node) usize {
        return n.pages.len / t.per;
    }

    pub fn deinit(t: *Tree) void {
        t.drop(t.root);
        t.holds.deinit(t.gpa);
    }

    /// Distinct pages the tree holds: what its page limit counts.
    pub fn heldPages(t: *const Tree) usize {
        return t.holds.count();
    }

    fn drop(t: *Tree, n: *Node) void {
        for (n.kids.items) |k| t.drop(k);
        n.kids.deinit(t.gpa);
        if (n.entry) |e| t.dropEntry(e);
        for (n.pages) |id| t.holds.drop(t.pages, id);
        t.gpa.free(n.tokens);
        t.gpa.free(n.pages);
        t.gpa.free(n.images);
        t.gpa.destroy(n);
    }

    pub fn touch(t: *Tree, n: *Node) void {
        t.clock += 1;
        n.used = t.clock;
    }

    /// Whether `k` repeats every word `n` holds, under its Ns and with its images (its parent's not checked).
    pub fn repeats(n: *const Node, k: Key) bool {
        const start = if (n.parent) |p| p.end else 0;
        if (!std.meta.eql(n.ns, k.ns) or k.tokens.len < start + n.tokens.len) return false;
        if (!std.mem.eql(u32, n.tokens, k.tokens[start..][0..n.tokens.len])) return false;
        return sameImages(n.images, k.images, start, start + n.tokens.len);
    }

    /// The kid of `n` whose first page `k` repeats from `at`: its words, `ahead` more, its images, its Ns.
    fn child(t: *const Tree, n: *const Node, k: Key, at: usize) ?*Node {
        const w = t.page + t.ahead;
        for (n.kids.items) |c| {
            if (!std.meta.eql(c.ns, k.ns) or !std.mem.eql(u32, c.tokens[0..w], k.tokens[at..][0..w])) continue;
            if (sameImages(c.images, k.images, at, at + w)) return c;
        }
        return null;
    }

    /// Whole pages of `n` (from `at`) that `k` repeats, words, `ahead` more and images each, at most `most`.
    fn common(t: *const Tree, n: *const Node, k: Key, at: usize, most: usize) usize {
        const w = t.page + t.ahead;
        var p: usize = 0;
        while (p < t.npages(n) and p < most) : (p += 1) {
            const lo = at + p * t.page;
            if (!std.mem.eql(u32, n.tokens[p * t.page ..][0..w], k.tokens[lo..][0..w])) break;
            if (!sameImages(n.images, k.images, lo, lo + w)) break;
        }
        return p;
    }

    /// The deepest node with a state at a resume point (`ok.ok(end)`) that `prompt` repeats up to `limit_at`.
    pub fn deepest(t: *const Tree, k: Key, limit_at: usize, ok: anytype) ?*Node {
        var node = t.root;
        var at: usize = 0;
        var best: ?*Node = null;
        while (at + t.page <= limit_at) {
            const c = t.child(node, k, at) orelse break;
            const p = t.common(c, k, at, (limit_at - at) / t.page);
            if (p < t.npages(c)) break;
            at += p * t.page;
            node = c;
            if (c.entry != null and ok.ok(c.end)) best = c;
        }
        return best;
    }

    /// The deepest page edge `prompt` repeats up to `limit_at`, inside a node too: where a page-only store resumes.
    pub fn deepestPage(t: *const Tree, k: Key, limit_at: usize, ok: anytype) ?Hit {
        var node = t.root;
        var at: usize = 0;
        var best: ?Hit = null;
        while (at + t.page <= limit_at) {
            const c = t.child(node, k, at) orelse break;
            const p = t.common(c, k, at, (limit_at - at) / t.page);
            if (p == 0) break;
            var e = p; // the deepest of its matched page edges that is a resume point
            while (e > 0 and !ok.ok(@intCast(at + e * t.page))) e -= 1;
            if (e > 0) best = .{ .node = c, .pages = e };
            if (p < t.npages(c)) break;
            at += p * t.page;
            node = c;
        }
        return best;
    }

    /// The ids of the path to `h`, each held once more for the caller; the nodes marked used.
    pub fn takeTo(t: *Tree, h: Hit, adopt: *std.ArrayList(u32)) !void {
        var path: std.ArrayList(*Node) = .empty;
        defer path.deinit(t.gpa);
        var w: ?*Node = h.node.parent;
        while (w) |x| : (w = x.parent) if (x.parent != null) try path.append(t.gpa, x);
        try adopt.ensureUnusedCapacity(t.gpa, h.at(t) / t.page * t.per);
        while (path.pop()) |x| t.hand(x, x.pages, adopt);
        t.hand(h.node, h.node.pages[0 .. h.pages * t.per], adopt);
    }

    fn hand(t: *Tree, n: *Node, ids: []const u32, adopt: *std.ArrayList(u32)) void {
        for (ids) |id| {
            t.pages.vtable.retain(t.pages.ptr, id);
            adopt.appendAssumeCapacity(id);
        }
        t.touch(n);
    }

    /// Logical pages of `tokens` (`np` whole pages and `ahead` more) the tree holds, their ids into `path` if any.
    pub fn heldPrefix(t: *const Tree, k: Key, np: usize, path: ?[]u32, last: *?*Node) usize {
        var node = t.root;
        var pi: usize = 0;
        while (pi < np) {
            const c = t.child(node, k, pi * t.page) orelse break;
            const p = t.common(c, k, pi * t.page, np - pi);
            if (path) |out| @memcpy(out[pi * t.per ..][0 .. p * t.per], c.pages[0 .. p * t.per]);
            pi += p;
            node = c;
            if (p < t.npages(c)) break;
        }
        last.* = if (node == t.root) null else node;
        return pi;
    }

    /// Pages of `k` (whole pages, then `ahead` words) the tree holds from the start.
    pub fn have(t: *const Tree, k: Key) usize {
        var last: ?*Node = null;
        return t.heldPrefix(k, (k.tokens.len -| t.ahead) / t.page, null, &last);
    }

    /// The pages of the path to `n`, each held once more for the caller; the nodes marked used.
    pub fn take(t: *Tree, n: *Node, adopt: *std.ArrayList(u32)) !void {
        var path: std.ArrayList(*Node) = .empty;
        defer path.deinit(t.gpa);
        var w: ?*Node = n;
        while (w) |x| : (w = x.parent) if (x.parent != null) try path.append(t.gpa, x);
        try adopt.ensureUnusedCapacity(t.gpa, n.end / t.page * t.per);
        while (path.pop()) |x| {
            for (x.pages) |id| {
                t.pages.vtable.retain(t.pages.ptr, id);
                adopt.appendAssumeCapacity(id);
            }
            t.touch(x);
        }
    }

    /// Puts `tokens` (whole pages, `mine` the caller's ids page-major, `ahead` more) in the tree; `path` gets its ids.
    pub fn insert(t: *Tree, k: Key, mine: []const u32, path: []u32) !*Node {
        const tokens = k.tokens;
        const np = mine.len / t.per;
        var node = t.root;
        var pi: usize = 0;
        while (pi < np) {
            const c = t.child(node, k, pi * t.page) orelse {
                const leaf = try t.gpa.create(Node);
                errdefer t.gpa.destroy(leaf);
                const own_tokens = try t.gpa.dupe(u32, tokens[pi * t.page ..]);
                errdefer t.gpa.free(own_tokens);
                const own_pages = try t.gpa.dupe(u32, mine[pi * t.per ..]);
                errdefer t.gpa.free(own_pages);
                const own_images = try t.gpa.dupe(Identity, overlapping(k.images, pi * t.page, tokens.len));
                errdefer t.gpa.free(own_images);
                leaf.* = .{ .parent = node, .end = @intCast(np * t.page), .tokens = own_tokens, .pages = own_pages, .images = own_images, .ns = k.ns };
                try t.holds.reserve(t.gpa, leaf.pages.len); // before the leaf is linked: what follows cannot fail
                try node.kids.append(t.gpa, leaf);
                for (leaf.pages) |id| t.holds.shareAssumeCapacity(t.pages, id);
                @memcpy(path[pi * t.per ..], leaf.pages);
                t.touch(leaf);
                return leaf;
            };
            const p = t.common(c, k, pi * t.page, np - pi);
            const up = if (p < t.npages(c)) try t.split(c, p) else c;
            @memcpy(path[pi * t.per ..][0 .. p * t.per], up.pages);
            t.touch(up);
            node = up;
            pi += p;
        }
        return node;
    }

    /// Splits `c` after its first `p` pages: a node of those pages now stands above it.
    fn split(t: *Tree, c: *Node, p: usize) !*Node {
        const up = try t.gpa.create(Node);
        errdefer t.gpa.destroy(up);
        const head_tokens = try t.gpa.dupe(u32, c.tokens[0 .. p * t.page + t.ahead]); // its pages, the tail's first `ahead`
        errdefer t.gpa.free(head_tokens);
        const head_pages = try t.gpa.dupe(u32, c.pages[0 .. p * t.per]);
        errdefer t.gpa.free(head_pages);
        const tail_tokens = try t.gpa.dupe(u32, c.tokens[p * t.page ..]);
        errdefer t.gpa.free(tail_tokens);
        const tail_pages = try t.gpa.dupe(u32, c.pages[p * t.per ..]);
        errdefer t.gpa.free(tail_pages);
        const parent = c.parent.?;
        const cut = parent.end + p * t.page; // each side keeps the images over its rows and `ahead` words
        const head_images = try t.gpa.dupe(Identity, overlapping(c.images, parent.end, cut + t.ahead));
        errdefer t.gpa.free(head_images);
        const tail_images = try t.gpa.dupe(Identity, overlapping(c.images, cut, c.end + t.ahead));
        errdefer t.gpa.free(tail_images);
        up.* = .{ .parent = parent, .end = @intCast(cut), .tokens = head_tokens, .pages = head_pages, .images = head_images, .ns = c.ns, .used = c.used };
        try up.kids.append(t.gpa, c);
        errdefer up.kids.deinit(t.gpa);
        const slot = std.mem.indexOfScalar(*Node, parent.kids.items, c).?;
        parent.kids.items[slot] = up;
        t.gpa.free(c.tokens);
        t.gpa.free(c.pages);
        t.gpa.free(c.images);
        c.tokens = tail_tokens;
        c.pages = tail_pages;
        c.images = tail_images;
        c.parent = up;
        return up;
    }

    /// An entry's snapshot, parts and prompt go.
    fn dropEntry(t: *Tree, e: Entry) void {
        if (e.saved) |saved| t.family.vtable.drop(t.family.ptr, saved);
        t.dropParts(e.parts);
        t.gpa.free(e.last);
    }

    /// Parts made by a keep go back: each id's name in `holds`, then the slices.
    pub fn dropParts(t: *Tree, ps: []parts.Part) void {
        t.dropEach(ps);
        t.gpa.free(ps);
    }

    /// Each part's ids go back and its id slice is freed; the caller frees `ps` itself.
    pub fn dropEach(t: *Tree, ps: []const parts.Part) void {
        for (ps) |part| {
            for (part.ids) |id| t.holds.drop(t.pages, id);
            t.gpa.free(part.ids);
        }
    }

    /// Frees the state at `n`, and the nodes below nothing keeps.
    pub fn evict(t: *Tree, n: *Node) void {
        const e = n.entry orelse return;
        std.debug.assert(n.pins == 0); // a keep or a restore in progress
        t.dropEntry(e);
        n.entry = null;
        t.snaps -= 1;
        t.counts.evicted += 1;
        t.prune(n);
    }

    /// Removes `start` and its parents while they hold no state and have nothing below them.
    pub fn prune(t: *Tree, start: *Node) void {
        var n = start;
        while (n.parent) |parent| {
            if (n.kids.items.len > 0 or n.entry != null or n.pins > 0) return;
            const slot = std.mem.indexOfScalar(*Node, parent.kids.items, n).?;
            _ = parent.kids.swapRemove(slot);
            t.drop(n);
            n = parent;
        }
    }

    /// Whether nothing but the tree holds the pages of `n`.
    fn idle(t: *const Tree, n: *const Node) bool {
        for (n.pages) |id| if (t.pages.vtable.holders(t.pages.ptr, id) != 1) return false;
        return true;
    }

    /// Whether leaf `a` is less valuable than `b`: never resumed before resumed, then the less recently used.
    fn lessValuable(a: *const Node, b: *const Node) bool {
        const hit_a = if (a.entry) |e| e.hit else false;
        const hit_b = if (b.entry) |e| e.hit else false;
        if (hit_a != hit_b) return !hit_a;
        return a.used < b.used;
    }

    fn weakestLeaf(t: *const Tree, n: *Node, best: *?*Node) void {
        if (n.parent != null and n.kids.items.len == 0 and n.pins == 0 and t.idle(n) and (best.* == null or lessValuable(n, best.*.?))) best.* = n;
        for (n.kids.items) |k| t.weakestLeaf(k, best);
    }

    /// Evicts the least valuable leaf nobody holds; false when there is none.
    pub fn evictLeaf(t: *Tree) bool {
        var best: ?*Node = null;
        t.weakestLeaf(t.root, &best);
        const n = best orelse return false;
        if (n.entry != null) t.evict(n) else t.prune(n);
        return true;
    }

    /// Bytes the tree holds: its pages and its snapshots.
    pub fn held(t: *const Tree) u64 {
        var n: u64 = t.heldPages() * t.pages.bytes;
        sum(t.root, &n);
        return n;
    }

    fn sum(n: *const Node, out: *u64) void {
        if (n.entry) |e| out.* += e.bytes;
        for (n.kids.items) |k| sum(k, out);
    }
};
