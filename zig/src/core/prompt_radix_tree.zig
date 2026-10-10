//! The paged prompt cache's tree: nodes are runs of whole pages keyed by tokens, each may end in a family snapshot.

const std = @import("std");
const Allocator = std.mem.Allocator;

const pc = @import("prompt_cache.zig");

pub const Saved = pc.Saved;
pub const Snapshots = pc.Snapshots;
pub const Pages = pc.Pages;
pub const Counts = pc.Counts;

/// The state kept at a node's end.
pub const Entry = struct {
    saved: Saved,
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
};

pub const Node = struct {
    parent: ?*Node,
    /// Tokens up to the end of this node.
    end: u32,
    /// The tokens of this node's pages, a page's worth each.
    tokens: []u32,
    pages: []u32,
    kids: std.ArrayList(*Node) = .empty,
    entry: ?Entry = null,
    used: u64 = 0,
};

pub const Tree = struct {
    gpa: Allocator,
    family: Snapshots,
    pages: Pages,
    /// Tokens a page holds.
    page: usize,
    root: *Node,
    clock: u64 = 0,
    /// Pages and snapshots the tree holds.
    held_pages: usize = 0,
    snaps: usize = 0,
    counts: Counts = .{},

    pub fn init(gpa: Allocator, family: Snapshots, pages: Pages, page: usize) !Tree {
        const root = try gpa.create(Node);
        root.* = .{ .parent = null, .end = 0, .tokens = &.{}, .pages = &.{} };
        return .{ .gpa = gpa, .family = family, .pages = pages, .page = page, .root = root };
    }

    pub fn deinit(t: *Tree) void {
        t.drop(t.root);
    }

    fn drop(t: *Tree, n: *Node) void {
        for (n.kids.items) |k| t.drop(k);
        n.kids.deinit(t.gpa);
        if (n.entry) |e| {
            t.family.vtable.drop(t.family.ptr, e.saved);
            t.gpa.free(e.last);
        }
        for (n.pages) |id| t.pages.vtable.release(t.pages.ptr, id);
        t.gpa.free(n.tokens);
        t.gpa.free(n.pages);
        t.gpa.destroy(n);
    }

    pub fn touch(t: *Tree, n: *Node) void {
        t.clock += 1;
        n.used = t.clock;
    }

    fn child(t: *const Tree, n: *const Node, tokens: []const u32) ?*Node {
        for (n.kids.items) |k| if (std.mem.eql(u32, k.tokens[0..t.page], tokens[0..t.page])) return k;
        return null;
    }

    /// Whole pages of `n` that `tokens` repeats, at most `most`.
    fn common(t: *const Tree, n: *const Node, tokens: []const u32, most: usize) usize {
        var p: usize = 0;
        while (p < n.pages.len and p < most and std.mem.eql(u32, n.tokens[p * t.page ..][0..t.page], tokens[p * t.page ..][0..t.page])) p += 1;
        return p;
    }

    /// The deepest node with a state that `prompt` repeats whole up to `limit_at` (a page edge).
    pub fn deepest(t: *const Tree, prompt: []const u32, limit_at: usize) ?*Node {
        var node = t.root;
        var at: usize = 0;
        var best: ?*Node = null;
        while (at + t.page <= limit_at) {
            const c = t.child(node, prompt[at..]) orelse break;
            const p = t.common(c, prompt[at..], (limit_at - at) / t.page);
            if (p < c.pages.len) break;
            at += p * t.page;
            node = c;
            if (c.entry != null) best = c;
        }
        return best;
    }

    /// Pages of `tokens` (`np` whole pages) the tree holds from the start, into `path` if any; `last` the end node.
    pub fn heldPrefix(t: *const Tree, tokens: []const u32, np: usize, path: ?[]u32, last: *?*Node) usize {
        var node = t.root;
        var pi: usize = 0;
        while (pi < np) {
            const c = t.child(node, tokens[pi * t.page ..]) orelse break;
            const p = t.common(c, tokens[pi * t.page ..], np - pi);
            if (path) |out| @memcpy(out[pi..][0..p], c.pages[0..p]);
            pi += p;
            node = c;
            if (p < c.pages.len) break;
        }
        last.* = if (node == t.root) null else node;
        return pi;
    }

    /// Pages of `tokens` (whole pages) the tree holds from the start.
    pub fn have(t: *const Tree, tokens: []const u32) usize {
        var last: ?*Node = null;
        return t.heldPrefix(tokens, tokens.len / t.page, null, &last);
    }

    /// The pages of the path to `n`, each held once more for the caller; the nodes marked used.
    pub fn take(t: *Tree, n: *Node, adopt: *std.ArrayList(u32)) !void {
        var path: std.ArrayList(*Node) = .empty;
        defer path.deinit(t.gpa);
        var w: ?*Node = n;
        while (w) |x| : (w = x.parent) if (x.parent != null) try path.append(t.gpa, x);
        try adopt.ensureUnusedCapacity(t.gpa, n.end / t.page);
        while (path.pop()) |x| {
            for (x.pages) |id| {
                t.pages.vtable.retain(t.pages.ptr, id);
                adopt.appendAssumeCapacity(id);
            }
            t.touch(x);
        }
    }

    /// Puts `tokens` (whole pages, `mine` the caller's) in the tree, returns the end node; `path` gets the tree's page.
    pub fn insert(t: *Tree, tokens: []const u32, mine: []const u32, path: []u32) !*Node {
        const np = mine.len;
        var node = t.root;
        var pi: usize = 0;
        while (pi < np) {
            const c = t.child(node, tokens[pi * t.page ..]) orelse {
                const leaf = try t.gpa.create(Node);
                errdefer t.gpa.destroy(leaf);
                leaf.* = .{ .parent = node, .end = @intCast(np * t.page), .tokens = try t.gpa.dupe(u32, tokens[pi * t.page ..]), .pages = try t.gpa.dupe(u32, mine[pi..]) };
                errdefer t.gpa.free(leaf.tokens);
                errdefer t.gpa.free(leaf.pages);
                try node.kids.append(t.gpa, leaf);
                for (leaf.pages) |id| t.pages.vtable.retain(t.pages.ptr, id);
                @memcpy(path[pi..], leaf.pages);
                t.held_pages += leaf.pages.len;
                t.touch(leaf);
                return leaf;
            };
            const p = t.common(c, tokens[pi * t.page ..], np - pi);
            const up = if (p < c.pages.len) try t.split(c, p) else c;
            @memcpy(path[pi..][0..p], up.pages);
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
        const head_tokens = try t.gpa.dupe(u32, c.tokens[0 .. p * t.page]);
        errdefer t.gpa.free(head_tokens);
        const head_pages = try t.gpa.dupe(u32, c.pages[0..p]);
        errdefer t.gpa.free(head_pages);
        const tail_tokens = try t.gpa.dupe(u32, c.tokens[p * t.page ..]);
        errdefer t.gpa.free(tail_tokens);
        const tail_pages = try t.gpa.dupe(u32, c.pages[p..]);
        errdefer t.gpa.free(tail_pages);
        const parent = c.parent.?;
        up.* = .{ .parent = parent, .end = parent.end + @as(u32, @intCast(p * t.page)), .tokens = head_tokens, .pages = head_pages, .used = c.used };
        try up.kids.append(t.gpa, c);
        errdefer up.kids.deinit(t.gpa);
        const slot = std.mem.indexOfScalar(*Node, parent.kids.items, c).?;
        parent.kids.items[slot] = up;
        t.gpa.free(c.tokens);
        t.gpa.free(c.pages);
        c.tokens = tail_tokens;
        c.pages = tail_pages;
        c.parent = up;
        return up;
    }

    /// Frees the state at `n`, and the nodes below nothing keeps.
    pub fn evict(t: *Tree, n: *Node) void {
        const e = n.entry orelse return;
        t.family.vtable.drop(t.family.ptr, e.saved);
        t.gpa.free(e.last);
        n.entry = null;
        t.snaps -= 1;
        t.counts.evicted += 1;
        t.prune(n);
    }

    /// Removes `start` and its parents while they hold no state and have nothing below them.
    pub fn prune(t: *Tree, start: *Node) void {
        var n = start;
        while (n.parent) |parent| {
            if (n.kids.items.len > 0 or n.entry != null) return;
            const slot = std.mem.indexOfScalar(*Node, parent.kids.items, n).?;
            _ = parent.kids.swapRemove(slot);
            t.held_pages -= n.pages.len;
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
        if (n.parent != null and n.kids.items.len == 0 and t.idle(n) and (best.* == null or lessValuable(n, best.*.?))) best.* = n;
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
        var n: u64 = t.held_pages * t.pages.bytes;
        sum(t.root, &n);
        return n;
    }

    fn sum(n: *const Node, out: *u64) void {
        if (n.entry) |e| out.* += e.bytes;
        for (n.kids.items) |k| sum(k, out);
    }
};
