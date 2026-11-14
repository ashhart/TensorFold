//! The radix store's eviction order: which kept state goes first, an entry at a node's end or one of its tails.
const std = @import("std");
const tree_mod = @import("prompt_radix_tree.zig");

const Tree = tree_mod.Tree;
const Node = tree_mod.Node;
const Entry = tree_mod.Entry;
const Found = tree_mod.Found;
const Key = tree_mod.Key;

/// Whether `x` is a less valuable state than `y`: one never resumed before one that was, then the older.
fn worse(x: *const Entry, y: *const Entry) bool {
    if (x.hit != y.hit) return !x.hit;
    return x.used < y.used;
}

/// Whether `k` repeats every word `n` and its parents hold, with their images and namespace: one conversation.
fn extends(n: *const Node, k: Key) bool {
    const parent = n.parent orelse return true;
    return Tree.repeats(n, k) and extends(parent, k); // its words, images and namespace: the same conversation
}

/// Whether a later prompt kept a state below `n`.
pub fn superseded(n: *const Node, born: u32) bool {
    for (n.kids.items) |k| {
        if (k.entry != null and k.entry.?.born > born) return true;
        for (k.tails.items) |e| if (e.born > born) return true;
        if (superseded(k, born)) return true;
    }
    return false;
}

/// Whether a later prompt kept a state past `f`: below its node, or for a tail only where it repeats its key.
pub fn supersededAt(f: Found, born: u32) bool {
    const n = f.node;
    const own = n.tokens[n.end - (if (n.parent) |p| p.end else 0) ..]; // the `ahead` words past its end (none at the root)
    const tl = f.entry().tail orelse {
        for (n.tails.items) |o| if (o.born > born and std.mem.startsWith(u32, o.tail.?.key, own)) return true; // inside the next page
        return superseded(n, born);
    };
    for (n.tails.items) |o| if (o.tail.?.r > tl.r and o.born > born and repeats(tl, n.end, o.tail.?.key, o.tail.?.images, o.tail.?.ns)) return true;
    for (n.kids.items) |k| {
        if (!repeats(tl, n.end, k.tokens, k.images, k.ns)) continue;
        if ((k.entry != null and k.entry.?.born > born) or superseded(k, born)) return true;
        for (k.tails.items) |o| if (o.born > born) return true;
    }
    return false;
}

/// Whether words from `at` (and their images and Ns) repeat tail `tl`'s key: its rows and the `ahead` words past them.
fn repeats(tl: tree_mod.Tail, at: usize, words: []const u32, images: []const tree_mod.Identity, ns: tree_mod.Ns) bool {
    if (!std.meta.eql(ns, tl.ns) or !std.mem.startsWith(u32, words, tl.key)) return false;
    return tree_mod.sameImages(images, tl.images, at, at + tl.key.len);
}

/// Each state of `n` in order, its end's entry first, then its tails.
fn each(n: *Node, ctx: anytype, comptime f: fn (@TypeOf(ctx), Found) void) void {
    if (n.entry != null) f(ctx, .{ .node = n });
    for (0..n.tails.items.len) |i| f(ctx, .{ .node = n, .tail = i });
}

const Pick = struct {
    k: Key,
    moved: ?Found = null,
    oldest: ?Found = null,

    fn consider(p: *Pick, f: Found) void {
        const e = f.entry();
        const prompt = p.k.tokens;
        const along = if (e.tail) |tl| tl.matches(f.node.end, p.k) else true; // a tail: its own words too
        // a state its conversation moved past goes first; a shared cut or a branch point only as the least valuable
        const gone = !e.shared and !e.branched and ((prompt.len > e.born and prompt.len > f.at() and along and extends(f.node, p.k)) or supersededAt(f, e.born));
        if (gone and (p.moved == null or e.used < p.moved.?.entry().used)) p.moved = f;
        if (p.oldest == null or worse(e, p.oldest.?.entry())) p.oldest = f;
    }
};

fn pick(n: *Node, p: *Pick, skip: ?*Node) void {
    if (n != skip and n.pins == 0) each(n, p, Pick.consider);
    for (n.kids.items) |kid| pick(kid, p, skip);
}

/// The state to free first, never at `skip`: one a later prompt extends (no branch point), oldest first; else worst.
pub fn victim(t: *const Tree, k: Key, skip: ?*Node) ?Found {
    var p: Pick = .{ .k = k };
    pick(t.root, &p, skip);
    return p.moved orelse p.oldest;
}

const Parts = struct {
    t: *const Tree,
    best: ?Found = null,

    fn consider(p: *Parts, f: Found) void {
        const e = f.entry();
        if (p.frees(e) and (p.best == null or worse(e, p.best.?.entry()))) p.best = f;
    }

    /// Whether `e` has a part or tail page only the store holds.
    fn frees(p: *const Parts, e: *const Entry) bool {
        const pages = p.t.pages;
        for (e.parts) |part| for (part.ids) |id| if (pages.vtable.holders(pages.ptr, id) == 1) return true;
        if (e.tail) |tl| for (tl.ids) |id| if (pages.vtable.holders(pages.ptr, id) == 1) return true;
        return false;
    }
};

fn pickParts(n: *Node, p: *Parts) void {
    if (n.pins == 0) each(n, p, Parts.consider);
    for (n.kids.items) |k| pickParts(k, p);
}

/// The least valuable state with a part or tail page only the store holds, never a pinned one.
pub fn partVictim(t: *const Tree) ?Found {
    var p: Parts = .{ .t = t };
    pickParts(t.root, &p);
    return p.best;
}
