//! Writable tails (C4, #2444): a checkpoint inside a page keeps that page's valid rows as copies the store holds.
const std = @import("std");
const parts = @import("prompt_radix_parts.zig");
const tree_mod = @import("prompt_radix_tree.zig");

const Tree = tree_mod.Tree;
const Key = tree_mod.Key;
const Ns = tree_mod.Ns;
const Identity = tree_mod.Identity;

/// A checkpoint's rows past its page edge: what it matches, and its partial page as one held copy a prefix group.
pub const Tail = struct {
    /// Rows past the edge (1 to a page less one).
    r: u32,
    /// The words past the edge: its `r` rows and the tree's `ahead` more.
    key: []u32,
    /// The images overlapping them, at prompt positions.
    images: []Identity,
    /// The Ns its rows were computed under.
    ns: Ns,
    /// Page-major like a node's: one held copy a prefix group, its first `r` rows valid; never written again.
    ids: []u32,

    /// Whether `k` repeats this tail past `edge`: its words, the `ahead` more, its images and its Ns.
    pub fn matches(tl: Tail, edge: usize, k: Key) bool {
        if (!std.meta.eql(tl.ns, k.ns) or k.tokens.len < edge + tl.key.len) return false;
        if (!std.mem.eql(u32, tl.key, k.tokens[edge..][0..tl.key.len])) return false;
        return tree_mod.sameImages(tl.images, k.images, edge, edge + tl.key.len);
    }
};

/// The tail of a keep `r` rows past `edge`: a held page a prefix group, rows [0, r) copied from `partial`; fenced.
pub fn make(t: *Tree, hooks: parts.PartHooks, owner: ?*anyopaque, first: u32, k: Key, edge: usize, r: u32, partial: []const u32) !Tail {
    std.debug.assert(partial.len == t.per and r > 0 and r < t.page);
    const key = try t.gpa.dupe(u32, k.tokens[edge..][0 .. r + t.ahead]);
    errdefer t.gpa.free(key);
    const images = try t.gpa.dupe(Identity, tree_mod.overlapping(k.images, edge, edge + key.len));
    errdefer t.gpa.free(images);
    const ids = try t.gpa.alloc(u32, t.per);
    errdefer t.gpa.free(ids);
    try t.holds.reserve(t.gpa, ids.len);
    const hold = t.pages.vtable.hold orelse return error.NoCachePages;
    if (!hold(t.pages.ptr, ids)) return error.NoCachePages;
    for (ids) |id| t.holds.adoptHeldAssumeCapacity(id);
    errdefer {
        hooks.vtable.fence(hooks.ptr, owner); // queued copies into the pages complete before they go back
        for (ids) |id| t.holds.drop(t.pages, id);
    }
    for (partial, ids, 0..) |src, dst, g| try hooks.vtable.copy_rows(hooks.ptr, owner, first + @as(u32, @intCast(g)), src, dst, r);
    hooks.vtable.fence(hooks.ptr, owner); // the copies are the checkpoint's from now on
    return .{ .r = r, .key = key, .images = images, .ns = k.ns, .ids = ids };
}

/// The tail's pages back to the tree's holds, and its slices freed.
pub fn free(t: *Tree, tl: Tail) void {
    for (tl.ids) |id| t.holds.drop(t.pages, id);
    t.gpa.free(tl.ids);
    t.gpa.free(tl.key);
    t.gpa.free(tl.images);
}

/// A resume's tail copies into `out`, one a prefix group, to page `index`; each source held for the stream to `finish`.
pub fn handOut(t: *Tree, tl: Tail, first: u32, index: u32, out: []parts.Copy) void {
    for (tl.ids, out, 0..) |id, *c, g| {
        t.pages.vtable.retain(t.pages.ptr, id);
        c.* = .{ .group = first + @as(u32, @intCast(g)), .index = index, .src = id, .rows = tl.r };
    }
}
