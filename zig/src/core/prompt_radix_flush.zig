//! Administrative shared-store flush: keep the newest complete states and the prefix pages they need.
const tree = @import("prompt_radix_tree.zig");
const Node = tree.Node;
const Found = tree.Found;

pub fn retain(t: *tree.Tree, count: u32) void {
    if (count == 0 or t.snaps == 0) return t.clear();
    while (t.snaps > count) {
        var oldest: ?Found = null;
        visit(t.root, &oldest);
        t.evictAt(oldest.?);
    }
    while (empty(t.root)) |n| t.prune(n);
}

fn consider(f: Found, oldest: *?Found) void {
    if (oldest.* == null or f.entry().used < oldest.*.?.entry().used) oldest.* = f;
}

fn visit(n: *Node, oldest: *?Found) void {
    if (n.entry != null) consider(.{ .node = n }, oldest);
    for (n.tails.items, 0..) |_, i| consider(.{ .node = n, .tail = i }, oldest);
    for (n.kids.items) |k| visit(k, oldest);
}

fn empty(n: *Node) ?*Node {
    for (n.kids.items) |k| if (empty(k)) |leaf| return leaf;
    return if (n.parent != null and n.entry == null and n.tails.items.len == 0 and n.kids.items.len == 0) n else null;
}
