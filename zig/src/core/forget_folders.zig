//! Bounded, owned diagnostics for sibling identities under the current learned root.
pub const Reason = enum { unowned_match, unindexed, appending, unreadable };
pub const Issue = struct { identity: u64, reason: Reason };
pub const Folders = struct {
    items: [8]Issue = @splat(.{ .identity = 0, .reason = .unowned_match }),
    len: u8 = 0,
    omitted: u32 = 0, // Omitted observations, not unique folders; repeated span calls can count the same folder.

    pub fn add(f: *Folders, identity: u64, reason: Reason) void {
        for (f.items[0..f.len]) |item| {
            if (item.identity == identity and item.reason == reason) return;
        }
        if (f.len == f.items.len) {
            f.omitted +|= 1;
            return;
        }
        f.items[f.len] = .{ .identity = identity, .reason = reason };
        f.len += 1;
    }

    pub fn merge(f: *Folders, other: Folders) void {
        for (other.items[0..other.len]) |item| f.add(item.identity, item.reason);
        f.omitted +|= other.omitted;
    }
};
