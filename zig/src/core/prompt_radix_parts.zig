//! The radix store's manifest (C3, #2444): a family's cache groups, a checkpoint's parts, and one hold a page.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Allocator = std.mem.Allocator;

/// What a cache group holds; the core pool's kinds (kv_groups.zig), named here for the store.
pub const Kind = enum { full, window, state, draft };

/// How a checkpoint group's part is made: the stream's own read-only pages, or store-held pages its bytes go to.
pub const Make = enum { retained, copied };

/// A checkpoint group: its kind, how its part is made, and (retained) its rows a page.
pub const PartSpec = struct { kind: Kind, make: Make, rows: u32 = 0 };

/// A family's cache groups as its store sees them, declared once when the store opens.
pub const Groups = struct {
    /// Rows a page of each prefix group; empty: one prefix group of the store's page.
    prefix_rows: []const u32 = &.{},
    /// The checkpoint groups, in order.
    parts: []const PartSpec = &.{},
    /// A legacy snapshot (the family's Snapshots) beside the parts.
    legacy: bool = true,
    /// Checkpoints inside a page (C4): each keeps that page's rows as a held copy (needs copy_rows and Pages.hold).
    tails: bool = false,

    /// Prefix groups: page ids a logical page on the tree's path.
    pub fn prefixCount(g: Groups) usize {
        return @max(1, g.prefix_rows.len);
    }

    /// No checkpoint at all (pure full attention): any matched page edge resumes.
    pub fn none(g: Groups) bool {
        return g.parts.len == 0 and !g.legacy;
    }
};

/// One checkpoint group's pages in one entry; page `i` sits at logical index `base + i` in the group's table.
pub const Part = struct { base: u32, ids: []u32, bytes: u64 };

/// A keep's input for one retained group: its pages from logical index `base`, the last partial if rows fall short.
pub const Retained = struct { base: u32, ids: []const u32 };

/// A held page's rows [0, rows) into the stream's page `index` of `group` (parts first, then the prefix groups).
pub const Copy = struct { group: u32, index: u32, src: u32, rows: u32 };

/// What the store asks a family that declares parts: the bytes, and the device copies (each on the owner's stream).
pub const PartHooks = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// A copied part's bytes at `at` tokens.
        bytes: *const fn (ptr: *anyopaque, group: u32, at: u32) u64,
        /// The owner's live bytes of `group` after `at` tokens into the held pages of `dst`.
        copy: *const fn (ptr: *anyopaque, owner: ?*anyopaque, group: u32, at: u32, dst: Part) anyerror!void,
        /// The held pages of `src` into the owner's live `group` (the next prompt chunk starts at its position).
        restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, group: u32, src: Part) anyerror!void,
        /// Rows [0, rows) of every layer of `group` (numbered as in Copy) from page `src` into page `dst`.
        copy_rows: *const fn (ptr: *anyopaque, owner: ?*anyopaque, group: u32, src: u32, dst: u32, rows: u32) anyerror!void,
        /// Every copy queued for `owner` has completed, on every rank that runs it.
        fence: *const fn (ptr: *anyopaque, owner: ?*anyopaque) void,
    };
};

/// The store's references: one pool reference a page, however many nodes and entries name it.
pub const Holds = struct {
    counts: std.AutoHashMapUnmanaged(u32, u32) = .empty,

    pub fn deinit(h: *Holds, gpa: Allocator) void {
        h.counts.deinit(gpa);
    }

    /// Room for `n` more pages, so the registrations after it cannot fail.
    pub fn reserve(h: *Holds, gpa: Allocator, n: usize) error{OutOfMemory}!void {
        try h.counts.ensureUnusedCapacity(gpa, @intCast(n));
    }

    /// The store names a page a stream holds: its first name takes one pool reference.
    pub fn shareAssumeCapacity(h: *Holds, pages: pc.Pages, id: u32) void {
        const slot = h.counts.getOrPutAssumeCapacity(id);
        if (slot.found_existing) {
            slot.value_ptr.* += 1;
            return;
        }
        slot.value_ptr.* = 1;
        pages.vtable.retain(pages.ptr, id);
    }

    /// The store names a page it took itself (Pages.hold): that hold is its pool reference.
    pub fn adoptHeldAssumeCapacity(h: *Holds, id: u32) void {
        const slot = h.counts.getOrPutAssumeCapacity(id);
        std.debug.assert(!slot.found_existing); // a page just held is new to the store
        slot.value_ptr.* = 1;
    }

    /// One name fewer; the last gives the pool reference back.
    pub fn drop(h: *Holds, pages: pc.Pages, id: u32) void {
        const slot = h.counts.getPtr(id) orelse std.debug.panic("prompt cache: page {d} dropped while not held", .{id});
        slot.* -= 1;
        if (slot.* > 0) return;
        _ = h.counts.remove(id);
        pages.vtable.release(pages.ptr, id);
    }

    pub fn contains(h: *const Holds, id: u32) bool {
        return h.counts.contains(id);
    }

    /// Distinct pages the store holds: what its page limit counts.
    pub fn count(h: *const Holds) usize {
        return h.counts.count();
    }
};

test {
    _ = @import("prompt_radix_parts_test.zig");
}
