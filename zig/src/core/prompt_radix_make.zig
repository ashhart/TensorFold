//! A checkpoint's parts (C3, #2444): made at a keep, restored at a resume, all or nothing, the copies fenced.
const std = @import("std");
const parts = @import("prompt_radix_parts.zig");
const tree_mod = @import("prompt_radix_tree.zig");
const Allocator = std.mem.Allocator;

const Part = parts.Part;
const Retained = parts.Retained;
const Copy = parts.Copy;
const Tree = tree_mod.Tree;

/// A store's part declaration: its groups' specs, and the family's hooks for them.
pub const Decl = struct {
    specs: []const parts.PartSpec = &.{},
    hooks: ?parts.PartHooks = null,

    /// Logical pages of a retained group below or at `at`: whole ones, and one more when its rows fall short.
    fn spans(rows: u32, at: u32) u32 {
        return (at + rows - 1) / rows;
    }

    /// Whether `retained` gives each retained group its pages through `at` (one entry a part, in order).
    pub fn fits(d: Decl, at: u32, retained: []const Retained) bool {
        if (retained.len != d.specs.len) return false;
        for (d.specs, retained) |spec, r| switch (spec.make) {
            .copied => if (r.ids.len != 0) return false,
            .retained => if (r.ids.len == 0 or r.base + r.ids.len != spans(spec.rows, at)) return false,
        };
        return true;
    }

    /// Ids a keep at `at` adds to the tree's holds: copied parts' pages, retained ids not held yet, partial copies.
    pub fn ids(d: Decl, t: *const Tree, at: u32, retained: []const Retained) usize {
        var n: usize = 0;
        for (d.specs, retained, 0..) |spec, r, g| switch (spec.make) {
            .copied => n += copiedPages(d, t, @intCast(g), at),
            .retained => {
                const partial = at % spec.rows != 0;
                const whole = r.ids[0 .. r.ids.len - @intFromBool(partial)];
                for (whole) |id| n += @intFromBool(!t.holds.contains(id));
                n += @intFromBool(partial);
            },
        };
        return n;
    }

    /// Ids a keep at `at` adds that a plan knows before the keep: copied parts' pages, one per retained partial page.
    pub fn planIds(d: Decl, t: *const Tree, at: u32) usize {
        var n: usize = 0;
        for (d.specs, 0..) |spec, g| switch (spec.make) {
            .copied => n += copiedPages(d, t, @intCast(g), at),
            .retained => n += @intFromBool(at % spec.rows != 0),
        };
        return n;
    }

    fn copiedPages(d: Decl, t: *const Tree, g: u32, at: u32) usize {
        const h = d.hooks.?;
        const bytes = h.vtable.bytes(h.ptr, g, at);
        return @intCast((bytes + t.pages.bytes - 1) / t.pages.bytes);
    }

    /// The parts of a checkpoint at `at`, each id named in the tree's holds; on failure, what it took goes back.
    pub fn make(d: Decl, t: *Tree, owner: ?*anyopaque, at: u32, retained: []const Retained) ![]Part {
        if (d.specs.len == 0) return &.{};
        const h = d.hooks.?;
        const out = try t.gpa.alloc(Part, d.specs.len);
        var made: usize = 0;
        errdefer {
            h.vtable.fence(h.ptr, owner); // queued copies into the pages complete before they go back
            t.dropEach(out[0..made]);
            t.gpa.free(out);
        }
        try t.holds.reserve(t.gpa, d.ids(t, at, retained));
        for (d.specs, retained, 0..) |spec, r, gi| {
            const g: u32 = @intCast(gi);
            switch (spec.make) {
                .copied => {
                    const n = copiedPages(d, t, g, at);
                    const held = try t.gpa.alloc(u32, n);
                    if (!hold(t, held)) {
                        t.gpa.free(held);
                        return error.NoCachePages;
                    }
                    for (held) |id| t.holds.adoptHeldAssumeCapacity(id);
                    out[made] = .{ .base = 0, .ids = held, .bytes = h.vtable.bytes(h.ptr, g, at) };
                    made += 1;
                    try h.vtable.copy(h.ptr, owner, g, at, out[made - 1]);
                },
                .retained => {
                    const rows_left = at % spec.rows;
                    const whole = r.ids.len - @intFromBool(rows_left != 0);
                    var copy: [1]u32 = undefined; // the partial page: a copy the store holds
                    if (rows_left != 0 and !hold(t, &copy)) return error.NoCachePages;
                    const own = t.gpa.alloc(u32, r.ids.len) catch |err| {
                        if (rows_left != 0) t.pages.vtable.release(t.pages.ptr, copy[0]);
                        return err;
                    };
                    for (r.ids[0..whole], own[0..whole]) |id, *o| {
                        t.holds.shareAssumeCapacity(t.pages, id);
                        o.* = id;
                    }
                    if (rows_left != 0) {
                        own[whole] = copy[0];
                        t.holds.adoptHeldAssumeCapacity(copy[0]);
                    }
                    out[made] = .{ .base = r.base, .ids = own, .bytes = 0 };
                    made += 1;
                    if (rows_left != 0) try h.vtable.copy_rows(h.ptr, owner, g, r.ids[whole], copy[0], rows_left);
                },
            }
        }
        h.vtable.fence(h.ptr, owner); // every copy into the parts is done: they are the checkpoint's from now on
        return out;
    }

    fn hold(t: *Tree, out: []u32) bool {
        const f = t.pages.vtable.hold orelse return false;
        return f(t.pages.ptr, out);
    }

    /// The copied parts of `e` into the owner's live state, fenced; the store keeps their pages.
    pub fn restore(d: Decl, owner: ?*anyopaque, ps: []const Part) !void {
        if (d.specs.len == 0) return;
        const h = d.hooks.?;
        defer h.vtable.fence(h.ptr, owner); // the copies out of the store's pages are done, failed or not
        for (d.specs, ps, 0..) |spec, part, g| if (spec.make == .copied) try h.vtable.restore(h.ptr, owner, @intCast(g), part);
    }

    /// The retained parts a resume at `at` maps: whole pages held once for the stream, partial ones as copies to make.
    pub fn handOut(d: Decl, t: *Tree, a: Allocator, at: u32, ps: []const Part) !struct { parts: []Retained, copies: []Copy } {
        var nr: usize = 0;
        var nc: usize = 0;
        for (d.specs) |spec| if (spec.make == .retained) {
            nr += 1;
            nc += @intFromBool(at % spec.rows != 0);
        };
        const out = try a.alloc(Retained, nr);
        var owned: usize = 0; // retained entries whose ids `a` owns
        errdefer {
            for (out[0..owned]) |r| a.free(r.ids);
            a.free(out);
        }
        const copies = try a.alloc(Copy, nc);
        errdefer a.free(copies);
        nr = 0;
        for (d.specs, ps) |spec, part| if (spec.make == .retained) {
            const whole = part.ids.len - @intFromBool(at % spec.rows != 0);
            // the plan's own copy: eviction frees part.ids
            out[nr] = .{ .base = part.base, .ids = try a.dupe(u32, part.ids[0..whole]) };
            nr += 1;
            owned = nr;
        };
        nr = 0;
        nc = 0;
        for (d.specs, ps, 0..) |spec, part, g| if (spec.make == .retained) {
            const rows_left = at % spec.rows;
            const whole = part.ids.len - @intFromBool(rows_left != 0);
            // the stream's hold on each, until it adopts or finishes
            for (part.ids) |id| t.pages.vtable.retain(t.pages.ptr, id);
            nr += 1;
            if (rows_left == 0) continue;
            copies[nc] = .{ .group = @intCast(g), .index = at / spec.rows, .src = part.ids[whole], .rows = rows_left };
            nc += 1;
        };
        return .{ .parts = out, .copies = copies };
    }
};
