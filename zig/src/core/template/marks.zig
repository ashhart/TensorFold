//! Where each request message's own output starts in a render (template.zig `renderMarked`), or nothing when unsure.
const std = @import("std");
const v = @import("value.zig");
const Allocator = std.mem.Allocator;

const Mark = struct { at: usize, k: u32 };
const Key = struct { ptr: usize, len: usize };

/// A loop iteration over a message, or one that binds it first, marks that message where its output begins.
pub const Marks = struct {
    a: Allocator,
    index: std.AutoHashMapUnmanaged(*const v.Dict, u32) = .empty,
    starts: []?usize,
    root: *const v.Buffer,
    bound: ?u32 = null, // the message the current iteration set to a name or passed to a macro first
    level: ?*const v.Buffer = null, // where the current iteration writes; a bind made inside a capture is not its own
    open: std.AutoHashMapUnmanaged(*const v.Buffer, std.ArrayList(Mark)) = .empty, // marks in captures being written
    held: std.AutoHashMapUnmanaged(Key, []const Mark) = .empty, // marks of finished captures, by the exact string

    pub const Saved = struct { bound: ?u32, level: ?*const v.Buffer };

    pub fn init(a: Allocator, items: []const v.Value, root: *const v.Buffer) Allocator.Error!Marks {
        var m: Marks = .{ .a = a, .starts = &.{}, .root = root };
        for (items) |item| if (item == .dict) try m.index.put(a, item.dict, m.index.count());
        m.starts = try a.alloc(?usize, m.index.count());
        @memset(m.starts, null);
        return m;
    }

    fn of(m: *const Marks, item: v.Value) ?u32 {
        return if (item == .dict) m.index.get(item.dict) else null;
    }

    pub fn bind(m: *Marks, out: *const v.Buffer, x: v.Value) void {
        if (m.bound == null and out == m.level) m.bound = m.of(x);
    }

    /// An iteration starts writing to `out`.
    pub fn begin(m: *Marks, out: *const v.Buffer) Saved {
        const saved: Saved = .{ .bound = m.bound, .level = m.level };
        m.bound = null;
        m.level = out;
        return saved;
    }

    /// The iteration over `item` that began at byte `at` of `out` ended; one that wrote nothing marks nothing.
    pub fn end(m: *Marks, saved: Saved, out: *const v.Buffer, at: usize, item: v.Value) Allocator.Error!void {
        const k = if (out.text.items.len > at) (m.of(item) orelse m.bound) else null;
        m.bound = saved.bound;
        m.level = saved.level;
        if (k) |x| try m.mark(out, at, x);
    }

    fn mark(m: *Marks, out: *const v.Buffer, at: usize, k: u32) Allocator.Error!void {
        if (out == m.root) {
            if (m.starts[k] == null or at < m.starts[k].?) m.starts[k] = at;
            return;
        }
        const slot = try m.open.getOrPut(m.a, out);
        if (!slot.found_existing) slot.value_ptr.* = .empty;
        try slot.value_ptr.append(m.a, .{ .at = at, .k = k });
    }

    /// A capture into `buf` finished as `s`: its marks travel with that exact string, and with nothing else.
    pub fn close(m: *Marks, buf: *const v.Buffer, s: v.Str) Allocator.Error!void {
        const kv = m.open.fetchRemove(buf) orelse return;
        try m.held.put(m.a, keyOf(s), kv.value.items);
    }

    /// `s` was written to `out` at `base`; a string changed after its capture carries no marks (fail closed).
    pub fn carry(m: *Marks, out: *const v.Buffer, s: v.Str, base: usize) Allocator.Error!void {
        if (m.held.count() == 0) return;
        for (m.held.get(keyOf(s)) orelse return) |x| try m.mark(out, base + x.at, x.k);
    }

    /// `l` and `r` joined into `s` (`~` or `+`): both sides' marks travel with it.
    pub fn joined(m: *Marks, l: v.Str, r: v.Str, s: v.Str) Allocator.Error!void {
        if (m.held.count() == 0) return;
        const left = m.held.get(keyOf(l)) orelse &.{};
        const right = m.held.get(keyOf(r)) orelse &.{};
        if (left.len + right.len == 0) return;
        const both = try m.a.alloc(Mark, left.len + right.len);
        @memcpy(both[0..left.len], left);
        for (right, both[left.len..]) |x, *y| y.* = .{ .at = l.s.len + x.at, .k = x.k };
        try m.held.put(m.a, keyOf(s), both);
    }

    fn keyOf(s: v.Str) Key {
        return .{ .ptr = @intFromPtr(s.s.ptr), .len = s.s.len };
    }
};
