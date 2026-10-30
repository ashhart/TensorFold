//! Counts only allocator operations routed through evaluator concatenations.
const std = @import("std");

pub const Counts = struct {
    alloc_calls: usize = 0,
    alloc_requested: usize = 0,
    alloc_success: usize = 0,
    alloc_bytes: usize = 0,
    alloc_failed: usize = 0,
    resize_calls: usize = 0,
    resize_growth_requested: usize = 0,
    resize_success: usize = 0,
    resize_growth: usize = 0,
    resize_failed: usize = 0,
    remap_calls: usize = 0,
    remap_growth_requested: usize = 0,
    remap_success: usize = 0,
    remap_growth: usize = 0,
    remap_failed: usize = 0,
};

pub var string_alloc: Counts = .{};
pub var list_alloc: Counts = .{};
pub var string_calls: usize = 0;
pub var string_completed: usize = 0;
pub var string_reuse: usize = 0;
pub var string_new: usize = 0;
pub var prefix_bytes: usize = 0;
pub var prefix_copied: usize = 0;
pub var list_attempts: usize = 0;
pub var list_success: usize = 0;
pub var list_elements: usize = 0;

pub fn reset() void {
    string_alloc = .{};
    list_alloc = .{};
    string_calls = 0;
    string_completed = 0;
    string_reuse = 0;
    string_new = 0;
    prefix_bytes = 0;
    prefix_copied = 0;
    list_attempts = 0;
    list_success = 0;
    list_elements = 0;
}

pub fn stringJoin(prefix: usize, reuse: bool) void {
    string_calls += 1;
    if (reuse) {
        string_reuse += 1;
    } else {
        string_new += 1;
        prefix_bytes += prefix;
    }
}

pub const Probe = struct {
    child: std.mem.Allocator,
    counts: *Counts,

    pub fn allocator(self: *Probe) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn alloc(ctx: *anyopaque, n: usize, al: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *Probe = @ptrCast(@alignCast(ctx));
        self.counts.alloc_calls += 1;
        self.counts.alloc_requested += n;
        const ptr = self.child.rawAlloc(n, al, ret) orelse {
            self.counts.alloc_failed += 1;
            return null;
        };
        self.counts.alloc_success += 1;
        self.counts.alloc_bytes += n;
        return ptr;
    }

    fn resize(ctx: *anyopaque, bytes: []u8, al: std.mem.Alignment, n: usize, ret: usize) bool {
        const self: *Probe = @ptrCast(@alignCast(ctx));
        const growth = n -| bytes.len;
        self.counts.resize_calls += 1;
        self.counts.resize_growth_requested += growth;
        if (!self.child.rawResize(bytes, al, n, ret)) {
            self.counts.resize_failed += 1;
            return false;
        }
        self.counts.resize_success += 1;
        self.counts.resize_growth += growth;
        return true;
    }

    fn remap(ctx: *anyopaque, bytes: []u8, al: std.mem.Alignment, n: usize, ret: usize) ?[*]u8 {
        const self: *Probe = @ptrCast(@alignCast(ctx));
        const growth = n -| bytes.len;
        self.counts.remap_calls += 1;
        self.counts.remap_growth_requested += growth;
        const ptr = self.child.rawRemap(bytes, al, n, ret) orelse {
            self.counts.remap_failed += 1;
            return null;
        };
        self.counts.remap_success += 1;
        self.counts.remap_growth += growth;
        return ptr;
    }

    fn free(ctx: *anyopaque, bytes: []u8, al: std.mem.Alignment, ret: usize) void {
        const self: *Probe = @ptrCast(@alignCast(ctx));
        self.child.rawFree(bytes, al, ret);
    }
};

fn printCounts(label: []const u8, c: Counts) void {
    std.debug.print("ALLOC {s} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d}\n", .{
        label,                    c.alloc_calls,             c.alloc_requested, c.alloc_success, c.alloc_bytes,   c.alloc_failed,
        c.resize_calls,           c.resize_growth_requested, c.resize_success,  c.resize_growth, c.resize_failed, c.remap_calls,
        c.remap_growth_requested, c.remap_success,           c.remap_growth,    c.remap_failed,
    });
}

pub fn print() void {
    std.debug.print("JOINS {d} {d} {d} {d} {d} {d} {d} {d} {d}\n", .{
        string_calls,  string_completed, string_reuse,  string_new, prefix_bytes, prefix_copied,
        list_attempts, list_success,     list_elements,
    });
    printCounts("string", string_alloc);
    printCounts("list", list_alloc);
}
