//! Host tests of a forward's launches: a trace records tags instead of issuing kernels.
const std = @import("std");

/// Host tests record here; null in serving and in every other test.
pub var active: ?*Trace = null;

pub const Trace = struct {
    gpa: std.mem.Allocator,
    tags: std.ArrayListUnmanaged(Tag) = .empty,

    pub const Tag = union(enum) { norm, kv, o, attn, launch: usize };

    pub fn init(gpa: std.mem.Allocator) Trace {
        return .{ .gpa = gpa };
    }

    pub fn deinit(t: *Trace) void {
        t.tags.deinit(t.gpa);
    }

    pub fn note(t: *Trace, tag: Tag) !void {
        try t.tags.append(t.gpa, tag);
    }
};
