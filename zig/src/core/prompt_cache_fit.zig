//! Budget admission for the ordinary endpoint and rewind marks.
const std = @import("std");
const Allocator = std.mem.Allocator;
const pc = @import("prompt_cache.zig");

/// The marks the budget can hold (a new slice in `a`). Refuse the rest before any pass or peer copies them.
pub fn fitting(s: *pc.Store, a: Allocator, marks_: []const u32) ![]const u32 {
    defer a.free(marks_);
    var out: std.ArrayList(u32) = .empty;
    for (marks_) |at| {
        if (s.hold) break; // memory pressure: the pass keeps nothing new, so none is planned
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes <= s.budget) try out.append(a, at) else {
            s.counts.refused += 1;
            pc.note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
        }
    }
    return out.toOwnedSlice(a);
}

/// Fitted marks that the budget holds together: the history's first, then the rewind, then the rest in order.
pub fn fittingRewind(s: *pc.Store, a: Allocator, marks_: []const u32, endpoint: u32, rewind: u32, endpoint_bytes: u64) ![]const u32 {
    defer a.free(marks_);
    var out: std.ArrayList(u32) = .empty;
    var used: u64 = 0;
    if (endpoint_bytes <= s.budget and endpoint_bytes > 0) {
        used = endpoint_bytes;
        if (std.mem.indexOfScalar(u32, marks_, endpoint) != null) try out.append(a, endpoint);
    }
    if (s.active_rewind_at == rewind and std.mem.indexOfScalar(u32, marks_, rewind) != null) try out.append(a, rewind);
    if (s.active_rewind_at == rewind) used += s.family.vtable.bytes(s.family.ptr, rewind);
    for (marks_) |at| {
        if (at == endpoint or at == rewind) continue;
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes <= s.budget -| used) {
            try out.append(a, at);
            used += bytes;
        }
    }
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    return out.toOwnedSlice(a);
}
