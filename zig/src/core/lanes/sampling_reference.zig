//! Test and benchmark oracle: TensorFold's original full-row stable candidate sort, without selection shortcuts.
const std = @import("std");
pub const sampling = @import("sampling.zig");
const Candidate = sampling.Candidate;

fn before(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

pub fn ordered(gpa: std.mem.Allocator, values: []const f64, ids: []const u64) ![]Candidate {
    const order = try gpa.alloc(Candidate, ids.len);
    for (order, values, ids) |*o, v, id| o.* = .{ .value = v, .id = id };
    std.mem.sort(Candidate, order, {}, before);
    return order;
}

/// Only candidate selection changes in the optimized path; both paths use the unchanged ordered draw.
pub fn choose(gpa: std.mem.Allocator, values: []const f64, ids: []const u64, position: u64, s: sampling.Sampling) !u64 {
    const order = try ordered(gpa, values, ids);
    defer gpa.free(order);
    return sampling.chooseOrdered(gpa, order, position, s);
}
