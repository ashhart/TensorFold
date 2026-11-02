//! A grammar stream's first token: the prompt pass draws it among the bits that Backend.first_masked is then given.
const std = @import("std");
const lanes = @import("lanes");
const Stream = lanes.Stream;
const ln = @import("lanes.zig");

/// Before `prefill`, `finish` or `join`: the grammar's first-token bits, kept in `st` for the target (empty: none).
pub fn take(x: *ln.Lanes, st: *ln.St, s: *Stream) ![]const u32 {
    st.first_bits.clearRetainingCapacity();
    const g = s.grammar orelse return &.{};
    // Constraint.next: the same bits Engine.drawFirst asks for after the pass; the grammar takes no token in between
    const bits = (try g.next(x.scratch.allocator())) orelse return &.{};
    try st.first_bits.appendSlice(x.gpa, bits);
    return st.first_bits.items;
}

/// Backend.first_masked's guarantee: `mask` is the bits the pass drew the pick under, and the pick is among them.
pub fn check(st: *const ln.St, mask: []const u32) !void {
    if (!std.mem.eql(u32, mask, st.first_bits.items)) return error.FirstMaskChanged;
    if (!lanes.grammar.allows(mask, st.first)) return error.FirstNotMasked;
}
