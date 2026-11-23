//! fn_int4.cu's controls on the host side: the controls image's scalar gate/up (tf_ctl_int4_swiglu).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const grouped = cuda.grouped;
const int4 = cuda.int4;

const g = core.contract.gates;

/// tf_ctl_int4_swiglu's `fault`: fn_int4.cu's epilogue on fp32 sums (the epilogue's model), or a planted one.
pub const Fault = enum(i32) { epilogue = 0, swapped = 1, cut = 2, bf16_sums = 3 };

/// The gate a planted scalar gate/up must fail; the epilogue's model is a kernel and fails none.
pub fn gate(f: ?Fault) ?g.Gate {
    return switch (f orelse return null) {
        .epilogue => null,
        .swapped, .cut => .shapes,
        .bf16_sums => .sums,
    };
}

/// The scalar gate/up on a plan's pairs: x [R, D] token rows -> out [R * slots, width] bf16, laid out as `gateUp` does.
pub fn launch(s: cuda.Stream, f: cuda.Function, x: u64, shape: core.contract.spec.Shape, slots: usize, ex: int4.Experts, p: grouped.Plan, out: u64, pairs: usize, fault: Fault) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i32, @intCast(shape.k)));
    a.add(@as(i32, @intCast(slots)));
    a.add(ex.up);
    a.add(ex.up_s);
    a.add(@as(i32, @intCast(shape.k)));
    a.add(@as(i32, @intCast(shape.n)));
    a.add(p.items);
    a.add(p.counts);
    a.add(p.members);
    a.add(out);
    a.add(@as(i32, @intCast(shape.n)));
    a.add(@backingInt(fault));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((shape.n + 255) / 256), .y = @intCast(pairs), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
}
