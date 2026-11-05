//! The joint schema head's contract entry: its kernels and planted controls through gates 1-4.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const controls = @import("hip_controls");
const Profile = @import("contract_entry.zig").Profile;
const js_rows_k = @import("contract_js_rows.zig");
const js_spans_k = @import("contract_js_spans.zig");
const js_point_k = @import("contract_js_point.zig");
const js_heads_k = @import("contract_js_heads.zig");

const g = core.contract.gates;

/// Every joint schema kernel at its fixed shapes; both profiles run the same cases (each seeded by its name).
pub fn run(a: std.mem.Allocator, _: std.Io, _: Profile) anyerror!g.Tally {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var ctl = try hip.Module.load(&r, &controls.bytes);
    defer ctl.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    const js = try hip.joint_schema.Fns.load(&l);
    const js_which = comptime std.enums.values(js_rows_k.Which);
    var js_rows: [js_which.len]js_rows_k.Row = undefined;
    for (&js_rows, js_which) |*x, w| x.* = try .init(a, &r, s, js, ctl, w);
    defer for (&js_rows) |*x| x.deinit();
    for (&js_rows) |*x| try kernels.append(a, x.kernel());
    var js_spans: [3]js_spans_k.Spans = undefined;
    for (&js_spans, std.enums.values(js_spans_k.Which)) |*x, w| x.* = .init(a, &r, s, js, w);
    defer for (&js_spans) |*x| x.deinit();
    for (&js_spans) |*x| try kernels.append(a, x.kernel());
    var js_points: [std.enums.values(js_point_k.Which).len]js_point_k.Point = undefined;
    for (&js_points, std.enums.values(js_point_k.Which)) |*x, w| x.* = try .init(a, &r, s, js, ctl, w);
    defer for (&js_points) |*x| x.deinit();
    for (&js_points) |*x| try kernels.append(a, x.kernel());
    var js_heads: [3]js_heads_k.Heads = undefined;
    for (&js_heads, std.enums.values(js_heads_k.Which)) |*x, w| x.* = .init(a, &r, s, js, w);
    defer for (&js_heads) |*x| x.deinit();
    for (&js_heads) |*x| try kernels.append(a, x.kernel());
    var t: g.Tally = .{};
    try g.runAll(a, kernels.items, 0x6b63, &t);
    return t;
}
