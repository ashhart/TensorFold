//! GLM-OCR's contract entry: its pointwise and norm kernels from the glm_ocr image through gates 1-4.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const Profile = @import("contract_entry.zig").Profile;
const ocr_ops = @import("glm_ocr_ops").Ops;
const ocr_points = @import("contract_glm_ocr_ops.zig");
const ocr_norms = @import("contract_glm_ocr_norm.zig");

const g = core.contract.gates;

/// Every GLM-OCR kernel at its fixed shapes; both profiles run the same cases.
pub fn run(a: std.mem.Allocator, _: std.Io, _: Profile) anyerror!g.Tally {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var ocr = try ocr_ops.load(&r);
    defer ocr.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    var pointwise: [std.enums.values(ocr_points.Which).len]ocr_points.Pointwise = undefined;
    for (&pointwise, std.enums.values(ocr_points.Which)) |*op, which_op| op.* = .init(a, &r, s, &ocr, which_op);
    defer for (&pointwise) |*op| op.deinit();
    for (&pointwise) |*op| try kernels.append(a, op.kernel());
    var norms: [std.enums.values(ocr_norms.Which).len]ocr_norms.Norm = undefined;
    for (&norms, std.enums.values(ocr_norms.Which)) |*op, which_op| op.* = .init(a, &r, s, &ocr, which_op);
    defer for (&norms) |*op| op.deinit();
    for (&norms) |*op| try kernels.append(a, op.kernel());
    var t: g.Tally = .{};
    try g.runAll(a, kernels.items, 0x6b63, &t);
    return t;
}
