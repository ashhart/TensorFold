//! Test-only geometry choices; serving imports neither this selector nor the forced M1 registration.
const std = @import("std");
const c = @import("contract").spec;
const cases = @import("nemotron_experts_cases.zig");
pub const Mode = enum { w3, rows2, rows4, forced_m1_rows2r8 };
pub const modes = std.enums.values(Mode);
pub const count = 66;
pub fn entry(proj: cases.Proj, mode: Mode, fault: cases.Fault) cases.Entry {
    const up = proj == .up;
    return .{ .proj = proj, .symbol = switch (mode) {
        .w3 => if (up) "tf_xup_w3" else "tf_xdown_w3",
        .rows2 => if (up) "tf_xup_rows2" else "tf_xdown_rows2",
        .rows4 => if (up) "tf_xup_rows4" else "tf_xdown_rows4",
        .forced_m1_rows2r8 => if (up) "tf_xup_rows2r8" else "tf_xdown_rows2r8",
    }, .rps = if (mode == .forced_m1_rows2r8) 8 else 4, .sg = if (mode == .w3) 4 else 2, .mb = switch (mode) {
        .w3 => 3,
        .rows4 => 4,
        else => 2,
    }, .fault = fault };
}
pub fn served(proj: cases.Proj) [count]c.Shape {
    const nk = cases.dims(proj);
    var shapes: [count]c.Shape = undefined;
    shapes[0] = .{ .rows = 16, .n = nk[0], .k = nk[1] };
    var at: usize = 1;
    for (1..65) |rows| if (rows != 16) {
        shapes[at] = .{ .rows = rows, .n = nk[0], .k = nk[1] };
        at += 1;
    };
    shapes[at] = .{ .rows = 127, .n = nk[0], .k = nk[1] };
    shapes[at + 1] = .{ .rows = 128, .n = nk[0], .k = nk[1] };
    return shapes;
}
pub fn visit(visitor: anytype, only: ?[]const u8) !void {
    for ([_]cases.Proj{ .up, .down }) |proj| for (modes) |mode| {
        try add(visitor, only, proj, mode, .none);
    };
    try add(visitor, only, .up, .forced_m1_rows2r8, .coupled);
    try add(visitor, only, .down, .w3, .fp32_sums);
}
fn add(visitor: anytype, only: ?[]const u8, proj: cases.Proj, mode: Mode, fault: cases.Fault) !void {
    const name = try std.fmt.allocPrint(visitor.a, "{s}metal nemotron experts verify plain bits {t} {t}", .{ if (fault == .none) "" else "control: ", proj, mode });
    defer visitor.a.free(name);
    if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
    try visitor.addPlain(entry(proj, mode, fault), name);
}
