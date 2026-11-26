//! Explicit diagnostic qualification requires the complete registered real/control inventory.
const std = @import("std");
const Diagnostic = struct { selector: []const u8, real: usize, controls: usize };
pub const diagnostics = [_]Diagnostic{
    .{ .selector = "nemotron verify attention legacy-window bits", .real = 2, .controls = 2 },
    .{ .selector = "nemotron verify attention one-row bits", .real = 2, .controls = 2 },
    .{ .selector = "nemotron experts verify plain bits", .real = 8, .controls = 2 },
    .{ .selector = "forced_m1_rows2r8", .real = 2, .controls = 1 },
};
pub fn validate(only: ?[]const u8, kernels: anytype) !void {
    const selected = only orelse return;
    if (kernels.len == 0) return error.EmptyExplicitSelection;
    const text = if (std.mem.startsWith(u8, selected, "metal ")) selected[6..] else selected;
    for (diagnostics) |d| if (std.mem.eql(u8, text, d.selector)) {
        var real: usize = 0;
        var controls: usize = 0;
        for (kernels) |k| {
            if (k.control == null) real += 1 else controls += 1;
        }
        if (real != d.real or controls != d.controls) return error.IncompleteDiagnosticSelection;
        return;
    };
}
