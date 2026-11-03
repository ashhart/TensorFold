//! Shader text belongs to the native family module, which owns the kernel sources rather than the core ABI module.
const std = @import("std");
const core = @import("core");
pub fn metal(a: std.mem.Allocator, comptime spec: type) ![]u8 {
    const text = @embedFile("../../../kernels/metal/core/living_rows.metal");
    const body = @embedFile("../../../kernels/common/living_rows.h");
    if (spec == core.living_rows) return std.mem.replaceOwned(u8, a, text, "// TF_LIVING_BODY", body);
    const end = std.mem.indexOf(u8, text, "// TF_LIVING_BODY") orelse return error.LivingSource;
    if (spec == core.living_gdn) return std.mem.concat(a, u8, &.{ text[0..end], body, "\n", @embedFile("../../../kernels/common/living_gdn.h"), "\n", @embedFile("../../../kernels/metal/core/living_gdn.metal") });
    if (spec == core.living_qsa) return std.mem.concat(a, u8, &.{ text[0..end], body, "\n", @embedFile("../../../kernels/common/living_qsa.h"), "\n", @embedFile("../../../kernels/metal/core/living_qsa.metal") });
    @compileError("unknown native learner spec");
}
pub fn controlled(a: std.mem.Allocator, comptime spec: type, controls: bool) ![]u8 {
    const text = try metal(a, spec);
    if (!controls) return text;
    defer a.free(text);
    return std.mem.concat(a, u8, &.{ "#define TF_LW_CONTROLS 1\n", text });
}
pub fn named(a: std.mem.Allocator, name: []const u8) ![]u8 {
    inline for (.{ core.living_rows, core.living_gdn, core.living_qsa }, .{ "rows", "gdn", "qsa" }) |spec, label| if (std.mem.startsWith(u8, name, "living-native " ++ label ++ " ")) return controlled(a, spec, std.mem.indexOf(u8, name, "_bad_") != null);
    return error.UnknownLivingNativeLaunch;
}
