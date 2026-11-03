//! Native learner shaders are embedded by their source owner, separate from the reusable core launch ABI.
const std = @import("std");
pub fn controlled(a: std.mem.Allocator, comptime spec: type, controls: bool) ![]u8 {
    const text = @embedFile("kernels/metal/core/living_rows.metal");
    const common = @embedFile("kernels/common/living_rows.h");
    if (std.mem.eql(u8, spec.names[0], "tf_lw_linear_back")) {
        const source = try std.mem.replaceOwned(u8, a, text, "// TF_LIVING_BODY", common);
        if (!controls) return source;
        defer a.free(source);
        return std.mem.concat(a, u8, &.{ "#define TF_LW_CONTROLS 1\n", source });
    }
    const end = std.mem.indexOf(u8, text, "// TF_LIVING_BODY") orelse return error.LivingSource;
    const body = if (std.mem.eql(u8, spec.names[0], "tf_lw_g4_copy")) @embedFile("kernels/common/living_gemma.h") else @embedFile("kernels/common/living_qsa.h");
    const entry = if (std.mem.eql(u8, spec.names[0], "tf_lw_g4_copy")) @embedFile("kernels/metal/core/living_gemma.metal") else @embedFile("kernels/metal/core/living_qsa.metal");
    return std.mem.concat(a, u8, &.{ if (controls) "#define TF_LW_CONTROLS 1\n" else "", text[0..end], common, "\n", body, "\n", entry });
}
