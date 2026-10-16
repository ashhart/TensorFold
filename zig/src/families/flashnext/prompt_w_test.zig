//! The prompt kernels at any width compile from one text, and fused members' offsets follow their own widths.
const std = @import("std");
const ks = @import("kernel_sources");

test "the any-width prompt text names its kernels and declares no width of its own" {
    const text = ks.flashnext_qmmw;
    for ([_][]const u8{ "tf_qmmw_t_nax", "tf_qmmw_splitk_nax", "tf_gather_guw_nax_##BM", "tf_gather_qmmw_nax_##BM" }) |name|
        try std.testing.expect(std.mem.indexOf(u8, text, name) != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "#define WB") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "#define WG") == null);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "#include \"../nax.h\""));
}
