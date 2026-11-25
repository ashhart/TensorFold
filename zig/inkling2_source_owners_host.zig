//! Pinned main library identities, plus the explicitly rebased DFlash2 libraries.
const std = @import("std");
const ink = @import("inkling2_metal");
const ks = @import("kernel_sources");
fn equal(source: []const u8, bytes: usize, expected: []const u8) !void {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(source, &hash, .{});
    try std.testing.expectEqual(bytes, source.len);
    try std.testing.expectEqualStrings(expected, &std.fmt.bytesToHex(hash, .lower));
}
test "unmodified Inkling and LFM2 libraries retain main and rebased DFlash2 sources have pinned identities" {
    try equal(ink.attention, 25176, "446df33916bbd384fb2dec029d0dc2c34acd068047eb6a6b7dca778aaf41345a");
    try equal(ink.prompt_attention, 36507, "2e690463846a822b5c087139eaa3dfbc9a980caa7753e36cb3562afc88a17abd");
    try equal(ink.prompt_attention_pre_m5, 36531, "28b71a25e1aab7174cf121f3ae05f84afb165970443a91f6978c06628630935b");
    try equal(ink.glue, 40891, "d38acec59342de2cf8f69824eb7b73c633b56fad03fdaa6cb1029d99c05ced2e");
    try equal(ink.experts, 36126, "9a59ac2de326eb3fcf0fd832f5e1923272f099505be165a06611a2e32e6e3bf6");
    try equal(ink.experts_mma, 48056, "4448aba61d8e99534d2c2c005fee440aa74f4d5098e57f4b2a1588ed22096b36");
    try equal(ink.dense_mma, 5389, "6d0b015dd66035efbd110ae4d8dd690ca677ed9c500a83837aee7269161b334a");
    try equal(ink.dense_tiled, 4324, "cdddebd86d265d9a79fd93b80894266750ca267e7ffeb76c481e6e73ab03c3ee");
    try equal(ink.loop, 18092, "7c721243e573ffd29ae6c3a594a193f760f8ce0672446e0c9b590842a3ef3793");
    try equal(ink.loop_mma, 23241, "bcb090e30b221ce5978dcc9ec28d8b163d1665b264543cfee848cd4ffb95c095");
    try equal(ink.runtime, 7440, "2ce8a74513b74e4bd9ffb3ab2b4e27268a6297ade723fbaa41813026d8f9cfad");
    try equal(ink.tree, 4183, "03cd2b1384788b46a6de79aacc8edcc854f9a978c81ac3cd2336ae979aab7d69");
    try equal(ink.inkling2_dspark, 34958, "280e181c0ea975da90252417352bd2cbbde2a653951e4e12ecb3dc436b882cb9");
    try equal(ink.inkling2_dspark_pre_m5, 34982, "ed6e11de7d7f14e4821132f3598c7fee0c63f1b211682726911769b2cd042932");
    try equal(ink.vision, 40124, "f3bccee002e5b0d7572dc25316538b384e397a4f5dba3f4ac99839e434d848c6");
    try equal(ink.inkling2_dflash2, 24617, "97ecd25c5c6cddb4434b5ee3659d7350b912a7c196eb3c3c15dbbd62aee53026");
    try equal(ink.inkling2_dflash2_fast, 18466, "9e0e807660a4fb1c2be258f7681beddf839f719f9a5f1ff1011dfd4d735910e7");
    try std.testing.expectEqualStrings(ks.inkling2.inkling2_dflash2_fast, ink.inkling2_dflash2_fast);
    try equal(ink.inkling2_dflash2_fast_pre_m5, 18490, "64ae54458a4296a30c3c1c3d16e0efc41069ac25c98bf5320a32a0d793ac4ccd");
    try std.testing.expectEqualStrings(ks.inkling2.inkling2_dflash2_fast_pre_m5, ink.inkling2_dflash2_fast_pre_m5);
    try equal(ink.sample, 10443, "747835e5a84d76d7d838958e2677aa5af62e4ebf2e612520bfc759e967d44c81");
    try equal(ks.inkling2_glue, 26246, "0d1f7bb8d2721671eb744a52a39e1e9c26dcb866a026ee1f9c5785ec7b8c1d06");
    try equal(ks.inkling2_sample, 10443, "747835e5a84d76d7d838958e2677aa5af62e4ebf2e612520bfc759e967d44c81");
    try equal(ks.lfm2_prompt_masked.source, 16617, "5b5edd3224410f5bebb192cb2781692e21a17baad26be73b8e842f2bee92b0f0");
    try equal(ks.inkling2_runtime, 7440, "2ce8a74513b74e4bd9ffb3ab2b4e27268a6297ade723fbaa41813026d8f9cfad");
}
