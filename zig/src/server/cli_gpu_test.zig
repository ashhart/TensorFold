//! GPU CLI admission, value bounds and backend-specific refusals.
const std = @import("std");
const cli = @import("cli.zig");
const Usage = cli.Usage;
const parse = cli.parse;
const request = cli.request;
const capabilities = cli.capabilities;
const Args = cli.Args;
const parallelFixed = cli.parallelFixed;
const cuda_build = cli.Test.cuda_build;
const native_serves = cli.Test.native_serves;
const servedBy = cli.Test.servedBy;
const cudaFlag = cli.Test.cudaFlag;
const gpuFlag = cli.Test.gpuFlag;
const ranks = cli.Test.ranks;

test "--device and --segments: CUDA builds serve them, values checked; --speed-up is the Mac build's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    if (cuda_build) {
        const args = try parse(a, &.{ "m", "--device", "1", "--segments=2" }, &u);
        try std.testing.expectEqual(@as(?u32, 1), args.device);
        try std.testing.expectEqual(@as(?u32, 2), args.segments);
    } else try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--device", "1" }, &u));
    if (cuda_build) try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--speed-up", "pair.json" }, &u)) else try std.testing.expectEqualStrings("pair.json", (try parse(a, &.{ "m", "--speed-up", "pair.json" }, &u)).speed_up.?);
    var out: Args = .{};
    try std.testing.expect(try cudaFlag(a, &out, "--device", "0", &u));
    try std.testing.expectEqual(@as(?u32, 0), out.device);
    try std.testing.expectError(error.Usage, cudaFlag(a, &out, "--device", "-1", &u));
    try std.testing.expectError(error.Usage, cudaFlag(a, &out, "--segments", "0", &u));
    try std.testing.expect(try cudaFlag(a, &out, "--segments", "4", &u));
    try std.testing.expectEqual(@as(?u32, 4), out.segments);
    try std.testing.expect(!try cudaFlag(a, &out, "--port", "1", &u));
    try std.testing.expect(try cudaFlag(a, &out, "--loops", "4", &u));
    try std.testing.expectEqual(@as(?u32, 4), (try request(a, "d", "dory", out)).loops);
    try std.testing.expectError(error.Usage, cudaFlag(a, &out, "--loops", "-1", &u));
    try std.testing.expect(parallelFixed("3") and !parallelFixed("auto") and !parallelFixed("x"));
    // --rope-yarn: a factor that reaches the engine's request; the families and engines that lack YaRN refuse it
    try std.testing.expect(try cudaFlag(a, &out, "--rope-yarn", "4", &u));
    try std.testing.expectEqual(@as(?f64, 4), (try request(a, "d", "qwen3_5_moe", out)).rope_yarn);
    try std.testing.expectError(error.Usage, cudaFlag(a, &out, "--rope-yarn", "four", &u));
    if (cuda_build) try std.testing.expectEqual(@as(?f64, 2), (try parse(a, &.{ "m", "--rope-yarn", "2" }, &u)).rope_yarn);
}

test "the GPU engine's flags: refused and unlisted unless the backend serves them, values checked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    // each flag, a value, and the `serves` entry that turns it on
    const gated = [_][3][]const u8{ .{ "--tp", "2", "--tp" }, .{ "--rank", "0", "--tp" }, .{ "--master", "node0", "--tp" }, .{ "--master-port", "29600", "--tp" }, .{ "--policy", "matrix=off", "--policy" }, .{ "--mtp-drafts", "2", "--mtp-drafts" }, .{ "--mtp-confidence", "0.5", "--mtp-confidence" }, .{ "--checkpoint-slots", "3", "--checkpoint-slots" }, .{ "--spill-gib", "2", "--spill-gib" } };
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "0" });
    for (gated) |g| {
        const listed = std.mem.indexOf(u8, out.written(), try std.fmt.allocPrint(a, "\"{s}\"", .{g[0]})) != null;
        try std.testing.expectEqual(servedBy(native_serves, g[2]), listed);
        if (!listed) try std.testing.expectError(error.Usage, parse(a, &.{ "m", g[0], g[1] }, &u));
    }
    try std.testing.expect(servedBy(&.{ "--tp", "--policy" }, "--tp") and !servedBy(&.{"--tp"}, "--policy") and !servedBy(&.{}, "--tp"));
    var args: Args = .{};
    for ([_][2][]const u8{ .{ "--tp", "4" }, .{ "--rank", "2" }, .{ "--master", "node0" }, .{ "--master-port", "29600" }, .{ "--mtp-drafts", "2" }, .{ "--mtp-confidence", "0.5" }, .{ "--checkpoint-slots", "3" }, .{ "--spill-gib", "1.5" }, .{ "--policy", "matrix=off" }, .{ "--policy", "kernels=reference" } }) |g| try std.testing.expect(try gpuFlag(a, &args, g[0], g[1], &u));
    try std.testing.expectEqual(@as(?f64, 1.5), args.spill_gib);
    try std.testing.expectEqual(@as(u32, 4), args.tp);
    try std.testing.expectEqual(@as(u32, 2), args.rank);
    try std.testing.expectEqualStrings("node0", args.master);
    try std.testing.expectEqual(@as(u16, 29600), args.master_port);
    try std.testing.expectEqual(@as(?u32, 2), args.mtp_drafts);
    try std.testing.expectEqual(@as(?f64, 0.5), args.mtp_confidence);
    try std.testing.expectEqual(@as(?i64, 3), args.checkpoint_slots);
    try std.testing.expectEqualStrings("matrix=off,kernels=reference", args.policy);
    try ranks(&u, a, args);
    args.rank = 4;
    try std.testing.expectError(error.Usage, ranks(&u, a, args));
    for ([_][2][]const u8{ .{ "--rank", "-1" }, .{ "--master-port", "70000" }, .{ "--mtp-drafts", "65" }, .{ "--mtp-confidence", "1.5" }, .{ "--checkpoint-slots", "-1" }, .{ "--spill-gib", "-1" }, .{ "--spill-gib", "nan" }, .{ "--policy", "matrix" } }) |g| try std.testing.expectError(error.Usage, gpuFlag(a, &args, g[0], g[1], &u));
    try std.testing.expect(!try gpuFlag(a, &args, "--port", "1", &u));
}
