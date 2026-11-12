//! Pins for the server's flag dependencies: a flag that needs another, a refusal, or a build gate.
const std = @import("std");
const cli = @import("cli.zig");
const Usage = cli.Usage;
const parse = cli.parse;

test "flag deps: --cfg-batched and --cfg-force-guided need a gamma switch" {
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest; // CFG flags: macOS only
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    for ([_][]const u8{ "--cfg-batched", "--cfg-force-guided" }) |flag| {
        try std.testing.expectError(error.Usage, parse(a, &.{ "m", flag }, &u));
        try std.testing.expectEqualStrings("--cfg-batched and --cfg-force-guided need --cfg-gamma or --cfg-allow-request-gamma", u.message);
        try std.testing.expectError(error.Usage, parse(a, &.{ "m", flag, "--cfg-gamma", "0" }, &u));
        try std.testing.expectEqualStrings("--cfg-batched and --cfg-force-guided need --cfg-gamma or --cfg-allow-request-gamma", u.message);
    }
    const gamma = try parse(a, &.{ "m", "--cfg-batched", "--cfg-force-guided", "--cfg-gamma", "1.5" }, &u);
    try std.testing.expect(gamma.cfg.batched and gamma.cfg.force_guided);
    try std.testing.expectEqual(@as(f32, 1.5), gamma.cfg.gamma);
    const requested = try parse(a, &.{ "m", "--cfg-force-guided", "--cfg-allow-request-gamma" }, &u);
    try std.testing.expect(requested.cfg.force_guided and requested.cfg.request_gamma);
    const alone = try parse(a, &.{ "m", "--cfg-allow-request-gamma" }, &u);
    try std.testing.expect(alone.cfg.request_gamma);
}

test "flag deps: flags the build does not serve are refused with the Python engine's pointer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const pointer = "is not served by the native engine yet; the Python engine 0.6.6 serves it: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6";
    const gated = "is not served by the native engine; the Python engine 0.6.6 serves it: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6";
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--vision-offload" }, &u));
    try std.testing.expectEqualStrings("--vision-offload " ++ pointer, u.message);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--kv-dtype", "int8" }, &u));
    try std.testing.expectEqualStrings("--kv-dtype " ++ pointer, u.message);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--no-prefill-fp8" }, &u));
    try std.testing.expectEqualStrings("--no-prefill-fp8 " ++ pointer, u.message);
    const macos = @import("builtin").os.tag == .macos;
    const native = if (macos) "mlx" else "cuda";
    const other = if (macos) "cuda" else "mlx";
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--backend", other }, &u));
    try std.testing.expectEqualStrings("--backend " ++ other ++ " " ++ gated, u.message);
    try std.testing.expectEqualStrings("auto", (try parse(a, &.{ "m", "--backend", "auto" }, &u)).backend);
    try std.testing.expectEqualStrings(native, (try parse(a, &.{ "m", "--backend", native }, &u)).backend);
}

test "flag deps: --backend cpu on a GPU build names the CPU engine's own build" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var u: Usage = .{};
    try std.testing.expectError(error.Usage, parse(arena.allocator(), &.{ "m", "--backend", "cpu" }, &u));
    try std.testing.expectEqualStrings("--backend cpu is served by the CPU engine's own build: zig build native-cpu puts its tensorfold-native in zig-out/native-cpu/bin", u.message);
}
