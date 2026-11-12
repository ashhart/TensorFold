//! `zig build bench-stream-scan`: zig/src/server/stream_scan_bench.zig's long replies, timed a chunk at a time.
const std = @import("std");
const Module = std.Build.Module;

/// The benchmark at the HTTP side's shipped optimization (ReleaseSafe) unless `-Dstream-scan-optimize` names another.
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, api: *Module, tokenizer: *Module, template: *Module, core: *Module) void {
    const about = "bench-stream-scan's build: safe (default, as tensorfold-native's server is built) or fast";
    const optimize = b.option(std.builtin.OptimizeMode, "stream-scan-optimize", about) orelse .safe;
    const bench = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/src/server/root.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "engine_api", .module = api },
                .{ .name = "tokenizer", .module = tokenizer },
                .{ .name = "template", .module = template },
                .{ .name = "core", .module = core },
            },
        }),
        .filters = &.{"stream scan bench:"},
    });
    _ = @import("server.zig").addTo(b, bench.root_module, target, optimize);
    const run = b.addRunArtifact(bench);
    run.has_side_effects = true; // a benchmark: its times are the point, so it runs every time
    b.step("bench-stream-scan", "A 32k-token thinking reply through the stream routes, each chunk's request-thread time").dependOn(&run.step);
}
