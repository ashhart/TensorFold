//! Host-only compile and policy checks for the guarded Qwen FP8 prompt path.
const std = @import("std");

pub fn steps(
    b: *std.Build,
    draft_ids: *std.Build.Module,
    comptime kernel_count: usize,
    comptime makeRuntime: anytype,
    comptime makeFamily: anytype,
    server: *std.Build.Module,
    native: *std.Build.Module,
    target: std.Build.ResolvedTarget,
) void {
    const prompt = b.step("test-fp8-prefill-host", "Qwen FP8 prompt flags, admission, allocation and launch ABI; no GPU");
    const family = makeFamily(b, target, .debug, makeRuntime(b, target, .debug, &.{}), draft_ids);
    for ([_]*std.Build.Module{ server, native, family.qwen35 }) |module| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = module, .filters = &.{"FP8 prefill:"} }));
        run.has_side_effects = true;
        prompt.dependOn(&run.step);
    }
    const empty = b.addWriteFiles().add("fp8-compile-only-empty-image", "");
    const images: [kernel_count]?std.Build.LazyPath = @splat(empty);
    const cuda = makeRuntime(b, target, .debug, &images);
    const modules = makeFamily(b, target, .debug, cuda, draft_ids);
    const object = b.addObject(.{ .name = "fp8-prefill-compile-only", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/qwen35/fp8_prefill_compile.zig"),
        .target = target,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = modules.qwen35 } },
    }) });
    _ = object.getEmittedBin();
    b.step("compile-fp8-prefill-host", "Compile GPU-enabled Qwen FP8 init, forward and cleanup with inert images; never run").dependOn(&object.step);
}
