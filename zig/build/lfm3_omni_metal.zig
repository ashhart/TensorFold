//! macOS: LFM3-Omni's metallib, the family module over Metal, `zig build lfm3-omni-generate` and its host tests.
const std = @import("std");

const dir = "zig/kernels/metal/lfm3_omni";

/// The kernels with safe math and precise fp32 functions (the CUDA port's rounding op for op) and the tensor-unit ones.
fn metallib(b: *std.Build, metal_std: []const u8) std.Build.LazyPath {
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const out = link.addOutputFileArg("lfm3_omni.metallib");
    for ([_][]const u8{ "omni", "omni_gemm", "omni_rows", "omni_nax" }) |name| {
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", b.fmt("-std={s}", .{metal_std}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise", "-c" });
        cc.addFileArg(b.path(b.fmt("{s}/{s}.metal", .{ dir, name })));
        cc.addFileInput(b.path(dir ++ "/om_common.h"));
        cc.addFileInput(b.path("zig/kernels/metal/nax.h"));
        cc.addFileInput(b.path("zig/kernels/metal/prefill/gemm_nax.metal"));
        cc.addArg("-o");
        link.addFileArg(cc.addOutputFileArg(b.fmt("{s}.air", .{name})));
    }
    return out;
}

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, metal_std: []const u8, test_step: *std.Build.Step) void {
    const lib = metallib(b, metal_std);
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(lib, "lfm3_omni.metallib");
    const bytes = b.createModule(.{ .root_source_file = wf.add("metallib.zig", "pub const bytes = @embedFile(\"lfm3_omni.metallib\");\n") });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    // the WAV reader / resampler is Drama 3's
    const wav = b.createModule(.{ .root_source_file = b.path("zig/src/families/drama3/wav.zig"), .target = target, .optimize = optimize });
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/lfm3_omni/omni_metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "wav", .module = wav }, .{ .name = "lfm3_omni_metallib", .module = bytes } },
    });
    const tool = b.createModule(.{
        .root_source_file = b.path("zig/src/families/lfm3_omni/generate.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "lfm3_omni", .module = family } },
    });
    const exe = b.addExecutable(.{ .name = "lfm3-omni-generate", .root_module = tool });
    b.step("lfm3-omni-generate", "LFM3-Omni speech in, text and speech out on Metal, with its stage dumps").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const host = b.addRunArtifact(b.addTest(.{ .root_module = family }));
    b.step("test-lfm3-omni-host", "LFM3-Omni's mel front end, prompt, overlap-add and Metal argument layouts on the host").dependOn(&host.step);
    test_step.dependOn(&host.step);
}
