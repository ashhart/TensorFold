//! macOS: MiniMax Music 3's metallib, the family module over Metal, `zig build music3-generate` and its host tests.
const std = @import("std");

const dir = "zig/kernels/metal/music3";
const sources = [_][]const u8{ "ar.metal", "attn.metal", "sample.metal", "gemm.metal", "dit.metal", "dit_attn.metal", "dav.metal" };

/// Each source compiled with `dir` on the include path (m3_common.h), safe math and precise fp32 functions.
fn metallib(b: *std.Build, metal_std: []const u8) std.Build.LazyPath {
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const out = link.addOutputFileArg("music3.metallib");
    for (sources) |name| {
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", b.fmt("-std={s}", .{metal_std}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise" });
        cc.addPrefixedDirectoryArg("-I", b.path(dir));
        cc.addFileInput(b.path(dir ++ "/m3_common.h"));
        cc.addFileInput(b.path("zig/kernels/metal/nax.h"));
        cc.addFileInput(b.path("zig/kernels/metal/prefill/gemm_nax.metal"));
        cc.addArg("-c");
        cc.addFileArg(b.path(b.fmt("{s}/{s}", .{ dir, name })));
        cc.addArg("-o");
        link.addFileArg(cc.addOutputFileArg(b.fmt("m3_{s}.air", .{std.fs.path.stem(name)})));
    }
    return out;
}

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, metal_std: []const u8, test_step: *std.Build.Step) void {
    const lib = metallib(b, metal_std);
    b.getInstallStep().dependOn(&b.addInstallFile(lib, "lib/music3.metallib").step);
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(lib, "music3.metallib");
    const bytes = b.createModule(.{ .root_source_file = wf.add("metallib.zig", "pub const bytes = @embedFile(\"music3.metallib\");\n") });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/music3/music3_metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "music3_metallib", .module = bytes } },
    });
    const tool = b.createModule(.{
        .root_source_file = b.path("zig/src/families/music3/generate.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "music3", .module = family } },
    });
    const exe = b.addExecutable(.{ .name = "music3-generate", .root_module = tool });
    b.installArtifact(exe);
    b.step("music3-generate", "MiniMax Music 3 lyrics and a caption to a song on Metal").dependOn(&b.addInstallArtifact(exe, .{}).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family })).step);
}
