//! macOS: Drama 3's metallib, the family module over Metal, `zig build drama3-generate` and its host tests.
const std = @import("std");

const dir = "zig/kernels/metal/drama3";
const sources = [_][]const u8{ "dense.metal", "gdn.metal", "attn.metal", "moe.metal", "nvfp4.metal", "guidance.metal", "sample.metal", "codec.metal" };

/// Drama in the Metal engines: its `core` is the engine's safetensors, the server's tokenizer and frame_rounds.
pub fn addNative(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, native: *std.Build.Module, metal_std: []const u8) void {
    const tokenizer = native.import_table.get("tokenizer") orelse tok: {
        const t = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
        native.addImport("tokenizer", t);
        break :tok t;
    };
    const rounds = b.createModule(.{ .root_source_file = b.path("zig/src/core/frame_rounds.zig"), .target = target, .optimize = optimize });
    const shim = b.addWriteFiles().add("drama3_core.zig", "pub const safetensors = @import(\"tensorfold\").safetensors;\npub const tokenizer = @import(\"tokenizer\");\npub const frame_rounds = @import(\"frame_rounds\");\n");
    const core = b.createModule(.{ .root_source_file = shim, .target = target, .optimize = optimize, .imports = &.{
        .{ .name = "tensorfold", .module = native.import_table.get("tensorfold").? },
        .{ .name = "tokenizer", .module = tokenizer },
        .{ .name = "frame_rounds", .module = rounds },
    } });
    const lib = metallib(b, metal_std);
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(lib, "drama3.metallib");
    const bytes = b.createModule(.{ .root_source_file = wf.add("metallib.zig", "pub const bytes = @embedFile(\"drama3.metallib\");\n") });
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/drama3/drama3_metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "metal", .module = metal }, .{ .name = "drama3_metallib", .module = bytes } },
    });
    native.addImport("core", core);
    native.addImport("drama3", family);
}

/// Each source compiled with `dir` on the include path (d3_common.h), safe math and precise fp32 functions.
fn metallib(b: *std.Build, metal_std: []const u8) std.Build.LazyPath {
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const out = link.addOutputFileArg("drama3.metallib");
    for (sources) |name| {
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", b.fmt("-std={s}", .{metal_std}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise" });
        cc.addPrefixedDirectoryArg("-I", b.path(dir));
        cc.addFileInput(b.path(dir ++ "/d3_common.h"));
        cc.addArg("-c");
        cc.addFileArg(b.path(b.fmt("{s}/{s}", .{ dir, name })));
        cc.addArg("-o");
        link.addFileArg(cc.addOutputFileArg(b.fmt("d3_{s}.air", .{std.fs.path.stem(name)})));
    }
    return out;
}

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, metal_std: []const u8, test_step: *std.Build.Step) void {
    const lib = metallib(b, metal_std);
    b.getInstallStep().dependOn(&b.addInstallFile(lib, "lib/drama3.metallib").step);
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(lib, "drama3.metallib");
    const bytes = b.createModule(.{ .root_source_file = wf.add("metallib.zig", "pub const bytes = @embedFile(\"drama3.metallib\");\n") });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/drama3/drama3_metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "drama3_metallib", .module = bytes } },
    });
    const quant_check = b.addExecutable(.{
        .name = "drama3-metal-quant-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests/drama_nvfp4_metal.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "drama3", .module = family }, .{ .name = "drama3_metallib", .module = bytes } },
        }),
    });
    b.step("drama3-metal-quant-check", "Build the NVFP4 Metal operator checker; does not execute GPU work").dependOn(&b.addInstallArtifact(quant_check, .{}).step);
    const tool = b.createModule(.{
        .root_source_file = b.path("zig/src/families/drama3/generate.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "drama3", .module = family } },
    });
    const exe = b.addExecutable(.{ .name = "drama3-generate", .root_module = tool });
    b.installArtifact(exe);
    b.step("drama3-generate", "Fish Audio Drama 3 text to speech on Metal").dependOn(&b.addInstallArtifact(exe, .{}).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family })).step);
}
