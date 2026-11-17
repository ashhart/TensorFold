//! macOS: ZONOS3 Mini's metallib, the family module over Metal and `zig build zonos3-generate`.
const std = @import("std");

const dir = "zig/kernels/metal/zonos3";
const sources = [_][]const u8{ "backbone.metal", "codec.metal", "encoder.metal", "teacher.metal" };

/// Each source compiled with `dir` on the include path (z3_common.h), safe math, precise fp32 functions and no
/// contraction: a sum's fused steps are the ones the source writes, as `--fmad=false` keeps them on CUDA.
fn metallib(b: *std.Build, metal_std: []const u8) std.Build.LazyPath {
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const out = link.addOutputFileArg("zonos3.metallib");
    for (sources) |name| {
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", b.fmt("-std={s}", .{metal_std}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise", "-ffp-contract=off" });
        cc.addPrefixedDirectoryArg("-I", b.path(dir));
        cc.addFileInput(b.path(dir ++ "/z3_common.h"));
        cc.addArg("-c");
        cc.addFileArg(b.path(b.fmt("{s}/{s}", .{ dir, name })));
        cc.addArg("-o");
        link.addFileArg(cc.addOutputFileArg(b.fmt("z3_{s}.air", .{std.fs.path.stem(name)})));
    }
    return out;
}

/// The family over Metal, the core and the lane core (its keyed sampler), with the metallib embedded.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, metal_std: []const u8) *std.Build.Module {
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(metallib(b, metal_std), "zonos3.metallib");
    const bytes = b.createModule(.{ .root_source_file = wf.add("metallib.zig", "pub const bytes = @embedFile(\"zonos3.metallib\");\n") });
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/zonos3/zonos3_metal.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "zonos3_metallib", .module = bytes } } });
}

/// ZONOS3 in the Metal engines as `zonos3`, over the core Drama's addNative gave them (torch_pth and wav).
pub fn addNative(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, native: *std.Build.Module, lanes: *std.Build.Module, metal_std: []const u8) void {
    native.addImport("zonos3", module(b, target, optimize, metal, native.import_table.get("core").?, lanes, metal_std));
}

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, lanes: *std.Build.Module, metal_std: []const u8, test_step: *std.Build.Step) void {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const family = module(b, target, optimize, metal, core, lanes, metal_std);
    const tool = b.createModule(.{ .root_source_file = b.path("zig/src/families/zonos3/generate.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "zonos3", .module = family } } });
    const exe = b.addExecutable(.{ .name = "zonos3-generate", .root_module = tool });
    b.step("zonos3-generate", "ZONOS3 Mini on Metal against the reference's dumps").dependOn(&b.addInstallArtifact(exe, .{}).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = family })).step);
}
