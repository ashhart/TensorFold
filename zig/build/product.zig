//! `zig build -Dgpu=<gpu>`: one stripped server for one hardware GPU, every model family, nothing else.
const std = @import("std");
const catalog = @import("cuda_kernels.zig");
const caps = @import("../src/hip/caps.zig");

/// The `-Dgpu` name, or null for the development build.
pub fn requested(b: *std.Build) ?[]const u8 {
    return b.option([]const u8, "gpu", "Build one server for one GPU: sm_80, sm_86, sm_89, sm_120, sm_121, spark, or a HIP arch (gfx1201 ...)");
}

const cuda_sms = [_][]const u8{ "80", "86", "89", "120", "121" };

/// A CUDA SM as `-Dsm` spells it, or an exact HIP architecture.
const Gpu = union(enum) { cuda: []const u8, hip: []const u8 };

/// The backend and architecture `name` means, or null.
fn parseGpu(name: []const u8) ?Gpu {
    if (std.mem.eql(u8, name, "spark")) return .{ .cuda = "121" };
    if (std.mem.startsWith(u8, name, "sm_")) {
        for (cuda_sms) |sm| if (std.mem.eql(u8, name[3..], sm)) return .{ .cuda = sm };
        return null;
    }
    if (std.mem.indexOfScalar(u8, name, ':') == null and caps.Caps.of(name) != null) return .{ .hip = name };
    return null;
}

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

pub fn build(b: *std.Build, name: []const u8) void {
    const gpu = parseGpu(name) orelse fail("-Dgpu={s}: not a GPU this build knows; use sm_80, sm_86, sm_89, sm_120, sm_121, spark, or an exact HIP arch from zig/src/hip/caps.zig", .{name});
    // the development build's architecture options: -Dgpu sets the architecture, so they are refused here
    if (b.option([]const u8, "sm", "Refused with -Dgpu") != null) fail("-Dsm: -Dgpu sets the architecture", .{});
    if (b.option(bool, "hip", "Refused with -Dgpu") != null) fail("-Dhip: -Dgpu={s} chooses the backend", .{name});
    const hip_options = @import("hip.zig").options(b); // the only declaration of the HIP options, so they parse once
    if (hip_options.hip_arch != null) fail("-Dhip-arch: -Dgpu sets the architecture", .{});
    if (gpu == .cuda) for ([_][]const u8{ "hipcc", "hip-include", "hip-device-lib", "hip-relax-branches" }) |option| {
        if (b.user_input_options.contains(option)) fail("-D{s}: -Dgpu={s} builds CUDA, where the HIP options do not apply", .{ option, name });
    };
    const target = b.standardTargetOptions(.{});
    const os = target.result.os.tag;
    switch (gpu) {
        .cuda => if (os != .linux and os != .windows) fail("-Dgpu={s}: CUDA servers build for Linux and Windows targets", .{name}),
        .hip => if (os != .linux) fail("-Dgpu={s}: HIP servers build for Linux targets", .{name}),
    }
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "debug, safe, fast or small (default fast)") orelse .fast;
    const strip = b.option(bool, "strip", "No debug info in the server (default true with -Dgpu)") orelse true;
    const release_version = b.option([]const u8, "version", "Release version, exactly MAJOR.MINOR.PATCH");
    if (release_version) |v| if (!@import("dist.zig").validVersion(v)) fail("-Dversion must be exactly MAJOR.MINOR.PATCH", .{});
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", release_version orelse @import("../../build.zig.zon").version);
    build_options.addOption(bool, "slide_test_hooks", false);
    const exe = switch (gpu) {
        .cuda => |sm| cudaServer(b, target, optimize, build_options, name, sm),
        .hip => |arch| hipServer(b, target, optimize, build_options, hip_options, arch),
    };
    exe.root_module.strip = strip; // before the install step, which decides then whether a .pdb comes with it
    const dir = switch (gpu) {
        .cuda => "native/bin",
        .hip => "native-hip/bin",
    };
    b.getInstallStep().dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = dir } } }).step);
    @import("sys.zig").wire(b);
    @import("checkpoint_config.zig").wire(b);
    @import("save_guard.zig").wire(b);
}

/// The development `native` server for one SM, with every image built for it (nvcc) or checked (prebuilt).
fn cudaServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options, name: []const u8, sm: []const u8) *std.Build.Step.Compile {
    const cuda_build = @import("cuda.zig");
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the kernel images");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files, each checked for its one SASS profile");
    if ((nvcc == null) == (prebuilt == null)) fail("-Dgpu={s} needs exactly one of -Dnvcc and -Dfatbins", .{name});
    const version: ?std.Build.LazyPath = if (nvcc) |tool| blk: {
        const run = b.addSystemCommand(&.{ tool, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    const target_sm = catalog.Target.parse(sm);
    var images: [catalog.kernels.len]?std.Build.LazyPath = @splat(null);
    const check = b.addRunArtifact(b.addExecutable(.{ .name = "check-product-fatbins", .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_product_fatbins.zig"), .target = b.graph.host }) }));
    for (catalog.kernels, &images) |k, *image| {
        if (nvcc) |tool| {
            image.* = @import("cuda_fatbin.zig").fatbin(b, tool, version.?, k, sm);
        } else {
            const path = b.graph.cwdRelativePath(b.pathJoin(&.{ prebuilt.?, b.fmt("{s}.fatbin", .{k.name}) }));
            image.* = path;
            // an image below its kernel's min_sm is an inert empty file, as the nvcc path makes it
            if (catalog.profile(k, target_sm)) |p| {
                check.addArg(b.fmt("{d}{s}", .{ p.number, if (p.arch_specific) "a" else "" }));
                check.addFileArg(path);
            }
        }
    }
    const rt = @import("cuda_runtime.zig").runtime(b, target, optimize, &images);
    const ids_files = b.addWriteFiles();
    _ = ids_files.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
    const draft_ids = b.createModule(.{ .root_source_file = ids_files.add("draft_ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");\n") });
    const mods = cuda_build.family(b, target, optimize, rt, draft_ids);
    const exe = cuda_build.nativeServer(b, target, optimize, rt, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision, mods.tokenizer, mods.living, build_options, false, null);
    if (prebuilt != null) exe.step.dependOn(&check.step);
    return exe;
}

/// The development `native-hip` server for one arch.
fn hipServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options, o: @import("hip.zig").Options, arch: []const u8) *std.Build.Step.Compile {
    const hip_build = @import("hip.zig");
    const hipcc = o.hipcc orelse "hipcc";
    const hipcc_resolved = b.findProgram(.{ .names = &.{hipcc} }) orelse hipcc;
    const hip_include = o.hip_include orelse b.pathResolve(&.{ std.fs.path.dirname(hipcc_resolved) orelse "/opt/rocm/bin", "..", "include" });
    const objects = hip_build.kernelObjects(b, hipcc, hipcc_resolved, o.device_lib, o.relax_branches, arch);
    const e = hip_build.engineModules(b, target, optimize, objects, hip_include);
    return hip_build.server(b, target, e, build_options);
}
