//! libtfvideo: video input's optional runtime library (tf_video.c over the static FFmpeg build_ffmpeg.sh installs).

const std = @import("std");

/// The FFmpeg release the decoder is written and checked against (its frames against PyAV's).
pub const pinned = "9.0.2";

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, host_tests: *std.Build.Step) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/platform/video_library.zig"),
        .target = b.graph.host,
        .link_libc = true,
    }) });
    const run = b.addRunArtifact(tests);
    b.step("test-video-library-host", "Video candidate order with a recording opener, without real libraries").dependOn(&run.step);
    host_tests.dependOn(&run.step);
    const cross = b.step("check-video-layout-linux", "Compile the Linux video probe and stub without execution");
    inline for (.{ .aarch64, .x86_64 }) |arch| {
        const linux = b.resolveTargetQuery(.{ .cpu_arch = arch, .os_tag = .linux, .abi = .gnu });
        const probe = layoutProbe(b, linux);
        _ = probe.getEmittedBin();
        cross.dependOn(&probe.step);
        const stub = b.createModule(.{ .target = linux, .link_libc = true });
        stub.addCSourceFile(.{ .file = b.path("zig/tests/tfvideo_stub.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" } });
        const lib = b.addLibrary(.{ .name = "tfvideo", .linkage = .dynamic, .root_module = stub });
        _ = lib.getEmittedBin();
        cross.dependOn(&lib.step);
    }
    if (b.graph.host.result.os.tag == .linux) {
        const layout = b.addSystemCommand(&.{ "python3", "tools/zig/test_tfvideo_layout.py", "--probe" });
        layout.addArtifactArg(layoutProbe(b, b.graph.host));
        b.step("test-video-layout-linux", "Temporary bin/lib layout with an owned stub, no FFmpeg or GPU").dependOn(&layout.step);
    }
    const step = b.step(
        "video",
        "libtfvideo (video input) over FFmpeg " ++ pinned ++ ": -Dffmpeg=PREFIX from tools/zig/build_ffmpeg.sh",
    );
    const prefix = b.option(
        []const u8,
        "ffmpeg",
        "A static FFmpeg " ++ pinned ++ " install (tools/zig/build_ffmpeg.sh PREFIX), for `zig build video`",
    ) orelse {
        step.dependOn(&b.addFail("zig build video needs -Dffmpeg=PREFIX: " ++
            "tools/zig/build_ffmpeg.sh PREFIX builds FFmpeg " ++ pinned ++ " there").step);
        return;
    };
    const root: std.Build.LazyPath = .{ .cwd_relative = prefix };
    const mod = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libc = true });
    mod.addIncludePath(root.path(b, "include"));
    mod.addCSourceFile(.{ .file = b.path("zig/src/video/tf_video.c"), .flags = &.{ "-std=c11", "-Wall" } });
    for ([_][]const u8{
        "libavformat.a",
        "libavcodec.a",
        "libswscale.a",
        "libavutil.a",
    }) |a| mod.addObjectFile(root.path(b, b.pathJoin(&.{ "lib", a })));
    mod.linkSystemLibrary("m", .{});
    const lib = b.addLibrary(.{ .name = "tfvideo", .linkage = .dynamic, .root_module = mod });
    step.dependOn(&b.addInstallArtifact(lib, .{ .dest_dir = .{ .override = .{ .custom = "native/lib" } } }).step);
}

fn layoutProbe(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Step.Compile {
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/tfvideo_layout.zig"), .target = target, .link_libc = true });
    root.addImport("tfvideo", b.createModule(.{ .root_source_file = b.path("zig/src/core/tfvideo.zig"), .target = target, .link_libc = true }));
    return b.addExecutable(.{ .name = "tfvideo-layout", .root_module = root });
}
