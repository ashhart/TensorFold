//! libtfvideo: video input's optional runtime library (zig/src/video/tf_video.c over FFmpeg). `zig build video
//! -Dffmpeg=PREFIX` builds it into zig-out/native/lib from the static FFmpeg tools/zig/build_ffmpeg.sh installs at
//! PREFIX (the release PyAV 19.0.1 ships); the server loads it when the first video arrives and serves without it.

const std = @import("std");

/// The FFmpeg release the decoder is written and checked against (its frames against PyAV's).
pub const pinned = "9.0.2";

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const step = b.step("video", "libtfvideo (video input) over FFmpeg " ++ pinned ++ ": -Dffmpeg=PREFIX from tools/zig/build_ffmpeg.sh");
    const prefix = b.option([]const u8, "ffmpeg", "A static FFmpeg " ++ pinned ++ " install (tools/zig/build_ffmpeg.sh PREFIX), for `zig build video`") orelse {
        step.dependOn(&b.addFail("zig build video needs -Dffmpeg=PREFIX: tools/zig/build_ffmpeg.sh PREFIX builds FFmpeg " ++ pinned ++ " there").step);
        return;
    };
    const root: std.Build.LazyPath = .{ .cwd_relative = prefix };
    const mod = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libc = true });
    mod.addIncludePath(root.path(b, "include"));
    mod.addCSourceFile(.{ .file = b.path("zig/src/video/tf_video.c"), .flags = &.{ "-std=c11", "-Wall" } });
    for ([_][]const u8{ "libavformat.a", "libavcodec.a", "libswscale.a", "libavutil.a" }) |a| mod.addObjectFile(root.path(b, b.pathJoin(&.{ "lib", a })));
    mod.linkSystemLibrary("m", .{});
    const lib = b.addLibrary(.{ .name = "tfvideo", .linkage = .dynamic, .root_module = mod });
    step.dependOn(&b.addInstallArtifact(lib, .{ .dest_dir = .{ .override = .{ .custom = "native/lib" } } }).step);
}
