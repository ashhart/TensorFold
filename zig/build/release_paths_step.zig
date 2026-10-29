//! zig build check-release-paths: the release's own path check on the host's dist server, without packaging it.
const std = @import("std");
const dist = @import("dist.zig");
const privacy = @import("cuda_release_paths.zig");

/// The check package.sh runs on its stage (check_paths.sh, the checker, the build root), over a stage of one binary.
pub fn binary(b: *std.Build, tool: *std.Build.Step.Compile, bin: std.Build.LazyPath) *std.Build.Step.Run {
    const stage = b.addWriteFiles();
    _ = stage.addCopyFile(bin, "bin/tensorfold-native");
    return privacy.directory(b, tool, stage.getDirectory()).cast(std.Build.Step.Run).?;
}

pub fn steps(b: *std.Build, tool: *std.Build.Step.Compile, ids: *std.Build.Module, options: *std.Build.Step.Options, sdk_override: ?[]const u8) void {
    const step = b.step("check-release-paths", "Build this host's release server as dist does and run the release path check on it");
    const host = b.graph.host.result;
    for (dist.platforms) |p| {
        if (p.query.os_tag != host.os.tag or p.query.cpu_arch != host.cpu.arch) continue;
        const sdk = if (host.os.tag != .macos) "" else sdk_override orelse macosSdk(b) orelse {
            step.dependOn(&b.addFail("check-release-paths needs the macOS SDK: install Xcode or pass -Ddist-macos-sdk=DIR").step);
            return;
        };
        step.dependOn(&binary(b, tool, dist.server(b, b.resolveTargetQuery(p.query), ids, options, sdk, null).getEmittedBin()).step);
        return;
    }
    step.dependOn(&b.addFail("check-release-paths: dist ships no server for this host (macOS arm64, Linux x86_64 or aarch64)").step);
}

fn macosSdk(b: *std.Build) ?[]const u8 {
    var code: u8 = 0;
    const out = b.runAllowFail(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }, &code, .ignore) catch return null;
    return std.mem.trim(u8, out, "\r\n");
}

const head = "\xcf\xfa\xed\xfe\x0c\x00\x00\x01\x00\x00\x00\x00";
const comment = "// The build prefixes {s} under these Metal scalar and pointer definitions.\n#include <metal_stdlib>\n";
const planted = [_]struct { name: []const u8, text: []const u8 }{
    .{ .name = "zig-source", .text = "\x00/opt/tensorfold/server/main.zig\x00" },
    .{ .name = "home", .text = "\x00/home/example/models\x00" },
    .{ .name = "metal-header-1.1.7", .text = "\x00" ++ std.fmt.comptimePrint(comment, .{"weight_only.h"}) ++ "\x00" },
};
const clean = "\x00/dev/urandom\x00/v1/chat/completions\x00" ++ std.fmt.comptimePrint(comment, .{"the shared weight-only header"}) ++ "\x00";

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const step = b.step("test-check-release-paths", "check-release-paths refuses planted source, home and 1.1.7 header paths; passes a clean binary");
    all.dependOn(step);
    const tool = privacy.checker(b);
    inline for (planted) |fixture| {
        const run = binary(b, tool, b.addWriteFiles().add(fixture.name, head ++ fixture.text ++ "\x01\x02"));
        run.setName(b.fmt("check-release-paths refuses {s}", .{fixture.name}));
        run.expectExitCode(1);
        run.expectStdErrEqual("release privacy: refused tensorfold-native (PrivateBuildPath)\n");
        step.dependOn(&run.step);
    }
    const pass = binary(b, tool, b.addWriteFiles().add("clean", head ++ clean ++ "\x01\x02"));
    pass.setName("check-release-paths passes a clean binary");
    pass.expectExitCode(0);
    pass.expectStdErrEqual("");
    step.dependOn(&pass.step);
}
