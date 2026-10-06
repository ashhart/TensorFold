//! Release archives use the production native modules, not the development CLI or test programs.
const std = @import("std");
const root = @import("../../build.zig");
const cuda = @import("cuda.zig");

pub fn validVersion(v: []const u8) bool {
    var it = std.mem.splitScalar(u8, v, '.');
    var count: usize = 0;
    while (it.next()) |part| {
        if (part.len == 0 or (part.len > 1 and part[0] == '0')) return false;
        for (part) |c| if (!std.ascii.isDigit(c)) return false;
        _ = std.fmt.parseInt(u32, part, 10) catch return false;
        count += 1;
    }
    return count == 3;
}

pub fn targets(b: *std.Build, ids: *std.Build.Module, options: *std.Build.Step.Options, version: ?[]const u8) void {
    const package_tests = b.addSystemCommand(&.{"python3"});
    package_tests.addFileArg(b.path("tools/release/test_package.py"));
    package_tests.has_side_effects = true;
    b.step("test-dist-package", "CPU regression for repeated same-version archives and bundled notices").dependOn(&package_tests.step);
    const step = b.step("dist", "Package macOS arm64 (M1 CPU) and Linux x86_64/aarch64 CUDA servers");
    const smoke = b.step("dist-smoke", "Build archives and smoke the host-compatible archive without a checkout");
    smoke.dependOn(step);
    const fatbins = b.option([]const u8, "dist-fatbins", "Directory of qualified CUDA fatbins to embed for both Linux CPUs");
    const sdk_override = b.option([]const u8, "dist-macos-sdk", "macOS SDK path, default xcrun macosx SDK on a Mac");
    const aot = b.option([]const u8, "dist-cuda-aot", "Optional CUDA capture root with sm<capability>/aot.json and cubins/");
    const host_only = b.option(bool, "dist-host-only", "Verification only: build unusable CUDA archives labelled host-only") orelse false;
    if (version == null) {
        step.dependOn(&b.addFail("dist requires -Dversion=MAJOR.MINOR.PATCH; the release owner chooses the version").step);
        return;
    }
    if (fatbins == null and !host_only) {
        step.dependOn(&b.addFail("dist requires -Ddist-fatbins=DIR for CUDA inference; -Ddist-host-only=true is for CPU verification only").step);
        return;
    }
    if (fatbins != null and host_only) {
        step.dependOn(&b.addFail("dist-host-only and dist-fatbins cannot be combined").step);
        return;
    }
    if (host_only and aot != null) {
        step.dependOn(&b.addFail("dist-cuda-aot requires inference fatbins; it cannot be combined with dist-host-only").step);
        return;
    }
    const fatbin_check = if (fatbins) |dir| cuda.checkDistFatbins(b, dir) else null;
    const sdk = sdk_override orelse if (b.graph.host.result.os.tag == .macos)
        std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }), "\r\n")
    else {
        step.dependOn(&b.addFail("the macOS archive requires a macOS SDK; use a Mac builder or -Ddist-macos-sdk=DIR").step);
        return;
    };
    const platforms = [_]struct { label: []const u8, query: std.Target.Query }{
        .{ .label = "macos-arm64", .query = .{
            .cpu_arch = .aarch64,
            .os_tag = .macos,
            .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_m1 },
            .os_version_min = .{ .semver = .{ .major = 13, .minor = 0, .patch = 0 } },
        } },
        .{ .label = "linux-x86_64", .query = .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu, .glibc_version = .{ .major = 2, .minor = 28, .patch = 0 } } },
        .{ .label = "linux-aarch64", .query = .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu, .glibc_version = .{ .major = 2, .minor = 28, .patch = 0 } } },
    };
    for (platforms) |p| {
        const target = b.resolveTargetQuery(p.query);
        const exe = if (p.query.os_tag == .macos)
            root.distMetalServer(b, target, ids, options, sdk)
        else
            cuda.distServer(b, target, ids, options, fatbins);
        if (p.query.os_tag == .linux) if (fatbin_check) |check| exe.step.dependOn(check);
        const label = if (host_only and p.query.os_tag == .linux) b.fmt("{s}-host-only", .{p.label}) else p.label;
        const pack = b.addSystemCommand(&.{"sh"});
        pack.addFileArg(b.path("tools/release/package.sh"));
        pack.addArgs(&.{ version.?, label });
        pack.addArtifactArg(exe);
        pack.addFileArg(b.path("LICENSE"));
        pack.addFileArg(b.path("NOTICE"));
        pack.addFileArg(b.path("packaging/RUNTIME.md"));
        pack.addDirectoryArg(b.path("LICENSES"));
        pack.addFileArg(b.path("THIRD_PARTY_NOTICES.md"));
        const archives = pack.addOutputDirectoryArg("archives");
        if (aot != null and p.query.os_tag == .linux and !host_only) pack.addDirectoryArg(b.graph.cwdRelativePath(aot.?));
        step.dependOn(&b.addInstallDirectory(.{ .source_dir = archives, .install_dir = .prefix, .install_subdir = "dist" }).step);
        if (b.graph.host.result.os.tag == p.query.os_tag and b.graph.host.result.cpu.arch == p.query.cpu_arch) {
            const check = b.addSystemCommand(&.{"sh"});
            check.has_side_effects = true;
            check.addFileArg(b.path("tools/release/smoke.sh"));
            check.addFileArg(archives.path(b, b.fmt("tensorfold-{s}-{s}.tar.gz", .{ version.?, label })));
            check.addArg(version.?);
            smoke.dependOn(&check.step);
        }
    }
}

test "release versions have exactly three canonical numeric components" {
    for ([_][]const u8{ "0.0.0", "1.2.3", "123.456.789" }) |v| try std.testing.expect(validVersion(v));
    for ([_][]const u8{ "", "1.2", "1.2.3.4", "1..3", "01.2.3", "1.2.3-rc1", "1.2.3+build", "v1.2.3", "1.2.-3", "1.2.3/", "4294967296.0.0" }) |v| try std.testing.expect(!validVersion(v));
}
