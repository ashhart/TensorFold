//! Release archives use the production native modules, not the development CLI or test programs.
const std = @import("std");
const root = @import("../../build.zig");
const cuda = @import("cuda.zig");
const grammar = @import("grammar.zig");

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

pub const Platform = struct { label: []const u8, query: std.Target.Query };
pub const platforms = [_]Platform{
    .{ .label = "macos-arm64", .query = .{
        .cpu_arch = .aarch64,
        .os_tag = .macos,
        .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_m1 },
        .os_version_min = .{ .semver = .{ .major = 13, .minor = 0, .patch = 0 } },
    } },
    .{ .label = "linux-x86_64", .query = .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu, .glibc_version = .{ .major = 2, .minor = 28, .patch = 0 } } },
    .{ .label = "linux-aarch64", .query = .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu, .glibc_version = .{ .major = 2, .minor = 28, .patch = 0 } } },
    .{ .label = "windows-x86_64", .query = .{ .cpu_arch = .x86_64, .os_tag = .windows, .abi = .gnu } },
};

pub fn selected(label: []const u8, choice: ?[]const u8) bool {
    const value = choice orelse return !std.mem.startsWith(u8, label, "windows-");
    return std.mem.eql(u8, value, "all") or std.mem.eql(u8, label, value);
}

/// The server binary an archive ships: Metal over the macOS SDK, else CUDA with the given fatbins (null: host-only).
pub fn server(b: *std.Build, target: std.Build.ResolvedTarget, ids: *std.Build.Module, options: *std.Build.Step.Options, sdk: []const u8, fatbins: ?[]const u8) *std.Build.Step.Compile {
    return if (target.result.os.tag == .macos) root.distMetalServer(b, target, ids, options, sdk) else cuda.distServer(b, target, ids, options, fatbins);
}

pub fn targets(b: *std.Build, ids: *std.Build.Module, options: *std.Build.Step.Options, version: ?[]const u8, xgrammar: ?[]const u8) void {
    const privacy = @import("cuda_release_paths.zig").checker(b);
    const package_tests = b.addSystemCommand(&.{"python3"});
    package_tests.addFileArg(b.path("tools/release/check_package.py"));
    package_tests.addPrefixedFileArg("--path-checker=", privacy.getEmittedBin());
    package_tests.has_side_effects = true;
    b.step("test-dist-package", "CPU regression for repeated same-version archives and bundled notices").dependOn(&package_tests.step);
    const step = b.step("dist", "Package native servers; dist-platform selects Windows x86_64 or a Unix target");
    const smoke = b.step("dist-smoke", "Build archives and smoke the host-compatible archive without a checkout");
    smoke.dependOn(step);
    const fatbins = b.option([]const u8, "dist-fatbins", "Directory of qualified CUDA fatbins to embed for both Linux CPUs");
    const sdk_override = b.option([]const u8, "dist-macos-sdk", "macOS SDK path, default xcrun macosx SDK on a Mac");
    const aot = b.option([]const u8, "dist-cuda-aot", "Optional CUDA capture root with sm<capability>/aot.json and cubins/");
    const host_only = b.option(bool, "dist-host-only", "Verification only: build unusable CUDA archives labelled host-only") orelse false;
    const choice = b.option([]const u8, "dist-platform", "One platform label or all; default preserves the three Unix archives");
    const windows_runtime = b.option([]const u8, "dist-windows-runtime", "Reviewed Windows DLL overlay with manifests, licenses and notices");
    @import("release_paths_step.zig").steps(b, privacy, ids, options, sdk_override);
    if (choice) |value| if (!std.mem.eql(u8, value, "all")) {
        const known = for (platforms) |p| {
            if (std.mem.eql(u8, p.label, value)) break true;
        } else false;
        if (!known) {
            step.dependOn(&b.addFail("unknown dist-platform; use a platform label from packaging/README.md or all").step);
            return;
        }
    };
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
    const unix = selected("macos-arm64", choice) or selected("linux-x86_64", choice) or selected("linux-aarch64", choice);
    if (xgrammar == null and !host_only and unix) {
        step.dependOn(&b.addFail("dist requires -Dxgrammar=DIR (structured output's libtfgrammar): " ++ grammar.clone).step);
        return;
    }
    if (selected("windows-x86_64", choice) and !host_only and windows_runtime == null) {
        step.dependOn(&b.addFail("Windows dist requires -Ddist-windows-runtime=DIR with reviewed DLLs, licenses and notices").step);
        return;
    }
    const fatbin_check = if (fatbins) |dir| cuda.checkDistFatbins(b, dir, privacy) else null;
    const sdk = if (!selected("macos-arm64", choice)) "" else sdk_override orelse if (b.graph.host.result.os.tag == .macos)
        std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }), "\r\n")
    else {
        step.dependOn(&b.addFail("the macOS archive requires a macOS SDK; use a Mac builder or -Ddist-macos-sdk=DIR").step);
        return;
    };
    for (platforms) |p| {
        if (!selected(p.label, choice)) continue;
        const target = b.resolveTargetQuery(p.query);
        const exe = server(b, target, ids, options, sdk, fatbins);
        const windows = p.query.os_tag == .windows;
        if (p.query.os_tag != .macos) if (fatbin_check) |check| exe.step.dependOn(check);
        const label = if (host_only and p.query.os_tag != .macos) b.fmt("{s}-host-only", .{p.label}) else p.label;
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
        pack.addPrefixedFileArg("--path-checker=", privacy.getEmittedBin());
        pack.addArg(b.fmt("--build-root={s}", .{@import("cuda_release_paths.zig").absoluteRoot(b)}));
        if (aot != null and p.query.os_tag != .macos and !host_only) pack.addDirectoryArg(b.graph.cwdRelativePath(aot.?));
        if (windows) {
            const manifest = b.addSystemCommand(&.{"python3"});
            manifest.addFileArg(b.path("tools/release/windows_manifest.py"));
            manifest.addArg("--binary");
            manifest.addArtifactArg(exe);
            manifest.addArg("--output");
            const source = manifest.addOutputFileArg("SOURCE.json");
            if (host_only) manifest.addArg("--host-only");
            if (fatbins) |dir| manifest.addPrefixedDirectoryArg("--fatbins=", b.graph.cwdRelativePath(dir));
            pack.addPrefixedFileArg("--source-manifest=", source);
            if (windows_runtime) |dir| pack.addPrefixedDirectoryArg("--overlay=", b.graph.cwdRelativePath(dir));
        } else if (xgrammar) |dir| pack.addPrefixedDirectoryArg("--overlay=", grammar.overlay(b, target, dir) orelse {
            step.dependOn(&b.addFail(b.fmt("-Dxgrammar={s} is not an xgrammar " ++ grammar.pinned ++ " checkout: " ++ grammar.clone, .{dir})).step);
            return;
        });
        step.dependOn(&b.addInstallDirectory(.{ .source_dir = archives, .install_dir = .prefix, .install_subdir = "dist" }).step);
        if (b.graph.host.result.os.tag == p.query.os_tag and b.graph.host.result.cpu.arch == p.query.cpu_arch) {
            const check = b.addSystemCommand(if (windows) &.{ "pwsh", "-NoProfile", "-File" } else &.{"sh"});
            check.has_side_effects = true;
            check.addFileArg(b.path(if (windows) "tools/release/smoke-windows.ps1" else "tools/release/smoke.sh"));
            check.addFileArg(archives.path(b, b.fmt("tensorfold-{s}-{s}.{s}", .{ version.?, label, if (windows) "zip" else "tar.gz" })));
            check.addArg(version.?);
            if (windows and host_only) check.addArg("-AllowHostOnly");
            smoke.dependOn(&check.step);
        }
    }
}

test "Windows native installation and archive selection preserve Unix names and defaults" {
    try std.testing.expectEqualStrings("tensorfold.exe", cuda.serverInstallName(.windows));
    try std.testing.expectEqualStrings("tensorfold-native", cuda.serverInstallName(.linux));
    try std.testing.expectEqualStrings("tensorfold-native", cuda.serverInstallName(.macos));
    try std.testing.expect(!selected("windows-x86_64", null));
    try std.testing.expect(selected("linux-x86_64", null));
    try std.testing.expect(selected("windows-x86_64", "all"));
    try std.testing.expect(selected("windows-x86_64", "windows-x86_64"));
    try std.testing.expect(!selected("macos-arm64", "windows-x86_64"));
}

test "release versions have exactly three canonical numeric components" {
    for ([_][]const u8{ "0.0.0", "1.2.3", "123.456.789" }) |v| try std.testing.expect(validVersion(v));
    for ([_][]const u8{ "", "1.2", "1.2.3.4", "1..3", "01.2.3", "1.2.3-rc1", "1.2.3+build", "v1.2.3", "1.2.-3", "1.2.3/", "4294967296.0.0" }) |v| try std.testing.expect(!validVersion(v));
}
