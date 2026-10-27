//! libtfgrammar, structured output's optional runtime library (tf_grammar.cc over xgrammar), built beside bin/.

const std = @import("std");

/// The xgrammar release the shim is written and checked against (its masks against the Python package's).
pub const pinned = @import("../src/platform/grammar_library.zig").pinned;

pub const clone = @import("../src/platform/grammar_library.zig").clone;

/// The -Dxgrammar checkout, if given (`zig build grammar` and `zig build dist` both build from it).
pub fn option(b: *std.Build) ?[]const u8 {
    return b.option([]const u8, "xgrammar", "An xgrammar " ++ pinned ++ " checkout with its submodules (zig build grammar, dist)");
}

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, dir: ?[]const u8, lanes: *std.Build.Module) void {
    const step = b.step("grammar", "libtfgrammar (structured output) from an xgrammar " ++ pinned ++ " checkout: -Dxgrammar=DIR");
    const budget = b.step("test-grammar-budget", "grammar_budget's masks against brute force through a real libtfgrammar: -Dxgrammar=DIR");
    const root = dir orelse {
        step.dependOn(&b.addFail("zig build grammar needs -Dxgrammar=DIR: " ++ clone).step);
        budget.dependOn(&b.addFail("zig build test-grammar-budget needs -Dxgrammar=DIR: " ++ clone).step);
        return;
    };
    const lib = library(b, target, root) orelse {
        const fail = b.addFail(b.fmt("-Dxgrammar={s} is not an xgrammar " ++ pinned ++ " checkout: " ++ clone, .{root}));
        step.dependOn(&fail.step);
        budget.dependOn(&fail.step);
        return;
    };
    step.dependOn(&installNative(b, lib).step);
    const native = target.result.os.tag == b.graph.host.result.os.tag and target.result.cpu.arch == b.graph.host.result.cpu.arch;
    const host = if (native) lib else library(b, b.graph.host, root).?;
    const options = b.addOptions();
    options.addOptionPath("library", host.getEmittedBin());
    const check = b.createModule(.{ .root_source_file = b.path("zig/grammar_budget_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    check.addOptions("grammar_budget_library", options);
    budget.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = check })).step);
}

fn installNative(b: *std.Build, lib: *std.Build.Step.Compile) *std.Build.Step.InstallArtifact {
    return b.addInstallArtifact(lib, .{ .dest_dir = .{ .override = .{ .custom = "native/lib" } } });
}

/// Inspect the production installers without compiling a grammar library or fetching its sources.
pub fn hostTests(b: *std.Build, all: *std.Build.Step) void {
    const options = b.addOptions();
    var server_dir: ?std.Build.InstallDir = null;
    if (b.top_level_steps.get("native")) |native| for (native.step.dependencies.items) |dep| {
        if (dep.cast(std.Build.Step.InstallArtifact)) |install| server_dir = install.dest_dir;
    };
    options.addOption([]const u8, "server_dir", relativeDir(server_dir orelse .bin));
    inline for (.{ .linux, .macos, .windows }) |os| {
        const target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = os });
        const lib = b.addLibrary(.{ .name = "grammar-layout", .linkage = .dynamic, .root_module = b.createModule(.{ .target = target }) });
        const install = installNative(b, lib);
        options.addOption([]const u8, @tagName(os), relativeDir(install.dest_dir.?));
    }
    const root = b.createModule(.{ .root_source_file = b.path("zig/grammar_install_host.zig"), .target = b.graph.host });
    root.addOptions("grammar_layout", options);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    b.step("test-grammar-install-host", "Grammar installs beside the native server on every OS and prefix").dependOn(&run.step);
    all.dependOn(&run.step);
}

/// libtfgrammar for `target` from the checkout at `dir`; null unless `dir` is a git checkout at the pinned tag.
pub fn library(b: *std.Build, target: std.Build.ResolvedTarget, dir: []const u8) ?*std.Build.Step.Compile {
    var code: u8 = 0;
    const tag = b.runAllowFail(&.{ "git", "-C", dir, "describe", "--tags", "--exact-match" }, &code, .ignore) catch return null;
    if (!std.mem.eql(u8, std.mem.trim(u8, tag, "\r\n"), pinned)) return null;
    const root: std.Build.LazyPath = .{ .cwd_relative = dir };
    const mod = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libcpp = true, .strip = true });
    // xgrammar's checks keep __FILE__: the map keeps the checkout's absolute path out of the shipped library
    const map = b.fmt("-ffile-prefix-map={s}=xgrammar", .{dir});
    const flags = [_][]const u8{ "-std=c++17", "-DXGRAMMAR_ENABLE_CPPTRACE=0", "-fexceptions", "-Wno-everything", map };
    mod.addIncludePath(root.path(b, "include"));
    mod.addIncludePath(root.path(b, "3rdparty/picojson"));
    mod.addIncludePath(root.path(b, "3rdparty/dlpack/include"));
    mod.addIncludePath(root.path(b, "cpp"));
    mod.addCSourceFiles(.{ .root = root, .files = sources(b, dir), .flags = &flags });
    mod.addCSourceFile(.{ .file = b.path("zig/src/grammar/tf_grammar.cc"), .flags = &flags });
    mod.addCSourceFile(.{ .file = b.path("zig/src/grammar/tf_budget.cc"), .flags = &flags });
    return b.addLibrary(.{ .name = "tfgrammar", .linkage = .dynamic, .root_module = mod });
}

/// An archive's grammar files: lib/libtfgrammar for `target`, and xgrammar's license, notice and picojson's notice.
pub fn overlay(b: *std.Build, target: std.Build.ResolvedTarget, dir: []const u8) ?std.Build.LazyPath {
    const lib = library(b, target, dir) orelse return null;
    const files = b.addWriteFiles();
    const name = if (target.result.os.tag == .macos) "libtfgrammar.dylib" else "libtfgrammar.so";
    _ = files.addCopyFile(lib.getEmittedBin(), b.fmt("lib/{s}", .{name}));
    const root: std.Build.LazyPath = .{ .cwd_relative = dir };
    _ = files.addCopyFile(root.path(b, "LICENSE"), "LICENSES/xgrammar-Apache-2.0.txt");
    _ = files.addCopyFile(root.path(b, "NOTICE"), "LICENSES/xgrammar-NOTICE.txt");
    _ = files.addCopyFile(root.path(b, "3rdparty/dlpack/LICENSE"), "LICENSES/dlpack-Apache-2.0.txt");
    return files.getDirectory();
}

/// xgrammar's library sources as its CMakeLists globs them: every .cc under cpp but the Python bindings (cpp/tvm_ffi).
fn sources(b: *std.Build, dir: []const u8) []const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    const io = b.graph.io;
    var cpp = std.Io.Dir.cwd().openDir(io, b.pathJoin(&.{ dir, "cpp" }), .{ .iterate = true }) catch return &.{};
    defer cpp.close(io);
    var walk = cpp.walk(b.allocator) catch @panic("OOM");
    defer walk.deinit();
    while (walk.next(io) catch null) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.path, ".cc") or std.mem.startsWith(u8, e.path, "tvm_ffi")) continue;
        out.append(b.allocator, b.pathJoin(&.{ "cpp", e.path })) catch @panic("OOM");
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    return out.items;
}

fn relativeDir(dir: std.Build.InstallDir) []const u8 {
    return switch (dir) {
        .custom => |path| path,
        .bin => "bin",
        .lib => "lib",
        .header => "include",
        .prefix => "",
    };
}
