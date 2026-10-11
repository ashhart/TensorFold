//! libtfgrammar: structured output's optional runtime library (zig/src/grammar/tf_grammar.cc over xgrammar's C++
//! library). `zig build grammar -Dxgrammar=DIR` builds it from an xgrammar checkout at the pinned tag into
//! zig-out/native/lib; the server loads it when the first structured request arrives and serves text without it.

const std = @import("std");

/// The xgrammar release the shim is written and checked against (its masks against the Python package's).
pub const pinned = "v0.2.8";

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const step = b.step("grammar", "libtfgrammar (structured output) from an xgrammar " ++ pinned ++ " checkout: -Dxgrammar=DIR");
    const dir = b.option([]const u8, "xgrammar", "An xgrammar " ++ pinned ++ " checkout with its submodules, for `zig build grammar`") orelse {
        step.dependOn(&b.addFail("zig build grammar needs -Dxgrammar=DIR: git clone --branch " ++ pinned ++ " --recurse-submodules https://github.com/mlc-ai/xgrammar").step);
        return;
    };
    const root: std.Build.LazyPath = .{ .cwd_relative = dir };
    const mod = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libcpp = true });
    const flags = [_][]const u8{ "-std=c++17", "-DXGRAMMAR_ENABLE_CPPTRACE=0", "-fexceptions", "-Wno-everything" };
    mod.addIncludePath(root.path(b, "include"));
    mod.addIncludePath(root.path(b, "3rdparty/picojson"));
    mod.addIncludePath(root.path(b, "3rdparty/dlpack/include"));
    mod.addCSourceFiles(.{ .root = root, .files = sources(b, dir), .flags = &flags });
    mod.addCSourceFile(.{ .file = b.path("zig/src/grammar/tf_grammar.cc"), .flags = &flags });
    const lib = b.addLibrary(.{ .name = "tfgrammar", .linkage = .dynamic, .root_module = mod });
    step.dependOn(&b.addInstallArtifact(lib, .{ .dest_dir = .{ .override = .{ .custom = "native/lib" } } }).step);
}

/// xgrammar's library sources as its CMakeLists globs them: cpp/**/*.cc without the Python bindings (cpp/tvm_ffi).
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
