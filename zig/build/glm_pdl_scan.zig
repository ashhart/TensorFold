//! test-glm-pdl-scan: the GLM kernel sources copied beside the scan test, which embeds each one (no GPU, no nvcc).
const std = @import("std");

pub fn add(b: *std.Build) void {
    const files = b.addWriteFiles();
    var names: std.ArrayList(u8) = .empty;
    const io = b.graph.io;
    var dir = std.Io.Dir.cwd().openDir(io, "zig/kernels/cuda", .{ .iterate = true }) catch @panic("zig/kernels/cuda");
    defer dir.close(io);
    var found: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (it.next(io) catch null) |e| {
        if (e.kind != .file or !std.mem.startsWith(u8, e.name, "glm_")) continue;
        if (!std.mem.endsWith(u8, e.name, ".cu") and !std.mem.endsWith(u8, e.name, ".cuh")) continue;
        found.append(b.allocator, b.dupe(e.name)) catch @panic("OOM");
    }
    std.mem.sort([]const u8, found.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    for (found.items) |name| {
        _ = files.addCopyFile(b.path(b.pathJoin(&.{ "zig/kernels/cuda", name })), name);
        names.appendSlice(b.allocator, name) catch @panic("OOM");
        names.append(b.allocator, '\n') catch @panic("OOM");
    }
    _ = files.addCopyFile(b.path("zig/kernels/cuda/qmmf_body.cuh"), "qmmf_body.cuh"); // qmmf's weight-first prologue
    _ = files.add("files.txt", names.items);
    const root = files.addCopyFile(b.path("zig/tests/glm_pdl_scan.zig"), "main.zig");
    const test_ = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = root, .target = b.graph.host, .optimize = .debug }) });
    b.step("test-glm-pdl-scan", "Every GLM kernel opens with the dependency wait (dependent launches stay safe)").dependOn(&b.addRunArtifact(test_).step);
}
