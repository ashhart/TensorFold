//! Compile the actual CPU pocket-fit tool against its real fitter without loading an engine or shaders.
const std = @import("std");
pub fn steps(b: *std.Build) void {
    const fit = b.createModule(.{ .root_source_file = b.path("zig/src/living/pocket_fit.zig"), .target = b.graph.host });
    const files = b.addWriteFiles();
    const facade = b.createModule(.{ .root_source_file = files.add("pocket-fit-family.zig", "pub const nemotron = struct { pub const pocket_fit = @import(\"living_fit\"); };\n"), .target = b.graph.host, .imports = &.{.{ .name = "living_fit", .module = fit }} });
    const sys = b.createModule(.{ .root_source_file = b.path("zig/src/platform/sys.zig"), .target = b.graph.host, .link_libc = true });
    const tool = b.addExecutable(.{ .name = "pocket-fit-host", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/pocket_fit_check.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "tensorfold", .module = facade }, .{ .name = "sys", .module = sys } } }) });
    b.step("compile-pocket-fit-host", "Compile the actual pocket-fit tool with its real CPU fitter and no shaders").dependOn(&tool.step);
}
