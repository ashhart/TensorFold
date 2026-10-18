//! Production image ownership. Host compile checks use separate synthetic modules and never invoke this graph.
const std = @import("std");
const cuda_build = @import("cuda.zig");
const list = @import("dsv41_units.zig");
pub const Images = struct { units: [list.units.len]std.Build.LazyPath, optional: [list.optional.len]?std.Build.LazyPath };
pub fn images(b: *std.Build, nvcc: ?[]const u8, prebuilt: ?[]const u8) ?*const Images {
    const step = b.step("fatbins-dsv41", "Build or install the complete family images, sm_121 only");
    const result = b.allocator.create(Images) catch @panic("OOM");
    if (prebuilt) |dir| {
        for (list.units, &result.units) |unit, *path| {
            const file = b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{unit.name}) });
            std.Io.Dir.cwd().access(b.graph.io, file, .{}) catch {
                step.dependOn(&b.addFail(b.fmt("Missing DeepSeek image: {s}", .{file})).step);
                return null;
            };
            path.* = b.graph.cwdRelativePath(file);
        }
        for (list.optional, &result.optional) |unit, *path| {
            const file = b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{unit.name}) });
            path.* = if (std.Io.Dir.cwd().access(b.graph.io, file, .{})) |_| b.graph.cwdRelativePath(file) else |_| null;
        }
    } else if (nvcc) |compiler| {
        const version = b.addSystemCommand(&.{ compiler, "--version" }).captureStdOut(.{});
        for (list.units, &result.units) |unit, *path| path.* = cuda_build.fatbin(b, compiler, version, unit, "121");
        for (list.optional, &result.optional) |unit, *path| path.* = cuda_build.fatbin(b, compiler, version, unit, "121");
    } else {
        step.dependOn(&b.addFail("fatbins-dsv41 needs -Ddsv41-nvcc=PATH or -Ddsv41-fatbins=DIR").step);
        return null;
    }
    for (list.units, result.units) |unit, path| step.dependOn(&b.addInstallFile(path, b.fmt("fatbin/{s}.fatbin", .{unit.name})).step);
    for (list.optional, result.optional) |unit, path| if (path) |file| {
        step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{unit.name})).step);
    };
    return result;
}
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, embedded: ?*const Images) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", embedded != null);
    options.addOption(u32, "family_scope", 41);
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kernels.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addOptions("dsv41_kernel_options", options);
    if (embedded) |im| for (list.units, im.units) |unit, path| m.addAnonymousImport(b.fmt("dsv41_fatbin_{s}", .{unit.name}), .{ .root_source_file = path });
    const empty = b.addWriteFiles().add("empty-dsv41.fatbin", "");
    for (list.optional, 0..) |unit, i| {
        const path = if (embedded) |im| im.optional[i] orelse empty else empty;
        m.addAnonymousImport(b.fmt("dsv41_fatbin_{s}", .{unit.name}), .{ .root_source_file = path });
    }
    return m;
}
