//! DeepSeek contract images use the same built artifacts as their named installs.
const std = @import("std");
const list = @import("dsv41_units.zig");
pub fn images(b: *std.Build, runner: *std.Build.Step.Compile, install: *std.Build.Step, kernels: anytype, files: []const ?std.Build.LazyPath, prebuilt: ?[]const u8, nvcc: ?[]const u8, version: ?std.Build.LazyPath, sms: []const u8, cuobjdump: ?[]const u8) void {
    const empty = b.addWriteFiles().add("empty-contract-deepseek.fatbin", "");
    for (list.units ++ list.optional) |k| {
        const common = for (kernels, 0..) |known, i| {
            if (std.mem.eql(u8, known.name, k.name)) break i;
        } else null;
        const file = if (common) |i| files[i] else if (prebuilt) |dir|
            b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }))
        else if (nvcc) |tool|
            @import("cuda.zig").fatbin(b, tool, version.?, k, sms)
        else
            null;
        runner.root_module.addAnonymousImport(b.fmt("contract_fatbin_{s}", .{k.name}), .{ .root_source_file = file orelse empty });
        if (common != null) continue;
        if (file) |path| {
            install.dependOn(&b.addInstallFile(path, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
            if (std.mem.eql(u8, k.name, "dsv41_x3ld")) @import("contract_symbols.zig").check(b, path, sms, install, cuobjdump);
        }
    }
}
pub fn inert(b: *std.Build, m: *std.Build.Module) void {
    const files = b.addWriteFiles();
    for (list.units ++ list.optional) |k| m.addAnonymousImport(b.fmt("contract_fatbin_{s}", .{k.name}), .{
        .root_source_file = files.add(b.fmt("{s}.fatbin", .{k.name}), k.name),
    });
}
