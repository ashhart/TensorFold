//! Drama host tests and code generation without CUDA, model weights or a Metal compiler.
const std = @import("std");

pub fn compile(b: *std.Build, step: *std.Build.Step, family: *std.Build.Module, name: []const u8) void {
    const obj = b.addObject(.{ .name = name, .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/drama_compile.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "drama3", .module = family }},
    }) });
    step.dependOn(&obj.step);
    const tool = b.createModule(.{ .root_source_file = b.path("zig/src/families/drama3/generate.zig"), .target = b.graph.host, .link_libc = true });
    tool.addImport("drama3", family);
    tool.addImport("core", family.import_table.get("core").?);
    const wrapper = b.createModule(.{ .root_source_file = b.path("zig/tests/drama_tool_compile.zig"), .target = b.graph.host, .link_libc = true });
    wrapper.addImport("drama_generate", tool);
    step.dependOn(&b.addObject(.{ .name = b.fmt("{s}_tool", .{name}), .root_module = wrapper }).step);
}

pub fn native(b: *std.Build, step: *std.Build.Step, family: *std.Build.Module, core: *std.Build.Module, api: *std.Build.Module, cuda: ?*std.Build.Module, name: []const u8) void {
    const mod = b.createModule(.{ .root_source_file = b.path("zig/src/native/drama_compile.zig"), .target = b.graph.host, .link_libc = true });
    mod.addImport("drama3", family);
    mod.addImport("core", core);
    mod.addImport("engine_api", api);
    if (cuda) |c| mod.addImport("cuda", c);
    step.dependOn(&b.addObject(.{ .name = name, .root_module = mod }).step);
}

pub fn metal(b: *std.Build, core: *std.Build.Module, api: *std.Build.Module) void {
    if (b.graph.host.result.os.tag != .macos) return;
    const runtime = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    const bytes = b.addWriteFiles().add("drama3_host_metallib.zig", "pub const bytes: []const u8 = &.{};\n");
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/drama3/drama3_metal.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = runtime }, .{ .name = "core", .module = core }, .{ .name = "drama3_metallib", .module = b.createModule(.{ .root_source_file = bytes }) } },
    });
    const step = b.step("check-drama-metal-host", "Compile Drama's Metal host entry points without compiling shaders");
    compile(b, step, family, "drama_metal_host");
    native(b, step, family, core, api, null, "drama_metal_native_host");
}
