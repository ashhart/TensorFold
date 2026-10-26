//! Muse Spark (families/muse_spark) on the maintained CUDA runtime: the family module, tf-muse and its host tests.
const std = @import("std");

/// Its forward, prompt path, DFlash2 drafter and native lane backend, over TP sessions with the RoCE mailbox kernel.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/muse_spark/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("tp", @import("tp.zig").module(b, target, optimize, cuda));
    m.addImport("tp_kernel", @import("tp.zig").mailboxModule(b));
    return m;
}

/// tf-muse: rank 1 behind the server (`follow`) and the family's device checks; installed, and its own step.
pub fn tool(b: *std.Build, m: *std.Build.Module, strip: ?bool) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/muse_spark/m1_main.zig"), .target = m.resolved_target, .optimize = m.optimize, .link_libc = true, .strip = strip });
    for ([_][]const u8{ "cuda", "core", "tp", "tp_kernel" }) |name| root.addImport(name, m.import_table.get(name).?);
    root.addImport("muse_spark", m);
    const exe = b.addExecutable(.{ .name = "tf-muse", .root_module = root });
    b.installArtifact(exe);
    b.step("tf-muse", "Muse Spark's rank 1 (follow) and device checks").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// The host parts: config, rank cuts, hello words, prompt chunk plans, slot choice, sampling.
pub fn host(b: *std.Build, m: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-muse", "Muse Spark family host tests").dependOn(&run.step);
    all.dependOn(&run.step);
}
