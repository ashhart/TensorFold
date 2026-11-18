//! macOS test allocations avoid the debug unwinder; the compiler runner still owns test and leak reporting.
const std = @import("std");
const Step = std.Build.Step;

fn runner(b: *std.Build) std.Build.LazyPath {
    const generator = b.addExecutable(.{
        .name = "test-allocator-runner-source",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/build/test_runner_source.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });
    const source = b.addRunArtifact(generator);
    source.addFileArg(std.Build.LazyPath.zig_lib.path(b, "compiler/test_runner.zig"));
    return source.addOutputFileArg("test_runner.zig");
}

pub fn checks(b: *std.Build) void {
    const step = b.step("test-test-allocator-host", "Zero-frame test allocations retain threaded safety and reject a planted leak");
    const source = runner(b);
    for ([_]bool{ false, true }) |leak| {
        const options = b.addOptions();
        options.addOption(bool, "leak", leak);
        const module = b.createModule(.{
            .root_source_file = b.path("zig/tests/test_allocator_probe.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
            .link_libc = true,
        });
        module.addOptions("allocator_probe", options);
        if (!leak) {
            const server_test = b.addTest(.{ .root_module = module, .test_runner = .{ .path = source, .mode = .server } });
            step.dependOn(&b.addRunArtifact(server_test).step);
        }
        const test_exe = b.addTest(.{ .root_module = module, .test_runner = .{ .path = source, .mode = .simple } });
        const run = b.addRunArtifact(test_exe);
        run.setName(if (leak) "test allocator rejects planted leak" else "test allocator balances threaded allocations");
        run.expectExitCode(if (leak) 1 else 0);
        run.expectStdErrMatch(if (leak) "1 tests leaked memory." else "All 1 tests passed.");
        step.dependOn(&run.step);
    }
}

pub fn wire(b: *std.Build) void {
    var seen: std.AutoHashMapUnmanaged(*Step, void) = .empty;
    var source: ?std.Build.LazyPath = null;
    for (b.top_level_steps.values()) |top| visit(b, &top.step, &seen, &source);
}

fn visit(b: *std.Build, step: *Step, seen: *std.AutoHashMapUnmanaged(*Step, void), source: *?std.Build.LazyPath) void {
    if ((seen.getOrPut(b.allocator, step) catch @panic("OOM")).found_existing) return;
    if (step.cast(Step.Compile)) |compile| {
        if (compile.kind == .@"test" and compile.root_module.resolved_target.?.result.os.tag == .macos and
            compile.test_runner == null)
        {
            if (source.* == null) source.* = runner(b);
            compile.test_runner = .{ .path = source.*.?, .mode = .server };
            source.*.?.addStepDependencies(step);
        }
    }
    for (step.dependencies.items) |dep| visit(b, dep, seen, source);
}
