//! `-Drerun-tests`: every check runs again whatever the cache holds: CI's kept cache saves compiles, never a result.
const std = @import("std");
const Step = std.Build.Step;

pub fn wire(b: *std.Build) void {
    var seen: std.AutoHashMapUnmanaged(*Step, void) = .empty;
    if (b.option(bool, "name-tests", "Probe: name each test run by its root file") orelse false) for (b.top_level_steps.values()) |top| name(b, &top.step, &seen);
    if (!(b.option(bool, "rerun-tests", "Run every check again, whatever the build cache holds") orelse false)) return;
    seen = .empty;
    for (b.top_level_steps.values()) |top| mark(b, &top.step, &seen);
}

fn name(b: *std.Build, step: *Step, seen: *std.AutoHashMapUnmanaged(*Step, void)) void {
    const entry = seen.getOrPut(b.allocator, step) catch @panic("OOM");
    if (entry.found_existing) return;
    if (step.cast(Step.Run)) |run| if (run.producer) |exe| if (exe.root_module.root_source_file) |root| {
        run.step.name = b.fmt("RUN {s}", .{root.getDisplayName()});
    };
    if (step.cast(Step.Compile)) |exe| if (exe.root_module.root_source_file) |root| {
        exe.step.name = b.fmt("CC {s}", .{root.getDisplayName()});
    };
    for (step.dependencies.items) |dep| name(b, dep, seen);
}

fn mark(b: *std.Build, step: *Step, seen: *std.AutoHashMapUnmanaged(*Step, void)) void {
    const entry = seen.getOrPut(b.allocator, step) catch @panic("OOM");
    if (entry.found_existing) return;
    if (step.cast(Step.Run)) |run| run.has_side_effects = run.has_side_effects or isCheck(run);
    for (step.dependencies.items) |dep| mark(b, dep, seen);
}

/// A run whose result only says pass or fail: nothing later reads its output.
fn isCheck(run: *const Step.Run) bool {
    if (run.captured_stdout != null or run.captured_stderr != null) return false;
    for (run.argv.items) |arg| switch (arg) {
        .output_file, .output_directory => return false,
        else => {},
    };
    return true;
}
