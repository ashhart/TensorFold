//! Files tests open while they run: a cached pass says nothing about one unless the build knows about it.
const std = @import("std");
const Step = std.Build.Step;

const reads = [_]struct { root: []const u8, files: []const []const u8 }{
    .{ .root = "zig/src/core/root.zig", .files = &.{
        "zig/tests/fixtures/exl3_oracle.bin",
        "zig/tests/fixtures/lfm2/images.json",
        "zig/tests/fixtures/picture/edges.json",
        "zig/tests/fixtures/picture/trns.json",
        "zig/tests/fixtures/picture/webp.json",
    } },
    .{ .root = "zig/src/core/exlr8_oracle_test.zig", .files = &.{
        "zig/tests/fixtures/exl3_oracle.bin",
        "zig/tests/fixtures/exlr8/oracle.json",
    } },
    .{ .root = "zig/src/server/root.zig", .files = &.{
        "zig/src/families/deepseek_v41/fixtures/serve/mini_tokenizer.json",
        "zig/src/families/deepseek_v41/fixtures/serve/template.jsonl",
    } },
    .{ .root = "zig/src/families/lfm2/lfm2.zig", .files = &.{"zig/tests/fixtures/lfm2/images.json"} },
};

pub fn wire(b: *std.Build) void {
    var seen: std.AutoHashMapUnmanaged(*Step, void) = .empty;
    for (b.top_level_steps.values()) |top| declare(b, &top.step, &seen);
}

fn declare(b: *std.Build, step: *Step, seen: *std.AutoHashMapUnmanaged(*Step, void)) void {
    const entry = seen.getOrPut(b.allocator, step) catch @panic("OOM");
    if (entry.found_existing) return;
    if (step.cast(Step.Run)) |run| if (run.producer) |exe| if (exe.root_module.root_source_file) |root| {
        const name = b.allocator.dupe(u8, root.getDisplayName()) catch @panic("OOM");
        std.mem.replaceScalar(u8, name, '\\', '/');
        for (reads) |read| if (std.mem.eql(u8, name, read.root)) for (read.files) |file| run.addFileInput(b.path(file));
    };
    for (step.dependencies.items) |dep| declare(b, dep, seen);
}
