//! One prompt-capacity option shared by every CUDA caller, including the single-GPU product build.
const std = @import("std");
const policy = @import("../src/core/qwen_prefill_config.zig");

pub fn option(b: *std.Build) usize {
    return policy.resolve(b.option(usize, "qwen-prefill-max-rows", "CUDA Qwen prompt capacity: 4096 (default) or experimental 8192")) catch
        @panic("-Dqwen-prefill-max-rows must be 4096 or 8192");
}
fn config(b: *std.Build, target: std.Build.ResolvedTarget, rows: usize) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(usize, "max_rows", rows);
    const module = b.createModule(.{ .root_source_file = b.path("zig/src/core/qwen_prefill_config.zig"), .target = target });
    module.addOptions("qwen_prefill_options", options);
    return module;
}
pub fn tests(b: *std.Build, parent: *std.Build.Step) void {
    const step = b.step("test-qwen-prefill-config-host", "Default and wide CUDA prompt capacity, refusals and warmup shapes");
    for ([_]usize{ 4096, 8192 }) |rows|
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = config(b, b.graph.host, rows) })).step);
    parent.dependOn(step);
}
pub fn wire(b: *std.Build, rows: usize) void {
    var seen_steps: std.AutoHashMapUnmanaged(*std.Build.Step, void) = .empty;
    var seen_modules: std.AutoHashMapUnmanaged(*std.Build.Module, void) = .empty;
    var modules: std.StringHashMapUnmanaged(*std.Build.Module) = .empty;
    var walk: Walk = .{ .b = b, .rows = rows, .seen_steps = &seen_steps, .seen_modules = &seen_modules, .modules = &modules };
    for (b.top_level_steps.values()) |top| walk.step(&top.step);
}
const Walk = struct {
    b: *std.Build,
    rows: usize,
    seen_steps: *std.AutoHashMapUnmanaged(*std.Build.Step, void),
    seen_modules: *std.AutoHashMapUnmanaged(*std.Build.Module, void),
    modules: *std.StringHashMapUnmanaged(*std.Build.Module),
    fn step(w: *Walk, item: *std.Build.Step) void {
        if ((w.seen_steps.getOrPut(w.b.allocator, item) catch @panic("OOM")).found_existing) return;
        if (item.cast(std.Build.Step.Compile)) |compile| w.module(compile.root_module, compile.root_module.resolved_target);
        for (item.dependencies.items) |dependency| w.step(dependency);
    }
    fn module(w: *Walk, item: *std.Build.Module, inherited: ?std.Build.ResolvedTarget) void {
        if ((w.seen_modules.getOrPut(w.b.allocator, item) catch @panic("OOM")).found_existing) return;
        const target = item.resolved_target orelse inherited orelse w.b.graph.host;
        if (item.root_source_file) |root| if (root == .src_path and
            !item.import_table.contains("qwen_prefill_config") and
            !std.mem.endsWith(u8, root.src_path.sub_path, "core/qwen_prefill_config.zig"))
        {
            const key = target.result.zigTriple(w.b.allocator) catch @panic("OOM");
            const entry = w.modules.getOrPut(w.b.allocator, key) catch @panic("OOM");
            if (!entry.found_existing) entry.value_ptr.* = config(w.b, target, w.rows);
            item.addImport("qwen_prefill_config", entry.value_ptr.*);
        };
        for (item.import_table.values()) |dependency| w.module(dependency, target);
    }
};
