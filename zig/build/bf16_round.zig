//! One std-only `bf16_round` module for every build consumer; its file has a single module owner.

const std = @import("std");

/// `test-bf16-round` under `parent`, then the module for every module the declared steps compile.
pub fn wire(b: *std.Build, parent: ?*std.Build.Step) void {
    if (parent) |p| {
        const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/bf16_round.zig"), .target = b.graph.host }) });
        const step = b.step("test-bf16-round", "fp32 and float64 to bf16, ties to even, against the copies it replaced");
        step.dependOn(&b.addRunArtifact(tests).step);
        p.dependOn(step);
    }
    var seen_steps: std.AutoHashMapUnmanaged(*std.Build.Step, void) = .empty;
    var seen_mods: std.AutoHashMapUnmanaged(*std.Build.Module, void) = .empty;
    var per_target: std.StringHashMapUnmanaged(*std.Build.Module) = .empty;
    var w: Walk = .{ .b = b, .seen_steps = &seen_steps, .seen_mods = &seen_mods, .per_target = &per_target };
    for (b.top_level_steps.values()) |top| w.step(&top.step);
}

const Walk = struct {
    b: *std.Build,
    seen_steps: *std.AutoHashMapUnmanaged(*std.Build.Step, void),
    seen_mods: *std.AutoHashMapUnmanaged(*std.Build.Module, void),
    per_target: *std.StringHashMapUnmanaged(*std.Build.Module),

    fn step(w: *Walk, s: *std.Build.Step) void {
        if ((w.seen_steps.getOrPut(w.b.allocator, s) catch @panic("OOM")).found_existing) return;
        if (s.cast(std.Build.Step.Compile)) |compile| w.module(compile.root_module, compile.root_module.resolved_target);
        for (s.dependencies.items) |dep| w.step(dep);
    }

    fn module(w: *Walk, m: *std.Build.Module, inherited: ?std.Build.ResolvedTarget) void {
        if ((w.seen_mods.getOrPut(w.b.allocator, m) catch @panic("OOM")).found_existing) return;
        const target = m.resolved_target orelse inherited orelse w.b.graph.host;
        // only TensorFold's own sources see it; the module itself and generated modules do not
        if (m.root_source_file) |root| if (root == .src_path and !m.import_table.contains("bf16_round") and !std.mem.endsWith(u8, root.src_path.sub_path, "core/bf16_round.zig"))
            m.addImport("bf16_round", w.rounding(target));
        for (m.import_table.values()) |dep| w.module(dep, target);
    }

    fn rounding(w: *Walk, target: std.Build.ResolvedTarget) *std.Build.Module {
        const key = target.result.zigTriple(w.b.allocator) catch @panic("OOM");
        const slot = w.per_target.getOrPut(w.b.allocator, key) catch @panic("OOM");
        if (!slot.found_existing) slot.value_ptr.* = w.b.createModule(.{ .root_source_file = w.b.path("zig/src/core/bf16_round.zig"), .target = target });
        return slot.value_ptr.*;
    }
};
