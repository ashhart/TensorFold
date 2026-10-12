//! `sys` (zig/src/platform/sys.zig) for every module the build compiles: one walk of the step graph after it is
//! declared, so no module's declaration needs to name it.

const std = @import("std");

pub fn wire(b: *std.Build) void {
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
        // only TensorFold's own sources see it; the sys module itself and generated modules do not
        if (m.root_source_file) |root| if (root == .src_path and !m.import_table.contains("sys") and !std.mem.endsWith(u8, root.src_path.sub_path, "platform/sys.zig"))
            m.addImport("sys", w.sys(target));
        for (m.import_table.values()) |dep| w.module(dep, target);
    }

    fn sys(w: *Walk, target: std.Build.ResolvedTarget) *std.Build.Module {
        const key = target.result.zigTriple(w.b.allocator) catch @panic("OOM");
        const slot = w.per_target.getOrPut(w.b.allocator, key) catch @panic("OOM");
        if (!slot.found_existing) slot.value_ptr.* = w.b.createModule(.{ .root_source_file = w.b.path("zig/src/platform/sys.zig"), .target = target, .link_libc = true });
        return slot.value_ptr.*;
    }
};
