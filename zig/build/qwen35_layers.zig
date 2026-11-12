//! The std-only shared modules (`qwen35_layers`, `qwen35_rules`, `rope_config`) for every build consumer.

const std = @import("std");

/// A shared module: its import name, its file under zig/src, and its test step.
const Shared = struct { name: []const u8, path: []const u8, step: []const u8, about: []const u8 };

const shared = [_]Shared{
    .{
        .name = "qwen35_layers",
        .path = "core/qwen35_layers.zig",
        .step = "test-qwen35-layers",
        .about = "Qwen3.5's layer pattern from layer_types and full_attention_interval, one rule",
    },
    .{
        .name = "qwen35_rules",
        .path = "core/qwen35_rules.zig",
        .step = "test-qwen35-rules",
        .about = "Qwen3.5's computed-part refusals: one table, a column per backend",
    },
    .{
        .name = "rope_config",
        .path = "core/rope_config.zig",
        .step = "test-rope-config",
        .about = "A tower's RoPE keys: rope_parameters, the legacy rope_scaling and the top-level keys, read one way",
    },
};

/// Each module's test step under `parent`, then the modules for every module the declared steps compile.
pub fn wire(b: *std.Build, parent: ?*std.Build.Step) void {
    for (shared) |m| wireOne(b, parent, m);
}

fn wireOne(b: *std.Build, parent: ?*std.Build.Step, m: Shared) void {
    const file = b.fmt("zig/src/{s}", .{m.path});
    if (parent) |p| {
        const root = b.createModule(.{ .root_source_file = b.path(file), .target = b.graph.host });
        const step = b.step(m.step, m.about);
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root })).step);
        p.dependOn(step);
    }
    var seen_steps: std.AutoHashMapUnmanaged(*std.Build.Step, void) = .empty;
    var seen_mods: std.AutoHashMapUnmanaged(*std.Build.Module, void) = .empty;
    var per_target: std.StringHashMapUnmanaged(*std.Build.Module) = .empty;
    var w: Walk = .{
        .b = b,
        .name = m.name,
        .path = m.path,
        .file = file,
        .seen_steps = &seen_steps,
        .seen_mods = &seen_mods,
        .per_target = &per_target,
    };
    for (b.top_level_steps.values()) |top| w.step(&top.step);
}

const Walk = struct {
    b: *std.Build,
    name: []const u8,
    path: []const u8,
    file: []const u8,
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
        if (m.root_source_file) |root| if (root == .src_path and !m.import_table.contains(w.name) and
            !std.mem.endsWith(u8, root.src_path.sub_path, w.path)) m.addImport(w.name, w.reader(target));
        for (m.import_table.values()) |dep| w.module(dep, target);
    }

    fn reader(w: *Walk, target: std.Build.ResolvedTarget) *std.Build.Module {
        const key = target.result.zigTriple(w.b.allocator) catch @panic("OOM");
        const slot = w.per_target.getOrPut(w.b.allocator, key) catch @panic("OOM");
        if (!slot.found_existing) slot.value_ptr.* = w.b.createModule(.{ .root_source_file = w.b.path(w.file), .target = target });
        return slot.value_ptr.*;
    }
};
