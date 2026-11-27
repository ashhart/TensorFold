//! `-Denv-graph=PATH`: each compile step's modules, roots and imports as text lines, for tools/zig/env_names.py.
const std = @import("std");

/// Writes the graph once every step is wired; without the option it does nothing.
pub fn write(b: *std.Build) void {
    const path = b.option([]const u8, "env-graph", "Write every compile step's module roots and imports to PATH") orelse return;
    var w: Walk = .{ .b = b };
    // dist first, so each compile step the release archives build is marked as theirs
    if (b.top_level_steps.get("dist")) |dist| w.step(&dist.step);
    w.release = false;
    for (b.top_level_steps.values()) |top| w.step(&top.step);
    std.Io.Dir.cwd().writeFile(b.graph.io, .{ .sub_path = path, .data = w.out.items }) catch @panic("cannot write -Denv-graph");
}

const Walk = struct {
    b: *std.Build,
    release: bool = true,
    out: std.ArrayList(u8) = .empty,
    steps: std.AutoHashMapUnmanaged(*std.Build.Step, void) = .empty,
    ids: std.AutoHashMapUnmanaged(*std.Build.Module, usize) = .empty,

    fn print(w: *Walk, comptime fmt: []const u8, args: anytype) void {
        w.out.print(w.b.allocator, fmt, args) catch @panic("OOM");
    }

    fn step(w: *Walk, s: *std.Build.Step) void {
        if ((w.steps.getOrPut(w.b.allocator, s) catch @panic("OOM")).found_existing) return;
        if (s.cast(std.Build.Step.Compile)) |c| {
            const id = w.module(c.root_module);
            w.print("C\t{s}\t{s}\t{d}\t{s}\n", .{ @tagName(c.kind), c.name, id, if (w.release) "dist" else "-" });
        }
        for (s.dependencies.items) |dep| w.step(dep);
    }

    /// The module's id, printing `M id root` (`-`: generated) and an `I id name dep` per import the first time.
    fn module(w: *Walk, m: *std.Build.Module) usize {
        if (w.ids.get(m)) |id| return id;
        const id = w.ids.count();
        w.ids.put(w.b.allocator, m, id) catch @panic("OOM");
        const root = if (m.root_source_file) |r| if (r == .src_path) r.src_path.sub_path else "-" else "-";
        w.print("M\t{d}\t{s}\n", .{ id, root });
        for (m.import_table.keys(), m.import_table.values()) |name, dep| {
            const dep_id = w.module(dep);
            w.print("I\t{d}\t{s}\t{d}\n", .{ id, name, dep_id });
        }
        return id;
    }
};
