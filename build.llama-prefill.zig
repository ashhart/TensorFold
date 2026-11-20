const std = @import("std");
const metal = @import("zig/build/metal_targets.zig");

pub fn build(b: *std.Build) void {
    const ids_files = b.addWriteFiles();
    _ = ids_files.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
    const draft_ids = b.createModule(.{ .root_source_file = ids_files.add("draft_ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");") });
    const mods = metal.metalEngineModules(b, b.graph.host, .Debug, draft_ids);
    const empty = b.addWriteFiles().add("llama-prefill-empty.metallib", "");
    mods.sources.addImport("nemotron_packed_metallib", metal.embedded(b, empty, "nemotron_packed.metallib"));
    const llama = b.createModule(.{ .root_source_file = b.path("zig/src/llama_prefill_host.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = mods.metal }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "kernel_sources", .module = mods.sources } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = llama, .filters = &.{"llama prefill:"} }));
    b.step("test", "Llama's cancelled-prefill checkpoint without GPU work").dependOn(&run.step);
    @import("zig/build/profile.zig").walk(b);
}
