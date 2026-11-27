//! Execute the native post-init statements verbatim, with recording replacements only for GPU surfaces.
const std = @import("std");
pub fn wire(b: *std.Build, qwen: *std.Build.Module, fake: *std.Build.Module) void {
    const source = @embedFile("../src/families/qwen3_5/cuda_native.zig");
    const begin = std.mem.indexOf(u8, source, "    const e = try E.Engine.init(") orelse @panic("native engine init missing");
    const end = std.mem.indexOfPos(u8, source, begin, "    if (o.prompt_int8) {") orelse @panic("native post-init boundary missing");
    const body = source[begin..end];
    const core = qwen.import_table.get("core").?;
    const cuda = qwen.import_table.get("cuda").?;
    const rope = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda_rope.zig"), .target = b.graph.host, .imports = &.{.{ .name = "core", .module = core }} });
    const identity = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda_round_identity.zig"), .target = b.graph.host, .imports = &.{.{ .name = "core", .module = core }} });
    const fixture = b.createModule(.{ .root_source_file = b.path("zig/qwen35_round_identity_fixture.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "rope", .module = rope }, .{ .name = "cuda", .module = cuda } } });
    const files = b.addWriteFiles();
    _ = files.add("cuda_rope_table.zig", "pub const set = @import(\"fixture\").setRope;\n");
    _ = files.add("cuda_round_identity.zig", "pub const read = @import(\"identity\").read;\n");
    const header =
        \\const std = @import("std");
        \\const cuda_rope = @import("rope");
        \\const E = @import("fixture");
        \\const fitOf = E.fitOf;
        \\pub fn run(gpa: std.mem.Allocator, io: std.Io, ctx: E.Context, dir: []const u8, o: E.Options,
        \\    attribution: ?*@import("cuda").round_attribution.Trace) !*E.Engine {
        \\
    ;
    const code = std.fmt.allocPrint(b.allocator, "{s}{s}    return e;\n}}\n", .{ header, body }) catch @panic("out of memory");
    const generated = b.createModule(.{ .root_source_file = files.add("qwen35_native_post_init.zig", code), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "rope", .module = rope }, .{ .name = "identity", .module = identity }, .{ .name = "fixture", .module = fixture }, .{ .name = "cuda", .module = cuda } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/qwen35_round_identity_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "native_post_init", .module = generated }, .{ .name = "native_init_fixture", .module = fixture }, .{ .name = "attribution_fake", .module = fake } } });
    b.step("test-qwen35-round-identity-host", "Actual native post-init YaRN resolution and published identity without GPU allocation/upload").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"attribution review: resolved YaRN"} })).step);
}
