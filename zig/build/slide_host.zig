//! Learning admission and the unchanged Metal lesson fixtures, with inert images and no GPU work.
const std = @import("std");

pub fn metal(b: *std.Build, m: anytype) void {
    const target = b.graph.host;
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = m.lanes }} });
    const native = b.createModule(.{ .root_source_file = b.path("zig/src/native/metal.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = m.metal }, .{ .name = "tensorfold", .module = m.engine }, .{ .name = "engine_api", .module = api } } });
    native.addImport("living", m.living);
    @import("qwen4.zig").metalNative(b, target, native);
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .link_libc = true });
    const rounds = b.createModule(.{ .root_source_file = b.path("zig/src/core/frame_rounds.zig"), .target = target });
    const files = b.addWriteFiles();
    const core = b.createModule(.{ .root_source_file = files.add("slide-drama-core.zig", "pub const safetensors = @import(\"tensorfold\").safetensors;\npub const tokenizer = @import(\"tokenizer\");\npub const frame_rounds = @import(\"frame_rounds\");\n"), .target = target, .imports = &.{ .{ .name = "tensorfold", .module = m.engine }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "frame_rounds", .module = rounds } } });
    const bytes = b.createModule(.{ .root_source_file = files.add("slide-inert-metallib.zig", "pub const bytes: []const u8 = &.{};\n") });
    const drama = b.createModule(.{ .root_source_file = b.path("zig/src/families/drama3/drama3_metal.zig"), .target = target, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = m.metal }, .{ .name = "core", .module = core }, .{ .name = "drama3_metallib", .module = bytes } } });
    native.addImport("drama3", drama);
    native.addImport("core", core);
    native.addImport("tokenizer", tokenizer);
    b.step("test-slide-metal-host", "Metal learning admission and registration without shaders or devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{ "slide admission:", "Metal learner:" } })).step);
    const fixtures = b.createModule(.{ .root_source_file = b.path("zig/nemotron_living_host.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "living", .module = m.living }} });
    b.step("test-living-metal-host", "Unchanged Nemotron lesson and pocket shard fixtures without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fixtures })).step);
}
