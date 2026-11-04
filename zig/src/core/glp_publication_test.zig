//! What the GLP container says about itself: the keys a consumer reads before it reads the directions.

const std = @import("std");
const glp = @import("glp.zig");
const t = @import("glp_test.zig");

test "a file that names itself carries what it said, and the model it came from" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{
        .base_model = "TensorFold/Qwen3.8-27B-MLX-4bit",
        .base_revision = "22d8d538154e0e5b6f8dcb1fcc74a60a0798104c",
        .publication = .{
            .name = "Qwen3.8-27B-refusal-cvec-per_layer-L10-58-a1",
            .description = "A projective refusal-ablation control vector.",
            .author = "TensorFold",
            .license = "MIT",
            .model_hint = "qwen3_5",
            .measured_effect = "alpha 0.5 to 3 is flat; alpha 4 empties the replies",
            .created = "2026-10-27",
        },
        .layers = &layers,
    });
    defer a.free(bytes);
    var read = try t.readOk(bytes);
    defer read.deinit();
    const p = read.vector.publication;
    try std.testing.expectEqualStrings("Qwen3.8-27B-refusal-cvec-per_layer-L10-58-a1", p.name.?);
    try std.testing.expectEqualStrings("A projective refusal-ablation control vector.", p.description.?);
    try std.testing.expectEqualStrings("TensorFold", p.author.?);
    try std.testing.expectEqualStrings("MIT", p.license.?);
    try std.testing.expectEqualStrings("qwen3_5", p.model_hint.?);
    try std.testing.expectEqualStrings("alpha 0.5 to 3 is flat; alpha 4 empties the replies", p.measured_effect.?);
    try std.testing.expectEqualStrings("2026-10-27", p.created.?);

    // The container itself carries the keys a consumer reads, not only the reader.
    var parsed = try t.parseFile(a, bytes);
    defer parsed.deinit();
    try std.testing.expectEqualStrings("Qwen3.8-27B-refusal-cvec-per_layer-L10-58-a1", parsed.file.str("general.name").?);
    try std.testing.expectEqualStrings("TensorFold", parsed.file.str("general.author").?);
    try std.testing.expectEqualStrings("qwen3_5", parsed.file.str("controlvector.model_hint").?);
    try std.testing.expectEqual(@as(u64, 1), parsed.file.num("controlvector.layer_count").?);
    try std.testing.expectEqual(@as(u64, 1), parsed.file.num("general.base_model.count").?);
    try std.testing.expectEqualStrings("Qwen3.8-27B-MLX-4bit", parsed.file.str("general.base_model.0.name").?);
    try std.testing.expectEqualStrings("TensorFold", parsed.file.str("general.base_model.0.organization").?);
    try std.testing.expectEqualStrings("22d8d538154e0e5b6f8dcb1fcc74a60a0798104c", parsed.file.str("general.base_model.0.version").?);
    try std.testing.expectEqualStrings("https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit", parsed.file.str("general.base_model.0.repo_url").?);
}

test "a vector written again keeps what the file said about itself" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{
        .base_model = "Qwen/Qwen3.8-27B",
        .publication = .{ .name = "a name", .author = "an author" },
        .layers = &layers,
    });
    defer a.free(bytes);
    var read = try t.readOk(bytes);
    defer read.deinit();
    const again = try glp.writeVector(a, read.vector);
    defer a.free(again);
    var reread = try t.readOk(again);
    defer reread.deinit();
    try std.testing.expectEqualStrings("a name", reread.vector.publication.name.?);
    try std.testing.expectEqualStrings("an author", reread.vector.publication.author.?);
}

test "a file that names no model of its own keeps the base model out of the header" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{ .layers = &layers });
    defer a.free(bytes);
    var parsed = try t.parseFile(a, bytes);
    defer parsed.deinit();
    try std.testing.expect(parsed.file.str("general.base_model.0.repo_url") == null);
    try std.testing.expectEqual(@as(u64, 1), parsed.file.num("controlvector.layer_count").?);
}

test "a file that names no other site names the site it is applied at" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{ .hook_point = glp.residual_site, .layers = &layers });
    defer a.free(bytes);
    var parsed = try t.parseFile(a, bytes);
    defer parsed.deinit();
    // glp.derived_at is the site the direction was captured at, not a time.
    try std.testing.expectEqualStrings("residual_stream_post_layer", parsed.file.str("glp.derived_at").?);
    var read = try t.readOk(bytes);
    defer read.deinit();
    try std.testing.expect(!read.vector.transferred);
}
