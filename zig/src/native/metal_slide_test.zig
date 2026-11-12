//! Metal registration preserves Nemotron's adapter identity and effective-width admission without GPU work.
const std = @import("std");
const nemo = @import("tensorfold").nemotron;
const living = @import("living");
const registration = @import("metal_slide.zig");

test "Metal learner: Nemotron keeps its backend, adapter hooks and initial lesson mode" {
    const a = std.testing.allocator;
    var backend: nemo.backend.Metal = undefined;
    const Adapter = @import("slide_adapter.zig").Adapter(nemo.slide);
    var before = Adapter.init(a, std.testing.io, &backend);
    defer before.deinit();
    const after = try registration.init(nemo, a, std.testing.io, &backend, "unused-checkpoint");
    defer after.deinit(a);
    const typed: *Adapter = @ptrCast(@alignCast(after.ctx));
    try std.testing.expect(typed.learner.b == &backend);
    try std.testing.expectEqual(before.learner.whole, typed.learner.whole);
    try std.testing.expectEqual(before.learner.method, typed.learner.method);
    try std.testing.expectEqual(before.learner.readsConversations(), after.conversations);
    const hook = before.hook();
    try std.testing.expectEqual(hook.begin, after.hook.begin);
    try std.testing.expectEqual(hook.step, after.hook.step);
    try std.testing.expectEqual(hook.abort, after.hook.abort);
}

test "Metal learner: descriptor widths match Nemotron's existing mixed-width admission" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const paths = [_][]const u8{
        "backbone.embeddings",                    "lm_head",                                                       "backbone.layers.0.mixer.in_proj",                  "backbone.layers.0.mixer.out_proj",
        "backbone.layers.1.mixer.switch_mlp.fc1", "backbone.layers.1.mixer.shared_experts.up_proj",                "backbone.layers.1.mixer.shared_experts.down_proj", "backbone.layers.2.mixer.q_proj",
        "language_model.backbone.embeddings",     "language_model.backbone.layers.1.mixer.shared_experts.up_proj", "unused.module",
    };
    const values = [_][]const u8{ "true", "false", "7", "{\"bits\":8}", "{\"group_size\":32}", "{\"bits\":4,\"group_size\":64}" };
    var checked: usize = 0;
    for ([_]u8{ 4, 8 }) |bits| for (paths) |path| for (values) |value| {
        const q = try std.fmt.allocPrint(a, "\"quantization\": {{\"group_size\":64,\"bits\":{d},\"{s}\":{s}}}", .{ bits, path, value });
        const text = try std.mem.replaceOwned(u8, a, nemo.config.lightning, "\"quantization\": {\"group_size\": 64, \"bits\": 4}", q);
        var why: nemo.config.Why = .{};
        const old = nemo.config.parse(a, text, &why) catch continue; // the actual loader continues to reject invalid configs
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const got = try living.widths.other(a, living.families.nemotron_h, doc.object);
        var buf: [128]u8 = undefined;
        if (old.other(&buf)) |want| {
            const bad = got orelse return error.MissingWidthRefusal;
            try std.testing.expectEqualStrings(want.name, bad.name);
            try std.testing.expectEqual(want.width.bits, bad.quant.bits);
            try std.testing.expectEqual(want.width.group, bad.quant.group);
        } else try std.testing.expect(got == null);
        checked += 1;
    };
    try std.testing.expect(checked > 100);
    const fallback = try std.mem.replaceOwned(u8, a, nemo.config.lightning, "\"quantization\": {\"group_size\": 64, \"bits\": 4}", "\"quantization\": {}, \"quantization_config\": {\"group_size\":64,\"bits\":8}");
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, fallback, .{});
    const got = (try living.widths.other(a, living.families.nemotron_h, doc.object)).?;
    try std.testing.expectEqualStrings("backbone.embeddings", got.name);
    try std.testing.expectEqual(@as(u8, 8), got.quant.bits);
}
