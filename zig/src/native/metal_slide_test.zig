//! Metal registration preserves Nemotron's adapter identity and effective-width admission without GPU work.
const std = @import("std");
const nemo = @import("tensorfold").nemotron;
const living = @import("living");
const registration = @import("metal_slide.zig");
const api = @import("engine_api");

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

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// `name` set to `value` (null: unset); what it was, copied first, so a deferred call puts it back.
fn setTo(a: std.mem.Allocator, name: [:0]const u8, value: ?[:0]const u8) !?[:0]const u8 {
    const before: ?[:0]const u8 = if (std.c.getenv(name)) |v| try a.dupeSentinel(u8, std.mem.span(v), 0) else null;
    _ = if (value) |v| setenv(name, v, 1) else unsetenv(name);
    return before;
}

test "slide admission: TF_KV_SHARE refuses Nemotron's --learn by name before any load or learned folder" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const open = @import("metal.zig").open;
    const refusal = @import("tensorfold").kv_prompts.learnRefusal;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const learned = try std.fs.path.join(a, &.{ try tmp.dir.realPathFileAlloc(io, ".", a), "learned" });
    const share_was = try setTo(a, "TF_KV_SHARE", "4");
    defer _ = setTo(a, "TF_KV_SHARE", share_was) catch null;
    const pages_was = try setTo(a, "TF_NEMOTRON_METAL_KV_PAGES", "1");
    defer _ = setTo(a, "TF_NEMOTRON_METAL_KV_PAGES", pages_was) catch null;
    const missing = "missing-kv-share-checkpoint";
    for ([_]bool{ false, true }) |talks| {
        var why: []const u8 = "";
        const o: api.Open = .{ .dir = missing, .model_type = "nemotron_h", .learn = learned, .learn_conversations = talks };
        try std.testing.expect((try open(a, std.testing.allocator, io, o, &why)) == null);
        try std.testing.expectEqualStrings(refusal(true, talks).?, why); // not the load's failure: no load ran
        try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "learned", .{}));
    }
    // control: without TF_KV_SHARE the same start reaches the load, which fails on the missing checkpoint
    _ = try setTo(a, "TF_KV_SHARE", null);
    var why: []const u8 = "";
    try std.testing.expect((try open(a, std.testing.allocator, io, .{ .dir = missing, .model_type = "nemotron_h", .learn = learned }, &why)) == null);
    try std.testing.expectStringStartsWith(why, "the native engine cannot load " ++ missing);
}
