//! MOSS-Image's serving host without a device: recognition, refusals, limits, the queue, the splice, the 8-bit rule.
const std = @import("std");
const api = @import("engine_api");
const spec = @import("moss_image_spec.zig");
const host = @import("moss_image_host.zig");

/// A checkpoint folder holding `text` as its model_index.json: the path's sentinel is the allocation `a.free` expects.
fn checkpoint(tmp: *std.testing.TmpDir, io: std.Io, a: std.mem.Allocator, text: []const u8) ![:0]u8 {
    try tmp.dir.writeFile(io, .{ .sub_path = "model_index.json", .data = text });
    return tmp.dir.realPathFileAlloc(io, ".", a);
}

const pipeline =
    \\{"_class_name": "MOSSImagePipeline", "_diffusers_version": "0.38.0",
    \\ "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"],
    \\ "text_encoder": ["transformers", "Qwen3_5Model"], "tokenizer": ["transformers", "Qwen2Tokenizer"],
    \\ "transformer": ["pipeline_moss_image", "MOSSImageTransformer"], "vae": ["diffusers", "AutoencoderKLFlux2"]}
;

test "MOSS-Image: only a model_index.json naming the publisher's pipeline is recognised" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try checkpoint(&tmp, io, a, pipeline);
    defer a.free(dir);
    try std.testing.expect(spec.recognize(a, io, dir));
    // a token family's checkpoint has no model_index.json at all: the CUDA dispatch stays on its own path
    var other = std.testing.tmpDir(.{});
    defer other.cleanup();
    const family = try other.dir.realPathFileAlloc(io, ".", a);
    defer a.free(family);
    try other.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3_5\"}" });
    try std.testing.expect(!spec.recognize(a, io, family));
    // another diffusers pipeline is not this one either
    const video = try std.mem.replaceOwned(u8, a, pipeline, "MOSSImagePipeline", "WanPipeline");
    defer a.free(video);
    const moved = try checkpoint(&tmp, io, a, video);
    defer a.free(moved);
    try std.testing.expect(!spec.recognize(a, io, moved));
}

test "MOSS-Image: the limits a picture request is held to are the API's own, unloosened" {
    const l = spec.limits();
    try std.testing.expectEqual(@as(u16, 1536), l.max_side);
    try std.testing.expectEqual(@as(u32, 1_572_864), l.max_pixels);
    try std.testing.expectEqual(@as(u32, 200), l.max_steps);
    try std.testing.expect(l.max_prompt_tokens > 0 and l.max_prompt_tokens < 4096);
    // 1,536 x 1,024 is the largest geometry the transformer's one scratch is planned for
    try std.testing.expectEqual(l.max_pixels, 1536 * 1024);
}

test "MOSS-Image: every text-engine option is refused by name, and a bare open is not" {
    const bare: api.Open = .{ .dir = ".", .model_type = "unknown" };
    try std.testing.expect(spec.refusal(bare) == null);
    const flags = [_]struct { api.Open, []const u8 }{
        .{ .{ .dir = ".", .model_type = "unknown", .drafter = "dir" }, "--drafter" },
        .{ .{ .dir = ".", .model_type = "unknown", .mtp_drafts = 4 }, "--mtp-drafts" },
        .{ .{ .dir = ".", .model_type = "unknown", .prompt_cache_gib = 4 }, "--prompt-cache-gib" },
        .{ .{ .dir = ".", .model_type = "unknown", .keep = 8 }, "--checkpoint-slots" },
        .{ .{ .dir = ".", .model_type = "unknown", .learn = "dir" }, "--learn" },
        .{ .{ .dir = ".", .model_type = "unknown", .learn_conversations = true }, "--learn-conversations" },
        .{ .{ .dir = ".", .model_type = "unknown", .slide = true }, "--slide" },
        .{ .{ .dir = ".", .model_type = "unknown", .recall = true }, "--recall" },
        .{ .{ .dir = ".", .model_type = "unknown", .vision = .{} }, "--vision" },
        .{ .{ .dir = ".", .model_type = "unknown", .tp = 2 }, "--tp" },
        .{ .{ .dir = ".", .model_type = "unknown", .record = "dir" }, "--record" },
        .{ .{ .dir = ".", .model_type = "unknown", .kv8 = true }, "--kv-cache" },
        .{ .{ .dir = ".", .model_type = "unknown", .context = 8192 }, "--context" },
        .{ .{ .dir = ".", .model_type = "unknown", .device = 1 }, "--device" },
        .{ .{ .dir = ".", .model_type = "unknown", .lanes = 4, .lanes_fixed = true }, "--parallel" },
        .{ .{ .dir = ".", .model_type = "unknown", .segments = 4 }, "--segments" },
        .{ .{ .dir = ".", .model_type = "unknown", .policy = "x" }, "--policy" },
    };
    for (flags) |case| {
        const why = spec.refusal(case[0]) orelse return error.NotRefused;
        try std.testing.expect(std.mem.indexOf(u8, why, case[1]) != null);
    }
    // values that ask for nothing are not refusals: a prompt cache of 0 and one segment are the defaults
    try std.testing.expect(spec.refusal(.{ .dir = ".", .model_type = "unknown", .prompt_cache_gib = 0, .keep = 0, .segments = 1, .device = 0 }) == null);
}

test "MOSS-Image: the chat template's ids bracket a prompt, cut at the encoder's longest" {
    const a = std.testing.allocator;
    const before = [_]u32{ 1, 2, 3 };
    const after = [_]u32{ 8, 9 };
    const ids = try host.bracket(a, &before, &[_]u32{ 5, 6 }, &after);
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 5, 6, 8, 9 }, ids);
    // the empty negative prompt is the template alone, as the pipeline's encode_prompt("") is
    const none = try host.bracket(a, &before, &.{}, &after);
    defer a.free(none);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 8, 9 }, none);
    // a prompt past the encoder's longest keeps its head, as the pipeline's tokenizer call truncates
    const long = try a.alloc(u32, 4096);
    defer a.free(long);
    for (long, 0..) |*v, i| v.* = @intCast(i % 7);
    const cut = try host.bracket(a, &before, long, &after);
    defer a.free(cut);
    try std.testing.expectEqual(@as(usize, 4096), cut.len);
    try std.testing.expectEqualSlices(u32, &before, cut[0..before.len]);
}

test "MOSS-Image: the queue holds one picture and one arrival, serves them in order and frees a slot" {
    var q: host.Queue = .{};
    try std.testing.expectEqual(@as(usize, 2), host.Queue.depth);
    const first = q.admit(11) orelse return error.NotAdmitted;
    const second = q.admit(12) orelse return error.NotAdmitted;
    try std.testing.expectEqual(@as(usize, 2), q.held());
    try std.testing.expect(q.admit(13) == null); // the third request is the 503 the route answers
    // arrival order, not slot order: the first request begun is the first admitted
    try std.testing.expectEqual(first, q.next().?);
    try std.testing.expectEqual(second, q.next().?);
    try std.testing.expect(q.next() == null);
    // both slots stay held while their pictures run; releasing one admits the waiting request
    try std.testing.expectEqual(@as(usize, 2), q.held());
    try std.testing.expectEqual(@as(?usize, first), q.find(11));
    try std.testing.expectEqual(@as(?usize, second), q.find(12));
    try std.testing.expectEqual(@as(?usize, null), q.find(13));
    q.release(first);
    try std.testing.expectEqual(@as(usize, 1), q.held());
    const third = q.admit(13) orelse return error.NotAdmitted;
    try std.testing.expectEqual(third, first);
    try std.testing.expectEqual(third, q.next().?);
}

test "MOSS-Image: the pipeline's 8-bit rule is the decoder kernel's clamp and round to even" {
    // clamp to [-1, 1], then * 0.5 + 0.5 to [0, 1], then * 255 with rint
    const pixels = [_]f32{ -1, 1, 0, 0.5, -0.5, 2, -2, 0.25 };
    var out: [pixels.len]u8 = undefined;
    host.quantize(&out, &pixels);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 255, 128, 191, 64, 255, 0, 159 }, &out);
    // a tie goes to the nearer even number, as metal::rint does
    for ([_]struct { f32, u8 }{
        .{ 0.5, 0 },
        .{ 1.5, 2 },
        .{ 2.5, 2 },
        .{ 3.5, 4 },
        .{ 254.5, 254 },
        .{ 0.4999, 0 },
        .{ 0.6, 1 },
        .{ 255, 255 },
    }) |case| try std.testing.expectEqual(case[1], host.roundEven(case[0]));
}
