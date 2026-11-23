//! Opt-in Mini checks: TF_INKLING2_MINI_SMOKE's dir (24 layers, 3 passes, NVFP4/MXFP8 experts, mtp.safetensors).
const std = @import("std");
const tf = @import("tensorfold");
const environment = @import("sys").environment;
const mtl = @import("metal");
const api = @import("engine_api");
const fl = tf.inkling2_metal;
const lanes = tf.lanes;
const native = @import("native/inkling2_backend.zig");
const kept_states = @import("native/inkling2_kept.zig");
const t = std.testing;

fn fixture() ![]const u8 {
    const raw = environment.rawText("TF_INKLING2_MINI_SMOKE");
    return (try environment.path("TF_INKLING2_MINI_SMOKE", raw)) orelse error.SkipZigTest;
}

fn logits(m: *fl.Model, rows: usize) []const f32 {
    return m.logits.slice(f32, rows * m.config.text.unpadded_vocab);
}

fn imageEncode(ptr: *anyopaque, image: lanes.stream.Image) !mtl.Buffer {
    const tower: *fl.vision.Tower = @ptrCast(@alignCast(ptr));
    return tower.encodeInput(image.pixels, image.gh, image.gw);
}

fn reply(engine: *lanes.Engine, ids: []const u32, images: []const lanes.stream.Image, drafts: bool, settings: ?lanes.Sampling) ![]u32 {
    var stream = try lanes.Stream.init(t.allocator, .{ .id = "inkling2-smoke", .prompt = ids, .max_new = 16, .images = images, .drafts = drafts, .sampling = settings });
    defer stream.deinit(t.allocator);
    try engine.addStream(&stream);
    while (engine.activeCount() != 0) try engine.step();
    if (stream.reason == .@"error") return error.InklingReplyFailed;
    try t.expectEqual(@as(usize, 16), stream.emitted().len);
    try t.expectEqual(drafts, stream.drafted > 0);
    try t.expect(stream.accepted <= stream.drafted);
    return t.allocator.dupe(u32, stream.emitted());
}

/// A stream prefilled alone on `backend`: its first token, drawn and read, then the first logprob row.
fn firstOf(backend: lanes.backend.Backend, stream: *lanes.Stream) !struct { token: u32, row: lanes.LogprobRow } {
    try backend.prefill(stream);
    defer backend.release(stream);
    const token = try backend.read(try backend.first(stream, stream.prompt_len));
    return .{ .token = token, .row = try backend.firstRow(stream) };
}

test "inkling2 Mini Metal loop forward, shared rows, exact rollback, snapshots, images and DSpark" {
    const dir = try fixture();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try fl.Model.load(t.allocator, t.io, dir, .{ .capacity = 128, .streams = 4, .rows = 32, .drafts = true, .vision = true });
    defer m.deinit();
    // the loop is real: more logical visits than physical layers, every visit with its own KV and convolution state
    try t.expect(m.config.text.executionLayers() > m.config.text.layers);
    var a = try m.newState();
    defer a.deinit();
    var b = try m.newState();
    defer b.deinit();
    var c = try m.newState();
    defer c.deinit();
    var d = try m.newState();
    defer d.deinit();
    var ids: [32]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1 + i);
    const vocab = m.config.text.unpadded_vocab;
    try m.forward(&.{.{ .state = &a, .ids = ids[0..16], .logit_rows = 16, .taps = true }});
    const first = try t.allocator.dupe(f32, logits(m, 16));
    defer t.allocator.free(first);
    for (ids[0..16], 0..) |id, i| {
        try m.forward(&.{.{ .state = &b, .ids = &.{id} }});
        try t.expectEqualSlices(f32, first[i * vocab ..][0..vocab], logits(m, 1));
    }
    try m.forward(&.{ .{ .state = &c, .ids = ids[0..16], .logit_rows = 16 }, .{ .state = &d, .ids = ids[0..16], .logit_rows = 16 } });
    try t.expectEqualSlices(f32, first, logits(m, 32)[0..first.len]);
    try t.expectEqualSlices(f32, first, logits(m, 32)[first.len..]);
    // Advance the prompt, then keep a partial verify window.
    try m.forward(&.{.{ .state = &a, .ids = ids[16..32], .logit_rows = 16 }});
    const second = try t.allocator.dupe(f32, logits(m, 16));
    defer t.allocator.free(second);
    for (ids[16..32], 0..) |id, i| {
        try m.forward(&.{.{ .state = &b, .ids = &.{id} }});
        try t.expectEqualSlices(f32, second[i * vocab ..][0..vocab], logits(m, 1));
    }
    var snap = try m.save(&a);
    defer snap.deinit();
    try m.forward(&.{.{ .state = &a, .ids = &.{ 33, 34, 35, 36 } }});
    try m.keep(&a, 34);
    try m.forward(&.{.{ .state = &a, .ids = &.{70} }});
    const kept = try t.allocator.dupe(f32, logits(m, 1));
    defer t.allocator.free(kept);
    try m.forward(&.{.{ .state = &b, .ids = &.{ 33, 34, 70 } }});
    try t.expectEqualSlices(f32, kept, logits(m, 1));
    try m.restore(&c, &snap);
    try m.forward(&.{.{ .state = &c, .ids = &.{ 33, 34, 70 } }});
    try t.expectEqualSlices(f32, kept, logits(m, 1));
    // The tower's real output replaces a template image row, rather than treating it as an ordinary token.
    const tower = try fl.vision.Tower.loadFromCheckpoint(t.allocator, m.device, m.queue, &m.config, .{ .max_tokens = 4 }, &m.checkpoint);
    defer tower.deinit();
    const pixels = try t.allocator.alloc(f32, 4 * 1200);
    defer t.allocator.free(pixels);
    for (pixels, 0..) |*p, i| p.* = @as(f32, @floatFromInt(i % 17)) / 8 - 1;
    const encoded = try tower.encodeInput(pixels, 1, 1);
    defer encoded.deinit();
    const prompt = [_]u32{ 1, m.config.image_token, 2, 3 };
    m.reset(&c);
    try m.forward(&.{.{ .state = &c, .ids = &prompt, .images = &.{.{ .row = 1, .rows = 1, .buf = encoded, .off = 0 }} }});
    const pictured = try t.allocator.dupe(f32, logits(m, 1));
    defer t.allocator.free(pictured);
    m.reset(&d);
    try m.forward(&.{.{ .state = &d, .ids = &prompt }});
    try t.expect(!std.mem.eql(f32, pictured, logits(m, 1)));
    const target = try native.Target.init(t.allocator, m, .{ .capacity = 128, .streams = 4, .rows = 32, .drafts = true, .images = .{ .ctx = tower, .encode = imageEncode } });
    defer target.deinit();
    const backend = target.backend();
    // A masked first draw outside the alternatives must retain its raw logprob, not the argmax's row.
    {
        var logged = try lanes.Stream.init(t.allocator, .{ .id = "first-logprob", .prompt = &prompt, .max_new = 1, .logprobs = 0 });
        defer logged.deinit(t.allocator);
        try backend.prefill(&logged);
        defer backend.release(&logged);
        const best = try backend.read(try backend.first(&logged, prompt.len));
        const alternative = (best + 1) % vocab;
        const mask = try t.allocator.alloc(u32, (vocab + 31) / 32);
        defer t.allocator.free(mask);
        @memset(mask, 0);
        mask[alternative / 32] = @as(u32, 1) << @intCast(alternative % 32);
        const picked = try backend.read(try backend.vtable.first_masked.?(backend.ptr, &logged, prompt.len, mask));
        const metadata = try backend.firstRow(&logged);
        try t.expectEqual(alternative, picked);
        try t.expectEqual(picked, metadata.token);
        try t.expectEqual(@as(u8, 0), metadata.count);
        try t.expect(std.math.isFinite(metadata.logprob));
    }
    // A kept prompt state resumes bit for bit: the restored lane draws what a fresh pass over the whole prompt draws.
    {
        const long = [_]u32{ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
        var head = try lanes.Stream.init(t.allocator, .{ .id = "keep-head", .prompt = long[0..16], .max_new = 1 });
        defer head.deinit(t.allocator);
        try backend.prefill(&head);
        const snaps = target.snapshots();
        const saved = try snaps.vtable.save(snaps.ptr, &head, 16);
        defer snaps.vtable.drop(snaps.ptr, saved);
        backend.release(&head);
        var fresh = try lanes.Stream.init(t.allocator, .{ .id = "keep-fresh", .prompt = &long, .max_new = 1, .logprobs = 3 });
        defer fresh.deinit(t.allocator);
        const want = try firstOf(backend, &fresh);
        var resumed = try lanes.Stream.init(t.allocator, .{ .id = "keep-resumed", .prompt = &long, .max_new = 1, .logprobs = 3, .reuse = .{ .saved = saved, .at = 16 } });
        defer resumed.deinit(t.allocator);
        const got = try firstOf(backend, &resumed);
        try t.expectEqual(@as(u32, 16), resumed.cached);
        try t.expect(!resumed.reuse_failed);
        try t.expectEqual(want.token, got.token);
        try t.expectEqual(want.row.logprob, got.row.logprob);
        try t.expectEqualSlices(u32, want.row.ids[0..3], got.row.ids[0..3]);

        var wrong = try lanes.Stream.init(t.allocator, .{ .id = "keep-wrong", .prompt = &long, .max_new = 1, .reuse = .{ .saved = saved, .at = 15 } });
        defer wrong.deinit(t.allocator);
        const retried = try firstOf(backend, &wrong);
        try t.expect(wrong.reuse_failed);
        try t.expectEqual(@as(u32, 0), wrong.cached);
        try t.expectEqual(want.token, retried.token);
        try t.expectEqual(want.row.logprob, retried.row.logprob);
    }
    const images = [_]lanes.stream.Image{.{ .pixels = pixels, .gh = 1, .gw = 1, .at = 1, .tokens = 1 }};
    var plain_cfg = try lanes.Config.init(t.allocator, target.facts(), 16, 15);
    defer plain_cfg.deinit(t.allocator);
    var clock = lanes.backend.WallClock{ .io = t.io };
    var engine = lanes.Engine.init(t.allocator, &plain_cfg, backend, clock.clock());
    defer engine.deinit();
    const expected = try reply(&engine, &prompt, &images, false, null);
    defer t.allocator.free(expected);
    const sampled: lanes.Sampling = .{ .seed = 7, .temperature = 0.7, .top_k = 20, .top_p = 0.95 };
    const expected_sampled = try reply(&engine, &prompt, &images, false, sampled);
    defer t.allocator.free(expected_sampled);
    // a lane released and pooled again serves the same reply
    const again = try reply(&engine, &prompt, &images, false, null);
    defer t.allocator.free(again);
    try t.expectEqualSlices(u32, expected, again);
    const draft = try fl.draft.Adapter.init(t.allocator, t.io, dir, m.device, m.queue, &m.config, .{ .embed = m.weights.embed, .embed_norm = m.weights.embed_norm, .unembed = m.weights.unembed }, .{ .streams = 4, .capacity = 128 });
    defer draft.deinit();
    var wrapped = try lanes.drafted.Drafted.init(t.allocator, backend, draft.drafter());
    defer wrapped.deinit();
    wrapped.first_rows = true;
    var drafted_cfg = try lanes.Config.init(t.allocator, wrapped.facts(target.facts()), 16, 15);
    defer drafted_cfg.deinit(t.allocator);
    var drafted = lanes.Engine.init(t.allocator, &drafted_cfg, wrapped.backend(), clock.clock());
    defer drafted.deinit();
    const got = try reply(&drafted, &prompt, &images, true, null);
    defer t.allocator.free(got);
    try t.expectEqualSlices(u32, expected, got);
    const got_sampled = try reply(&drafted, &prompt, &images, true, sampled);
    defer t.allocator.free(got_sampled);
    try t.expectEqualSlices(u32, expected_sampled, got_sampled);
    const other_prompt = [_]u32{ 7, m.config.image_token, 8, 9 };
    const other_sampled: lanes.Sampling = .{ .seed = 11, .temperature = 0.7, .top_k = 20, .top_p = 0.95 };
    const other = try reply(&drafted, &other_prompt, &images, true, other_sampled);
    defer t.allocator.free(other);
    var one = try lanes.Stream.init(t.allocator, .{ .id = "one", .prompt = &prompt, .max_new = 16, .images = &images, .drafts = true, .sampling = sampled });
    defer one.deinit(t.allocator);
    var two = try lanes.Stream.init(t.allocator, .{ .id = "two", .prompt = &other_prompt, .max_new = 16, .images = &images, .drafts = true, .sampling = other_sampled });
    defer two.deinit(t.allocator);
    try drafted.addStream(&one);
    try drafted.addStream(&two);
    while (drafted.activeCount() != 0) try drafted.step();
    try t.expectEqualSlices(u32, expected_sampled, one.emitted());
    try t.expectEqualSlices(u32, other, two.emitted());
}

/// Keeps the target's state, and the head's context beside it, where a prompt pass stands at a mark.
const Keeper = struct {
    target: *native.Target,
    saved: ?*anyopaque = null,

    fn at(ptr: *anyopaque, s: *lanes.Stream, n: u32) void {
        const k: *Keeper = @ptrCast(@alignCast(ptr));
        const snaps = k.target.snapshots();
        k.saved = snaps.vtable.save(snaps.ptr, s, n) catch null;
    }
};

const Run = struct { tokens: []u32, drafted: u64, accepted: u64, cached: u32 };

fn run(engine: *lanes.Engine, prompt: []const u32, drafts: bool, settings: ?lanes.Sampling, reuse: lanes.stream.Reuse) !Run {
    var stream = try lanes.Stream.init(t.allocator, .{ .id = "inkling2-kept", .prompt = prompt, .max_new = 24, .drafts = drafts, .sampling = settings, .reuse = reuse });
    defer stream.deinit(t.allocator);
    try engine.addStream(&stream);
    while (engine.activeCount() != 0) try engine.step();
    if (stream.reason == .@"error") return error.InklingReplyFailed;
    return .{ .tokens = try t.allocator.dupe(u32, stream.emitted()), .drafted = stream.drafted, .accepted = stream.accepted, .cached = stream.cached };
}

test "inkling2 Mini resumes a drafted stream from a kept state: its tokens and draft counts equal a fresh pass's" {
    const dir = try fixture();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try fl.Model.load(t.allocator, t.io, dir, .{ .capacity = 128, .streams = 2, .rows = 32, .drafts = true });
    defer m.deinit();
    const target = try native.Target.init(t.allocator, m, .{ .capacity = 128, .streams = 2, .rows = 32, .drafts = true });
    defer target.deinit();
    const draft = try fl.draft.Adapter.init(t.allocator, t.io, dir, m.device, m.queue, &m.config, .{ .embed = m.weights.embed, .embed_norm = m.weights.embed_norm, .unembed = m.weights.unembed }, .{ .streams = 2, .capacity = 128 });
    defer draft.deinit();
    target.head = draft;
    var wrapped = try lanes.drafted.Drafted.init(t.allocator, target.backend(), draft.drafter());
    defer wrapped.deinit();
    wrapped.first_rows = true;
    var plain_cfg = try lanes.Config.init(t.allocator, target.facts(), 16, 15);
    defer plain_cfg.deinit(t.allocator);
    var drafted_cfg = try lanes.Config.init(t.allocator, wrapped.facts(target.facts()), 16, 15);
    defer drafted_cfg.deinit(t.allocator);
    var clock = lanes.backend.WallClock{ .io = t.io };
    var plain = lanes.Engine.init(t.allocator, &plain_cfg, target.backend(), clock.clock());
    defer plain.deinit();
    var drafted = lanes.Engine.init(t.allocator, &drafted_cfg, wrapped.backend(), clock.clock());
    defer drafted.deinit();
    var prompt: [70]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(5 + i % 20); // a cycle: the head drafts and the target accepts some
    const marks = [_]u32{32};
    const sampled: lanes.Sampling = .{ .seed = 7, .temperature = 0.7, .top_k = 20, .top_p = 0.95 };
    for ([_]?lanes.Sampling{ null, sampled }) |settings| {
        var keeper: Keeper = .{ .target = target };
        const fresh = try run(&drafted, &prompt, true, settings, .{ .marks = &marks, .hook = .{ .ptr = &keeper, .at = Keeper.at } });
        defer t.allocator.free(fresh.tokens);
        const snaps = target.snapshots();
        const saved = keeper.saved orelse return error.NothingKept;
        defer snaps.vtable.drop(snaps.ptr, saved);
        try t.expect(@as(*const kept_states.Kept, @ptrCast(@alignCast(saved))).head != null); // the drafting stream's head had a context
        try t.expect(fresh.drafted > 0);
        const resumed = try run(&drafted, &prompt, true, settings, .{ .saved = saved, .at = 32 });
        defer t.allocator.free(resumed.tokens);
        try t.expectEqual(@as(u32, 32), resumed.cached);
        try t.expectEqualSlices(u32, fresh.tokens, resumed.tokens);
        try t.expectEqual(fresh.drafted, resumed.drafted); // the head drafted from the same context, so rounds are the same
        try t.expectEqual(fresh.accepted, resumed.accepted);
        const serial = try run(&plain, &prompt, false, settings, .{});
        defer t.allocator.free(serial.tokens);
        try t.expectEqualSlices(u32, serial.tokens, resumed.tokens);
        // a state kept by a stream that did not draft holds no head context: same tokens, the head begins past it
        var bare: Keeper = .{ .target = target };
        const plain_kept = try run(&plain, &prompt, false, settings, .{ .marks = &marks, .hook = .{ .ptr = &bare, .at = Keeper.at } });
        defer t.allocator.free(plain_kept.tokens);
        const headless = bare.saved orelse return error.NothingKept;
        defer snaps.vtable.drop(snaps.ptr, headless);
        try t.expect(@as(*const kept_states.Kept, @ptrCast(@alignCast(headless))).head == null);
        const begun = try run(&drafted, &prompt, true, settings, .{ .saved = headless, .at = 32 });
        defer t.allocator.free(begun.tokens);
        try t.expectEqual(@as(u32, 32), begun.cached);
        try t.expectEqualSlices(u32, serial.tokens, begun.tokens);
    }
}

test {
    _ = @import("native/inkling2_backend.zig");
}

test "inkling2 Mini keeps every expert resident: SSD expert pools are refused for it on Metal" {
    var o: api.Open = .{ .dir = "", .model_type = "inkling2_loop", .ssd_experts_gib = 4 };
    try t.expect(o.fsrRefusal("metal") != null);
    try t.expect(o.fsrRefusal("cuda") != null);
    o.ssd_experts_gib = null;
    o.ple_on_ssd = true;
    try t.expect(o.fsrRefusal("metal") != null);
    o.ple_on_ssd = false;
    o.fsr_trace = "trace";
    try t.expect(o.fsrRefusal("metal") != null);
}
