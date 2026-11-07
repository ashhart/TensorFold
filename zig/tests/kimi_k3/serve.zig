//! Cut-down synthetic K3 through the real lane adapter; no checkpoint or K3.1 reference is used.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const lanes = @import("lanes");
const Gpu = @import("gpu.zig").Gpu;
const Synth = @import("synth.zig").Synth;
const M = k3.backend.Metal;
const A = std.mem.Allocator;

const Proposal = struct {
    ids: [7]u32 = .{ 1, 2, 3, 4, 5, 6, 7 },
    widest: u32 = 0,
    rounds: usize = 0,
    fn interface(p: *@This()) lanes.proposer.Proposer {
        return .{ .ptr = p, .vtable = &.{ .propose = propose, .last_match = match, .round = observed, .priced = true } };
    }
    fn propose(ptr: *anyopaque, _: []const u32, most: i64) ![]const u32 {
        const p: *@This() = @ptrCast(@alignCast(ptr));
        return p.ids[0..@min(p.ids.len, @as(usize, @intCast(@max(0, most))))];
    }
    fn match(_: *anyopaque) i64 {
        return 16;
    }
    fn observed(ptr: *anyopaque, rows: u32, _: u32) void {
        const p: *@This() = @ptrCast(@alignCast(ptr));
        p.widest = @max(p.widest, rows);
        p.rounds += 1;
    }
};

fn tokens(a: A, io: std.Io, m: *M, prompt: []const u32, proposal: ?*Proposal, saved: ?*k3.snapshot.Snap, sample: ?lanes.Sampling) ![]u32 {
    var config = try lanes.Config.init(a, m.facts(), k3.backend.window, k3.backend.window - 1);
    defer config.deinit(a);
    config.timed = false;
    var clock: lanes.backend.WallClock = .{ .io = io };
    var engine = lanes.Engine.init(a, &config, m.backend(), clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "synthetic", .prompt = prompt, .max_new = 8, .sampling = sample, .drafts = proposal != null, .proposer = if (proposal) |p| p.interface() else null, .reuse = if (saved) |p| .{ .saved = p, .at = p.position.pos } else .{} });
    defer s.deinit(a);
    try engine.addStream(&s);
    var rounds: usize = 0;
    while (engine.activeCount() > 0) : (rounds += 1) {
        if (rounds > 64) return error.SyntheticRoundLimit;
        try engine.step();
    }
    try std.testing.expectEqual(@as(usize, 8), s.emitted().len);
    if (saved) |p| try std.testing.expectEqual(p.position.pos, s.cached);
    return a.dupe(u32, s.emitted());
}

fn together(a: A, io: std.Io, m: *M, prompts: []const []const u32, answers: []const []const u32, sample: ?lanes.Sampling) !void {
    var config = try lanes.Config.init(a, m.facts(), k3.backend.window, k3.backend.window - 1);
    defer config.deinit(a);
    config.timed = false;
    var clock: lanes.backend.WallClock = .{ .io = io };
    var engine = lanes.Engine.init(a, &config, m.backend(), clock.clock());
    defer engine.deinit();
    var streams: [2]lanes.Stream = undefined;
    var ready: usize = 0;
    defer for (streams[0..ready]) |*s| s.deinit(a);
    for (prompts, 0..) |p, i| {
        streams[i] = try lanes.Stream.init(a, .{ .id = if (i == 0) "a" else "b", .prompt = p, .max_new = 8, .sampling = sample, .drafts = false });
        ready += 1;
        try engine.addStream(&streams[i]);
    }
    var rounds: usize = 0;
    while (engine.activeCount() > 0) : (rounds += 1) {
        if (rounds > 64) return error.SyntheticRoundLimit;
        try engine.step();
    }
    for (&streams, answers) |*s, want| try std.testing.expectEqualSlices(u32, want, s.emitted());
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu = try Gpu.init();
    defer gpu.deinit();
    var source: Synth = .{ .gpa = a, .device = gpu.device, .queue = gpu.queue, .k = &gpu.k };
    defer source.deinit();
    var c: k3.config.Config = .{};
    c.layers = 4;
    c.experts = 16;
    c.vocab = 512;
    c.full_attn = .empty;
    c.full_attn.set(3);
    var graph = try k3.graph.Graph.load(a, gpu.device, gpu.queue, &gpu.k, c, source.source());
    defer graph.deinit();
    source.sync();
    const m = try M.init(a, &graph, gpu.device, gpu.queue, &gpu.k, 3, 128, source.bufs.items);
    defer m.deinit();
    const p0 = [_]u32{ 1, 7, 3, 1, 7, 3, 1, 7, 3, 1, 7, 3, 1, 7, 3, 1, 7 };
    const p1 = [_]u32{ 5, 2, 4, 8, 5, 2, 4, 8, 5 };
    const plain = try tokens(a, init.io, m, &p0, null, null, null);
    defer a.free(plain);
    const first_logits = try a.dupe(u16, m.logits(0));
    defer a.free(first_logits);
    for (first_logits) |bits| try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(@as(u32, bits) << 16))));
    const solo = try tokens(a, init.io, m, &p1, null, null, null);
    defer a.free(solo);
    try std.testing.expect(!std.mem.eql(u16, first_logits, m.logits(0)));
    std.debug.print("KIMI_NATIVE_TOKENS plain={any} other_prompt={any}; finite logits differ across prompts\n", .{ plain, solo });
    var proposal: Proposal = .{};
    const drafted = try tokens(a, init.io, m, &p0, &proposal, null, null);
    defer a.free(drafted);
    try std.testing.expectEqualSlices(u32, plain, drafted);
    try std.testing.expect(proposal.widest > 1 and proposal.rounds > 0);
    try together(a, init.io, m, &.{ &p0, &p1 }, &.{ plain, solo }, null);
    const sample: lanes.Sampling = .{ .seed = 973, .temperature = 0.7, .top_k = 8, .top_p = 0.9 };
    const sampled_plain = try tokens(a, init.io, m, &p0, null, null, sample);
    defer a.free(sampled_plain);
    const sampled_other = try tokens(a, init.io, m, &p1, null, null, sample);
    defer a.free(sampled_other);
    var sampled_proposal: Proposal = .{};
    const sampled_drafted = try tokens(a, init.io, m, &p0, &sampled_proposal, null, sample);
    defer a.free(sampled_drafted);
    try std.testing.expectEqualSlices(u32, sampled_plain, sampled_drafted);
    try std.testing.expect(sampled_proposal.widest > 1);
    try together(a, init.io, m, &.{ &p0, &p1 }, &.{ sampled_plain, sampled_other }, sample);
    std.debug.print("KIMI_NATIVE_SAMPLED seed=973 plain={any} other_prompt={any}\n", .{ sampled_plain, sampled_other });
    const backend = m.backend();
    for ([_]usize{ 1, 4, 15, 16, 17 }) |cut| {
        var prefix = try lanes.Stream.init(a, .{ .id = "prefix", .prompt = p0[0..cut], .max_new = 0, .drafts = false });
        defer prefix.deinit(a);
        try backend.vtable.prefill(backend.ptr, &prefix);
        const slot = try m.find(&prefix);
        const saved = try k3.snapshot.save(m, slot, prefix.prompt());
        defer saved.deinit(a);
        try std.testing.expect(saved.charge() <= k3.snapshot.bound(&c, @intCast(cut), k3.backend.window));
        backend.vtable.release(backend.ptr, &prefix);
        const resumed = try tokens(a, init.io, m, &p0, null, saved, null);
        defer a.free(resumed);
        try std.testing.expectEqualSlices(u32, plain, resumed);
    }
    // Capture after keeping only part of a verify window, while KDA's state still needs its logged replay.
    var base = try lanes.Stream.init(a, .{ .id = "rollback", .prompt = p0[0..11], .max_new = 8, .drafts = false });
    defer base.deinit(a);
    try backend.vtable.prefill(backend.ptr, &base);
    var positions = [_]u64{ 12, 13, 14, 15, 16, 17 };
    const held_ids = [_]u32{ 9, 4, 3, 2, 1 };
    const windows = [_]lanes.backend.Window{.{ .stream = &base, .pending = 7, .held = 0, .tokens = &held_ids, .parents = null, .positions = &positions }};
    var drawn: [6]u32 = undefined;
    var held: [5]u32 = undefined;
    var verified = [_]lanes.backend.Verified{.{ .sampled = &drawn, .drafts = &held }};
    try backend.vtable.verify(backend.ptr, &windows, &verified);
    try std.testing.expectEqualSlices(u32, m.scratch.tokens.slice(u32, drawn.len), &drawn);
    try backend.vtable.keep(backend.ptr, &windows, &.{&.{ 0, 1, 2 }});
    const rollback_slot = try m.find(&base);
    try std.testing.expectEqual(@as(u32, 3), m.state.streams[rollback_slot].kept);
    var forced: [14]u32 = undefined;
    @memcpy(forced[0..11], p0[0..11]);
    @memcpy(forced[11..], &[_]u32{ 7, 9, 4 });
    const replay = try k3.snapshot.save(m, rollback_slot, &forced);
    defer replay.deinit(a);
    backend.vtable.release(backend.ptr, &base);
    const fresh = try tokens(a, init.io, m, &forced, null, null, null);
    defer a.free(fresh);
    const restored = try tokens(a, init.io, m, &forced, null, replay, null);
    defer a.free(restored);
    try std.testing.expectEqualSlices(u32, fresh, restored);
    const sampled_fresh = try tokens(a, init.io, m, &forced, null, null, sample);
    defer a.free(sampled_fresh);
    const sampled_restored = try tokens(a, init.io, m, &forced, null, replay, sample);
    defer a.free(sampled_restored);
    try std.testing.expectEqualSlices(u32, sampled_fresh, sampled_restored);
    const Cancel = struct {
        fn check(ptr: *anyopaque) bool {
            return @as(*const bool, @ptrCast(@alignCast(ptr))).*;
        }
    };
    var cancelled = false;
    var cancel_stream = try lanes.Stream.init(a, .{ .id = "cancel", .prompt = &p0, .max_new = 8, .drafts = false, .cancel_check = .{ .ptr = &cancelled, .check = Cancel.check } });
    defer cancel_stream.deinit(a);
    defer backend.vtable.release(backend.ptr, &cancel_stream);
    try std.testing.expect(!try backend.vtable.prefill_step.?(backend.ptr, &cancel_stream, 4));
    cancelled = true;
    try std.testing.expectError(error.Cancelled, backend.vtable.prefill_step.?(backend.ptr, &cancel_stream, 4));
    try std.testing.expectEqual(@as(u32, 4), cancel_stream.fill_at.load(.acquire));
    backend.vtable.release(backend.ptr, &cancel_stream);
    const after_cancel = try tokens(a, init.io, m, &p0, null, null, null);
    defer a.free(after_cancel);
    try std.testing.expectEqualSlices(u32, plain, after_cancel);
    std.debug.print("KIMI_NATIVE_SYNTHETIC PASS: four layers, KDA/MLA and MXFP4 MoE; greedy and keyed sampled drafted=plain, concurrent=solo; five cuts and partial-verify replay resumed=fresh; cancelled prefill leaves fresh output exact; widest verify {d}\n", .{proposal.widest});
}
