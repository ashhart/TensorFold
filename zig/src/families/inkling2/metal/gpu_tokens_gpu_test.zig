//! Metal GPU tests for Inkling 2 Mini gpu_tokens=true vs gpu_tokens=false, sampling boundaries, and async ring edges.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const flight = @import("flight.zig");
const backend_ = @import("backend.zig");
const draws = @import("draw.zig");
const sample = @import("sample.zig");
const tree_gpu = @import("tree_gpu_test.zig");

const gpa = std.testing.allocator;

const CopyDrafter = struct {
    oracle: []const u32,
    prompt_len: usize,
    held_toks: std.AutoHashMapUnmanaged(*lanes.Stream, [4]u32) = .empty,
    held_len: std.AutoHashMapUnmanaged(*lanes.Stream, usize) = .empty,

    fn drafter(x: *CopyDrafter) lanes.drafter.Drafter {
        return .{ .ptr = x, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release, .tree = treeFn } };
    }
    fn self(ptr: *anyopaque) *CopyDrafter {
        return @ptrCast(@alignCast(ptr));
    }
    fn taps(_: *anyopaque) []const u32 {
        return &.{ 5, 23, 41, 48 };
    }
    fn facts(_: *anyopaque) lanes.drafter.Facts {
        return .{ .depth = 4, .step_ms = 0.01, .tree_nodes = 4, .prior = &.{ 0.9, 0.9, 0.9, 0.9 } };
    }
    fn open(ptr: *anyopaque, s: *lanes.Stream) !void {
        const x = self(ptr);
        try x.held_toks.put(gpa, s, .{ 1, 2, 3, 4 });
        try x.held_len.put(gpa, s, 0);
    }
    fn absorb(_: *anyopaque, _: []const lanes.drafter.Absorb) !void {}
    fn hold(ptr: *anyopaque, list: []const lanes.drafter.Hold) !void {
        const x = self(ptr);
        for (list) |it| {
            const p = it.pending;
            const step = it.position -| x.prompt_len;
            const g0 = if (step < x.oracle.len) x.oracle[step] else (p + 1) % 100;
            const g1 = if (step + 1 < x.oracle.len) x.oracle[step + 1] else (p + 2) % 100;
            x.held_toks.getPtr(it.stream).?.* = .{ (g0 + 3) % 100, g0, g1, (g1 + 5) % 100 };
            x.held_len.getPtr(it.stream).?.* = @min(@as(usize, it.depth), 4);
        }
    }
    fn held(ptr: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) !void {
        const x = self(ptr);
        for (streams, out) |s, dst| @memcpy(dst, x.held_toks.get(s).?[0..dst.len]);
    }
    fn treeFn(ptr: *anyopaque, s: *lanes.Stream, alloc: std.mem.Allocator) !?lanes.stream.Held {
        const x = self(ptr);
        const n = x.held_len.get(s) orelse 0;
        if (n < 4) return null;
        const src = x.held_toks.get(s) orelse return null;
        const toks = try alloc.dupe(u32, src[0..4]);
        errdefer alloc.free(toks);
        const ps = try alloc.dupe(i32, &.{ -1, -1, 1, 2 });
        return .{ .count = 4, .tokens = toks, .parents = ps };
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const x = self(ptr);
        _ = x.held_toks.remove(s);
        _ = x.held_len.remove(s);
    }
    fn deinit(x: *CopyDrafter) void {
        x.held_toks.deinit(gpa);
        x.held_len.deinit(gpa);
    }
};

fn runPlain(target: *backend_.Metal, id: []const u8, prompt: []const u32, sampling: ?lanes.Sampling, max_new: usize, snap: ?*anyopaque, snap_at: u32) ![]u32 {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 1.0 + 0.01 * @as(f64, @floatFromInt(w)) };
    var f = target.facts();
    f.window_costs = &costs;
    var cfg = try lanes.Config.init(gpa, f, 16, 16);
    defer cfg.deinit(gpa);
    var clk: lanes.fake.FixedClock = .{};
    var eng = lanes.Engine.init(gpa, &cfg, target.backend(), clk.clock());
    defer eng.deinit();
    var s = try lanes.Stream.init(gpa, .{
        .id = id,
        .prompt = prompt,
        .max_new = @intCast(max_new),
        .sampling = sampling,
        .reuse = if (snap) |sv| .{ .saved = sv, .at = snap_at } else .{},
    });
    defer s.deinit(gpa);
    try eng.addStream(&s);
    while (eng.activeCount() > 0) try eng.step();
    return gpa.dupe(u32, s.emitted());
}

fn checkBoundaryOne(cpu_be: *backend_.Metal, gpu_be: *backend_.Metal, samp: lanes.Sampling, norm_order: bool, want_tok: u32) !void {
    const vocab = cpu_be.model.config.text.unpadded_vocab;
    const scratch = try gpa.alloc(f32, vocab);
    defer gpa.free(scratch);
    const prompt: [42]u32 = @splat(11);
    var s_cpu = try lanes.Stream.init(gpa, .{ .id = "b_cpu", .prompt = &prompt, .max_new = 4, .sampling = samp });
    defer s_cpu.deinit(gpa);
    var s_gpu = try lanes.Stream.init(gpa, .{ .id = "b_gpu", .prompt = &prompt, .max_new = 4, .sampling = samp });
    defer s_gpu.deinit(gpa);
    try cpu_be.backend().prefill(&s_cpu);
    defer cpu_be.backend().release(&s_cpu);
    try gpu_be.backend().prefill(&s_gpu);
    defer gpu_be.backend().release(&s_gpu);
    const l_cpu = cpu_be.caches.get(&s_cpu).?.logits.slice(f32, vocab);
    const l_gpu = gpu_be.caches.get(&s_gpu).?.logits.slice(f32, vocab);
    @memset(l_cpu, -50.0);
    if (norm_order) {
        l_cpu[9] = 0.0;
        l_cpu[1035] = -16.7;
        @memset(l_cpu[1056..1088], -16.7);
    } else {
        l_cpu[3122] = 0.0;
        l_cpu[1035] = -4.5;
        @memset(l_cpu[1040..1058], -4.5);
    }
    @memcpy(l_gpu, l_cpu);
    const want = try draws.draw(gpa, l_cpu, scratch, samp, 42, &.{});
    try std.testing.expectEqual(want_tok, want);
    try std.testing.expectEqual(want, try lanes.gpu_full.draw(gpa, l_cpu, samp, 42, null));
    const h_cpu = try cpu_be.backend().first(&s_cpu, 42);
    const h_gpu = try gpu_be.backend().first(&s_gpu, 42);
    try std.testing.expectEqual(want, try cpu_be.backend().read(h_cpu));
    try std.testing.expectEqual(want, try gpu_be.backend().read(h_gpu));
}

fn checkBoundaryDraws(cpu_be: *backend_.Metal, gpu_be: *backend_.Metal) !void {
    try checkBoundaryOne(cpu_be, gpu_be, .{ .seed = 18508, .temperature = 1.0, .top_k = 40, .top_p = 0.9999995 }, true, 9);
    try checkBoundaryOne(cpu_be, gpu_be, .{ .seed = 18508, .temperature = 1.0, .top_k = 100, .top_p = 0.9999995 }, true, 9);
    try checkBoundaryOne(cpu_be, gpu_be, .{ .seed = 18508, .temperature = 1.0, .top_k = 0, .top_p = 0.9999995 }, true, 1035);
    try checkBoundaryOne(cpu_be, gpu_be, .{ .seed = 18508, .temperature = 1.0, .top_k = 20, .top_p = 0.9 }, false, 1035);
    try checkBoundaryOne(cpu_be, gpu_be, .{ .seed = 18508, .temperature = 1.0, .top_k = 100, .top_p = 0.9 }, false, 1035);
    try checkBoundaryOne(cpu_be, gpu_be, .{ .seed = 18508, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, false, 3122);
}

fn checkAsyncRingBoundary(gpu_be: *backend_.Metal) !void {
    const b = gpu_be.backend();
    var s1 = try lanes.Stream.init(gpa, .{ .id = "ring1", .prompt = &.{11}, .max_new = 8 });
    defer s1.deinit(gpa);
    var s2 = try lanes.Stream.init(gpa, .{ .id = "ring2", .prompt = &.{22}, .max_new = 8 });
    defer s2.deinit(gpa);
    var s_ref = try lanes.Stream.init(gpa, .{ .id = "ring_ref", .prompt = &.{11}, .max_new = 8 });
    defer s_ref.deinit(gpa);
    try b.prefill(&s1);
    defer b.release(&s1);
    try b.prefill(&s2);
    defer b.release(&s2);
    try b.prefill(&s_ref);
    defer b.release(&s_ref);

    const h0 = try b.queue(&s1, .{ .value = 13 }, 2);
    const tok0 = try b.read(h0);
    for (0..flight.ring - 3) |_| _ = try b.first(&s2, 1);
    const h_flight = try b.queue(&s2, .{ .value = 17 }, 2);
    try std.testing.expectEqual(h0 + flight.ring - 1, h_flight);
    try std.testing.expectEqual(h0 + flight.ring, gpu_be.ring.next);

    // Accept next - h0 == 4096 while a flight is live and read slot h0 % ring before overwriting.
    const h_ok = try b.queue(&s1, .{ .handle = h0 }, 3);
    try std.testing.expectEqual(h0 + flight.ring, h_ok);
    try std.testing.expectEqual(h0 + flight.ring + 1, gpu_be.ring.next);

    // Reject next - h0 == 4097 in queueFn and readFn while keeping next - (h0 + 1) == 4096 valid.
    try std.testing.expectError(error.InvalidInklingTokenHandle, b.queue(&s2, .{ .handle = h0 }, 3));
    try std.testing.expectError(error.InvalidInklingTokenHandle, b.read(h0));
    try std.testing.expectEqual(try b.read(h0 + 1), try b.read(h0 + 1));

    const got_tok = try b.read(h_ok);
    _ = try b.read(try b.queue(&s_ref, .{ .value = 13 }, 2));
    const want_tok = try b.read(try b.queue(&s_ref, .{ .value = tok0 }, 3));
    try std.testing.expectEqual(want_tok, got_tok);
    try std.testing.expectEqual(want_tok, try b.read(h_ok));
}

fn checkModesAndConcurrency(cpu_be: *backend_.Metal, gpu_be: *backend_.Metal) !void {
    const prompt = [_]u32{ 11, 22, 33, 11, 22, 44, 11, 22, 55, 11, 22, 33, 11, 22, 44, 11, 22 };
    const samps = [_]?lanes.Sampling{
        null,
        .{ .seed = 104, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 105, .temperature = 0.9, .top_k = 100, .top_p = 0.9 },
        .{ .seed = 101, .temperature = 1.0, .top_k = 0, .top_p = 0.9 },
    };
    for (samps, 0..) |samp, idx| {
        const cpu_toks = try runPlain(cpu_be, "cpu_plain", &prompt, samp, 8, null, 0);
        defer gpa.free(cpu_toks);
        const gpu_toks = try runPlain(gpu_be, "gpu_plain", &prompt, samp, 8, null, 0);
        defer gpa.free(gpu_toks);
        try std.testing.expectEqualSlices(u32, cpu_toks, gpu_toks);
        if (idx != 1) continue;

        // Exercise drafted, resumed, and concurrent solo-equivalent equality on gpu_tokens = true.
        var half_s = try lanes.Stream.init(gpa, .{ .id = "half", .prompt = prompt[0..8], .max_new = 8, .sampling = samp });
        defer half_s.deinit(gpa);
        try gpu_be.backend().prefill(&half_s);
        defer gpu_be.backend().release(&half_s);
        const snap = try backend_.Metal.snapSave(gpu_be, &half_s, 8);
        defer backend_.Metal.snapDrop(gpu_be, snap);

        const resumed_toks = try runPlain(gpu_be, "gpu_resumed", &prompt, samp, 8, snap, 8);
        defer gpa.free(resumed_toks);
        try std.testing.expectEqualSlices(u32, cpu_toks, resumed_toks);

        var drafter = CopyDrafter{ .oracle = cpu_toks, .prompt_len = prompt.len };
        defer drafter.deinit();
        var wrapped = try lanes.drafted.Drafted.init(gpa, gpu_be.backend(), drafter.drafter());
        defer wrapped.deinit();
        var costs: [16]lanes.config.Cost = undefined;
        for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 1.0 + 0.01 * @as(f64, @floatFromInt(w)) };
        var d_model = wrapped.facts(gpu_be.facts());
        d_model.window_costs = &costs;
        var d_cfg = try lanes.Config.init(gpa, d_model, 16, 16);
        defer d_cfg.deinit(gpa);
        d_cfg.fill_lanes = 8;
        var clk: lanes.fake.FixedClock = .{};
        var eng = lanes.Engine.init(gpa, &d_cfg, wrapped.backend(), clk.clock());
        defer eng.deinit();
        var ds = try lanes.Stream.init(gpa, .{ .id = "gpu_drafted", .prompt = &prompt, .max_new = 8, .drafts = true, .sampling = samp });
        defer ds.deinit(gpa);
        var cs = try lanes.Stream.init(gpa, .{ .id = "gpu_conc", .prompt = &prompt, .max_new = 8, .drafts = true, .sampling = samp, .reuse = .{ .saved = snap, .at = 8 } });
        defer cs.deinit(gpa);
        try eng.addStream(&ds);
        try eng.addStream(&cs);
        while (eng.activeCount() > 0) try eng.step();
        try std.testing.expectEqualSlices(u32, cpu_toks, ds.emitted());
        try std.testing.expectEqualSlices(u32, cpu_toks, cs.emitted());
        try std.testing.expect(ds.drafted > 0 and ds.accepted > 0);
    }
}

fn checkMixedMaskedVerify(cpu_be: *backend_.Metal, gpu_be: *backend_.Metal) !void {
    const vocab = cpu_be.model.config.text.vocab;
    const words = draws.minMaskWords(vocab);
    const masks = try gpa.alloc(u32, 2 * words);
    defer gpa.free(masks);
    @memset(masks, 0);
    for (0..2) |r| {
        masks[r * words + 10] = 0x5555_5555;
        masks[r * words + 11] = 0x00FF_00FF;
    }
    const prompt = [_]u32{ 11, 22, 33, 44 };
    const samp: lanes.Sampling = .{ .seed = 99, .temperature = 0.8, .top_k = 40, .top_p = 0.9 };
    var c0 = try lanes.Stream.init(gpa, .{ .id = "mc0", .prompt = &prompt, .max_new = 4, .sampling = samp });
    defer c0.deinit(gpa);
    var c1 = try lanes.Stream.init(gpa, .{ .id = "mc1", .prompt = &prompt, .max_new = 4, .sampling = samp });
    defer c1.deinit(gpa);
    var g0 = try lanes.Stream.init(gpa, .{ .id = "mg0", .prompt = &prompt, .max_new = 4, .sampling = samp });
    defer g0.deinit(gpa);
    var g1 = try lanes.Stream.init(gpa, .{ .id = "mg1", .prompt = &prompt, .max_new = 4, .sampling = samp });
    defer g1.deinit(gpa);
    for ([_]*lanes.Stream{ &c0, &c1 }) |s| try cpu_be.backend().prefill(s);
    defer for ([_]*lanes.Stream{ &c0, &c1 }) |s| cpu_be.backend().release(s);
    for ([_]*lanes.Stream{ &g0, &g1 }) |s| try gpu_be.backend().prefill(s);
    defer for ([_]*lanes.Stream{ &g0, &g1 }) |s| gpu_be.backend().release(s);

    var cs0: [2]u32 = undefined;
    var cs1: [2]u32 = undefined;
    var cd0: [1]u32 = undefined;
    var cd1: [1]u32 = undefined;
    var gs0: [2]u32 = undefined;
    var gs1: [2]u32 = undefined;
    var gd0: [1]u32 = undefined;
    var gd1: [1]u32 = undefined;
    const cw = [_]lanes.backend.Window{
        .{ .stream = &c0, .pending = 15, .held = 0, .tokens = &.{25}, .parents = null, .masks = masks, .positions = &.{ 5, 6 } },
        .{ .stream = &c1, .pending = 15, .held = 0, .tokens = &.{25}, .parents = null, .positions = &.{ 5, 6 } },
    };
    const gw = [_]lanes.backend.Window{
        .{ .stream = &g0, .pending = 15, .held = 0, .tokens = &.{25}, .parents = null, .masks = masks, .positions = &.{ 5, 6 } },
        .{ .stream = &g1, .pending = 15, .held = 0, .tokens = &.{25}, .parents = null, .positions = &.{ 5, 6 } },
    };
    var c_out = [_]lanes.backend.Verified{ .{ .sampled = &cs0, .drafts = &cd0 }, .{ .sampled = &cs1, .drafts = &cd1 } };
    var g_out = [_]lanes.backend.Verified{ .{ .sampled = &gs0, .drafts = &gd0 }, .{ .sampled = &gs1, .drafts = &gd1 } };
    try cpu_be.backend().verify(&cw, &c_out);
    try gpu_be.backend().verify(&gw, &g_out);
    try std.testing.expectEqualSlices(u32, &cs0, &gs0);
    try std.testing.expectEqualSlices(u32, &cs1, &gs1);
}

fn checkConcurrentDrawRows(gpu_be: *backend_.Metal) !void {
    const vocab = gpu_be.model.config.text.unpadded_vocab;
    const scratch = try gpa.alloc(f32, vocab);
    defer gpa.free(scratch);
    const all = gpu_be.scratch.logits_all.slice(f32, 4 * vocab);
    @memset(all, -50.0);
    const r0 = all[0..vocab];
    r0[9] = 0.0;
    r0[1035] = -16.7;
    @memset(r0[1056..1088], -16.7);
    const r1 = all[vocab .. 2 * vocab];
    r1[77] = 0.0;
    const r3 = all[3 * vocab .. 4 * vocab];
    r3[3122] = 0.0;
    r3[1035] = -4.5;
    @memset(r3[1040..1058], -4.5);

    const s0: lanes.Sampling = .{ .seed = 101, .temperature = 0.8, .top_k = 40, .top_p = 0.9 };
    const s3: lanes.Sampling = .{ .seed = 202, .temperature = 1.0, .top_k = 20, .top_p = 0.9 };
    const want0 = try draws.draw(gpa, r0, scratch, s0, 7, &.{});
    const want1: u32 = 77;
    const want3 = try draws.draw(gpa, r3, scratch, s3, 11, &.{});
    const batch_draws = [4]?sample.Draw{
        .{ .how = sample.drawHow(s0, 7), .out = .{ .buf = gpu_be.scratch.picks, .off = 0 } },
        .{ .how = sample.drawHow(null, 9), .out = .{ .buf = gpu_be.scratch.picks, .off = 4 } },
        null,
        .{ .how = sample.drawHow(s3, 11), .out = .{ .buf = gpu_be.scratch.picks, .off = 12 } },
    };
    const ck = gpu_be.scratch.sample_ck.slice(u32, st.sample_slices * st.sample_kmax);
    const ci = gpu_be.scratch.sample_ci.slice(u32, st.sample_slices * st.sample_kmax);
    const picks = gpu_be.scratch.picks.slice(u32, 4);
    for (0..16) |_| {
        @memset(picks, 0);
        @memset(ck, 0xff00_0000);
        @memset(ci, 42);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = gpu_be.model.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        sample.drawRows(&gpu_be.model.sample, &gpu_be.model.glue, e, &gpu_be.scratch, .{ .buf = gpu_be.scratch.logits_all }, &batch_draws, vocab);
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        try std.testing.expectEqual(want0, picks[0]);
        try std.testing.expectEqual(want1, picks[1]);
        try std.testing.expectEqual(@as(u32, 0), picks[2]);
        try std.testing.expectEqual(want3, picks[3]);
    }
}

test "Inkling 2 Mini gpu_tokens=true matches gpu_tokens=false across boundaries, drafts, resume, concurrency, and ring 4096" {
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    dev.deinit();
    const m = try tree_gpu.buildSyntheticModel();
    defer m.deinit();
    const cpu_be = try backend_.Metal.init(gpa, m, .{ .capacity = 256, .streams = 4, .budget = 8 * st.cacheBytes(256), .gpu_tokens = false });
    defer cpu_be.deinit();
    try cpu_be.warm();
    const gpu_be = try backend_.Metal.init(gpa, m, .{ .capacity = 256, .streams = 4, .budget = 8 * st.cacheBytes(256), .gpu_tokens = true });
    defer gpu_be.deinit();
    try gpu_be.warm();

    try checkBoundaryDraws(cpu_be, gpu_be);
    try checkMixedMaskedVerify(cpu_be, gpu_be);
    try checkConcurrentDrawRows(gpu_be);
    try checkModesAndConcurrency(cpu_be, gpu_be);
    try checkAsyncRingBoundary(gpu_be);
}
