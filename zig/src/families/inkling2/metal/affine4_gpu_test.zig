//! GPU tests for Inkling 2 Mini MLX 4-bit affine g64: CPU serial reference and 6-mode bit-equality.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const src = @import("inkling2_metal");
const model_ = @import("model.zig");
const model_affine4 = @import("model_affine4.zig");
const affine4 = @import("affine4.zig");
const serial_ref = @import("affine4_serial_ref.zig");
const synthetic = @import("affine4_synthetic_test.zig");
const st = @import("state.zig");
const backend_ = @import("backend.zig");

const gpa = std.testing.allocator;
const taps_ids = [_]u32{ 5, 23, 41, 48 };

fn featureRowBytes(b: lanes.backend.Backend, s: *lanes.Stream, start: u64) ![st.feature_row_bytes]u8 {
    const f = try b.features(s, &taps_ids, start, 1);
    const buf: mtl.Buffer = .{ .id = @ptrFromInt(f.buffer) };
    const bytes = buf.slice(u8, @intCast(f.offset + f.row_bytes));
    var out: [st.feature_row_bytes]u8 = undefined;
    @memcpy(&out, bytes[@intCast(f.offset)..][0..st.feature_row_bytes]);
    return out;
}

/// Captures each stream's final recurrent state (logits, KV, conv, taps) right before `lanes.Engine` releases it.
const CaptureOnRelease = struct {
    target: *backend_.Metal,
    inner: lanes.backend.Backend,
    saved: std.AutoHashMapUnmanaged(*lanes.Stream, serial_ref.StateBytes) = .empty,
    failed: ?anyerror = null,

    fn deinit(self: *CaptureOnRelease) void {
        var it = self.saved.valueIterator();
        while (it.next()) |sb| sb.deinit(gpa);
        self.saved.deinit(gpa);
    }

    fn backend(self: *CaptureOnRelease) lanes.backend.Backend {
        return .{
            .keep_full_lone = self.inner.keep_full_lone,
            .ptr = self,
            .vtable = &.{
                .prefill = prefill,
                .prepare_features = prepareFeatures,
                .features = features,
                .prefill_features = prefillFeatures,
                .first = first,
                .first_masked = firstMasked,
                .queue = queue,
                .read = read,
                .verify = verify,
                .keep = keep,
                .draft = draft,
                .release = release,
                .first_row = firstRow,
                .held = held,
            },
        };
    }

    fn cast(ptr: *anyopaque) *CaptureOnRelease {
        return @ptrCast(@alignCast(ptr));
    }
    fn prefill(p: *anyopaque, s: *lanes.Stream) !void {
        return cast(p).inner.prefill(s);
    }
    fn prepareFeatures(p: *anyopaque, ts: []const u32) !void {
        return cast(p).inner.vtable.prepare_features.?(cast(p).inner.ptr, ts);
    }
    fn features(p: *anyopaque, s: *lanes.Stream, ts: []const u32, start: u64, n: u32) !lanes.backend.Features {
        return cast(p).inner.features(s, ts, start, n);
    }
    fn prefillFeatures(p: *anyopaque, s: *lanes.Stream, ts: []const u32, sink: lanes.backend.FeatureSink) !void {
        return cast(p).inner.vtable.prefill_features.?(cast(p).inner.ptr, s, ts, sink);
    }
    fn first(p: *anyopaque, s: *lanes.Stream, pos: u64) !u64 {
        return cast(p).inner.first(s, pos);
    }
    fn firstMasked(p: *anyopaque, s: *lanes.Stream, pos: u64, mask: []const u32) !u64 {
        return cast(p).inner.vtable.first_masked.?(cast(p).inner.ptr, s, pos, mask);
    }
    fn queue(p: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, pos: u64) !u64 {
        return cast(p).inner.queue(s, feed, pos);
    }
    fn read(p: *anyopaque, h: u64) !u32 {
        return cast(p).inner.read(h);
    }
    fn verify(p: *anyopaque, w: []const lanes.backend.Window, o: []lanes.backend.Verified) !void {
        return cast(p).inner.verify(w, o);
    }
    fn keep(p: *anyopaque, w: []const lanes.backend.Window, ps: []const []const u32) !void {
        return cast(p).inner.keep(w, ps);
    }
    fn draft(p: *anyopaque, r: []const lanes.backend.DraftRequest) !void {
        return cast(p).inner.vtable.draft(cast(p).inner.ptr, r);
    }
    fn firstRow(p: *anyopaque, s: *lanes.Stream) !lanes.LogprobRow {
        return cast(p).inner.firstRow(s);
    }
    fn held(p: *anyopaque, s: *lanes.Stream, out: []u32) !void {
        return cast(p).inner.vtable.held.?(cast(p).inner.ptr, s, out);
    }
    fn release(p: *anyopaque, s: *lanes.Stream) void {
        const self = cast(p);
        if (self.target.caches.get(s)) |cache| {
            var tap_bytes: [st.feature_row_bytes]u8 = @splat(0);
            if (cache.len > 0) {
                if (featureRowBytes(self.target.backend(), s, cache.len - 1)) |tb| {
                    tap_bytes = tb;
                } else |_| {}
            }
            if (serial_ref.captureState(gpa, self.target.model.device, self.target.model.queue, cache, self.target.model.config.text.unpadded_vocab, &tap_bytes)) |sb| {
                self.saved.put(gpa, s, sb) catch |err| {
                    sb.deinit(gpa);
                    self.failed = err;
                };
            } else |err| self.failed = err;
        }
        self.inner.release(s);
    }
};

const TreeDrafter = struct {
    oracle: []const u32,
    prompt_len: usize,
    held_toks: std.AutoHashMapUnmanaged(*lanes.Stream, [5]u32) = .empty,
    held_len: std.AutoHashMapUnmanaged(*lanes.Stream, usize) = .empty,
    saw_non_prefix_rows: bool = false,

    fn drafter(x: *TreeDrafter) lanes.drafter.Drafter {
        return .{ .ptr = x, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release, .tree = treeFn } };
    }
    fn self(ptr: *anyopaque) *TreeDrafter {
        return @ptrCast(@alignCast(ptr));
    }
    fn taps(_: *anyopaque) []const u32 {
        return &taps_ids;
    }
    fn facts(_: *anyopaque) lanes.drafter.Facts {
        return .{ .depth = 5, .step_ms = 0.01, .tree_nodes = 5, .prior = &.{ 0.9, 0.9, 0.9, 0.9, 0.9 } };
    }
    fn open(ptr: *anyopaque, s: *lanes.Stream) !void {
        const x = self(ptr);
        try x.held_toks.put(gpa, s, .{ 1, 2, 3, 4, 5 });
        try x.held_len.put(gpa, s, 0);
    }
    fn absorb(ptr: *anyopaque, items: []const lanes.drafter.Absorb) !void {
        const x = self(ptr);
        for (items) |it| if (it.rows) |rows| for (rows, 0..) |rv, idx| if (rv != idx) {
            x.saw_non_prefix_rows = true;
        };
    }
    fn hold(ptr: *anyopaque, items: []const lanes.drafter.Hold) !void {
        const x = self(ptr);
        for (items) |it| {
            const p = it.pending;
            const step = it.position -| x.prompt_len;
            const g1 = if (step < x.oracle.len) x.oracle[step] else (p + 1) % 100;
            const g2 = if (step + 1 < x.oracle.len) x.oracle[step + 1] else (p + 2) % 100;
            x.held_toks.getPtr(it.stream).?.* = .{ (g1 + 7) % 100, g1, (g1 + 11) % 100, g2, (g2 + 13) % 100 };
            x.held_len.getPtr(it.stream).?.* = @min(@as(usize, it.depth), 5);
        }
    }
    fn held(ptr: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) !void {
        const x = self(ptr);
        for (streams, out) |s, dst| @memcpy(dst, x.held_toks.get(s).?[0..dst.len]);
    }
    fn treeFn(ptr: *anyopaque, s: *lanes.Stream, alloc: std.mem.Allocator) !?lanes.stream.Held {
        const x = self(ptr);
        if ((x.held_len.get(s) orelse 0) < 5) return null;
        const toks = try alloc.dupe(u32, x.held_toks.get(s).?[0..5]);
        errdefer alloc.free(toks);
        return .{ .count = 5, .tokens = toks, .parents = try alloc.dupe(i32, &.{ -1, -1, 0, 1, 3 }) };
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const x = self(ptr);
        _ = x.held_toks.remove(s);
        _ = x.held_len.remove(s);
    }
    fn deinit(x: *TreeDrafter) void {
        x.held_toks.deinit(gpa);
        x.held_len.deinit(gpa);
    }
};

fn verifySerialReference(m: *model_.Model, prompt: []const u32, decode_steps: u32, check_planted_fault: bool) !void {
    const vocab = m.config.text.unpadded_vocab;
    const cap = prompt.len + decode_steps + 16;
    const target = try backend_.Metal.init(gpa, m, .{ .capacity = cap, .streams = 1, .budget = 2 * st.cacheBytes(cap) });
    defer target.deinit();
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &taps_ids);

    // Arm A: Production GPU batched prompt prefill + decode.
    var gpu_s = try lanes.Stream.init(gpa, .{ .id = "gpu-affine4", .prompt = prompt, .max_new = decode_steps + 1 });
    defer gpu_s.deinit(gpa);
    try b.prefill(&gpu_s);
    defer b.release(&gpu_s);
    const gpu_tok0 = try b.read(try b.first(&gpu_s, prompt.len));
    const gpu_prompt_tap = try featureRowBytes(b, &gpu_s, prompt.len - 1);
    const gpu_cache = target.caches.get(&gpu_s).?;
    const gpu_prompt_state = try serial_ref.captureState(gpa, m.device, m.queue, gpu_cache, vocab, &gpu_prompt_tap);
    defer gpu_prompt_state.deinit(gpa);

    // Arm B: Named serial single-row CPU-affine4 reference (`cpu_affine4_serial_reference`).
    var ref_scratch = try st.Scratch.init(m.device, 1);
    defer ref_scratch.deinit();
    const ref_tap_buf = try m.device.buffer(st.feature_row_bytes, mtl.ResourceOptions.shared);
    defer ref_tap_buf.deinit();
    var ref_cache = try st.Cache.init(m.device, cap, &m.weight_plan);
    defer ref_cache.deinit();
    for (prompt, 0..) |tok, i| {
        try serial_ref.stepToken(m, &ref_scratch, &ref_cache, tok, i + 1 == prompt.len, .{ .buf = ref_tap_buf });
    }
    const ref_tok0 = serial_ref.argmax(ref_cache.logits.slice(f32, vocab));
    const ref_prompt_state = try serial_ref.captureState(gpa, m.device, m.queue, &ref_cache, vocab, ref_tap_buf.slice(u8, st.feature_row_bytes));
    defer ref_prompt_state.deinit(gpa);

    try std.testing.expectEqual(ref_tok0, gpu_tok0);
    try serial_ref.expectStateBitEqual(ref_prompt_state, gpu_prompt_state);

    var tok = ref_tok0;
    for (0..decode_steps) |step| {
        const gpu_next = try b.read(try b.queue(&gpu_s, .{ .value = tok }, prompt.len + 1 + step));
        try serial_ref.stepToken(m, &ref_scratch, &ref_cache, tok, true, .{ .buf = ref_tap_buf });
        const ref_next = serial_ref.argmax(ref_cache.logits.slice(f32, vocab));
        try std.testing.expectEqual(ref_next, gpu_next);
        const gpu_tap = try featureRowBytes(b, &gpu_s, prompt.len + step);
        const gpu_dec = try serial_ref.captureState(gpa, m.device, m.queue, gpu_cache, vocab, &gpu_tap);
        defer gpu_dec.deinit(gpa);
        const ref_dec = try serial_ref.captureState(gpa, m.device, m.queue, &ref_cache, vocab, ref_tap_buf.slice(u8, st.feature_row_bytes));
        defer ref_dec.deinit(gpa);
        try serial_ref.expectStateBitEqual(ref_dec, gpu_dec);
        tok = ref_next;
    }

    // Planted control: corrupt affine4.metal nibble order and verify GPU diverges from cpu_affine4_serial_reference.
    if (check_planted_fault) {
        const from = "const float dw0 = float((word >> sh) & 0xfu) * sc + bi;\n    const float dw1 = float((word >> (sh + 4u)) & 0xfu) * sc + bi;";
        const to = "const float dw0 = float((word >> (sh + 4u)) & 0xfu) * sc + bi;\n    const float dw1 = float((word >> sh) & 0xfu) * sc + bi;";
        const bad_src = try std.mem.replaceOwned(u8, gpa, src.affine4, from, to);
        defer gpa.free(bad_src);
        const bad_pipes = try affine4.Pipelines.loadSource(m.device, bad_src);
        defer bad_pipes.deinit();
        const saved_pipes = m.affine4;
        m.affine4 = bad_pipes;
        defer m.affine4 = saved_pipes;
        var bad_s = try lanes.Stream.init(gpa, .{ .id = "bad-q4", .prompt = prompt, .max_new = 2 });
        defer bad_s.deinit(gpa);
        try b.prefill(&bad_s);
        defer b.release(&bad_s);
        const bad_cache = target.caches.get(&bad_s).?;
        try std.testing.expect(!std.mem.eql(u8, ref_prompt_state.logits, bad_cache.logits.slice(u8, vocab * 4)));
    }
}

fn verifySixModes(m: *model_.Model) !void {
    const prompts = [_][]const u32{
        &.{ 11, 22, 33, 11, 22, 44, 11, 22, 55, 11, 22, 33, 11, 22, 44, 11 },
        &.{ 7, 19, 31, 43, 59, 71, 83, 97, 101, 109, 127, 131 },
        &.{ 13, 29, 47, 61, 73, 89, 103, 113, 137, 149, 157, 163, 173, 181 },
        &.{ 5, 17, 37, 53, 67, 79, 107, 139, 151, 167 },
    };
    const max_new: u32 = 8;
    const cap: usize = 64;
    const target = try backend_.Metal.init(gpa, m, .{ .capacity = cap, .streams = 4, .budget = 8 * st.cacheBytes(cap) });
    defer target.deinit();
    try target.warm();
    try target.backend().vtable.prepare_features.?(target.backend().ptr, &taps_ids);

    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 1.0 + 0.01 * @as(f64, @floatFromInt(w)) };

    // 1. Plain + Fresh + Solo (c=1) for all 4 prompts (stream 0 is primary, streams 1..3 are concurrent baselines).
    var solo_tokens: [4][]u32 = undefined;
    var solo_states: [4]serial_ref.StateBytes = undefined;
    var solo_made: usize = 0;
    defer for (0..solo_made) |i| {
        gpa.free(solo_tokens[i]);
        solo_states[i].deinit(gpa);
    };
    for (prompts, 0..) |p, idx| {
        var cap_wrap: CaptureOnRelease = .{ .target = target, .inner = target.backend() };
        defer cap_wrap.deinit();
        var plain_cfg = try lanes.Config.init(gpa, target.facts(), 16, 16);
        defer plain_cfg.deinit(gpa);
        var clk: lanes.fake.FixedClock = .{};
        var eng = lanes.Engine.init(gpa, &plain_cfg, cap_wrap.backend(), clk.clock());
        defer eng.deinit();
        var s = try lanes.Stream.init(gpa, .{ .id = "solo", .prompt = p, .max_new = max_new });
        defer s.deinit(gpa);
        try eng.addStream(&s);
        while (eng.activeCount() > 0) try eng.step();
        if (cap_wrap.failed) |err| return err;
        solo_tokens[idx] = try gpa.dupe(u32, s.emitted());
        solo_states[idx] = cap_wrap.saved.fetchRemove(&s).?.value;
        solo_made += 1;
    }

    // Save snapshot at prefix 8 of prompt 0 for Resumed (plain and drafted) arms.
    var snap_s = try lanes.Stream.init(gpa, .{ .id = "snap", .prompt = prompts[0][0..8], .max_new = max_new });
    defer snap_s.deinit(gpa);
    try target.backend().prefill(&snap_s);
    defer target.backend().release(&snap_s);
    const snap = try backend_.Metal.snapSave(target, &snap_s, 8);
    defer backend_.Metal.snapDrop(target, snap);

    // 2. Resumed + Plain + Solo (c=1).
    {
        var cap_wrap: CaptureOnRelease = .{ .target = target, .inner = target.backend() };
        defer cap_wrap.deinit();
        var plain_cfg = try lanes.Config.init(gpa, target.facts(), 16, 16);
        defer plain_cfg.deinit(gpa);
        var clk: lanes.fake.FixedClock = .{};
        var eng = lanes.Engine.init(gpa, &plain_cfg, cap_wrap.backend(), clk.clock());
        defer eng.deinit();
        var rs = try lanes.Stream.init(gpa, .{ .id = "resumed-plain", .prompt = prompts[0], .max_new = max_new, .reuse = .{ .saved = snap, .at = 8 } });
        defer rs.deinit(gpa);
        try eng.addStream(&rs);
        while (eng.activeCount() > 0) try eng.step();
        if (cap_wrap.failed) |err| return err;
        try std.testing.expectEqual(@as(u32, 8), rs.cached);
        try std.testing.expectEqualSlices(u32, solo_tokens[0], rs.emitted());
        try serial_ref.expectStateBitEqual(solo_states[0], cap_wrap.saved.get(&rs).?);
    }

    // 3 & 4. Drafted (tree-verify) Fresh + Drafted Resumed (snapshot).
    {
        var cap_wrap: CaptureOnRelease = .{ .target = target, .inner = target.backend() };
        defer cap_wrap.deinit();
        var drafter: TreeDrafter = .{ .oracle = solo_tokens[0], .prompt_len = prompts[0].len };
        defer drafter.deinit();
        var wrapped = try lanes.drafted.Drafted.init(gpa, cap_wrap.backend(), drafter.drafter());
        defer wrapped.deinit();
        var tree_model = wrapped.facts(target.facts());
        tree_model.window_costs = &costs;
        var tree_cfg = try lanes.Config.init(gpa, tree_model, 16, 16);
        defer tree_cfg.deinit(gpa);
        tree_cfg.fill_lanes = 8;
        var clk: lanes.fake.FixedClock = .{};
        var eng = lanes.Engine.init(gpa, &tree_cfg, wrapped.backend(), clk.clock());
        defer eng.deinit();

        var fresh_d = try lanes.Stream.init(gpa, .{ .id = "draft-fresh", .prompt = prompts[0], .max_new = max_new, .drafts = true });
        defer fresh_d.deinit(gpa);
        try eng.addStream(&fresh_d);
        while (eng.activeCount() > 0) try eng.step();
        if (cap_wrap.failed) |err| return err;
        try std.testing.expect(fresh_d.drafted > 0 and fresh_d.accepted > 0 and drafter.saw_non_prefix_rows);
        try std.testing.expectEqualSlices(u32, solo_tokens[0], fresh_d.emitted());
        try serial_ref.expectStateBitEqual(solo_states[0], cap_wrap.saved.get(&fresh_d).?);

        var resumed_d = try lanes.Stream.init(gpa, .{ .id = "draft-resumed", .prompt = prompts[0], .max_new = max_new, .drafts = true, .reuse = .{ .saved = snap, .at = 8 } });
        defer resumed_d.deinit(gpa);
        try eng.addStream(&resumed_d);
        while (eng.activeCount() > 0) try eng.step();
        if (cap_wrap.failed) |err| return err;
        try std.testing.expectEqual(@as(u32, 8), resumed_d.cached);
        try std.testing.expect(resumed_d.drafted > 0);
        try std.testing.expectEqualSlices(u32, solo_tokens[0], resumed_d.emitted());
        try serial_ref.expectStateBitEqual(solo_states[0], cap_wrap.saved.get(&resumed_d).?);
    }

    // 5. Concurrent (c=4): all 4 streams run simultaneously and must match their c=1 solo baselines bit-for-bit.
    {
        var cap_wrap: CaptureOnRelease = .{ .target = target, .inner = target.backend() };
        defer cap_wrap.deinit();
        var conc_cfg = try lanes.Config.init(gpa, target.facts(), 16, 16);
        defer conc_cfg.deinit(gpa);
        var clk: lanes.fake.FixedClock = .{};
        var eng = lanes.Engine.init(gpa, &conc_cfg, cap_wrap.backend(), clk.clock());
        defer eng.deinit();
        var streams: [4]lanes.Stream = undefined;
        var made: usize = 0;
        defer for (0..made) |i| streams[i].deinit(gpa);
        for (prompts, 0..) |p, i| {
            streams[i] = try lanes.Stream.init(gpa, .{ .id = "conc", .prompt = p, .max_new = max_new });
            made += 1;
            try eng.addStream(&streams[i]);
        }
        while (eng.activeCount() > 0) try eng.step();
        if (cap_wrap.failed) |err| return err;
        for (0..4) |i| {
            try std.testing.expectEqualSlices(u32, solo_tokens[i], streams[i].emitted());
            try serial_ref.expectStateBitEqual(solo_states[i], cap_wrap.saved.get(&streams[i]).?);
        }
    }
}

test "quantized-pack Inkling 2 Mini matches cpu_affine4_serial_reference and is bit-equal across all 6 modes" {
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    dev.deinit();
    const m = try synthetic.build(gpa);
    defer m.deinit();
    try verifySerialReference(m, &.{ 11, 22, 33, 44, 55, 66 }, 3, true);
    try verifySixModes(m);
}

test "real Inkling-2-Mini-4bit checkpoint pinned digests, serial reference, and 6-mode bit-equality (TF_INKLING2_4BIT_DIR)" {
    const dir = @import("sys").getenv("TF_INKLING2_4BIT_DIR") orelse return error.SkipZigTest;
    const io = std.testing.io;
    try model_affine4.verifyPinnedDigests(gpa, io, dir);
    const m = try model_.Model.load(gpa, io, dir);
    defer m.deinit();
    try verifySerialReference(m, &.{ 1000, 1037, 1074, 1111 }, 2, false);
    try verifySixModes(m);
}
