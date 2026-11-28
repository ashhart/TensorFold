//! Direct BF16 dispatcher parity, switch-off path checks, and lanes.Engine target-oracle tests for DFlash 2.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const sys = @import("sys");
const Model = @import("model.zig").Model;
const fwd = @import("forward.zig");
const dense_mma = @import("dense.zig");
const dflash2 = @import("dflash2.zig");
const bf16_fast = @import("dflash2_bf16.zig");
const w = @import("dflash2_weights.zig");
const q4 = @import("draft_head_q4.zig");
const batch_test = @import("dflash2_batch_test.zig");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn fillTiledBf16(rng: std.Random, slice: []u16, cols: usize, scale: f32) void {
    const rows = slice.len / cols;
    const tile = @min(rows, 64);
    for (slice[0 .. tile * cols]) |*v| v.* = batch_test.bf16((rng.float(f32) * 2.0 - 1.0) * scale);
    var r: usize = tile;
    while (r < rows) : (r += tile) {
        const take = @min(tile, rows - r);
        @memcpy(slice[r * cols ..][0 .. take * cols], slice[0 .. take * cols]);
        for (0..take) |j| {
            const row = r + j;
            slice[row * cols + (row * 13) % cols] = batch_test.bf16(@as(f32, @floatFromInt((row % 9) + 1)) * 0.04 * scale);
        }
    }
}

fn checkBf16Shape(device: mtl.Device, queue: mtl.Queue, model: *const Model, fast: *const bf16_fast.Pipelines, n: usize, k: usize, row_list: []const usize, prefer_16: bool) !void {
    var max_rows: usize = 0;
    for (row_list) |r| max_rows = @max(max_rows, r);
    var prng = std.Random.DefaultPrng.init(0xbf16_0001 +% n +% k);
    const rng = prng.random();
    const x_buf = try device.buffer(max_rows * k * 2, opts);
    defer x_buf.deinit();
    fillTiledBf16(rng, x_buf.slice(u16, max_rows * k), k, 0.35);
    const w_buf = try device.buffer(n * k * 2, opts);
    defer w_buf.deinit();
    fillTiledBf16(rng, w_buf.slice(u16, n * k), k, 0.2);
    const ref_buf = try device.buffer(max_rows * n * 2, opts);
    defer ref_buf.deinit();
    const got_buf = try device.buffer(max_rows * n * 2, opts);
    defer got_buf.deinit();
    const wt = batch_test.makeTensor(w_buf, 0, &.{ n, k });

    for (row_list) |rows| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        fwd.dense(model, e, wt, .{ .buf = x_buf }, .{ .buf = ref_buf }, rows);
        fast.dispatch(model, e, wt, .{ .buf = x_buf }, .{ .buf = got_buf }, rows, prefer_16);
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        const want = ref_buf.slice(u16, rows * n);
        const got = got_buf.slice(u16, rows * n);
        for (want, got) |wb, gb| {
            const wv = q4.asFloat(wb);
            const gv = q4.asFloat(gb);
            try std.testing.expect(std.math.isFinite(gv));
            try std.testing.expect(@abs(wv - gv) <= @max(@abs(wv) * 0.02, 0.05));
        }
    }
}

test "DFlash2 fast BF16 dispatcher matches fwd.dense at served shapes, rows 17/33/64/256 and vocab tails" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var model: Model = undefined;
    model.device = device;
    model.queue = queue;
    model.dense = try batch_test.loadDensePipelines(device);
    defer for (model.dense) |p| p.deinit();
    model.dense_mma = try dense_mma.Mma.load(device);
    defer if (model.dense_mma) |m| m.deinit();
    model.dense_tiled = try dense_mma.Tiled.load(device);
    defer if (model.dense_tiled) |t| t.deinit();
    var fast = try bf16_fast.Pipelines.load(device);
    defer fast.deinit();

    const served_rows = [_]usize{ 17, 33, 64, 256 };
    const projections = [_][2]usize{
        .{ 2304, 9216 },
        .{ 4096, 2304 },
        .{ 3072, 2304 },
        .{ 1024, 2304 },
        .{ 6912, 2304 },
        .{ 9216, 2304 },
        .{ 576, 2304 },
        .{ 256, 2304 },
    };
    for (projections) |nk| {
        try checkBf16Shape(device, queue, &model, &fast, nk[0], nk[1], &served_rows, false);
    }
    try checkBf16Shape(device, queue, &model, &fast, 3072, 2304, &.{ 17, 33 }, true);
    try checkBf16Shape(device, queue, &model, &fast, 65536 + 58, 2304, &.{ 17, 33 }, false);
    try checkBf16Shape(device, queue, &model, &fast, w.valid_vocab, w.hidden, &.{ 17, 33, 64 }, true);
}

test "DFlash2 switch-off keeps serial path with zero fast pipelines and concurrent Q4 matches pre-change serial rings" {
    const saved = sys.environment.rawText(q4.env_name);
    const saved_copy = if (saved) |val| try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}", .{val}, 0) else null;
    defer {
        if (saved_copy) |val| {
            _ = sys.env.setenv(q4.env_name, val.ptr, 1);
            std.testing.allocator.free(val);
        } else _ = sys.env.unsetenv(q4.env_name);
    }
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var rig = try batch_test.SyntheticRig.init(std.testing.allocator, device, queue);
    defer rig.deinit();

    // Verify unset and "0" leave fast pipelines unloaded and encoder serial even after enableDraftHeadQ4.
    for ([_]?[:0]const u8{ null, "0" }) |val| {
        if (val) |v| _ = sys.env.setenv(q4.env_name, v.ptr, 1) else _ = sys.env.unsetenv(q4.env_name);
        const df = try rig.makeDrafter(0, false, 1);
        defer df.deinit();
        try df.enableDraftHeadQ4(0, 1 << 40);
        try std.testing.expect(!dflash2.Drafter.testing.hasFast(df));
        try std.testing.expectEqual(mtl.DispatchType.serial, dflash2.Drafter.testing.dispatchMode(df));
        try std.testing.expect(df.loaded.draft_head_q4 == null);
    }

    // Compare concurrent Q4 (shared-sums overlap) vs pre-change serial Q4 (fast == null) rings for N=1, 7, 9.
    const gpa = std.testing.allocator;
    const row_bytes = w.taps_n * w.hidden * 2;
    const ring_bytes = w.ring_size * w.kv_width * 2;
    const tap_buf = try device.buffer(512 * row_bytes, opts);
    defer tap_buf.deinit();
    batch_test.fillTapRows(tap_buf, 0, 512, 0xdf2_5511);
    const dummy_follow: [64]u32 = @splat(3);
    const dummy_prompt: [16]u32 = @splat(11);
    const ring_a = try gpa.alloc(u8, ring_bytes);
    defer gpa.free(ring_a);
    const ring_b = try gpa.alloc(u8, ring_bytes);
    defer gpa.free(ring_b);

    for ([_]usize{ 1, 7, 9 }) |count| {
        const serial = try rig.makeDrafter(4, device.tensorUnits(), 1);
        defer serial.deinit();
        const fast_q4 = try rig.makeDrafter(4, device.tensorUnits(), dflash2.max_batch_streams);
        defer fast_q4.deinit();
        try dflash2.Drafter.testing.enableDraftHeadQ4Direct(fast_q4);
        try std.testing.expect(!dflash2.Drafter.testing.hasFast(serial));
        try std.testing.expectEqual(mtl.DispatchType.serial, dflash2.Drafter.testing.dispatchMode(serial));
        try std.testing.expect(dflash2.Drafter.testing.hasFast(fast_q4));
        try std.testing.expectEqual(mtl.DispatchType.concurrent, dflash2.Drafter.testing.dispatchMode(fast_q4));

        const s_ser = try gpa.alloc(*lanes.Stream, count);
        defer gpa.free(s_ser);
        const s_fst = try gpa.alloc(*lanes.Stream, count);
        defer gpa.free(s_fst);
        for (0..count) |i| {
            s_ser[i] = try gpa.create(lanes.Stream);
            s_ser[i].* = try lanes.Stream.init(gpa, .{ .id = "ser", .prompt = &dummy_prompt, .max_new = 8 });
            s_ser[i].prompt_len = 25;
            try serial.drafter().open(s_ser[i]);
            s_fst[i] = try gpa.create(lanes.Stream);
            s_fst[i].* = try lanes.Stream.init(gpa, .{ .id = "fst", .prompt = &dummy_prompt, .max_new = 8 });
            s_fst[i].prompt_len = 25;
            try fast_q4.drafter().open(s_fst[i]);
        }
        defer for (0..count) |i| {
            serial.drafter().release(s_ser[i]);
            s_ser[i].deinit(gpa);
            gpa.destroy(s_ser[i]);
            fast_q4.drafter().release(s_fst[i]);
            s_fst[i].deinit(gpa);
            gpa.destroy(s_fst[i]);
        };

        var batch_abs = try gpa.alloc(lanes.drafter.Absorb, count);
        defer gpa.free(batch_abs);
        for (0..count) |i| {
            const rows_i: u32 = @intCast(5 + (i % 7) * 3);
            const off_i = (i * 29) * row_bytes;
            try serial.drafter().absorb(&.{.{
                .stream = s_ser[i],
                .features = .{ .buffer = @intFromPtr(tap_buf.id), .offset = off_i, .rows = rows_i, .row_bytes = row_bytes, .space = .device, .dtype = .bf16 },
                .start = 0,
                .follow = dummy_follow[0..rows_i],
                .prompt = true,
            }});
            batch_abs[i] = .{
                .stream = s_fst[i],
                .features = .{ .buffer = @intFromPtr(tap_buf.id), .offset = off_i, .rows = rows_i, .row_bytes = row_bytes, .space = .device, .dtype = .bf16 },
                .start = 0,
                .follow = dummy_follow[0..rows_i],
                .prompt = true,
            };
        }
        try fast_q4.drafter().absorb(batch_abs);
        // Settle async absorb via a quick hold on both drafters before copying private rings.
        var hold_ser = try gpa.alloc(lanes.drafter.Hold, count);
        defer gpa.free(hold_ser);
        var hold_fst = try gpa.alloc(lanes.drafter.Hold, count);
        defer gpa.free(hold_fst);
        for (0..count) |i| {
            const rows_i: u64 = 5 + (i % 7) * 3;
            hold_ser[i] = .{ .stream = s_ser[i], .pending = @intCast(100 + i), .position = rows_i + 1, .depth = 7 };
            hold_fst[i] = .{ .stream = s_fst[i], .pending = @intCast(100 + i), .position = rows_i + 1, .depth = 7 };
            try serial.drafter().hold(hold_ser[i .. i + 1]);
        }
        try fast_q4.drafter().hold(hold_fst);
        for (0..count) |i| {
            for (0..w.layers_n) |layer| {
                const staging_a = try device.buffer(ring_bytes, opts);
                defer staging_a.deinit();
                const staging_b = try device.buffer(ring_bytes, opts);
                defer staging_b.deinit();
                for ([_]bool{ false, true }) |is_v| {
                    const src_a = dflash2.Drafter.testing.ring(serial, s_ser[i], layer, is_v).?;
                    const src_b = dflash2.Drafter.testing.ring(fast_q4, s_fst[i], layer, is_v).?;
                    const cb = queue.commandBuffer();
                    const b = cb.blit();
                    b.copy(src_a, 0, staging_a, 0, ring_bytes);
                    b.copy(src_b, 0, staging_b, 0, ring_bytes);
                    b.end();
                    cb.commit();
                    cb.wait();
                    try std.testing.expect(cb.failure() == null);
                    @memcpy(ring_a, staging_a.contents()[0..ring_bytes]);
                    @memcpy(ring_b, staging_b.contents()[0..ring_bytes]);
                    try std.testing.expectEqualSlices(u8, ring_a, ring_b);
                }
            }
        }
    }
}

const OracleTarget = struct {
    gpa: std.mem.Allocator,
    tap_buf: mtl.Buffer,
    oracle: []const u32,
    prompt_len: usize,
    lanes_map: std.AutoHashMapUnmanaged(*lanes.Stream, std.ArrayList(u32)) = .empty,
    drawn: std.ArrayList(u32) = .empty,
    verify_calls: usize = 0,
    keep_calls: usize = 0,

    fn deinit(self: *OracleTarget) void {
        var it = self.lanes_map.valueIterator();
        while (it.next()) |hist| hist.deinit(self.gpa);
        self.lanes_map.deinit(self.gpa);
        self.drawn.deinit(self.gpa);
    }

    fn facts(_: *const OracleTarget) lanes.Model {
        const costs = comptime blk: {
            var c: [16]lanes.config.Cost = undefined;
            for (&c, 1..) |*slot, width| slot.* = .{ .width = @intCast(width), .ms = 20.0 + 0.2 * @as(f64, @floatFromInt(width)) };
            break :blk c;
        };
        return .{ .exact_width = 16, .batch_rows = 64, .max_streams = 4, .streams_exact = true, .window_costs = &costs };
    }

    fn backend(self: *OracleTarget) lanes.backend.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .prepare_features = prepareFeaturesFn,
            .features = featuresFn,
            .release = releaseFn,
        } };
    }

    fn cast(ptr: *anyopaque) *OracleTarget {
        return @ptrCast(@alignCast(ptr));
    }

    fn targetToken(self: *const OracleTarget, history_len: usize) u32 {
        if (history_len >= self.prompt_len) {
            const idx = history_len - self.prompt_len;
            if (idx < self.oracle.len) return self.oracle[idx];
        }
        return @intCast(500 + (history_len * 17) % 10000);
    }

    fn prepareFeaturesFn(_: *anyopaque, _: []const u32) anyerror!void {}

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = cast(ptr);
        var hist: std.ArrayList(u32) = .empty;
        try hist.appendSlice(self.gpa, s.prompt());
        try self.lanes_map.put(self.gpa, s, hist);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, _: u64) anyerror!u64 {
        const self = cast(ptr);
        const hist = self.lanes_map.getPtr(s) orelse return error.UnknownStream;
        const tok = self.targetToken(hist.items.len);
        const h: u64 = self.drawn.items.len;
        try self.drawn.append(self.gpa, tok);
        return h;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, _: u64) anyerror!u64 {
        const self = cast(ptr);
        const hist = self.lanes_map.getPtr(s) orelse return error.UnknownStream;
        const tok_in = switch (feed) {
            .value => |v| v,
            .handle => |h| self.drawn.items[@intCast(h)],
        };
        try hist.append(self.gpa, tok_in);
        const tok = self.targetToken(hist.items.len);
        const h: u64 = self.drawn.items.len;
        try self.drawn.append(self.gpa, tok);
        return h;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return cast(ptr).drawn.items[@intCast(handle)];
    }

    fn verifyFn(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const self = cast(ptr);
        self.verify_calls += 1;
        for (windows, out) |win, *res| {
            const hist = self.lanes_map.getPtr(win.stream) orelse return error.UnknownStream;
            const base_len = hist.items.len;
            try hist.append(self.gpa, win.pending);
            res.sampled[0] = self.targetToken(base_len + 1);
            for (win.tokens, 0..) |dtok, i| {
                try hist.append(self.gpa, dtok);
                res.drafts[i] = dtok;
                res.sampled[i + 1] = self.targetToken(base_len + 2 + i);
            }
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
        const self = cast(ptr);
        self.keep_calls += 1;
        for (windows, paths) |win, path| {
            const hist = self.lanes_map.getPtr(win.stream) orelse return error.UnknownStream;
            const base_len = hist.items.len - win.rows();
            hist.shrinkRetainingCapacity(base_len + path.len);
        }
    }

    fn draftFn(_: *anyopaque, _: []const lanes.backend.DraftRequest) anyerror!void {}

    fn featuresFn(ptr: *anyopaque, _: *lanes.Stream, _: []const u32, start: u64, count: u32) anyerror!lanes.backend.Features {
        const self = cast(ptr);
        const row_bytes: u32 = @intCast(w.taps_n * w.hidden * 2);
        return .{
            .buffer = @intFromPtr(self.tap_buf.id),
            .offset = start * row_bytes,
            .rows = count,
            .row_bytes = row_bytes,
            .space = .device,
            .dtype = .bf16,
        };
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.lanes_map.fetchRemove(s)) |kv| {
            var hist = kv.value;
            hist.deinit(self.gpa);
        }
    }
};

test "DFlash2 target-oracle preserves final emitted tokens while proposing and accepting drafts" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = std.testing.allocator;
    var rig = try batch_test.SyntheticRig.init(gpa, device, queue);
    defer rig.deinit();

    const row_bytes = w.taps_n * w.hidden * 2;
    const tap_buf = try device.buffer(128 * row_bytes, opts);
    defer tap_buf.deinit();
    batch_test.fillTapRows(tap_buf, 0, 128, 0xdf2_0bac);
    const prompt = [_]u32{ 12, 34, 56, 78, 90, 123, 456, 789 };
    const max_new: usize = 16;

    // Test both switch OFF (Q4 head off, serial) and switch ON (Q4 head on, concurrent).
    for ([_]bool{ false, true }) |use_q4_head| {
        const df = try rig.makeDrafter(4, device.tensorUnits(), 4);
        defer df.deinit();
        if (use_q4_head) try dflash2.Drafter.testing.enableDraftHeadQ4Direct(df);

        // Probe first block proposed by this drafter so oracle matches 3 drafts then diverges on the 4th.
        var probe_s = try lanes.Stream.init(gpa, .{ .id = "probe", .prompt = &prompt, .max_new = max_new });
        defer probe_s.deinit(gpa);
        try df.drafter().open(&probe_s);
        const follow_p: [prompt.len]u32 = @splat(1);
        try df.drafter().absorb(&.{.{
            .stream = &probe_s,
            .features = .{ .buffer = @intFromPtr(tap_buf.id), .offset = 0, .rows = prompt.len, .row_bytes = row_bytes, .space = .device, .dtype = .bf16 },
            .start = 0,
            .follow = &follow_p,
            .prompt = true,
        }});
        try df.drafter().hold(&.{.{ .stream = &probe_s, .pending = 42, .position = prompt.len + 1, .depth = 7 }});
        var proposed: [7]u32 = undefined;
        try df.drafter().held(&.{&probe_s}, &.{&proposed});
        df.drafter().release(&probe_s);

        var oracle: [max_new]u32 = undefined;
        oracle[0] = 42;
        oracle[1] = proposed[0];
        oracle[2] = proposed[1];
        oracle[3] = proposed[2];
        oracle[4] = @intCast((proposed[3] + 1) % w.valid_vocab);
        for (5..max_new) |i| oracle[i] = @intCast(1000 + i * 31);

        // 1. Plain undrafted reference run.
        var plain_target: OracleTarget = .{ .gpa = gpa, .tap_buf = tap_buf, .oracle = &oracle, .prompt_len = prompt.len };
        defer plain_target.deinit();
        var plain_cfg = try lanes.Config.init(gpa, plain_target.facts(), 16, 15);
        defer plain_cfg.deinit(gpa);
        var clk0: lanes.fake.FixedClock = .{};
        var plain_eng = lanes.Engine.init(gpa, &plain_cfg, plain_target.backend(), clk0.clock());
        defer plain_eng.deinit();
        var s_plain = try lanes.Stream.init(gpa, .{ .id = "plain", .prompt = &prompt, .max_new = max_new, .drafts = false });
        defer s_plain.deinit(gpa);
        try plain_eng.addStream(&s_plain);
        while (plain_eng.activeCount() > 0) try plain_eng.step();
        try std.testing.expectEqualSlices(u32, &oracle, s_plain.emitted());

        // 2. Drafted multi-stream run through lanes.drafted.Drafted + lanes.Engine.
        var draft_target: OracleTarget = .{ .gpa = gpa, .tap_buf = tap_buf, .oracle = &oracle, .prompt_len = prompt.len };
        defer draft_target.deinit();
        var wrapped = try lanes.drafted.Drafted.init(gpa, draft_target.backend(), df.drafter());
        defer wrapped.deinit();
        var draft_cfg = try lanes.Config.init(gpa, wrapped.facts(draft_target.facts()), 16, 15);
        defer draft_cfg.deinit(gpa);
        var clk1: lanes.fake.FixedClock = .{};
        var draft_eng = lanes.Engine.init(gpa, &draft_cfg, wrapped.backend(), clk1.clock());
        defer draft_eng.deinit();
        var s0 = try lanes.Stream.init(gpa, .{ .id = "d0", .prompt = &prompt, .max_new = max_new, .drafts = true });
        defer s0.deinit(gpa);
        var s1 = try lanes.Stream.init(gpa, .{ .id = "d1", .prompt = &prompt, .max_new = max_new, .drafts = true });
        defer s1.deinit(gpa);
        try draft_eng.addStream(&s0);
        try draft_eng.addStream(&s1);
        while (draft_eng.activeCount() > 0) try draft_eng.step();
        try std.testing.expectEqualSlices(u32, s_plain.emitted(), s0.emitted());
        try std.testing.expectEqualSlices(u32, s_plain.emitted(), s1.emitted());
        try std.testing.expect(s0.drafted > 0 and s0.accepted >= 3);
        try std.testing.expect(s1.drafted > 0 and s1.accepted >= 3);
        try std.testing.expect(draft_target.verify_calls > 0 and draft_target.keep_calls > 0);
    }
}
