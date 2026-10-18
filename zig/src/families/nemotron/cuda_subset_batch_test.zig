//! Real restored lane chains through the production projection and draw launches, with CPU-recorded device operators.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Engine = @import("cuda_engine.zig").Engine;
const mtp = @import("cuda_mtp.zig");
const Head = mtp.Head;
const Backend = @import("cuda_lanes.zig").Cuda;
const reuse = @import("cuda_reuse.zig");
const common = @import("cuda_subset_restore_test.zig");
const Fake = common.Fake;
const Buffers = common.Buffers;
const t = std.testing;
const D = 64;
fn bf(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}
fn fp(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}
fn mem(comptime T: type, ptr: u64) [*]T {
    return @ptrFromInt(ptr);
}
const Frame = struct {
    var h: *Head = undefined;
    var buffers: [2]*Buffers = undefined;
    var full: [2]bool = undefined;
    var positions: [2]usize = undefined;
    var level: usize = 0;
    var live: [3]usize = @splat(0);
    var slots: [2]usize = undefined;
    var stride: usize = 0;
    var draws: usize = 0;
    var samples: usize = 0;
    var errors: usize = 0;
    var activations: [2]f32 = .{ 1, 2 };
    fn reset() void {
        level = 0;
        live = @splat(0);
        draws = 0;
        samples = 0;
        errors = 0;
        activations = .{ 1, 2 };
    }
    fn network(head: *Head, spans: []const mtp.Span, tail: bool) !void {
        try t.expect(tail and spans.len <= 2 and level < 3);
        h = head;
        live[level] = spans.len;
        level += 1;
        draws = 0;
        stride = 0;
        for (spans, 0..) |span, k| {
            const slot: usize = if (span.k_cache == @intFromPtr(&buffers[0].hk)) 0 else 1;
            slots[k] = slot;
            try t.expectEqual(@intFromPtr(&buffers[slot].hk), span.k_cache);
            try t.expectEqual(@as(u8, if (slot == 0) 0xa5 else 0xb6), mem(u8, span.k_cache)[0]);
            try t.expectEqual(@as(u32, @intCast(positions[slot] + level - 1)), mem(u32, span.meta)[0]);
            const input = mem(u16, head.hin)[(span.row0 + span.rows - 1) * D];
            try t.expectEqual(activations[slot], fp(input));
            activations[slot] = fp(input) + 1;
            @memset(mem(u16, head.row)[k * D ..][0..D], 0);
            mem(u16, head.row)[k * D] = bf(activations[slot]);
            const width: usize = if (full[slot]) 128 else 64;
            if (k == 0) stride = width else if (stride != width) stride = 128;
        }
        @memset(mem(u16, head.logits)[0..272], 0xffff);
    }
    fn arg(comptime T: type, args: [*]?*anyopaque, at: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args[at].?))).*;
    }
    fn packedIndex(row: usize, col: usize, k: usize) usize {
        const tg = (row / 64) * (k / 64) + col / 64;
        const jj = (row % 64) / 8;
        const kt = (col % 64) / 16;
        const lane = (row % 8) * 4 + (col % 8) / 2;
        const e = col % 2 + 2 * ((col % 16) / 8);
        return ((((tg * 8 + jj) * 4 + kt) * 32 + lane) * 4 + e);
    }
    fn kernel(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const args = params.?;
        if (@intFromPtr(f) == 1) {
            const x = mem(u16, arg(u64, args, 0));
            const w = mem(u16, arg(u64, args, 1));
            const out = mem(u16, arg(u64, args, 4));
            const rows: usize = @intCast(arg(i32, args, 6));
            const n: usize = @intCast(arg(i32, args, 7));
            const k: usize = @intCast(arg(i32, args, 8));
            const ld: usize = @intCast(arg(i32, args, 11));
            std.debug.assert(k == D and (n == 64 or n == 128));
            for (0..rows) |r| for (0..n) |c| {
                var sum: f32 = 0;
                for (0..k) |i| sum += fp(x[r * ld + i]) * fp(w[packedIndex(c, i, k)]);
                out[r * n + c] = bf(sum);
            };
            return 0;
        }
        const subset = @intFromPtr(f) == 3;
        std.debug.assert(subset or @intFromPtr(f) == 2);
        const ptr = arg(u64, args, 0);
        const n = arg(u32, args, 1);
        const slot = slots[draws];
        const width: usize = if (full[slot]) 128 else 64;
        errors += @intFromBool(n != width or ptr != h.logits + draws * stride * 2);
        const logits = mem(u16, ptr);
        for (0..width) |c| {
            const expected: f32 = if (c == 5) 3 * activations[slot] else if (c == 97) 10 * activations[slot] else 0;
            errors += @intFromBool(logits[c] != bf(expected));
        }
        const rule = mem(cuda.sample.Rule, arg(u64, args, 2))[0];
        std.debug.assert(rule.top_k == 1);
        errors += @intFromBool(mem(u32, arg(u64, args, 3))[0] != positions[slot] + 1);
        errors += @intFromBool(arg(i32, args, 4) != level - 1);
        var pick: u32 = 0;
        for (1..n) |c| if (fp(logits[c]) > fp(logits[pick])) {
            pick = @intCast(c);
        };
        const out = arg(u64, args, 5);
        errors += @intFromBool(out != @intFromPtr(&buffers[slot].ids) + level * 4);
        const ids = if (subset) mem(u32, arg(u64, args, 6)) else null;
        mem(u32, out)[0] = if (ids) |table| table[pick] else pick;
        const prob = arg(u64, args, if (subset) 7 else 6);
        mem(f32, prob)[0] = 1;
        const used = live[level - 1] * stride;
        for (0..live[level - 1]) |r| {
            const row_width: usize = if (full[slots[r]]) 128 else 64;
            for (mem(u16, h.logits)[r * stride + row_width ..][0 .. stride - row_width]) |v| errors += @intFromBool(v != 0xffff);
        }
        for (mem(u16, h.logits)[used..272]) |v| errors += @intFromBool(v != 0xffff);
        draws += 1;
        samples += 1;
        return 0;
    }
};
const Scratch = struct {
    pinned: [5000]u32 align(16) = @splat(0),
    meta: [33]u32 = @splat(0),
    metas: [66]u32 = @splat(0),
    hin: [2 * D]u16 = @splat(0),
    tok: [2]u32 = @splat(0),
    row: [2 * D]u16 = @splat(0),
    outs: [2 * D]u16 = @splat(0),
    out: [D]u16 = @splat(0),
    logits: [272]u16 = @splat(0xffff),
    xs: [2]f32 = @splat(0),
    fn head(s: *Scratch, e: *Engine, d: *cuda.Driver) Head {
        var h: Head = undefined;
        h.e = e;
        h.pos = 0;
        h.pinned = .{ .d = d, .bytes = std.mem.sliceAsBytes(&s.pinned) };
        h.copied = .{ .d = d, .handle = @ptrFromInt(1) };
        for (&h.ready) |*event| event.* = h.copied;
        h.absorb_graphs = @splat(null);
        h.first_graphs = @splat(@splat(null));
        h.chain_graphs = @splat(@splat(null));
        inline for (.{ "meta", "metas", "hin", "tok", "row", "outs", "out", "logits" }) |field| @field(h, field) = @intFromPtr(&@field(s, field));
        h.rxs = @intFromPtr(&s.xs);
        return h;
    }
};
fn request(s: *lanes.Stream, depth: u32) lanes.backend.DraftRequest {
    return .{ .stream = s, .follow = &.{}, .first = .{ .value = 1 }, .rows = null, .start = s.prompt_len, .position = s.prompt_len + 1, .depth = depth };
}
fn restore(e: *Engine, h: *Head, seq: *@import("cuda_state.zig").Seq, s: *lanes.Stream) !void {
    e.bind(seq);
    var target: reuse.Target = .{ .e = e, .head = h };
    try reuse.restore(&target, null, s.reuse.saved.?);
    try e.setSampling(.{ .seed = 90, .temperature = 1, .top_k = 1 });
}
test "restored head: 15 and 16 token transitions reach mixed shrinking real chains and projection draws" {
    var d = Fake.driver();
    d.api.cuLaunchKernel = Frame.kernel;
    const ctx: cuda.Context = .{ .d = &d, .handle = null, .device = 0 };
    cuda.copy_source.setCurrent(null);
    const e = try t.allocator.create(Engine);
    defer t.allocator.destroy(e);
    Fake.e = e;
    Fake.used = @splat(false);
    e.* = undefined;
    e.gpa = t.allocator;
    e.ctx = &ctx;
    e.stream = .{ .d = &d, .handle = null };
    defer e.stream.synchronize() catch unreachable;
    e.c = std.mem.zeroes(@import("config.zig").Config);
    e.c.hidden = D;
    e.c.kv_heads = 1;
    e.c.head_dim = 1;
    e.c.layers = 1;
    e.c.kinds[0] = .attention;
    e.c.vocab = 128;
    e.max_len = 64;
    e.k.d = &d;
    e.k.ck = null;
    e.k.ggml = null;
    const f: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(1) };
    const lane: cuda.qmmf.Lane = .{ .mod = undefined, .fns = @splat(@splat(f)), .f32_fns = @splat(@splat(f)), .reduce = @splat(f), .clusters = false };
    e.k.modelopt = .{ .lane = lane, .gemm = undefined, .experts = undefined, .pack = undefined };
    e.k.draws.draw_fn = .{ .d = &d, .handle = @ptrFromInt(2) };
    e.k.draws.draw_ids_fn = .{ .d = &d, .handle = @ptrFromInt(3) };
    e.b = undefined;
    e.b.lane_part = 0;
    e.w.draft_count = 64;
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i);
    e.w.draft_coverage = try @import("draft_ids.zig").Coverage.init(t.allocator, 128, &ids);
    defer e.w.draft_coverage.?.deinit(t.allocator);
    e.w.draft_ids = @intFromPtr(&ids);
    var raw: [128 * D]u16 = @splat(0);
    raw[5 * D] = bf(3);
    raw[97 * D] = bf(10);
    var full: [128 * D]u16 = undefined;
    var subset: [128 * D]u16 = undefined;
    cuda.qmmf.packBf16(&full, &raw, 128, D);
    cuda.qmmf.packBf16(&subset, raw[0 .. 64 * D], 64, D);
    e.w.head = .{ .bf16 = .{ .mode = .bf16, .codes = @intFromPtr(&full), .scales = 0, .n = 128, .k = D, .npad = 128 } };
    e.w.draft_head = .{ .bf16 = .{ .mode = .bf16, .codes = @intFromPtr(&subset), .scales = 0, .n = 64, .k = D, .npad = 128 } };
    e.pos = 0;
    e.parity = 0;
    e.prev_keep = 0;
    e.rows = 0;
    e.sampling = null;
    e.slot = 0;
    e.decide = -1;
    e.epoch = 0;
    e.stats = .{};
    var own: Buffers = .{};
    var other: Buffers = .{};
    own.hidden[0] = bf(1);
    other.hidden[0] = bf(2);
    Frame.buffers = .{ &own, &other };
    e.own = own.seq();
    e.bound = &e.own;
    var neighbor = other.seq();
    var scratch: Scratch = .{};
    var h = scratch.head(e, &d);
    e.head = &h;
    e.bind(&e.own);
    Head.test_network = Frame.network;
    defer Head.test_network = null;
    var b: Backend = undefined;
    b.gpa = t.allocator;
    b.e = e;
    b.head = &h;
    b.lanes = .empty;
    defer b.lanes.deinit(t.allocator);
    b.pinned = h.pinned;
    const covered = @as([16]u32, @splat(1));
    const uncovered = @as([15]u32, @splat(1)) ++ [_]u32{97};
    const cases = .{
        .{ covered[0..15], true, &covered, false },
        .{ &covered, false, &uncovered, true },
        .{ &uncovered, true, &covered, false },
        .{ &covered, false, &uncovered, true },
        .{ &covered, false, &covered, false },
        .{ covered[0..15], true, &uncovered, true },
    };
    var saved_own: [160]u8 align(16) = @splat(0xa5);
    var saved_other: [160]u8 align(16) = @splat(0xb6);
    inline for (cases) |case| {
        var copy: reuse.Copy = .{ .mamba = .{ .d = &d, .ptr = 0, .len = 0 }, .kv = .{ .d = &d, .ptr = @intFromPtr(&saved_own), .len = saved_own.len }, .head = true, .pos = case[0].len, .parity = 1, .rows = 1 };
        var other_copy = copy;
        other_copy.pos = case[2].len;
        other_copy.kv.ptr = @intFromPtr(&saved_other);
        Frame.full = .{ case[1], case[3] };
        Frame.positions = .{ case[0].len, case[2].len };
        var s = try lanes.Stream.init(t.allocator, .{ .id = "restored", .prompt = case[0], .max_new = 8, .reuse = .{ .saved = &copy, .at = @intCast(copy.pos) } });
        defer s.deinit(t.allocator);
        var n = try lanes.Stream.init(t.allocator, .{ .id = "neighbor", .prompt = case[2], .max_new = 8, .reuse = .{ .saved = &other_copy, .at = @intCast(other_copy.pos) } });
        defer n.deinit(t.allocator);
        try b.lanes.put(t.allocator, &s, .{ .seq = &e.own });
        defer _ = b.lanes.remove(&s);
        try b.lanes.put(t.allocator, &n, .{ .seq = &neighbor });
        defer _ = b.lanes.remove(&n);
        try restore(e, &h, &e.own, &s);
        Frame.reset();
        try b.backend().draft(&.{request(&s, 3)});
        var solo: [3]u32 = undefined;
        try b.backend().vtable.held.?(b.backend().ptr, &s, &solo);
        try t.expectEqualSlices(u32, &@as([3]u32, @splat(if (case[1]) 97 else 5)), &solo);
        try t.expectEqual(@as(usize, 0), Frame.errors);
        try restore(e, &h, &neighbor, &n);
        Frame.reset();
        try b.backend().draft(&.{request(&n, 2)});
        var neighbor_solo: [2]u32 = undefined;
        try b.backend().vtable.held.?(b.backend().ptr, &n, &neighbor_solo);
        try t.expectEqual(@as(usize, 0), Frame.errors);
        try restore(e, &h, &e.own, &s);
        try restore(e, &h, &neighbor, &n);
        @memset(&own.ids, 0xf00dbaad);
        @memset(&other.ids, 0xf00dbaad);
        Frame.reset();
        try b.backend().draft(&.{ request(&n, 2), request(&s, 3) });
        var batch: [3]u32 = undefined;
        var neighbor_batch: [2]u32 = undefined;
        try b.backend().vtable.held.?(b.backend().ptr, &s, &batch);
        try b.backend().vtable.held.?(b.backend().ptr, &n, &neighbor_batch);
        try t.expectEqualSlices(u32, &solo, &batch);
        try t.expectEqualSlices(u32, &neighbor_solo, &neighbor_batch);
        try t.expectEqualSlices(usize, &.{ 2, 2, 1 }, &Frame.live);
        try t.expectEqual(@as(usize, 5), Frame.samples);
        try t.expectEqual(@as(usize, 0), Frame.errors);
        try t.expectEqual(@as(u32, 0xf00dbaad), own.ids[0]);
        for (own.ids[4..]) |v| try t.expectEqual(@as(u32, 0xf00dbaad), v);
        for (other.ids[3..]) |v| try t.expectEqual(@as(u32, 0xf00dbaad), v);
        try e.stream.synchronize();
    }
    for (Fake.used) |live| try t.expect(!live);
}
