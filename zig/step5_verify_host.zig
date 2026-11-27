//! Real Step-5 geometry and verify code with Metal objects left inert or recorded by the host fixture.
const std = @import("std");
pub const safetensors = @import("src/core/safetensors.zig");
pub const checkpoint_file_cache = @import("src/core/checkpoint_file_cache.zig");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("src/families/step5/config.zig");
const st = @import("src/families/step5/state.zig");
const wts = @import("src/families/step5/weights.zig");
const model = @import("src/families/step5/model.zig");
const backend = @import("src/families/step5/backend.zig");
const Model = model.Model;
const be = lanes.backend;
const a = std.testing.allocator;

test "step5 host: imports" {
    _ = @import("src/families/step5/config.zig");
    _ = @import("src/families/step5/weights.zig");
    _ = @import("src/families/step5/vision.zig");
}

fn geometry() cfg.Config {
    var c: cfg.Config = .{
        .hidden = 64,
        .layers = 2,
        .vocab = 64,
        .heads = 8,
        .kv_heads = 1,
        .head_dim = 32,
        .intermediate = 64,
        .moe_intermediate = 64,
        .shared_intermediate = 64,
        .experts = 4,
        .topk = 1,
        .window = 8,
        .index_heads = 4,
        .proxy = 32,
        .index_topk = 1,
        .region = 4,
    };
    for (c.layer[0..2], 0..) |*l, i| l.* = .{ .kind = if (i == 0) .sliding else .full, .span = 32, .theta = 10000, .limit = 7, .shared_limit = 7, .moe = false };
    return c;
}

fn linear(buffer: mtl.Buffer, n: usize, k: usize) wts.Linear {
    return .{ .w = buffer, .scales = buffer, .biases = buffer, .n = n, .k = k };
}

fn tensor(buffer: mtl.Buffer) wts.Tensor {
    return .{ .buffer = buffer, .offset = 0, .bytes = 4, .dtype = .f32 };
}

fn mlp(buffer: mtl.Buffer, hidden: usize, intermediate: usize) wts.Mlp {
    return .{ .gate = linear(buffer, intermediate, hidden), .up = linear(buffer, intermediate, hidden), .down = linear(buffer, hidden, intermediate) };
}

fn glue(g: model.Glue) usize {
    return @backingInt(g) + 1;
}

fn collect(r: *const mtl.Record, pipe: usize, out: []*const mtl.Pass) ![]*const mtl.Pass {
    var n: usize = 0;
    for (r.passes[0..r.pass_count]) |*p| if (p.pipe == pipe) {
        if (n == out.len) return error.TooManyRecordedPasses;
        out[n] = p;
        n += 1;
    };
    return out[0..n];
}

fn word(p: *const mtl.Pass, slot: usize, index: usize) !u32 {
    try std.testing.expect((index + 1) * 4 <= p.lengths[slot]);
    return std.mem.readInt(u32, p.values[slot][index * 4 ..][0..4], .little);
}

fn expectWord(p: *const mtl.Pass, slot: usize, index: usize, expected: u32) !void {
    try std.testing.expectEqual(expected, try word(p, slot, index));
}

fn expectBuffer(p: *const mtl.Pass, slot: usize, buffer: mtl.Buffer, off: usize) !void {
    try std.testing.expectEqual(@as(usize, @intFromPtr(buffer.id)), p.buffers[slot].id);
    try std.testing.expectEqual(off, p.buffers[slot].off);
}

fn checkVerify(
    record: *const mtl.Record,
    engine: *backend.Metal,
    caches: *const [2]*st.Cache,
    tokens: *const [2][15]u32,
    sampled: *const [2][16]u32,
    drafts: *const [2][15]u32,
    expected_ids: []const u32,
    starts: *const [2]usize,
    rows: usize,
    shared: bool,
) !void {
    // The opt-in fail-before is a runtime observation, not a reference to a not-yet-existing runShared.
    try std.testing.expectEqual(@as(usize, if (shared) 1 else 2), record.commits);
    try std.testing.expectEqual(@as(usize, 0), record.bad);
    const c = &engine.model.config;
    const total = 2 * rows;
    var heads: [4]*const mtl.Pass = undefined;
    var head_count: usize = 0;
    for (record.passes[0..record.pass_count]) |*p| if (p.buffers[5].id == record.logits and p.lengths[4] == 24) {
        heads[head_count] = p;
        head_count += 1;
    };
    try std.testing.expectEqual(@as(usize, if (shared) 1 else 2), head_count);
    for (heads[0..head_count]) |p| {
        try expectWord(p, 4, 3, @intCast(if (shared) total else rows));
        try std.testing.expectEqual(@as(usize, 0), p.buffers[5].off);
    }

    var copy_store: [4]*const mtl.Pass = undefined;
    const copies = try collect(record, glue(.copy32), &copy_store);
    try std.testing.expectEqual(@as(usize, 2), copies.len);
    for (copies, 0..) |p, seg| {
        try expectWord(p, 1, 0, @intCast(rows));
        try expectWord(p, 1, 1, c.hidden);
        try expectWord(p, 1, 6, c.hidden);
        const base = if (shared) seg * rows * c.hidden * 4 else 0;
        try expectBuffer(p, 0, engine.scratch.resid, base);
        try expectBuffer(p, 2, caches[seg].hidden, 0);
    }
    try std.testing.expect(caches[0] != caches[1]);
    try std.testing.expect(caches[0].hidden.id != caches[1].hidden.id);
    try std.testing.expect(caches[0].kv.id != caches[1].kv.id);
    try std.testing.expect(caches[0].index.id != caches[1].index.id);

    if (shared) {
        try std.testing.expectEqualSlices(u32, expected_ids, engine.scratch.ids.slice(u32, total));
        var attend_store: [8]*const mtl.Pass = undefined;
        const attends = try collect(record, glue(.attend), &attend_store);
        try std.testing.expectEqual(@as(usize, 4), attends.len);
        const qrow = @as(usize, c.heads) * c.head_dim * 2;
        var at: usize = 0;
        for (0..2) |layer| for (0..2) |seg| {
            const p = attends[at];
            at += 1;
            try expectWord(p, 4, 0, @intCast(rows));
            try expectWord(p, 4, 1, @intCast(starts[seg]));
            try expectWord(p, 4, 5, @intCast(caches[seg].ring(c, layer)));
            try expectWord(p, 4, 6, if (c.layer[layer].kind == .sliding) c.window else 0);
            try expectBuffer(p, 0, engine.scratch.qn, seg * rows * qrow);
            try expectBuffer(p, 1, caches[seg].kv, caches[seg].keys(layer));
            try expectBuffer(p, 2, caches[seg].kv, caches[seg].values(c, layer));
            try expectBuffer(p, 5, engine.scratch.o, seg * rows * qrow);
        };

        var score_store: [4]*const mtl.Pass = undefined;
        const scores = try collect(record, glue(.score), &score_store);
        try std.testing.expectEqual(@as(usize, if (rows == 1) 1 else 2), scores.len);
        for (scores, 0..) |p, score| {
            const seg = if (rows == 1) score + 1 else score;
            try expectWord(p, 3, 0, @intCast(rows));
            try expectWord(p, 3, 1, @intCast(starts[seg]));
            try expectWord(p, 3, 7, @intCast((starts[seg] + rows - 1) / c.region));
            try expectWord(p, 3, 9, c.index_topk);
            try expectBuffer(p, 0, engine.scratch.iqn, seg * rows * c.index_heads * c.proxy * 2);
            try expectBuffer(p, 1, engine.scratch.iw, seg * rows * c.index_heads * 4);
            try expectBuffer(p, 2, caches[seg].index, caches[seg].summaries(1));
        }

        var norm_store: [8]*const mtl.Pass = undefined;
        const norms = try collect(record, glue(.index_norm), &norm_store);
        var index_segments: usize = 0;
        for (norms) |p| {
            if (try word(p, 3, 11) != 1) continue;
            const seg = index_segments;
            index_segments += 1;
            try expectWord(p, 3, 0, @intCast(rows));
            try expectWord(p, 3, 1, @intCast(starts[seg]));
            try expectWord(p, 3, 5, @intCast(st.index_ring));
            try expectWord(p, 3, 11, 1);
            try expectBuffer(p, 0, engine.scratch.ik, seg * rows * c.proxy * 2);
            try expectBuffer(p, 4, caches[seg].index, caches[seg].indexKeys(c, 1));
        }
        try std.testing.expectEqual(@as(usize, 2), index_segments);
    }

    for (0..2) |seg| {
        try std.testing.expectEqual(starts[seg] + rows, caches[seg].len);
        try std.testing.expectEqual(@as(usize, 0), caches[seg].nheld);
        try std.testing.expectEqualSlices(u32, tokens[seg][0 .. rows - 1], drafts[seg][0 .. rows - 1]);
        for (0..rows) |row| {
            const global_row = if (shared) seg * rows + row else row;
            try std.testing.expectEqual(@as(u32, @intCast(global_row + 1)), sampled[seg][row]);
        }
    }
}

fn verifyCase(verify_shared: bool, rows: usize, shared: bool, moe: bool) !void {
    var record: mtl.Record = .{ .gpa = a, .capture = true };
    defer record.deinit();
    var m: Model = undefined;
    m.gpa = a;
    m.device = record.device();
    m.queue = record.queue();
    m.config = geometry();
    for (&m.glue, 0..) |*p, i| p.* = .{ .id = @ptrFromInt(i + 1) };
    m.rows = undefined;
    m.rows.wide = false;
    for (&m.rows.plain) |*p| p.* = .{ .id = @ptrFromInt(100) };
    for (&m.rows.indexed) |*p| p.* = .{ .id = @ptrFromInt(101) };
    for (&m.route) |*p| p.* = .{ .id = @ptrFromInt(102) };
    m.attend_nax = null;
    m.vision = null;
    m.mms = 0;
    const weight = try m.device.buffer(64, mtl.ResourceOptions.shared);
    const c = &m.config;
    const layers = try a.alloc(wts.Layer, c.layers);
    defer a.free(layers);
    const mtp = try a.alloc(wts.Mtp, 0);
    defer a.free(mtp);
    for (layers, 0..) |*l, i| {
        var attention: wts.Attention = .{
            .q = linear(weight, c.heads * c.head_dim, c.hidden),
            .k = linear(weight, c.kvWidth(), c.hidden),
            .v = linear(weight, c.kvWidth(), c.hidden),
            .o = linear(weight, c.hidden, c.heads * c.head_dim),
            .g = linear(weight, c.heads, c.hidden),
            .q_norm = tensor(weight),
            .k_norm = tensor(weight),
        };
        if (c.layer[i].kind == .full) attention.indexer = .{
            .q = linear(weight, c.index_heads * c.proxy, c.hidden),
            .k = linear(weight, c.proxy, c.hidden),
            .z = linear(weight, c.proxy, c.hidden),
            .w = tensor(weight),
            .q_norm = tensor(weight),
            .k_norm = tensor(weight),
            .k_bias = tensor(weight),
        };
        l.* = .{ .input_norm = tensor(weight), .post_norm = tensor(weight), .attn = attention, .ffn = .{ .dense = mlp(weight, c.hidden, c.intermediate) } };
        if (moe) l.ffn = .{ .moe = .{
            .router = weight,
            .bias = tensor(weight),
            .gate = .{ .one = linear(weight, c.moe_intermediate, c.hidden), .experts = c.experts },
            .up = .{ .one = linear(weight, c.moe_intermediate, c.hidden), .experts = c.experts },
            .down = .{ .one = linear(weight, c.hidden, c.moe_intermediate), .experts = c.experts },
            .shared = mlp(weight, c.hidden, c.shared_intermediate),
        } };
    }
    m.weights = .{ .embed = linear(weight, c.vocab, c.hidden), .head = linear(weight, c.vocab, c.hidden), .norm = tensor(weight), .layers = layers, .mtp = mtp };
    const engine = try backend.Metal.init(a, &m, .{ .capacity = 64, .budget = 1 << 30, .verify_shared = verify_shared });
    defer engine.deinit();

    var streams: [2]lanes.Stream = undefined;
    for (&streams, 0..) |*s, i| s.* = try lanes.Stream.init(a, .{ .id = if (i == 0) "left" else "right", .prompt = &.{1}, .max_new = 32 });
    defer for (&streams) |*s| s.deinit(a);
    const starts = [2]usize{ 7, 19 };
    var caches: [2]*st.Cache = undefined;
    for (&caches, &streams, starts) |*cache, *stream, start| {
        cache.* = try engine.createCache(64);
        cache.*.len = start;
        try engine.caches.put(a, stream, cache.*);
    }

    var tokens: [2][15]u32 = undefined;
    var positions: [2][16]u64 = undefined;
    var sampled: [2][16]u32 = undefined;
    var drafts: [2][15]u32 = undefined;
    var windows: [2]be.Window = undefined;
    var out: [2]be.Verified = undefined;
    var expected_ids: [2 * st.window_rows]u32 = undefined;
    var id_at: usize = 0;
    for (0..2) |seg| {
        for (0..rows) |row| positions[seg][row] = starts[seg] + row + 1;
        for (0..rows - 1) |row| tokens[seg][row] = @intCast(3 + seg * 4 + row);
        windows[seg] = .{ .stream = &streams[seg], .pending = @intCast(seg + 1), .held = 0, .tokens = tokens[seg][0 .. rows - 1], .parents = null, .positions = positions[seg][0..rows] };
        out[seg] = .{ .sampled = sampled[seg][0..rows], .drafts = drafts[seg][0 .. rows - 1] };
        expected_ids[id_at] = windows[seg].pending;
        id_at += 1;
        for (tokens[seg][0 .. rows - 1]) |token| {
            expected_ids[id_at] = token;
            id_at += 1;
        }
    }
    record.pass_count = 0;
    record.commits = 0;
    record.bad = 0;
    record.logits = @intFromPtr(engine.scratch.logits_all.id);
    try engine.backend().verify(&windows, &out);
    try checkVerify(&record, engine, &caches, &tokens, &sampled, &drafts, expected_ids[0..id_at], &starts, rows, shared);
    if (moe) {
        var passes: [16]*const mtl.Pass = undefined;
        const indexed = try collect(&record, 101, &passes);
        try std.testing.expectEqual(@as(usize, if (shared) 6 else 12), indexed.len);
        for (indexed) |p| try expectWord(p, 4, 3, @intCast((if (shared) 2 else @as(usize, 1)) * rows * c.topk));
    }
}

test "step5 host: verify shared switched off records two legacy forwards" {
    try verifyCase(false, 4, false, false);
}

test "step5 host: verify shared records one forward and independent segments" {
    try verifyCase(true, 4, true, false);
}

test "step5 host: verify shared over sixteen rows records two legacy forwards" {
    try verifyCase(true, 9, false, false);
}

test "step5 host: verify shared covers two one-row segments and the sixteen-row limit" {
    try verifyCase(true, 1, true, false);
    try verifyCase(true, 8, true, false);
}

test "step5 host: verify shared coalesces routed projections with unchanged per-segment cache geometry" {
    try verifyCase(false, 4, false, true);
    try verifyCase(true, 4, true, true);
}
