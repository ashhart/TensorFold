//! CPU CSA2 from DeepSeek AI inference/{model,kernel}.py, rev 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
const std = @import("std");
const tf = @import("tensorfold");
const m = @import("math.zig");
const projection = @import("projection.zig");
const Config = tf.deepseek_v41_config.Config;
const Attention = tf.deepseek_v41_weights.Attention;

pub const max_capacity = 4096;
/// Kernel-shaped key addressing: window rows at physical ring slots, then compressed rows.
pub const KeyTable = struct {
    window: []const u16,
    window_stride: usize = 512,
    compressed: []const u16 = &[_]u16{},
    compressed_stride: usize = 512,
    pub fn keyRow(t: KeyTable, at: usize) []const u16 {
        if (at < 128) return t.window[at * t.window_stride ..][0..512];
        const row = at - 128;
        return t.compressed[row * t.compressed_stride ..][0..512];
    }
};
fn narrow(comptime T: type, x: f64) T {
    return if (T == f32) @as(f32, @floatCast(x)) else x;
}
/// Compressor softmax pooling at reference precision T: FP32 is the source contract, FP64 the ideal.
pub fn poolOf(comptime T: type, kv: [2][]const T, gate: [2][]const T, out: []u16) void {
    @setFloatMode(.strict);
    for (out, 0..) |*o, d| {
        const high = @max(gate[0][d], gate[1][d]);
        const p0 = @exp(gate[0][d] - high);
        const p1 = @exp(gate[1][d] - high);
        const denominator = p0 + p1;
        const pooled = kv[0][d] * (p0 / denominator) + kv[1][d] * (p1 / denominator);
        o.* = if (T == f32) m.b(pooled) else m.b64(pooled);
    }
}
pub const State = struct {
    gpa: std.mem.Allocator,
    capacity: usize,
    window: []u16,
    sources: []u16,
    scores: []f32,
    candidates: []bool,
    blocks: []f32,
    active_owner: usize = 0,
    selected: [512]u32 = undefined,
    selected_len: usize = 0,
    partial_kv: [4][2][512]f32 = undefined,
    partial_gate: [4][2][512]f32 = undefined,
    qr: [1280]u16 = undefined,
    q: [32768]u16 = undefined,
    index_q: [4096]u16 = undefined,
    index_weights: [32]f32 = undefined,
    latent: [512]u16 = undefined,
    index_key: [128]u16 = undefined,
    kv: [512]u16 = undefined,
    o: [32768]u16 = undefined,
    grouped: [8192]u16 = undefined,
    observer: ?Observer = null,
    sparse_snapshot: ?[]u16 = null,
    /// The indexer-selection probe: per scored window, the contract and FP64-ideal margins, and the pick flips.
    score_probe: ?ScoreProbe = null,
    score_copy: []f32 = &.{},
    ideal_scores: []f32 = &.{},
    ideal_copy: []f32 = &.{},
    pick_work: []f32 = &.{},
    selected_ideal: []u32 = &.{},
    pub const ScoreProbe = struct {
        ctx: *anyopaque,
        record: *const fn (*anyopaque, layer: usize, position: usize, count: usize, selected_len: usize, margin: f32, margin_ideal: f32, flips: usize, lowest: f32, highest_unselected: f32) void,
    };
    pub const Observer = struct {
        ctx: *anyopaque,
        capture: *const fn (*anyopaque, tf.deepseek_v41.attention.DiagnosticStage, []const u16) void,
    };
    fn observe(self: *State, stage: tf.deepseek_v41.attention.DiagnosticStage, values: []const u16) void {
        if (self.observer) |observer| observer.capture(observer.ctx, stage, values);
    }

    pub fn init(gpa: std.mem.Allocator, capacity: usize) !State {
        if (capacity == 0 or capacity > max_capacity) return error.InvalidAttentionReferenceCapacity;
        const window = try gpa.alloc(u16, 40 * 128 * 512);
        errdefer gpa.free(window);
        const sources = try gpa.alloc(u16, 4 * capacity * 640);
        errdefer gpa.free(sources);
        const scores = try gpa.alloc(f32, capacity);
        errdefer gpa.free(scores);
        const candidates = try gpa.alloc(bool, capacity);
        errdefer gpa.free(candidates);
        const blocks = try gpa.alloc(f32, (capacity + 7) / 8);
        errdefer gpa.free(blocks);
        const score_copy = try gpa.alloc(f32, capacity);
        errdefer gpa.free(score_copy);
        const ideal_scores = try gpa.alloc(f32, capacity);
        errdefer gpa.free(ideal_scores);
        const ideal_copy = try gpa.alloc(f32, capacity);
        errdefer gpa.free(ideal_copy);
        const pick_work = try gpa.alloc(f32, capacity);
        errdefer gpa.free(pick_work);
        const selected_ideal = try gpa.alloc(u32, capacity);
        errdefer gpa.free(selected_ideal);
        @memset(window, 0);
        return .{ .gpa = gpa, .capacity = capacity, .window = window, .sources = sources, .scores = scores, .candidates = candidates, .blocks = blocks, .score_copy = score_copy, .ideal_scores = ideal_scores, .ideal_copy = ideal_copy, .pick_work = pick_work, .selected_ideal = selected_ideal };
    }
    pub fn deinit(self: *State) void {
        self.gpa.free(self.window);
        self.gpa.free(self.sources);
        self.gpa.free(self.scores);
        self.gpa.free(self.candidates);
        self.gpa.free(self.blocks);
        self.gpa.free(self.score_copy);
        self.gpa.free(self.ideal_scores);
        self.gpa.free(self.ideal_copy);
        self.gpa.free(self.pick_work);
        self.gpa.free(self.selected_ideal);
        self.* = undefined;
    }
    fn mainRow(self: *State, owner: usize, row: usize) []u16 {
        return self.sources[owner * self.capacity * 640 + row * 640 ..][0..512];
    }
    fn indexRow(self: *State, owner: usize, row: usize) []u16 {
        return self.sources[owner * self.capacity * 640 + row * 640 + 512 ..][0..128];
    }
    fn publish(self: *State, a: Attention, layer: usize, x: []const u16, position: usize, c: Config, scratch: *projection.Scratch) !bool {
        @setFloatMode(.strict);
        const compressor = a.compressor orelse return false;
        const owner = (layer - 2) / 6;
        self.active_owner = owner;
        const ratio: usize = c.compress_ratios.items()[layer];
        if (ratio == 1) {
            try projection.apply(compressor.wkv, x, &self.latent, scratch);
        } else {
            const slot = position % ratio;
            try projection.applyF32(compressor.wkv, x, &self.partial_kv[owner][slot], scratch);
            try projection.applyF32(compressor.wgate.?, x, &self.partial_gate[owner][slot], scratch);
            if ((position + 1) % ratio != 0) return false;
            poolOf(f32, .{ &self.partial_kv[owner][0], &self.partial_kv[owner][1] }, .{ &self.partial_gate[owner][0], &self.partial_gate[owner][1] }, &self.latent);
        }
        m.rms(&self.latent, compressor.norm, c.eps, &self.latent);
        const indexer = a.indexer.?;
        try projection.apply(indexer.wk.?, &self.latent, &self.index_key, scratch);
        m.rms(&self.index_key, indexer.k_norm.?, c.eps, &self.index_key);
        const group_position = position + 1 - ratio;
        m.rope(&self.index_key, group_position, true, false, c);
        m.quantize(&self.index_key, .index);
        const row = position / ratio;
        @memcpy(self.indexRow(owner, row), &self.index_key);
        m.rope(&self.latent, group_position, true, false, c);
        m.quantize(&self.latent, .main);
        @memcpy(self.mainRow(owner, row), &self.latent);
        return true;
    }
    fn chooseCandidates(self: *State, count: usize, c: Config) void {
        @memset(self.candidates[0..count], false);
        const nblocks = (count + 7) / 8;
        for (0..nblocks) |block| {
            var score: f32 = -std.math.inf(f32);
            for (block * 8..@min(block * 8 + 8, count)) |row| score = @max(score, self.scores[row]);
            self.blocks[block] = score;
        }
        if (nblocks == 0) return;
        self.blocks[nblocks - 1] = std.math.inf(f32);
        for (0..@min(nblocks, c.candidate_blocks)) |_| {
            var best: usize = 0;
            for (0..nblocks) |i| if (self.blocks[i] > self.blocks[best]) {
                best = i;
            };
            if (self.blocks[best] == -std.math.inf(f32)) break;
            for (best * 8..@min(best * 8 + 8, count)) |row| self.candidates[row] = true;
            self.blocks[best] = -std.math.inf(f32);
        }
    }
    fn selectPositions(self: *State, a: Attention, layer: usize, x: []const u16, position: usize, count: usize, c: Config, scratch: *projection.Scratch) !void {
        const indexer = a.indexer orelse return;
        self.selected_len = @min(count, c.index_topk);
        if (count == 0) return;
        try projection.apply(indexer.wq_b, &self.qr, &self.index_q, scratch);
        for (0..32) |head| {
            const q = self.index_q[head * 128 ..][0..128];
            m.rope(q, position, true, false, c);
            m.quantize(q, .index);
        }
        m.dense(indexer.weights_proj, x, &self.index_weights);
        const scale: f32 = 1 / @sqrt(@as(f32, 128 * 32));
        // Python scalar multiplication preserves the source weights projection's BF16 dtype.
        for (&self.index_weights) |*v| v.* = m.f(m.b(m.f(m.b(v.*)) * scale));
        for (0..count) |row| {
            const key = self.indexRow(self.active_owner, row);
            var score: f64 = 0;
            for (0..32) |head| {
                var dot: f64 = 0;
                for (key, 0..) |v, d| dot += @as(f64, m.f(v)) * m.f(self.index_q[head * 128 + d]);
                // einsum produces BF16, then BF16 multiply and head sum, in the pinned source.
                const rectified = @max(m.f(m.b(@floatCast(dot))), 0);
                const weighted = m.f(m.b(rectified * self.index_weights[head]));
                score += weighted;
            }
            self.scores[row] = m.f(m.b(@floatCast(score)));
        }
        if (layer == c.candidate_source) self.chooseCandidates(count, c);
        if (layer > c.candidate_source) for (0..count) |row| {
            if (!self.candidates[row]) self.scores[row] = -std.math.inf(f32);
        };
        @memcpy(self.score_copy[0..count], self.scores[0..count]);
        self.idealScores(a, count);
        for (0..self.selected_len) |pick| {
            var best: usize = 0;
            for (0..count) |i| if (self.scores[i] > self.scores[best]) {
                best = i;
            };
            self.selected[pick] = @intCast(best);
            self.scores[best] = -std.math.inf(f32);
        }
        std.mem.sort(u32, self.selected[0..self.selected_len], {}, std.sort.asc(u32));
        if (self.score_probe) |probe| self.reportSelection(probe, layer, position, count);
    }
    /// The same selection on FP64-ideal scores: same BF16 inputs, no intermediate roundings; pick flips are the signal.
    fn idealScores(self: *State, a: Attention, count: usize) void {
        @setFloatMode(.strict);
        if (a.indexer == null) return;
        for (0..count) |row| {
            if (self.score_copy[row] == -std.math.inf(f32)) {
                self.ideal_scores[row] = -std.math.inf(f32);
                continue;
            }
            const key = self.indexRow(self.active_owner, row);
            var score: f64 = 0;
            for (0..32) |head| {
                var dot: f64 = 0;
                for (key, 0..) |v, d| dot += @as(f64, m.f(v)) * m.f(self.index_q[head * 128 + d]);
                score += @max(dot, 0) * @as(f64, self.index_weights[head]);
            }
            self.ideal_scores[row] = @floatCast(score);
        }
    }
    fn pickTop(self: *State, scores: []const f32, into: []u32) void {
        @memcpy(self.pick_work[0..scores.len], scores);
        const work = self.pick_work[0..scores.len];
        for (into) |*pick| {
            var best: usize = 0;
            for (work, 0..) |v, i| if (v > work[best]) {
                best = i;
            };
            pick.* = @intCast(best);
            work[best] = -std.math.inf(f32);
        }
        std.mem.sort(u32, into, {}, std.sort.asc(u32));
    }
    fn pickedAt(selection: []const u32, row: usize) bool {
        var lo: usize = 0;
        var hi: usize = selection.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (selection[mid] == row) return true;
            if (selection[mid] < row) lo = mid + 1 else hi = mid;
        }
        return false;
    }
    fn reportSelection(self: *State, probe: ScoreProbe, layer: usize, position: usize, count: usize) void {
        var lowest: f32 = std.math.inf(f32);
        var highest_unselected: f32 = -std.math.inf(f32);
        for (self.selected[0..self.selected_len]) |row| {
            lowest = @min(lowest, self.score_copy[row]);
        }
        for (0..count) |row| {
            if (!pickedAt(self.selected[0..self.selected_len], row) and self.score_copy[row] > -std.math.inf(f32))
                highest_unselected = @max(highest_unselected, self.score_copy[row]);
        }
        const margin: f32 = if (self.selected_len == 0 or lowest == std.math.inf(f32)) 0 else lowest - highest_unselected;
        // The ideal picks, on a copy so the probe's ideal scores survive for callers.
        @memcpy(self.ideal_copy[0..count], self.ideal_scores[0..count]);
        const ideal_len = @min(self.selected_len, self.selected_ideal.len);
        self.pickTop(self.ideal_copy[0..count], self.selected_ideal[0..ideal_len]);
        var flips: usize = 0;
        if (self.selected_len == ideal_len) {
            for (self.selected[0..self.selected_len], self.selected_ideal[0..ideal_len]) |s, i| {
                if (s != i) flips += 1;
            }
        } else flips = self.selected_len;
        var ideal_lowest: f32 = std.math.inf(f32);
        var ideal_highest_unselected: f32 = -std.math.inf(f32);
        for (self.selected_ideal[0..ideal_len]) |row| {
            ideal_lowest = @min(ideal_lowest, self.ideal_scores[row]);
        }
        for (0..count) |row| {
            if (!pickedAt(self.selected_ideal[0..ideal_len], row) and self.ideal_scores[row] > -std.math.inf(f32))
                ideal_highest_unselected = @max(ideal_highest_unselected, self.ideal_scores[row]);
        }
        const margin_ideal: f32 = if (ideal_len == 0 or ideal_lowest == std.math.inf(f32)) 0 else ideal_lowest - ideal_highest_unselected;
        probe.record(probe.ctx, layer, position, count, self.selected_len, margin, margin_ideal, flips, lowest, highest_unselected);
    }
    pub fn apply(self: *State, a: Attention, layer: usize, x: []const u16, position: usize, c: Config, out: []u16, scratch: *projection.Scratch) !void {
        const compressed = c.compress_ratios.items()[layer] != 0;
        try projection.apply(.{ .trellis = a.wq_a }, x, &self.qr, scratch);
        self.observe(.qr_projection, &self.qr);
        m.rms(&self.qr, a.q_norm, c.eps, &self.qr);
        self.observe(.qr_norm, &self.qr);
        try projection.apply(.{ .trellis = a.wq_b }, &self.qr, &self.q, scratch);
        self.observe(.q_projection, &self.q);
        for (0..64) |head| m.rope(self.q[head * 512 ..][0..512], position, compressed, false, c);
        self.observe(.q, &self.q);
        try projection.apply(.{ .trellis = a.wkv }, x, &self.kv, scratch);
        self.observe(.kv_projection, &self.kv);
        m.rms(&self.kv, a.kv_norm, c.eps, &self.kv);
        self.observe(.kv_norm, &self.kv);
        m.rope(&self.kv, position, compressed, false, c);
        self.observe(.kv_rope, &self.kv);
        m.quantize(&self.kv, .window);
        self.observe(.kv_quantized, &self.kv);
        const window = self.window[layer * 128 * 512 ..][0 .. 128 * 512];
        @memcpy(window[(position % 128) * 512 ..][0..512], &self.kv);
        var count: usize = 0;
        if (compressed) {
            _ = try self.publish(a, layer, x, position, c, scratch);
            count = (position + 1) / c.compress_ratios.items()[layer];
            try self.selectPositions(a, layer, x, position, count, c, scratch);
        }
        var indices: [640]i32 = undefined;
        var slots: usize = 0;
        if (position == 0) {
            indices[0] = 0;
            slots = 1;
        } else {
            const oldest = position % 128 + 1;
            for (0..128) |i| {
                const at = (oldest + i) % 128;
                indices[i] = if (at > position) -1 else @intCast(at);
            }
            slots = 128;
        }
        if (compressed) for (self.selected[0..self.selected_len]) |row| {
            indices[slots] = @intCast(128 + row);
            slots += 1;
        };
        for (0..64) |head| {
            self.sparseHead(self.q[head * 512 ..][0..512], window, indices[0..slots], m.value(a.sink, head), self.o[head * 512 ..][0..512]);
            self.observeHead(head);
            m.rope(self.o[head * 512 ..][0..512], position, compressed, true, c);
        }
        if (self.sparse_snapshot) |snapshot| self.observe(.sparse_o, snapshot);
        self.observe(.o, &self.o);
        for (0..8) |group| try projection.apply(.{ .trellis = a.wo_a[group] }, self.o[group * 4096 ..][0..4096], self.grouped[group * 1024 ..][0..1024], scratch);
        self.observe(.grouped, &self.grouped);
        try projection.apply(.{ .trellis = a.wo_b }, &self.grouped, out, scratch);
        self.observe(.wo_b, out);
    }
    fn observeHead(self: *State, head: usize) void {
        if (self.sparse_snapshot) |snapshot|
            @memcpy(snapshot[head * 512 ..][0..512], self.o[head * 512 ..][0..512]);
    }
    pub fn sparseHead(self: *State, q: []const u16, window: []const u16, indices: []const i32, sink: f32, out: []u16) void {
        // The source-contract FP32 instance; mainRow addresses compressed rows at the 640 stride.
        sparseHeadOf(f32, q, .{
            .window = window,
            .compressed = self.sources[self.active_owner * self.capacity * 640 ..],
            .compressed_stride = 640,
        }, indices, sink, out);
    }
};

/// Sparse online-softmax head: f32 reproduces the pinned source contract bit-for-bit; f64 makes every boundary ideal.
pub fn sparseHeadOf(comptime T: type, q: []const u16, keys: KeyTable, indices: []const i32, sink: f32, out: []u16) void {
    return sparseHeadPre(T, q, keys, indices, sink, out, null);
}

/// Same head; `pre` captures acc/denominator pre-rounding, so FP32 vs FP64 ULP is the true rounding level, not BF16's.
pub fn sparseHeadPre(
    comptime T: type,
    q: []const u16,
    keys: KeyTable,
    indices: []const i32,
    sink: f32,
    out: []u16,
    pre: ?[]T,
) void {
    @setFloatMode(.strict);
    const scale: T = if (T == f32) 1 / @sqrt(@as(f32, 512)) else 1 / @sqrt(@as(f64, 512));
    var acc: [512]T = @splat(0);
    var high: T = m.wide(T, -1e30);
    var denominator: T = 0;
    var base: usize = 0;
    while (base < indices.len) : (base += 64) {
        var probabilities: [64]T = undefined;
        const width = @min(64, indices.len - base);
        var next_high = high;
        for (0..width) |j| {
            const index = indices[base + j];
            var dot: f64 = 0;
            if (index >= 0) {
                const key = keys.keyRow(@intCast(index));
                for (q, key) |a, v| dot += m.d64(a) * m.d64(v);
            }
            probabilities[j] = if (index < 0) -std.math.inf(T) else narrow(T, dot) * scale;
            next_high = @max(next_high, probabilities[j]);
        }
        const correction = @exp(high - next_high);
        var sum: f64 = 0;
        for (0..width) |j| {
            probabilities[j] = @exp(probabilities[j] - next_high);
            sum += probabilities[j];
        }
        denominator = denominator * correction + narrow(T, sum);
        for (0..512) |d| {
            var accumulated: f64 = acc[d] * correction;
            for (0..width) |j| if (indices[base + j] >= 0) {
                // Online 64-column source kernel rounds unnormalized probabilities to BF16 before P*V.
                const p: f64 = if (T == f32) @as(f64, m.f(m.b(probabilities[j]))) else m.d64(m.b64(probabilities[j]));
                accumulated += p * m.d64(keys.keyRow(@intCast(indices[base + j]))[d]);
            };
            acc[d] = @floatCast(accumulated);
        }
        high = next_high;
    }
    denominator += @exp(m.wide(T, sink) - high);
    for (out, acc, 0..) |*v, a, i| {
        const quotient = a / denominator;
        if (pre) |cells| cells[i] = quotient;
        v.* = if (T == f32) m.b(quotient) else m.b64(quotient);
    }
}

test "online 64-column sparse attention rounds probabilities before value GEMM" {
    var state = try State.init(std.testing.allocator, 2);
    defer state.deinit();
    var q: [512]u16 = @splat(0);
    q[0] = m.b(8);
    state.window[1] = m.b(1);
    state.window[512] = m.b(1);
    var indices: [65]i32 = @splat(-1);
    indices[0] = 0;
    indices[1] = 1;
    indices[64] = 0;
    var out: [512]u16 = undefined;
    state.sparseHead(&q, state.window[0..1024], &indices, 0, &out);
    const probability = @exp(-@as(f32, 8) * (1 / @sqrt(@as(f32, 512))));
    var denominator = @as(f32, @floatCast(@as(f64, probability) + 1));
    denominator += probability;
    denominator += probability;
    const expected = m.b((2 * m.f(m.b(probability))) / denominator);
    try std.testing.expectEqual(expected, out[1]);
}

test "attention reference capacity rejects unbounded history before allocation" {
    try std.testing.expectError(error.InvalidAttentionReferenceCapacity, State.init(std.testing.allocator, 0));
    try std.testing.expectError(error.InvalidAttentionReferenceCapacity, State.init(std.testing.allocator, max_capacity + 1));
}
