//! CPU CSA2 from DeepSeek AI inference/{model,kernel}.py, rev 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
const std = @import("std");
const tf = @import("tensorfold");
const m = @import("math.zig");
const projection = @import("projection.zig");
const Config = tf.deepseek_v41_config.Text;
const Attention = tf.deepseek_v41_weights.Attention;

pub const max_capacity = 4096;
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
        @memset(window, 0);
        return .{ .gpa = gpa, .capacity = capacity, .window = window, .sources = sources, .scores = scores, .candidates = candidates, .blocks = blocks };
    }
    pub fn deinit(self: *State) void {
        self.gpa.free(self.window);
        self.gpa.free(self.sources);
        self.gpa.free(self.scores);
        self.gpa.free(self.candidates);
        self.gpa.free(self.blocks);
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
        const ratio: usize = c.compress_ratios[layer];
        if (ratio == 1) {
            try projection.apply(compressor.wkv, x, &self.latent, scratch);
        } else {
            const slot = position % ratio;
            try projection.applyF32(compressor.wkv, x, &self.partial_kv[owner][slot], scratch);
            try projection.applyF32(compressor.wgate.?, x, &self.partial_gate[owner][slot], scratch);
            if ((position + 1) % ratio != 0) return false;
            for (0..512) |d| {
                const high = @max(self.partial_gate[owner][0][d], self.partial_gate[owner][1][d]);
                const p0 = @exp(self.partial_gate[owner][0][d] - high);
                const p1 = @exp(self.partial_gate[owner][1][d] - high);
                const denominator = p0 + p1;
                const weighted0 = self.partial_kv[owner][0][d] * (p0 / denominator);
                const weighted1 = self.partial_kv[owner][1][d] * (p1 / denominator);
                const pooled = weighted0 + weighted1;
                self.latent[d] = m.b(pooled);
            }
        }
        m.rms(&self.latent, compressor.norm, c.rms_norm_eps, &self.latent);
        const indexer = a.indexer.?;
        try projection.apply(indexer.wk.?, &self.latent, &self.index_key, scratch);
        m.rms(&self.index_key, indexer.k_norm.?, c.rms_norm_eps, &self.index_key);
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
        for (0..@min(nblocks, c.candidate_topk_blocks)) |_| {
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
        if (layer == c.candidate_source_layer_id) self.chooseCandidates(count, c);
        if (layer > c.candidate_source_layer_id) for (0..count) |row| {
            if (!self.candidates[row]) self.scores[row] = -std.math.inf(f32);
        };
        for (0..self.selected_len) |pick| {
            var best: usize = 0;
            for (0..count) |i| if (self.scores[i] > self.scores[best]) {
                best = i;
            };
            self.selected[pick] = @intCast(best);
            self.scores[best] = -std.math.inf(f32);
        }
        std.mem.sort(u32, self.selected[0..self.selected_len], {}, std.sort.asc(u32));
    }
    pub fn apply(self: *State, a: Attention, layer: usize, x: []const u16, position: usize, c: Config, out: []u16, scratch: *projection.Scratch) !void {
        const compressed = c.compress_ratios[layer] != 0;
        try projection.apply(a.wq_a, x, &self.qr, scratch);
        self.observe(.qr_projection, &self.qr);
        m.rms(&self.qr, a.q_norm, c.rms_norm_eps, &self.qr);
        self.observe(.qr_norm, &self.qr);
        try projection.apply(a.wq_b, &self.qr, &self.q, scratch);
        self.observe(.q_projection, &self.q);
        for (0..64) |head| m.rope(self.q[head * 512 ..][0..512], position, compressed, false, c);
        self.observe(.q, &self.q);
        try projection.apply(a.wkv, x, &self.kv, scratch);
        self.observe(.kv_projection, &self.kv);
        m.rms(&self.kv, a.kv_norm, c.rms_norm_eps, &self.kv);
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
            count = (position + 1) / c.compress_ratios[layer];
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
        for (0..8) |group| try projection.apply(a.wo_a[group], self.o[group * 4096 ..][0..4096], self.grouped[group * 1024 ..][0..1024], scratch);
        self.observe(.grouped, &self.grouped);
        try projection.apply(a.wo_b, &self.grouped, out, scratch);
        self.observe(.wo_b, out);
    }
    fn observeHead(self: *State, head: usize) void {
        if (self.sparse_snapshot) |snapshot|
            @memcpy(snapshot[head * 512 ..][0..512], self.o[head * 512 ..][0..512]);
    }
    fn keyRow(self: *State, window: []const u16, key_id: i32) []const u16 {
        const at: usize = @intCast(key_id);
        return if (at < 128) window[at * 512 ..][0..512] else self.mainRow(self.active_owner, at - 128);
    }
    pub fn sparseHead(self: *State, q: []const u16, window: []const u16, indices: []const i32, sink: f32, out: []u16) void {
        @setFloatMode(.strict);
        var acc: [512]f32 = @splat(0);
        var high: f32 = -1e30;
        var denominator: f32 = 0;
        var base: usize = 0;
        while (base < indices.len) : (base += 64) {
            var probabilities: [64]f32 = undefined;
            const width = @min(64, indices.len - base);
            var next_high = high;
            for (0..width) |j| {
                const index = indices[base + j];
                var dot: f64 = 0;
                if (index >= 0) {
                    const key = self.keyRow(window, index);
                    for (q, key) |a, v| dot += @as(f64, m.f(a)) * m.f(v);
                }
                probabilities[j] = if (index < 0) -std.math.inf(f32) else @as(f32, @floatCast(dot)) * (1 / @sqrt(@as(f32, 512)));
                next_high = @max(next_high, probabilities[j]);
            }
            const correction = @exp(high - next_high);
            var sum: f64 = 0;
            for (0..width) |j| {
                probabilities[j] = @exp(probabilities[j] - next_high);
                sum += probabilities[j];
            }
            denominator = denominator * correction + @as(f32, @floatCast(sum));
            for (0..512) |d| {
                var accumulated: f64 = acc[d] * correction;
                for (0..width) |j| if (indices[base + j] >= 0) {
                    // Online 64-column source kernel rounds unnormalized probabilities to BF16 before P*V.
                    accumulated += @as(f64, m.f(m.b(probabilities[j]))) * m.f(self.keyRow(window, indices[base + j])[d]);
                };
                acc[d] = @floatCast(accumulated);
            }
            high = next_high;
        }
        denominator += @exp(sink - high);
        for (out, acc) |*v, a| v.* = m.b(a / denominator);
    }
};

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
