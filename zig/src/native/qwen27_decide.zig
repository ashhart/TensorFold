//! A slot_lm decision on the 27B engine's runner: the trunk once, each branch from its saved state, float32 slots.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const Runner = q.decode_round.Runner;

/// The LM head's rows for the answer labels, dequantized as MLX does (bf16 scale * q, + bias) and widened to f32.
pub const Head = struct {
    gpa: std.mem.Allocator,
    ck: tf.Checkpoint,
    weight: tf.safetensors.Tensor,
    scales: tf.safetensors.Tensor,
    biases: tf.safetensors.Tensor,
    hidden: usize,
    bits: u5,
    group: usize,
    rows: std.AutoHashMapUnmanaged(u32, []f32) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, hidden: usize) !Head {
        var ck = try tf.Checkpoint.openModelPrefix(gpa, io, dir, "language_model.lm_head.");
        errdefer ck.close();
        const weight = try ck.get("language_model.lm_head.weight");
        const scales = try ck.get("language_model.lm_head.scales");
        const biases = try ck.get("language_model.lm_head.biases");
        if (weight.dtype != .u32 or scales.dtype != .bf16 or biases.dtype != .bf16 or weight.rank != 2 or scales.rank != 2) return error.SlotHeadFormat;
        if (!std.mem.eql(usize, scales.shape[0..2], biases.shape[0..2]) or weight.shape[0] != scales.shape[0]) return error.SlotHeadFormat;
        const bits = weight.shape[1] * 32 / hidden;
        if ((bits != 4 and bits != 8) or weight.shape[1] * 32 != bits * hidden or hidden % scales.shape[1] != 0) return error.SlotHeadFormat;
        return .{ .gpa = gpa, .ck = ck, .weight = weight, .scales = scales, .biases = biases, .hidden = hidden, .bits = @intCast(bits), .group = hidden / scales.shape[1] };
    }

    pub fn deinit(h: *Head) void {
        var it = h.rows.valueIterator();
        while (it.next()) |r| h.gpa.free(r.*);
        h.rows.deinit(h.gpa);
        h.ck.close();
    }

    /// Token `id`'s head row in float32.
    pub fn row(h: *Head, id: u32) ![]const f32 {
        if (h.rows.get(id)) |hit| return hit;
        if (id >= h.weight.shape[0]) return error.SlotOutOfVocabulary;
        const out = try h.gpa.alloc(f32, h.hidden);
        errdefer h.gpa.free(out);
        const words: []align(1) const u32 = std.mem.bytesAsSlice(u32, h.weight.bytes[id * h.weight.shape[1] * 4 ..][0 .. h.weight.shape[1] * 4]);
        const groups = h.scales.shape[1];
        const per_word = 32 / @as(usize, h.bits);
        const mask: u32 = (@as(u32, 1) << h.bits) - 1;
        for (out, 0..) |*o, i| {
            const word = words[i / per_word];
            const v: f32 = @floatFromInt((word >> @intCast((i % per_word) * h.bits)) & mask);
            const g = id * groups + i / h.group;
            o.* = bf16(bf16(half(h.scales, g) * v) + half(h.biases, g));
        }
        try h.rows.put(h.gpa, id, out);
        return out;
    }
};

fn half(t: tf.safetensors.Tensor, i: usize) f32 {
    const bits = std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little);
    return @bitCast(@as(u32, bits) << 16);
}

/// f32 to the nearest bf16 (ties to even), as a bf16 multiply or add rounds its result.
pub fn bf16(x: f32) f32 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return x;
    const rounded = u +% (0x7fff + ((u >> 16) & 1));
    return @bitCast(rounded & 0xffff0000);
}

pub const Decider = struct {
    gpa: std.mem.Allocator,
    runner: *Runner,
    head: Head,
    store: q.block_forward.Store = .{},
    rewind: mtl.Buffer,
    chunk: u32,
    hidden: []f32,
    slot: u32 = 0,
    bidirectional: bool,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, runner: *Runner, chunk: u32, bidirectional: bool) !Decider {
        var head = try Head.open(gpa, io, dir, runner.model.config.hidden);
        errdefer head.deinit();
        const rewind = try runner.model.device.buffer(q.snapshot.rewindBytes(runner, null), mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
        errdefer rewind.deinit();
        return .{ .gpa = gpa, .runner = runner, .head = head, .rewind = rewind, .chunk = blockRows(runner.promptRows(chunk)), .hidden = try gpa.alloc(f32, runner.model.config.hidden), .bidirectional = bidirectional };
    }

    pub fn deinit(d: *Decider) void {
        d.store.deinit();
        d.rewind.deinit();
        d.head.deinit();
        d.gpa.free(d.hidden);
    }

    /// Each branch's label logits at its last row, after the shared trunk; slot `d.slot` must hold no stream.
    pub fn decide(d: *Decider, x: *api.Decision) !void {
        if (x.pictures.len > 0) return refuse(x, "this decision model reads text only: send no pictures");
        if (x.branches.len == 0 or x.counts.len != x.branches.len) return refuse(x, "a decision needs one or more questions");
        var labels: usize = 0;
        var longest: usize = 0;
        for (x.branches, x.counts) |b, n| {
            if (b.len == 0) return refuse(x, "each question needs a nonempty token sequence");
            labels += n;
            longest = @max(longest, b.len);
        }
        if (labels != x.labels.len or labels != x.logits.len) return error.BadDecision;
        if (x.trunk.len + longest > d.runner.capacity) return refuse(x, "the state and its longest question exceed this server's --context");
        const r = d.runner;
        try r.reset(d.slot);
        const trunk: u32 = @intCast(x.trunk.len);
        if (trunk > 0) {
            _ = try q.block_forward.run(r, &d.store, d.slot, x.trunk, d.chunk, d.bidirectional);
            try q.snapshot.saveRewind(d.gpa, r, d.slot, null, trunk, d.rewind);
        }
        var at: usize = 0;
        var read: usize = x.trunk.len;
        for (x.branches, x.counts, 0..) |b, n, i| {
            if (i > 0) if (trunk > 0) try q.snapshot.rewindSlot(d.gpa, r, d.slot, null, trunk, d.rewind) else try r.reset(d.slot);
            const rows = try q.block_forward.run(r, &d.store, d.slot, b, d.chunk, d.bidirectional);
            try q.block_forward.lastHidden(r, rows, d.hidden);
            for (x.labels[at..][0..n], x.logits[at..][0..n]) |id, *logit| {
                const w = try d.head.row(id);
                var acc: f64 = 0;
                for (w, d.hidden) |a, h| acc += @as(f64, a) * h;
                logit.* = acc;
            }
            at += n;
            read += b.len;
        }
        try r.reset(d.slot);
        x.read = read;
    }
};

/// TF_QWEN27_BLOCK_ROWS (16 to `most`, a multiple of 16) runs longer blocks layer-major at that width; a check's knob.
fn blockRows(most: u32) u32 {
    const text = std.mem.span(std.c.getenv("TF_QWEN27_BLOCK_ROWS") orelse return most);
    const n = std.fmt.parseInt(u32, text, 10) catch return most;
    return if (n >= 16 and n <= most and n % 16 == 0) n else most;
}

fn refuse(x: *api.Decision, words: []const u8) error{Refused} {
    x.problem = words;
    return error.Refused;
}

test "bf16 rounding is nearest, ties to even" {
    try std.testing.expectEqual(@as(f32, 1.0), bf16(1.0 + 1.0 / 512.0)); // below half an ulp (1/128)
    try std.testing.expectEqual(@as(f32, 1.0078125), bf16(1.0 + 3.0 / 512.0)); // above half: up
    try std.testing.expectEqual(@as(f32, 1.0), bf16(1.0 + 1.0 / 256.0)); // a tie: to the even mantissa
    try std.testing.expectEqual(@as(f32, 1.015625), bf16(1.0078125 + 1.0 / 256.0)); // a tie from odd: up
}
