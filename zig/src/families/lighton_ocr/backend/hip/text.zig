//! LightOnOCR's Qwen3 decoder on HIP: shared BF16 projections, independent causal caches and Torch rounding.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const config = @import("../../config.zig");
const weights = @import("../../weights.zig");
const pointwise = @import("ops.zig");
const vision = @import("vision.zig");
const CancelCheck = @import("lanes").stream.CancelCheck;
const cancels = @import("hip_serve").cancel;
const Allocator = std.mem.Allocator;
const Buffer = hip.DeviceBuffer;

pub const max_rows = 128;
pub const cache_slack = 16;

/// A slot's flattened chain, starting at its committed cache length.
pub const Span = struct { slot: u32, first: u32, rows: u32, committed: u32 };
/// Projected image rows replacing a range of token embeddings within one span.
pub const ImageRows = struct { first: u32, rows: u32, features: u64 };

const Meta = extern struct { ids: [max_rows]u32, pos: [max_rows]i32 };

pub const Scratch = struct {
    meta: usize,
    inv: usize,
    h: usize,
    normed: usize,
    branch: usize,
    qraw: usize,
    kraw: usize,
    vraw: usize,
    q: usize,
    k: usize,
    v: usize,
    qwide: usize,
    awide: usize,
    attention: usize,
    gate: usize,
    up: usize,
    act: usize,
    logits: usize,
    bytes: usize,

    pub fn init(c: config.Config, context: usize) !Scratch {
        try geometry(c, context);
        var at: usize = 0;
        const h = try product(&.{ max_rows, c.hidden, 2 });
        const qw = try product(&.{ max_rows, c.queryWidth(), 2 });
        const kw = try product(&.{ max_rows, c.kvWidth(), 2 });
        const mid = try product(&.{ max_rows, c.intermediate, 2 });
        return .{
            .meta = try take(&at, @sizeOf(Meta)),
            .inv = try take(&at, c.head_dim / 2 * 4),
            .h = try take(&at, h),
            .normed = try take(&at, h),
            .branch = try take(&at, h),
            .qraw = try take(&at, qw),
            .kraw = try take(&at, kw),
            .vraw = try take(&at, kw),
            .q = try take(&at, qw),
            .k = try take(&at, kw),
            .v = try take(&at, kw),
            .qwide = try take(&at, try std.math.mul(usize, qw, 2)),
            .awide = try take(&at, try std.math.mul(usize, qw, 2)),
            .attention = try take(&at, qw),
            .gate = try take(&at, mid),
            .up = try take(&at, mid),
            .act = try take(&at, mid),
            .logits = try take(&at, try product(&.{ max_rows, c.vocab, 2 })),
            .bytes = at,
        };
    }

    fn take(at: *usize, bytes: usize) !usize {
        const start = at.*;
        const end = try std.math.add(usize, start, bytes);
        at.* = try aligned(end);
        return start;
    }
};

fn aligned(n: usize) !usize {
    return (try std.math.add(usize, n, 255)) & ~@as(usize, 255);
}

fn product(dims: []const usize) !usize {
    var n: usize = 1;
    for (dims) |dim| n = try std.math.mul(usize, n, dim);
    return n;
}

fn geometry(c: config.Config, context: usize) !void {
    if (context == 0 or context > @as(usize, c.context) + cache_slack or c.layers == 0 or
        c.layers > 28 or c.hidden == 0 or c.intermediate == 0 or c.vocab == 0 or
        c.heads == 0 or c.kv_heads == 0 or c.heads % c.kv_heads != 0 or c.head_dim != 128)
        return error.TextShape;
}

/// One slot's K/V histories over every layer, including the caller's physical verification slack.
pub fn streamBytes(c: config.Config, context: usize) !usize {
    try geometry(c, context);
    return product(&.{ c.layers, 2, context, c.kvWidth(), 2 });
}

pub fn scratchBytes(c: config.Config, context: usize) !usize {
    return (try Scratch.init(c, context)).bytes;
}

pub fn hostBytes() usize {
    return @sizeOf(Meta);
}

pub const Text = struct {
    gpa: Allocator,
    r: *const hip.Runtime,
    s: hip.Stream,
    c: config.Config,
    context: usize,
    slots: usize,
    shared: *const hip.Launcher,
    point: pointwise.Kernels,
    layers: []weights.TextLayer,
    weight_base: u64,
    embedding: u64,
    final_norm: u64,
    kv: Buffer,
    arena: Buffer,
    staged: hip.HostBuffer,
    copied: hip.abi.Event,
    staged_pending: bool = false,
    x: Scratch,
    logits: u64,

    pub fn init(gpa: Allocator, r: *const hip.Runtime, s: hip.Stream, layout: *const weights.Layout, weight_base: u64, context: usize, slots: usize, shared: *const hip.Launcher) !Text {
        const c = layout.c;
        const x = try Scratch.init(c, context);
        if (r != s.r or slots == 0 or slots > max_rows or layout.text.len != c.layers or weight_base == 0 or
            weight_base % 16 != 0) return error.TextShape;
        const layers = try gpa.dupe(weights.TextLayer, layout.text);
        errdefer gpa.free(layers);
        var point = try pointwise.Kernels.load(r, hip.kernels.images[@backingInt(hip.kernels.Group.lighton_ocr)], shared);
        errdefer point.unload();
        var kv = try Buffer.alloc(r, try std.math.mul(usize, try streamBytes(c, context), slots));
        errdefer kv.free();
        var arena = try Buffer.alloc(r, x.bytes);
        errdefer arena.free();
        var staged = try hip.HostBuffer.alloc(r, hostBytes());
        errdefer staged.free();
        var copied: hip.abi.Event = null;
        try hip.runtime.check(r.api.hipEventCreateWithFlags(&copied, hip.abi.event_disable_timing));
        if (copied == null) return error.Invalid;
        errdefer _ = r.api.hipEventDestroy(copied);
        var inv: [64]f32 = undefined;
        inverseFrequencies(&inv, @floatCast(c.theta));
        try arena.upload(x.inv, std.mem.asBytes(&inv));
        return .{ .gpa = gpa, .r = r, .s = s, .c = c, .context = context, .slots = slots, .shared = shared, .point = point, .layers = layers, .weight_base = weight_base, .embedding = try weightAddress(weight_base, layout.embedding), .final_norm = layout.final_norm, .kv = kv, .arena = arena, .staged = staged, .copied = copied, .x = x, .logits = arena.base() + x.logits };
    }

    pub fn deinit(t: *Text) void {
        t.s.synchronize() catch @panic("LightOnOCR HIP decoder work did not drain before release");
        _ = t.r.api.hipEventDestroy(t.copied);
        t.staged.free();
        t.arena.free();
        t.kv.free();
        t.point.unload();
        t.gpa.free(t.layers);
        t.* = undefined;
    }

    /// The slot/layer's [kv_heads,context,head_dim] BF16 keys; snapshot copies take a prefix of each head.
    pub fn key(t: *const Text, slot: usize, layer: usize) u64 {
        std.debug.assert(slot < t.slots and layer < t.c.layers);
        return t.kv.base() + (layer * 2 * t.slots + slot) * t.context * t.c.kvWidth() * 2;
    }

    pub fn value(t: *const Text, slot: usize, layer: usize) u64 {
        return t.key(slot, layer) + t.slots * t.context * t.c.kvWidth() * 2;
    }

    /// Prepare the exact BF16 decoder input; capture callers may synchronize and read the returned device rows.
    pub fn embed(t: *Text, ids: []const u32, spans: []const Span, images: []const ImageRows) !u64 {
        try validate(t.c, t.context, t.slots, ids, spans, images);
        if (t.staged_pending) try hip.runtime.check(t.r.api.hipEventSynchronize(t.copied));
        errdefer t.s.synchronize() catch @panic("LightOnOCR HIP failed forward did not drain");
        const meta: *Meta = @ptrCast(@alignCast(t.staged.bytes.ptr));
        prepare(meta, ids, spans);
        try t.arena.uploadAsync(t.x.meta, t.staged, t.s);
        try hip.runtime.check(t.r.api.hipEventRecord(t.copied, t.s.handle));
        t.staged_pending = true;
        const n = ids.len;
        const c = t.c;
        try t.shared.tf_embed_dense(@ptrFromInt(t.embedding), @backingInt(hip.ops.Kind.f32), @ptrFromInt(t.addr(t.x.meta + @offsetOf(Meta, "ids"))), @intCast(n), @intCast(c.hidden), @ptrFromInt(t.addr(t.x.h)), @backingInt(hip.ops.Kind.bf16), t.s.handle);
        for (images) |image| try t.arena.copyFromAsync(t.x.h + @as(usize, image.first) * c.hidden * 2, image.features, @as(usize, image.rows) * c.hidden * 2, t.s);
        return t.addr(t.x.h);
    }

    /// Shared projections over every flattened stream; only attention addressing differs between spans.
    pub fn forward(t: *Text, ids: []const u32, spans: []const Span, images: []const ImageRows) !void {
        return t.forwardCancelable(ids, spans, images, null);
    }

    pub fn forwardCancelable(t: *Text, ids: []const u32, spans: []const Span, images: []const ImageRows, cancel: ?CancelCheck) !void {
        try cancels.poll(cancel);
        _ = try t.embed(ids, spans, images);
        errdefer t.s.synchronize() catch @panic("LightOnOCR HIP failed forward did not drain");
        const n = ids.len;
        const c = t.c;
        var no_alloc: hip.Arena = .{ .buf = null, .base = 0, .len = 0 };
        const o: hip.ops.Ops = .{ .l = t.shared, .bf16 = true, .stream = t.s.handle, .arena = &no_alloc, .prefill = true };
        for (t.layers, 0..) |layer, i| {
            try cancels.drain(t.s, cancel);
            try t.norm(t.x.h, t.x.normed, layer.input_norm, n);
            try t.project(o, t.x.normed, layer.q, t.x.qraw, n, c.queryWidth(), c.hidden);
            try t.project(o, t.x.normed, layer.k, t.x.kraw, n, c.kvWidth(), c.hidden);
            try t.project(o, t.x.normed, layer.v, t.x.vraw, n, c.kvWidth(), c.hidden);
            try t.point.textQk(t.s, t.addr(t.x.qraw), t.addr(t.x.kraw), t.addr(t.x.vraw), try t.weight(layer.q_norm), try t.weight(layer.k_norm), t.addr(t.x.inv), t.addr(t.x.meta + @offsetOf(Meta, "pos")), t.addr(t.x.q), t.addr(t.x.k), t.addr(t.x.v), n, .{ .h = c.heads, .hk = c.kv_heads, .dim = c.head_dim, .eps = c.eps });
            for (spans) |span| {
                const first: usize = span.first;
                const rows: usize = span.rows;
                const qfirst = first * c.queryWidth();
                const kfirst = first * c.kvWidth();
                try o.kvWrite(.{ .ptr = t.addr(t.x.k) + kfirst * 2, .kind = .bf16 }, t.key(span.slot, i), rows, c.kv_heads, c.head_dim, t.context, span.committed);
                try o.kvWrite(.{ .ptr = t.addr(t.x.v) + kfirst * 2, .kind = .bf16 }, t.value(span.slot, i), rows, c.kv_heads, c.head_dim, t.context, span.committed);
                try t.point.rowsToHeads(t.s, t.addr(t.x.q) + qfirst * 2, t.addr(t.x.qwide) + qfirst * 4, rows, c.heads, c.head_dim);
                const cache: hip.ops.Ops.Cache = .{ .k = t.key(span.slot, i), .v = t.value(span.slot, i), .kind = .bf16, .kv_heads = c.kv_heads, .total = t.context, .d = c.head_dim };
                const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(c.head_dim), -0.5));
                try o.causalPrefill(t.addr(t.x.qwide) + qfirst * 4, cache, t.addr(t.x.awide) + qfirst * 4, rows, @as(usize, span.committed) + rows, c.heads, scale, span.committed);
                try t.point.headsToRows(t.s, t.addr(t.x.awide) + qfirst * 4, t.addr(t.x.attention) + qfirst * 2, rows, c.heads, c.head_dim);
            }
            try t.project(o, t.x.attention, layer.o, t.x.branch, n, c.hidden, c.queryWidth());
            try t.point.residual(t.s, t.addr(t.x.branch), t.addr(t.x.h), n, c.hidden);
            try t.norm(t.x.h, t.x.normed, layer.post_norm, n);
            try t.project(o, t.x.normed, layer.gate, t.x.gate, n, c.intermediate, c.hidden);
            try t.project(o, t.x.normed, layer.up, t.x.up, n, c.intermediate, c.hidden);
            try t.point.torchSwiGlu(t.s, t.addr(t.x.gate), t.addr(t.x.up), t.addr(t.x.act), n, c.intermediate);
            try t.project(o, t.x.act, layer.down, t.x.branch, n, c.hidden, c.intermediate);
            try t.point.residual(t.s, t.addr(t.x.branch), t.addr(t.x.h), n, c.hidden);
        }
        try cancels.drain(t.s, cancel);
        try t.norm(t.x.h, t.x.normed, t.final_norm, n);
        try o.denseRows(.{ .ptr = t.addr(t.x.normed), .kind = .bf16 }, t.embedding, .{ .ptr = t.logits, .kind = .bf16 }, n, c.vocab, c.hidden);
        try cancels.drain(t.s, cancel);
    }

    fn addr(t: *const Text, offset: usize) u64 {
        return t.arena.base() + offset;
    }

    fn weight(t: *const Text, offset: u64) !u64 {
        return weightAddress(t.weight_base, offset);
    }

    fn norm(t: *const Text, x: usize, y: usize, offset: u64, rows: usize) !void {
        return t.point.rms(t.s, t.addr(x), try t.weight(offset), t.addr(y), rows, t.c.hidden, t.c.eps);
    }

    fn project(t: *const Text, o: hip.ops.Ops, x: usize, offset: u64, y: usize, rows: usize, n: usize, k: usize) !void {
        return o.denseRows(.{ .ptr = t.addr(x), .kind = .bf16 }, try t.weight(offset), .{ .ptr = t.addr(y), .kind = .bf16 }, rows, n, k);
    }
};

fn weightAddress(base: u64, offset: u64) !u64 {
    return std.math.add(u64, base, try std.math.mul(u64, offset, 2));
}

fn inverseFrequencies(out: *[64]f32, theta: f32) void {
    for (out, 0..) |*value, j| {
        const exponent: f32 = @as(f32, @floatFromInt(2 * j)) / 128;
        value.* = 1.0 / std.math.pow(f32, theta, exponent);
    }
}

/// Reject a complete frame before any metadata transfer, cache write or kernel launch.
pub fn validate(c: config.Config, context: usize, slots: usize, ids: []const u32, spans: []const Span, images: []const ImageRows) !void {
    try geometry(c, context);
    if (slots == 0 or slots > max_rows or ids.len == 0 or ids.len > max_rows or spans.len == 0 or
        spans.len > slots) return error.TextShape;
    var at: usize = 0;
    for (spans, 0..) |span, i| {
        if (span.slot >= slots or span.first != at or span.rows == 0 or span.rows > ids.len - at or
            span.committed > context or span.rows > context - span.committed) return error.TextShape;
        for (spans[0..i]) |prior| if (prior.slot == span.slot) return error.DuplicateTextSlot;
        at += span.rows;
    }
    if (at != ids.len) return error.TextShape;
    for (ids) |id| if (id >= c.vocab) return error.TextToken;
    for (images, 0..) |image, i| {
        if (image.first > ids.len or image.rows == 0 or image.rows > ids.len - image.first or
            image.features == 0 or image.features % 16 != 0) return error.ImageRows;
        _ = std.math.add(u64, image.features, @as(u64, image.rows) * c.hidden * 2) catch return error.ImageRows;
        var contained = false;
        for (spans) |span| if (image.first >= span.first and @as(u64, image.first) + image.rows <= @as(u64, span.first) + span.rows) {
            contained = true;
            break;
        };
        if (!contained) return error.ImageRows;
        for (images[0..i]) |prior| if (image.first < @as(u64, prior.first) + prior.rows and
            prior.first < @as(u64, image.first) + image.rows) return error.ImageRows;
    }
}

fn prepare(m: *Meta, ids: []const u32, spans: []const Span) void {
    @memset(std.mem.asBytes(m), 0);
    @memcpy(m.ids[0..ids.len], ids);
    for (spans) |span| {
        for (0..span.rows) |j| {
            m.pos[span.first + j] = @intCast(span.committed + j);
        }
    }
}

test {
    _ = @import("text_test.zig");
}
