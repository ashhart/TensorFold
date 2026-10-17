//! Structured output on shared rounds: verify rows masked to the grammar's tokens, the first token drawn among them.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const fz = @import("replay.zig");
const be = lanes.backend;
const Backend = @import("backend.zig").Backend;

/// A row's allowed-token bits over the head's logits: token t is bit t % 32 of word t / 32, as lanes/grammar.zig.
pub const words = (fz.VOCAB + 31) / 32;

pub const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// a token whose bit is off gets -inf, so the argmax and tf_sample_full never draw it
    \\kernel void fz_mask_rows(device bfloat* logits [[buffer(0)]], device const uint* bits [[buffer(1)]],
    \\    constant uint2& c [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    \\  if (g.x >= c.x) return;
    \\  if (((bits[g.y * c.y + g.x / 32] >> (g.x % 32)) & 1u) == 0) logits[g.y * c.x + g.x] = bfloat(-INFINITY);
    \\}
;

/// The round's mask rows on the GPU and each session's first row, kept from its prompt pass.
pub const Masks = struct {
    pipe: mtl.Pipeline,
    bits: fz.Buf, // [MAXR, words]
    first: []u16 = &.{}, // [sessions, VOCAB]: a grammar stream's last prompt row, read when its first token is drawn

    pub fn init(dev: mtl.Device) !Masks {
        const lib = try mtl.Library.fromSource(dev, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(dev, lib, "fz_mask_rows", false);
        errdefer pipe.deinit();
        return .{ .pipe = pipe, .bits = .{ .b = try dev.buffer(fz.MAXR * words * 4, fz.opts) } };
    }

    pub fn deinit(m: *Masks, gpa: std.mem.Allocator) void {
        gpa.free(m.first);
        m.bits.b.deinit();
        m.pipe.deinit();
    }

    /// The round's rows' bits (every token for a window without a grammar); false when no window has one.
    pub fn fill(m: *Masks, windows: []const be.Window, starts: []const usize) bool {
        var any = false;
        for (windows) |w| any = any or w.masks.len > 0;
        if (!any) return false;
        const all = m.bits.b.slice(u32, fz.MAXR * words);
        for (windows, starts) |w, row| for (0..w.rows()) |r| {
            const dst = all[(row + r) * words ..][0..words];
            if (w.masks.len == 0) {
                @memset(dst, 0xffff_ffff);
                continue;
            }
            const n = w.masks.len / w.rows(); // the compiler's words: its vocabulary may be under the head's
            const src = w.masks[r * n ..][0..n];
            const k = @min(n, words);
            @memcpy(dst[0..k], src[0..k]);
            @memset(dst[k..], 0);
        };
        return true;
    }

    /// fz_mask_rows over `rows` rows of `logits`, before the argmax (the caller's barrier after it).
    pub fn encode(m: *const Masks, enc: mtl.ComputeEncoder, logits: fz.Buf, rows: usize) void {
        const c = [2]u32{ fz.VOCAB, words };
        enc.setPipeline(m.pipe);
        enc.setBuffer(logits.b, logits.off, 0);
        enc.setBuffer(m.bits.b, m.bits.off, 1);
        enc.setBytes(std.mem.asBytes(&c), 2);
        enc.dispatchThreads(mtl.Size.of(fz.VOCAB, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// Keeps session `slot`'s last prompt row (`logits`' first) for its first token's draw.
    pub fn keepFirst(m: *Masks, gpa: std.mem.Allocator, slot: usize, sessions: usize, logits: fz.Buf) !void {
        if (m.first.len == 0) m.first = try gpa.alloc(u16, sessions * fz.VOCAB);
        const row: [*]const u16 = @ptrCast(@alignCast(logits.b.contents() + logits.off));
        @memcpy(m.first[slot * fz.VOCAB ..][0..fz.VOCAB], row[0..fz.VOCAB]);
    }
};

/// The first token among `mask`'s from a bf16 row: the argmax (lowest id on ties, as fz_argmax) or tf_sample_full's.
pub fn draw(gpa: std.mem.Allocator, row: []const u16, mask: []const u32, sampling: ?lanes.Sampling, position: u32) !u32 {
    const v = try gpa.alloc(f32, row.len);
    defer gpa.free(v);
    var any = false;
    for (v, row, 0..) |*x, h, i| {
        const on = lanes.grammar.allows(mask, i);
        x.* = if (on) @bitCast(@as(u32, h) << 16) else -std.math.inf(f32);
        any = any or on;
    }
    if (!any) return error.NoAllowedToken;
    if (sampling) |s| return lanes.gpu_full.sample(gpa, v, s, position, null);
    var at: u32 = 0;
    for (v, 0..) |x, i| if (x > v[at]) {
        at = @intCast(i);
    };
    return at;
}

/// Backend.first_masked: the first token from the row the prompt pass kept, among the grammar's.
pub fn firstMasked(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
    const b: *Backend = @ptrCast(@alignCast(ptr));
    const slot = b.by.get(stream) orelse return error.UnknownStream;
    if (position != b.sessions[slot].m.pos) return error.PositionMismatch;
    if (b.grammar.first.len == 0) return error.FirstRowNotKept;
    const p = @import("batch_sample.zig").active(stream.sampling);
    return try draw(b.gpa, b.grammar.first[slot * fz.VOCAB ..][0..fz.VOCAB], mask, p, @intCast(position));
}

/// Backend.held: the head's held drafts, host tokens already once its drafts are in (draft waits for them).
pub fn held(ptr: *anyopaque, stream: *lanes.Stream, out: []u32) anyerror!void {
    const b: *Backend = @ptrCast(@alignCast(ptr));
    const s = &b.sessions[b.by.get(stream) orelse return error.UnknownStream];
    if (out.len > s.held_n) return error.WindowOutOfStep;
    @memcpy(out, s.held[0..out.len]);
}

test "a masked row's draw on the GPU is the host's: fz_mask_rows then fz_argmax or tf_sample_full" {
    const ks = @import("kernel_sources");
    const gpa = std.testing.allocator;
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var m = try Masks.init(dev);
    defer m.deinit(gpa);
    const slib = try mtl.Library.fromSource(dev, ks.nemotron_sample, mtl.CompileOptions.mlx());
    defer slib.deinit();
    const draw_pipe = try mtl.Pipeline.init(dev, slib, "tf_sample_full", false);
    defer draw_pipe.deinit();
    const glib = try mtl.Library.fromSource(dev, fz.glue_source, mtl.CompileOptions.mlx());
    defer glib.deinit();
    const argmax_pipe = try mtl.Pipeline.init(dev, glib, "fz_argmax", false);
    defer argmax_pipe.deinit();
    const q = try dev.queue();
    defer q.deinit();
    const rows = 4;
    const logits = try dev.buffer(rows * fz.VOCAB * 2, fz.opts);
    defer logits.deinit();
    const pos = try dev.buffer(4 * fz.MAXR, fz.opts);
    defer pos.deinit();
    const picks = try dev.buffer(4 * fz.MAXR, fz.opts);
    defer picks.deinit();
    var rng = std.Random.DefaultPrng.init(7);
    const host = try gpa.alloc(u16, rows * fz.VOCAB);
    defer gpa.free(host);
    for (host) |*w| { // spread logits, a few strong, as the head's bf16 rows
        const x: f32 = rng.random().floatNorm(f32) * 2 + if (rng.random().uintLessThan(u32, 5000) == 0) @as(f32, 12) else 0;
        w.* = @truncate(@as(u32, @bitCast(x)) >> 16);
    }
    // a grammar's rows: a few tokens each (the strongest left out of one), from a compiler whose vocabulary is smaller
    const n = words - 3;
    const masks = try gpa.alloc(u32, rows * n);
    defer gpa.free(masks);
    @memset(masks, 0);
    for (0..rows) |r| for (0..40) |_| {
        const t = rng.random().uintLessThan(u32, n * 32);
        masks[r * n + t / 32] |= @as(u32, 1) << @intCast(t % 32);
    };
    const w: be.Window = .{ .stream = undefined, .pending = 0, .held = 0, .tokens = &.{ 1, 2, 3 }, .parents = null, .positions = &.{}, .masks = masks };
    const settings = [_]?lanes.Sampling{ null, .{ .seed = 11, .temperature = 0.8 }, .{ .seed = 12, .temperature = 1.0, .top_k = 20, .top_p = 0.9 } };
    for (settings) |s| {
        @memcpy(logits.slice(u16, rows * fz.VOCAB), host);
        try std.testing.expect(m.fill(&.{w}, &.{0}));
        for (pos.slice(u32, rows), 0..) |*p, i| p.* = @intCast(100 + i);
        const cb = q.commandBuffer();
        const enc = cb.compute(.serial);
        m.encode(enc, .{ .b = logits }, rows);
        if (s) |st| @import("sample.zig").draw(enc, draw_pipe, .{ .b = logits }, .{ .b = pos }, .{ .b = picks }, rows, fz.VOCAB, st) else {
            enc.setPipeline(argmax_pipe);
            const v: u32 = fz.VOCAB;
            enc.setBuffer(logits, 0, 0);
            enc.setBuffer(picks, 0, 1);
            enc.setBytes(std.mem.asBytes(&v), 2);
            enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
        }
        enc.end();
        cb.commit();
        cb.wait();
        for (0..rows) |r| {
            const got = picks.slice(u32, rows)[r];
            try std.testing.expect(lanes.grammar.allows(masks[r * n ..][0..n], got));
            try std.testing.expectEqual(try draw(gpa, host[r * fz.VOCAB ..][0..fz.VOCAB], masks[r * n ..][0..n], s, @intCast(100 + r)), got);
        }
    }
    var none: [words]u32 = @splat(0);
    try std.testing.expectError(error.NoAllowedToken, draw(gpa, host[0..fz.VOCAB], &none, null, 0));
}
