//! Gemma 4 E4B as an engine of the XPU lane host (native/xpu_lane.zig): one stream, card argmax or top-k, host draws.
const std = @import("std");
const fwd = @import("xpu_forward.zig");
const cfg = @import("xpu_config.zig");

pub const Runner = fwd.Runner;
pub const config = cfg;

pub const Engine = struct {
    r: *fwd.Runner,
    /// The lane backend reads the position here (the runner's, kept in step).
    m: struct { pos: u32 = 0 } = .{},
    max_len: usize,

    pub fn init(dir: []const u8, context: u32) !*Engine {
        const e = try std.heap.page_allocator.create(Engine);
        errdefer std.heap.page_allocator.destroy(e);
        e.* = .{ .r = try fwd.Runner.init(dir, context), .max_len = context };
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.r.deinit();
        std.heap.page_allocator.destroy(e);
    }

    pub fn deviceBytes(e: *const Engine) u64 {
        return e.r.d.allocated;
    }

    pub fn vocab(_: *const Engine) usize {
        return cfg.vocab;
    }

    pub fn reset(e: *Engine) !void {
        e.r.reset();
        e.m.pos = 0;
    }

    /// One token in; with `logits`, the next one's logits and greedy token kept.
    pub fn feed(e: *Engine, token: u32, logits: bool) !void {
        try e.r.feed(&.{token}, logits);
        e.m.pos = e.r.pos;
    }

    pub fn sync(e: *Engine) !void {
        try e.r.d.sync();
    }

    pub fn argmax(e: *Engine) !u32 {
        return e.r.last;
    }

    /// The kept row's logits after the softcap, fp32.
    pub fn fetchLogits(e: *Engine, out: []f32) !void {
        @memcpy(out, try e.r.logits(0));
    }

    /// The `k` (<= 64) largest logits of the kept row, chosen on the card; softcapped, descending, ties to lower id.
    pub fn candidates(e: *Engine, k: u32, vals: []f32, ids: []u32) !void {
        try e.r.topK(0, k, vals, ids);
    }

    /// Prompt tokens in windows (the runner cuts them in 16 rows), no logits.
    pub fn prefillWindows(e: *Engine, ids: []const u32, _: usize) !void {
        try e.r.feed(ids, false);
        e.m.pos = e.r.pos;
        try e.r.d.sync();
    }
};
