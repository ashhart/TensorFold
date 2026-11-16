//! Prompt-only Qwen3 on the dense Metal forward. Inputs run independently, including their chunk boundaries.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const lm = @import("../../llama/model.zig");
const st = @import("../../llama/state.zig");
const forward = @import("../../llama/forward.zig");
pub const max_inputs = 128;
pub const least_window = 2048;
pub fn checkBatch(tokens: []const u32, trunk: u32, ends: []const u32, out_len: usize, window: usize, vocab: usize, hidden: usize) !void {
    if (trunk != 0 or tokens.len == 0 or tokens.len > window or ends.len == 0 or ends.len > max_inputs or
        ends[ends.len - 1] != tokens.len or out_len != ends.len * hidden) return error.InvalidBatch;
    var begin: u32 = 0;
    for (ends) |end| {
        if (end <= begin) return error.InvalidBatch;
        begin = end;
    }
    for (tokens) |id| if (id >= vocab) return error.InvalidBatch;
}
pub const Engine = struct {
    gpa: std.mem.Allocator,
    m: struct { spec: cfg.Spec },
    model: *lm.Model,
    scratch: st.Scratch = undefined,
    prompt: st.Scratch = undefined,
    cache: st.Cache = undefined,
    window: usize = 0,
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Engine {
        const spec = try cfg.read(gpa, io, dir);
        const model = try lm.Model.loadPrepared(gpa, io, dir, try cfg.dense(spec), null, @import("weights.zig").load);
        errdefer model.deinit();
        const lib = try mtl.Library.fromSource(model.device, ks.qwen3_heads, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        model.head_norm = try mtl.Pipeline.init(model.device, lib, "q3_head_norm", false);
        const e = try gpa.create(Engine);
        e.* = .{ .gpa = gpa, .m = .{ .spec = spec }, .model = model };
        return e;
    }
    pub fn scratchBytes(e: *Engine, window: usize) anyerror!usize {
        return st.cacheBytes(&e.model.config, window) + st.scratchBytes(&e.model.config, st.batch_rows) + st.scratchBytes(&e.model.config, st.prompt_rows);
    }
    pub fn size(e: *Engine, window: usize) !void {
        if (e.window != 0 or window == 0 or window > e.m.spec.max_positions) return error.BadWindow;
        e.scratch = try st.Scratch.init(e.model.device, &e.model.config, st.batch_rows);
        errdefer e.scratch.deinit();
        e.prompt = try st.Scratch.init(e.model.device, &e.model.config, st.prompt_rows);
        errdefer e.prompt.deinit();
        e.cache = try st.Cache.init(e.model.device, &e.model.config, window);
        e.window = window;
    }
    pub fn run(e: *Engine, tokens: []const u32, trunk: u32, ends: []const u32, out: []f32) !void {
        try checkBatch(tokens, trunk, ends, out.len, e.window, e.m.spec.vocab, e.m.spec.hidden);
        var begin: usize = 0;
        for (ends, 0..) |end, row| {
            e.cache.len = 0;
            var at = begin;
            while (at < end) {
                const n = @min(st.prompt_rows, end - at);
                const scratch = if (n <= st.batch_rows) &e.scratch else &e.prompt;
                try forward.runHidden(e.model, scratch, &e.cache, tokens[at..][0..n]);
                e.cache.len += n;
                at += n;
                if (at == end) {
                    for (scratch.x.slice(u16, e.m.spec.hidden), out[row * e.m.spec.hidden ..][0..e.m.spec.hidden]) |bits, *value|
                        value.* = @bitCast(@as(u32, bits) << 16);
                }
            }
            begin = end;
        }
    }
    pub fn deinit(e: *Engine) void {
        if (e.window > 0) {
            e.cache.deinit();
            e.prompt.deinit();
            e.scratch.deinit();
        }
        e.model.deinit();
        e.gpa.destroy(e);
    }
};
test "embedding batches refuse bad bounds before any Metal work" {
    try checkBatch(&.{ 1, 2, 3 }, 0, &.{ 1, 3 }, 8, 3, 4, 4);
    const bad = error.InvalidBatch;
    try std.testing.expectError(bad, checkBatch(&.{ 1, 2, 3 }, 1, &.{ 1, 3 }, 8, 3, 4, 4));
    try std.testing.expectError(bad, checkBatch(&.{ 1, 2, 3 }, 0, &.{ 2, 2, 3 }, 12, 3, 4, 4));
    try std.testing.expectError(bad, checkBatch(&.{ 1, 4, 3 }, 0, &.{ 1, 3 }, 8, 3, 4, 4));
    try std.testing.expectError(bad, checkBatch(&.{ 1, 2, 3 }, 0, &.{ 1, 3 }, 8, 2, 4, 4));
    try std.testing.expectError(bad, checkBatch(&.{ 1, 2, 3 }, 0, &.{ 1, 2 }, 8, 3, 4, 4));
    try std.testing.expectError(bad, checkBatch(&.{ 1, 2, 3 }, 0, &.{ 1, 3 }, 7, 3, 4, 4));
    try std.testing.expectError(bad, checkBatch(&.{}, 0, &.{}, 0, 3, 4, 4));
}
