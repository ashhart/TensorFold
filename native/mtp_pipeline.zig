//! Speculate on every target row before reading its sampled token. Retain only the
//! accepted MTP prefix and reuse its last state/draw as the next chain's first step.
const mx = @import("mlx.zig");
const Sampling = @import("sampling.zig").Sampling;
const gpu = @import("gpu_sampling.zig");

pub fn Pipeline(comptime M: type) type {
    return struct {
        const Self = @This();
        cache: M.DraftCache = .{},
        hidden: mx.Array = mx.empty,
        first: mx.Array = mx.empty,
        pub fn deinit(p: *Self) void {
            p.cache.deinit();
            mx.free(p.hidden);
            mx.free(p.first);
            p.* = .{};
        }
        pub fn prepare(m: *M, s: *mx.Scope, cache: M.DraftCache, hidden: mx.Array, token: i32, position: i32, settings: Sampling) !Self {
            var out = Self{ .cache = try cache.clone() };
            errdefer out.deinit();
            const h = try m.draftStepArray(s, hidden, try s.ints(&.{token}), &out.cache, true);
            out.hidden = try mx.retain(h);
            out.first = try mx.retain(try gpu.sample(&m.kernels, s, try m.draftHead(s, h), &.{position + 1}, settings, m.weights.arrays.get("draft_ids")));
            return out;
        }
        pub fn propose(p: *Self, m: *M, s: *mx.Scope, budget: usize, position: i32, settings: Sampling, queued: bool) !mx.Array {
            if (budget == 0 or budget > 15 or p.first.ctx == null) return error.InvalidDraftBudget;
            var cache = try p.cache.clone();
            defer cache.deinit();
            var hidden = p.hidden;
            var proposals: [15]mx.Array = undefined;
            proposals[0] = p.first;
            if (!queued) try mx.eval(proposals[0]);
            for (1..budget) |j| {
                hidden = try m.draftStepArray(s, hidden, proposals[j - 1], &cache, queued);
                proposals[j] = try gpu.sample(&m.kernels, s, try m.draftHead(s, hidden), &.{position + @as(i32, @intCast(j)) + 1}, settings, m.weights.arrays.get("draft_ids"));
                if (!queued) try mx.eval(proposals[j]);
            }
            return s.cat(proposals[0..budget], 0);
        }
        pub const Speculation = struct {
            cache: M.DraftCache,
            hidden: mx.Array,
            firsts: mx.Array,
            rows: usize,
            pub fn deinit(p: *Speculation) void {
                p.cache.deinit();
            }
        };
        pub fn speculate(p: *Self, m: *M, s: *mx.Scope, hidden: mx.Array, tokens: mx.Array, position: i32, settings: Sampling) !Speculation {
            var cache = try p.cache.clone();
            errdefer cache.deinit();
            const rows: usize = @intCast(mx.dim(hidden, 0));
            if (rows == 0 or rows > 16) return error.InvalidDraftRows;
            const h = try m.draftStepArray(s, hidden, tokens, &cache, true);
            var positions: [16]i32 = undefined;
            for (0..rows) |j| positions[j] = position + @as(i32, @intCast(j)) + 2;
            const firsts = try gpu.sample(&m.kernels, s, try m.draftHead(s, h), positions[0..rows], settings, m.weights.arrays.get("draft_ids"));
            return .{ .cache = cache, .hidden = h, .firsts = firsts, .rows = rows };
        }
        pub fn settle(p: *Self, s: *mx.Scope, spec: Speculation, keep: usize) !void {
            if (keep == 0 or keep > spec.rows) return error.InvalidCommit;
            var next = Self{ .cache = try M.draftPrefix(s, spec.cache, spec.rows, keep) };
            errdefer next.deinit();
            const end: i32 = @intCast(keep);
            next.hidden = try mx.retain(try s.slice(spec.hidden, 0, end - 1, end));
            next.first = try mx.retain(try s.slice(spec.firsts, 0, end - 1, end));
            p.deinit();
            p.* = next;
        }
    };
}
