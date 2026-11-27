//! The MoE pass's native experts: NVFP4 or block-FP8 pairs (whole experts a rank at --tp), shared on rank 0, rank sum.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const mixed = @import("cuda_mixed_experts.zig");
const nvfp4_tile = @import("cuda_nvfp4_tile.zig");
const zs = @import("cuda_zero_slice.zig");
const shared = @import("cuda_batch_side.zig");
const exchange = @import("cuda_moe_exchange.zig");
const Kernels = @import("cuda_batch_kernels.zig").Kernels;

const dims = wt.dims;
const experts_n = wt.experts_n;

/// moeBlockKeep's native layer: each routed pair's fp32 row into y [pairs, dims], the shared slot's on rank 0.
pub fn run(self: anytype, native: *const mixed.Owned, n: usize, pairs: usize, plan: cuda.grouped.Plan) !void {
    const s = self.s;
    if (native.tfq3) |*tl| {
        const tk = if (self.k.tfq3) |*t| t else return error.UnsupportedQuantization;
        const y_bytes = exchange.clearedRows(native.world, n) * s.slots * dims * 4;
        if (zs.clearsY(native.world, native.rank)) try self.stream.d.check(self.stream.d.api.cuMemsetD8Async(s.y, 0, y_bytes, self.stream.handle), "cuMemsetD8Async");
        const side = try shared.fork(self, native, n);
        const sl: u32 = @intCast(s.slots);
        try tk.moe(self.stream, tl.layer, s.mixed, dims, s.picks, sl, s.y, sl, n, sl - 1, 0.0, 1, true);
        if (side != null) try shared.join(self, side, native, n) else try @import("cuda_shared_split.zig").run(self, native, n);
        return;
    }
    const owned = &native.experts;
    // The last id is skipped by routed kernels. At one rank the opt-in leaves it out of the routed plan entirely.
    const routed_only = self.k.forms.routed_plan and native.world == 1;
    const e = owned.count + @as(usize, @intFromBool(!routed_only));
    const item_pairs = pairs - @as(usize, @intFromBool(routed_only)) * n;
    const skip: i32 = @intCast(owned.count);
    const picks = if (native.world == 1) s.picks else try ownPicks(self, s.picks, s.picks_ep, pairs, owned.first, owned.count);
    const y_bytes = exchange.clearedRows(native.world, n) * s.slots * dims * 4;
    if (zs.clearsY(native.world, native.rank)) try self.stream.d.check(self.stream.d.api.cuMemsetD8Async(s.y, 0, y_bytes, self.stream.handle), "cuMemsetD8Async");
    const side = try shared.fork(self, native, n);
    switch (owned.format) {
        .nvfp4 => {
            const pt = nvfp4_tile.size(self.k, n);
            // Keep the interleaved 11-slot scan: only E and the routed launch bound change.
            try self.k.router.route(self.stream, picks, pairs, e, pt, plan);
            const items = cuda.grouped.maxItems(item_pairs, e, pt);
            try nvfp4_tile.run(self.k, pt, self.stream, .up, .{ .x = s.mixed, .stride = dims, .slots = s.slots }, owned.view(), plan, s.moe_act, items, skip);
            try nvfp4_tile.run(self.k, pt, self.stream, .down_f32, .{ .x = s.moe_act, .stride = owned.width }, owned.view(), plan, s.y, items, skip);
        },
        .fp8g => try fp8(self.k, self.stream, s.picks, s.mixed, owned.view(), plan, pairs, n, s.y, s.moe_act, s.slots, native.world),
        else => return error.UnsupportedQuantization,
    }
    if (side != null) try shared.join(self, side, native, n) else try @import("cuda_shared_split.zig").run(self, native, n);
    if (native.world != 1) {
        const collective = self.m.collective orelse return error.TensorParallelUnavailable;
        if (exchange.rowSplit(native.world, n)) return exchange.run(self, collective, n);
        try collective.sum(self.stream, s.y, pairs * dims);
    }
}

/// `picks` as this rank's expert ids into `out`: its own experts from 0, every other id (and the shared slot) `count`.
fn ownPicks(self: anytype, picks: u64, out: u64, pairs: usize, first: usize, count: usize) !u64 {
    var a: cuda.Args = .{};
    a.add(picks);
    a.add(out);
    for ([_]usize{ pairs, first, count }) |v| a.add(@as(i32, @intCast(v)));
    try cuda.launch.launch(self.k.ep_picks, .{ .grid = .{ .x = @intCast((pairs + 255) / 256) }, .block = .{ .x = 256 } }, self.stream, &a);
    return out;
}

/// Block-FP8 pairs (W8A16) at NVFP4's decode/prompt tile: `act` bf16, `out` fp32; the shared slot is skipped.
pub fn fp8(k: *const Kernels, stream: cuda.Stream, picks: u64, x: u64, layer: cuda.experts.Layer, plan: cuda.grouped.Plan, pairs: usize, n: usize, out: u64, act: u64, slot_n: usize, world: usize) !void {
    if (layer.format != .fp8g) return error.UnsupportedQuantization;
    const routed_only = k.forms.routed_plan and world == 1;
    const e = experts_n + @as(usize, @intFromBool(!routed_only));
    const item_pairs = pairs - @as(usize, @intFromBool(routed_only)) * n;
    const prompt = nvfp4_tile.size(k, n) != 16;
    const tile: usize = if (prompt) cuda.experts.promptTile(.fp8g) else 16;
    try k.router.route(stream, picks, pairs, e, tile, plan);
    const items = cuda.grouped.maxItems(item_pairs, e, tile);
    try k.fp8.run(stream, .up, .{ .x = x, .stride = layer.dims, .slots = slot_n }, layer, plan, prompt, act, items, experts_n);
    try k.fp8.run(stream, .down_f32, .{ .x = act, .stride = layer.width }, layer, plan, prompt, out, items, experts_n);
}
