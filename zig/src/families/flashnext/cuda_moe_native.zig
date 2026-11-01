//! The MoE pass's native experts: NVFP4 or block-FP8 pairs (whole experts a rank at --tp), shared on rank 0, rank sum.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const mixed = @import("cuda_mixed_experts.zig");
const nvfp4_tile = @import("cuda_nvfp4_tile.zig");
const zs = @import("cuda_zero_slice.zig");
const Kernels = @import("cuda_batch_kernels.zig").Kernels;

const dims = wt.dims;
const experts_n = wt.experts_n;
const slots = wt.slots;
const moe_w = wt.moe_w;

/// moeBlockKeep's native layer: each routed pair's fp32 row into y [pairs, dims], the shared slot's on rank 0.
pub fn run(self: anytype, native: *const mixed.Owned, n: usize, pairs: usize, plan: cuda.grouped.Plan) !void {
    const s = self.s;
    const owned = &native.experts;
    // the rank's experts and one skip id past them (the shared slot at world 1; every other rank's pick at --tp)
    const e = owned.count + 1;
    const skip: i32 = @intCast(owned.count);
    const picks = if (native.world == 1) s.picks else try ownPicks(self, s.picks, s.picks_ep, pairs, owned.first, owned.count);
    if (zs.clearsY(native.world, native.rank)) try self.stream.d.check(self.stream.d.api.cuMemsetD8Async(s.y, 0, pairs * dims * 4, self.stream.handle), "cuMemsetD8Async");
    switch (owned.format) {
        .nvfp4 => {
            const pt = nvfp4_tile.size(n);
            try self.k.router.route(self.stream, picks, pairs, e, pt, plan);
            const items = cuda.grouped.maxItems(pairs, e, pt);
            try nvfp4_tile.run(self.k, pt, self.stream, .up, .{ .x = s.mixed, .stride = dims, .slots = slots }, owned.view(), plan, s.moe_act, items, skip);
            try nvfp4_tile.run(self.k, pt, self.stream, .down_f32, .{ .x = s.moe_act, .stride = owned.width }, owned.view(), plan, s.y, items, skip);
        },
        .fp8g => try fp8(self.k, self.stream, s.picks, s.mixed, owned.view(), plan, pairs, n, s.y, s.moe_act),
        else => return error.UnsupportedQuantization,
    }
    try @import("cuda_shared_split.zig").run(self, native, n);
    if (native.world != 1) {
        const collective = self.m.collective orelse return error.TensorParallelUnavailable;
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
pub fn fp8(k: *const Kernels, stream: cuda.Stream, picks: u64, x: u64, layer: cuda.experts.Layer, plan: cuda.grouped.Plan, pairs: usize, n: usize, out: u64, act: u64) !void {
    if (layer.format != .fp8g) return error.UnsupportedQuantization;
    const e = experts_n + 1;
    const prompt = nvfp4_tile.size(n) != 16;
    const tile: usize = if (prompt) cuda.experts.promptTile(.fp8g) else 16;
    try k.router.route(stream, picks, pairs, e, tile, plan);
    const items = cuda.grouped.maxItems(pairs, e, tile);
    try k.fp8.run(stream, .up, .{ .x = x, .stride = layer.dims, .slots = slots }, layer, plan, prompt, act, items, experts_n);
    try k.fp8.run(stream, .down_f32, .{ .x = act, .stride = layer.width }, layer, plan, prompt, out, items, experts_n);
}
