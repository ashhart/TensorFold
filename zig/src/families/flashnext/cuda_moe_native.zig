//! The MoE pass's native experts: NVFP4 or block-FP8 routed pairs, the shared expert on rank 0, the rank sum.
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
    const e = experts_n + 1;
    const owned = &native.experts;
    if (zs.clearsY(native.world, native.rank)) try self.stream.d.check(self.stream.d.api.cuMemsetD8Async(s.y, 0, pairs * dims * 4, self.stream.handle), "cuMemsetD8Async");
    switch (owned.format) {
        .nvfp4 => {
            const pt = nvfp4_tile.size(n);
            try self.k.router.route(self.stream, s.picks, pairs, e, pt, plan);
            const items = cuda.grouped.maxItems(pairs, e, pt);
            try nvfp4_tile.run(self.k, pt, self.stream, .up, .{ .x = s.mixed, .stride = dims, .slots = slots }, owned.view(), plan, s.moe_act, items, experts_n);
            try nvfp4_tile.run(self.k, pt, self.stream, .down_f32, .{ .x = s.moe_act, .stride = owned.width }, owned.view(), plan, s.y, items, experts_n);
        },
        .fp8g => try fp8(self.k, self.stream, s.picks, s.mixed, owned.view(), plan, pairs, n, s.y, s.moe_act),
        else => return error.UnsupportedQuantization,
    }
    if (native.rank == 0) {
        const glue = self.tri.own orelse return error.NativeGlueRequired;
        try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[0].native.?, s.shared_gate, s.native_part);
        try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[1].native.?, s.shared_up, s.native_part);
        try glue.sharedSwiGLU(self.stream, s.shared_gate, s.shared_up, s.shared_act, n, moe_w);
        try self.k.native_lane.matmulF32(self.stream, s.shared_act, moe_w, n, native.shared[2].native.?, s.shared_out, s.native_part);
        try glue.sharedScatter(self.stream, s.shared_out, s.y, n, dims, slots);
    }
    if (native.world != 1) {
        const collective = self.m.collective orelse return error.TensorParallelUnavailable;
        try collective.sum(self.stream, s.y, pairs * dims);
    }
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
