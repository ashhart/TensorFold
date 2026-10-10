//! QKV projects straight into the pool's log and ZBA into the frame's mixed buffer, side by side from the same input.
const metal = @import("metal");
const Pool = @import("gdn_pool_gpu.zig").Pool;
const pool_gpu = @import("gdn_pool_gpu.zig");
const Plan = @import("gdn_plan.zig").Plan;
const Frame = @import("gpu_frame.zig").Frame;
const Weights = @import("gpu_weights.zig");
const Quant = @import("quant_gpu.zig").Kernels;
const Glue = @import("glue_gpu.zig").Kernels;

pub const Hook = struct {
    pool: *Pool,
    quant: *Quant,
    glue: Glue,
    plan: *const Plan,
    linear_index: []const u32,
    accepted_chain: bool = false,

    pub fn forward(ptr: *anyopaque, e: metal.ComputeEncoder, block: usize, weight: Weights.Gdn, frame: *Frame, rows: u32) !void {
        const hook: *Hook = @ptrCast(@alignCast(ptr));
        if (block >= hook.linear_index.len or hook.linear_index[block] >= hook.pool.budget.layers or rows != hook.plan.parents.len) return error.BadGdnHook;
        const layer = hook.linear_index[block];
        const sums = @import("forward.zig").sums(frame, .sums, rows);
        try hook.quant.quant(e, weight.qkv, frame.get(.input), sums, frame.get(.dims), try hook.pool.projected(layer), rows);
        try hook.quant.quant(e, weight.zba, frame.get(.input), sums, frame.get(.dims), frame.get(.mixed), rows);
        e.barrier();
        if (hook.accepted_chain) try hook.pool.forwardChain(layer, devicePlan(hook.plan.*, frame), try refs(weight, frame), e, hook.quant.prompt) else try hook.pool.forward(layer, devicePlan(hook.plan.*, frame), try refs(weight, frame), e);
    }

    pub fn commit(hook: *Hook, e: metal.ComputeEncoder, block: usize, frame: *Frame) !void {
        if (block >= hook.linear_index.len or hook.linear_index[block] >= hook.pool.budget.layers) return error.BadGdnHook;
        try hook.pool.commit(hook.linear_index[block], devicePlan(hook.plan.*, frame), e);
    }
};

/// z, b and a read in place from the fused zba rows.
fn refs(weight: Weights.Gdn, frame: *Frame) !pool_gpu.Refs {
    const width = frame.config.gdnValueDim();
    const heads = frame.config.v_heads;
    const mixed = frame.get(.mixed);
    return .{ .z = mixed, .b = .{ .buffer = mixed.buffer, .offset = mixed.offset + width * 2 }, .a = .{ .buffer = mixed.buffer, .offset = mixed.offset + (width + heads) * 2 }, .output = frame.get(.mixer_out), .scalars = try pool_gpu.Scalars.from(weight), .zba_stride = @intCast(width + 2 * heads) };
}

fn devicePlan(plan: Plan, frame: *Frame) pool_gpu.DevicePlan {
    return .{ .host = plan, .metadata = .{ .parents = frame.get(.parents), .windows = frame.get(.windows), .row_slots = frame.get(.row_slots), .segments = frame.get(.segments), .keeps = frame.get(.keeps), .kept_rows = frame.get(.kept_rows) } };
}
