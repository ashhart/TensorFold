//! Prompt chain output and final FP32 state are encoded without touching decoder replay arithmetic.
const metal = @import("metal");
const abi = @import("prompt_state.zig");
const Ref = @import("projection.zig").Ref;

pub const Kernels = struct {
    conv: metal.Pipeline,
    recur: metal.Pipeline,

    pub fn convolution(k: Kernels, e: metal.ComputeEncoder, projected: Ref, history: Ref, weight: Ref, output: Ref, next_history: Ref, p: abi.Conv) !void {
        try abi.checkConv(p);
        e.setPipeline(k.conv);
        e.setBuffer(projected.buffer, projected.offset, 0);
        e.setBuffer(history.buffer, history.offset, 1);
        e.setBuffer(weight.buffer, weight.offset, 2);
        e.setValue(p, 3);
        e.setBuffer(output.buffer, output.offset, 4);
        e.setBuffer(next_history.buffer, next_history.offset, 5);
        e.dispatchThreads(.{ .width = p.channels, .height = p.rows, .depth = p.streams }, .{ .width = 256 });
        e.barrier();
    }

    pub fn recurrence(k: Kernels, e: metal.ComputeEncoder, q: Ref, key: Ref, value: Ref, decay: Ref, beta: Ref, committed: Ref, output: Ref, next_state: Ref, p: abi.Recur) !void {
        try abi.checkRecur(p);
        e.setPipeline(k.recur);
        e.setBuffer(q.buffer, q.offset, 0);
        e.setBuffer(key.buffer, key.offset, 1);
        e.setBuffer(value.buffer, value.offset, 2);
        e.setBuffer(decay.buffer, decay.offset, 3);
        e.setBuffer(beta.buffer, beta.offset, 4);
        e.setBuffer(committed.buffer, committed.offset, 5);
        e.setValue(p, 6);
        e.setBuffer(output.buffer, output.offset, 7);
        e.setBuffer(next_state.buffer, next_state.offset, 8);
        e.dispatchThreads(.{ .width = 32, .height = p.dv / 8, .depth = p.nv * p.streams }, .{ .width = 32, .height = 2 });
        e.barrier();
    }
};
