//! Caller-owned prompt pipelines compose RMS and activation stages without changing decoder pipelines.
const metal = @import("metal");
const abi = @import("prompt_glue.zig");
const glue = @import("glue.zig");
const Ref = @import("projection.zig").Ref;

pub const Kernels = struct {
    head_norm: metal.Pipeline,
    product: metal.Pipeline,
    silu: metal.Pipeline,
    qk: metal.Pipeline,
    decay: metal.Pipeline,

    pub fn rms(k: Kernels, e: metal.ComputeEncoder, x: Ref, gain: Ref, output: Ref, a: glue.Head) !void {
        if (a.activation != .bf16 or a.gain != .bf16) return error.UnsupportedPromptStorage;
        const threads = try glue.headThreads(a);
        if (threads > k.head_norm.maxThreads()) return error.UnsupportedPromptGroup;
        e.setPipeline(k.head_norm);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(gain.buffer, gain.offset, 1);
        e.setValue(a, 2);
        e.setBuffer(output.buffer, output.offset, 3);
        e.dispatchGroups(.{ .width = a.rows }, .{ .width = threads });
        e.barrier();
    }

    pub fn activation(k: Kernels, e: metal.ComputeEncoder, gate: Ref, factor: Ref, output: Ref, a: abi.Product) !void {
        _ = try abi.product(a.width, a.rows, a.mode);
        e.setPipeline(k.product);
        bind(e, .{ gate, factor });
        e.setValue(a, 2);
        e.setBuffer(output.buffer, output.offset, 3);
        e.dispatchThreads(.{ .width = @as(usize, a.width) * a.rows }, .{ .width = 256 });
        e.barrier();
    }

    pub fn convolutionActivation(k: Kernels, e: metal.ComputeEncoder, input: Ref, output: Ref, a: abi.Product) !void {
        _ = try abi.product(a.width, a.rows, a.mode);
        e.setPipeline(k.silu);
        e.setBuffer(input.buffer, input.offset, 0);
        e.setValue(a, 1);
        e.setBuffer(output.buffer, output.offset, 2);
        e.dispatchThreads(.{ .width = @as(usize, a.width) * a.rows }, .{ .width = 256 });
        e.barrier();
    }

    pub fn normalizedQk(k: Kernels, e: metal.ComputeEncoder, activation_rows: Ref, output: Ref, a: abi.Qk) !void {
        try abi.check(a);
        e.setPipeline(k.qk);
        e.setBuffer(activation_rows.buffer, activation_rows.offset, 0);
        e.setValue(a, 1);
        e.setBuffer(output.buffer, output.offset, 2);
        e.dispatchGroups(.{ .width = a.heads, .height = a.rows }, .{ .width = 32 });
        e.barrier();
    }

    pub fn gdnDecay(k: Kernels, e: metal.ComputeEncoder, a_rows: Ref, b_rows: Ref, a_log: Ref, dt: Ref, g: Ref, beta: Ref, p: abi.Decay) !void {
        try abi.checkDecay(p);
        e.setPipeline(k.decay);
        bind(e, .{ a_rows, b_rows, a_log, dt });
        e.setValue(p, 4);
        e.setBuffer(g.buffer, g.offset, 5);
        e.setBuffer(beta.buffer, beta.offset, 6);
        e.dispatchThreads(.{ .width = @as(usize, p.rows) * p.heads }, .{ .width = 256 });
        e.barrier();
    }
};

fn bind(e: metal.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, index| e.setBuffer(r.buffer, r.offset, index);
}
