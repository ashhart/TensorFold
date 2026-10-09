//! Force analysis of native interfaces without creating a Metal device or reading a model.
const std = @import("std");
const q = @import("qwen27");

pub fn main() void {
    inline for (.{ &q.dflash.generation.Generation.init, &q.dflash.generation.Generation.prefill, &q.dflash.generation.Generation.run }) |function| std.mem.doNotOptimizeAway(function);
    inline for (.{ &q.dflash.runtime_backend.Backend.init, &q.dflash.runtime_backend.Backend.ops, &q.dflash.runtime_backend.Backend.deinit, &q.dflash.runtime_model.Model.load, &q.dflash.runtime_model.Model.absorb, &q.dflash.runtime_model.Model.propose, &q.dflash.runtime_model.Model.deinit }) |function| std.mem.doNotOptimizeAway(function);
    inline for (.{ &q.projection.prepare, &q.gpu_weights.load, &q.gpu_frame.Frame.init, &q.gpu_frame.Frame.setRows, &q.quant_gpu.Kernels.quant, &q.quant_gpu.Kernels.norm, &q.quant_gpu.Kernels.sum, &q.quant_gpu.Kernels.mlp, &q.quant_gpu.Kernels.deinit }) |function| std.mem.doNotOptimizeAway(function);
    inline for (.{ &q.glue_gpu.Kernels.init, &q.glue_gpu.Kernels.embedding, &q.glue_gpu.Kernels.headNorm, &q.glue_gpu.Kernels.unstack, &q.glue_gpu.Kernels.residual }) |function| std.mem.doNotOptimizeAway(function);
    inline for (.{ &q.model.Model.load, &q.model.Model.deinit, &q.forward.encode }) |function| std.mem.doNotOptimizeAway(function);
    inline for (.{ &q.taps.Taps.init, &q.taps.Taps.record, &q.taps.Taps.complete }) |function| std.mem.doNotOptimizeAway(function);
    std.mem.doNotOptimizeAway(&q.taps.Taps.accepted);
    inline for (.{ &q.gdn_pool_gpu.Pool.init, &q.gdn_pool_gpu.Pool.forward, &q.gdn_pool_gpu.Pool.commit, &q.gdn_pool_gpu.Pool.clearSlot, &q.gdn_hook.Hook.forward, &q.gdn_hook.Hook.commit }) |function| std.mem.doNotOptimizeAway(function);
    inline for (.{ &q.attention.Attention.init, &q.attention.Attention.preprocess, &q.attention.Attention.encode, &q.attention.Attention.keep, &q.attention.Scratch.init, &q.attention.Scratch.upload, &q.attention.Binding.upload, &q.attention.Binding.uploadKeep }) |function| std.mem.doNotOptimizeAway(function);
    inline for (.{ &q.decode_round.Runner.init, &q.decode_round.Runner.verify, &q.decode_round.Runner.keep, &q.decode_round.Runner.reset }) |function| std.mem.doNotOptimizeAway(function);
    inline for (.{ &q.prompt_gpu.Kernels.rms, &q.prompt_gpu.Kernels.activation, &q.prompt_gpu.Kernels.normalizedQk, &q.prompt_gpu.Kernels.gdnDecay, &q.prompt_state_gpu.Kernels.convolution, &q.prompt_state_gpu.Kernels.recurrence }) |function| std.mem.doNotOptimizeAway(function);
}
