//! ROUND-480: EVERY shared Zig/Metal argument struct in the DeepSeek V4.1 family, checked against its Metal
//! twin at COMPTIME. One table, so a struct no forward happens to reference is still checked: `force()` is
//! called from `model.zig`'s comptime block, which the engine build always analyses.
//!
//! `index_mirror.check` compares the Metal declaration's field names, order, C types, array lengths and
//! C-layout offsets to the Zig struct's own, and its total to `@sizeOf`. A field added, removed, renamed,
//! reordered or retyped on EITHER side is a build error -- the ROUND-468 bug class (a silently shifted
//! `scale` turned every index score into 0.0) can no longer reach a serve.
//!
//! The per-struct checks ROUND-473/478 put inside the struct bodies stay where they are; this file is the
//! complete registry, including the pairs that used to be anonymous literals at the callsite.

const mirror = @import("metal/index_mirror.zig");
const sources = @import("kernel_sources");

const csa2_index = @import("csa2_index.zig");
const csa2 = @import("csa2.zig");
const engram = @import("engram.zig");
const engram_hash = @import("engram_hash.zig");
const kv = @import("kv.zig");
const mhc = @import("mhc.zig");
const ops = @import("ops.zig");
const rope = @import("rope.zig");
const route = @import("route.zig");
const attn = @import("metal/csa2_attn.zig");
const cand = @import("metal/cand.zig");
const compress = @import("metal/compress.zig");
const m_engram = @import("metal/engram.zig");
const experts = @import("metal/experts.zig");
const experts_mma = @import("metal/experts_mma.zig");
const fold = @import("metal/fold.zig");
const fp8_mma = @import("metal/fp8_mma.zig");
const fp8_rows = @import("metal/fp8_rows.zig");
const fused = @import("metal/fused.zig");
const glue = @import("metal/glue.zig");
const glue_k = @import("metal/glue_k.zig");
const hc = @import("metal/hc.zig");
const m_index = @import("metal/index.zig");
const mv = @import("metal/mv.zig");

/// Checks every pair in the table. Called from a `comptime` block so each call is evaluated, not emitted.
pub fn force() void {
    comptime {
        // The attention, the KV pool and the indexer: the family's largest argument records.
        mirror.check(csa2_index.Params, sources.deepseek_v41_index, "Dsv41Index");
        mirror.check(csa2.Params, sources.deepseek_v41_csa2, "Dsv41Attention");
        mirror.check(csa2.DenseKv, sources.deepseek_v41_csa2_nax, "DenseKv");
        mirror.check(kv.Pack, sources.deepseek_v41_kv, "Dsv41Pack");
        mirror.check(kv.Pool, sources.deepseek_v41_kv, "Dsv41Pool");
        mirror.check(kv.Glue, sources.deepseek_v41_kv_glue, "Dsv41KvGlue");
        mirror.check(kv.Pipelines.RingMove, sources.deepseek_v41_kv, "Dsv41RingMove");
        mirror.check(rope.Params, sources.deepseek_v41_rope, "Dsv41Rope");
        mirror.check(attn.Meta, sources.dsv41.csa2_attn, "TfAttnMeta");
        mirror.check(attn.Args, sources.dsv41.csa2_attn, "TfAttnArgs");
        mirror.check(m_index.Args, sources.dsv41.index, "TfDsv41IxArgs");

        // Engram: the hash, the gather and the gate, plus the per-layer hash record.
        mirror.check(engram.HashArgs, sources.deepseek_v41_engram, "EngramHashArgs");
        mirror.check(engram.GatherArgs, sources.deepseek_v41_engram, "EngramGatherArgs");
        mirror.check(engram.GateArgs, sources.deepseek_v41_engram, "EngramGateArgs");
        mirror.check(engram_hash.Layer, sources.deepseek_v41_engram, "EngramLayer");
        mirror.check(m_engram.Args, sources.dsv41.engram, "TfEngramArgs");

        // mHC and the hyper-connection boundary.
        mirror.check(mhc.BoundaryArgs, sources.deepseek_v41_hc_boundary, "HcArgs");
        mirror.check(mhc.Rms, sources.deepseek_v41_mhc, "Rms");
        mirror.check(mhc.MixDense, sources.deepseek_v41_mhc, "MixDense");
        mirror.check(mhc.ColDense, sources.deepseek_v41_mhc, "ColDense");
        mirror.check(mhc.GateTree, sources.deepseek_v41_mhc, "GateTree");
        mirror.check(mhc.GateTree4, sources.deepseek_v41_mhc, "GateTree4");
        mirror.check(mhc.Split, sources.deepseek_v41_mhc, "Split");
        mirror.check(mhc.Collapse, sources.deepseek_v41_mhc, "Collapse");
        mirror.check(mhc.Post, sources.deepseek_v41_mhc, "Post");
        mirror.check(mhc.Expand, sources.deepseek_v41_mhc, "Expand");
        mirror.check(hc.MixArgs, sources.dsv41.hc, "TfHcMixArgs");
        mirror.check(hc.RowArgs, sources.dsv41.hc, "TfHcArgs2");

        // The pointwise glue: the ops kernels, the route's args, the rope and quant launches.
        mirror.check(ops.RmsArgs, sources.deepseek_v41_ops, "Rms");
        mirror.check(ops.Embedding, sources.deepseek_v41_ops, "Embedding");
        mirror.check(ops.StreamMean, sources.deepseek_v41_ops, "StreamMean");
        mirror.check(ops.Add, sources.deepseek_v41_ops, "Add");
        mirror.check(ops.Markov, sources.deepseek_v41_ops, "MarkovArgs");
        mirror.check(route.Args, sources.deepseek_v41_route, "Shape");
        mirror.check(glue.RouteArgs, sources.dsv41.route, "Dsv41RouteArgs");
        mirror.check(glue.GNormArgs, sources.dsv41.glue_tmp, "TfGNormArgs");
        mirror.check(glue.GHcArgs, sources.dsv41.glue_tmp, "TfGHcArgs");
        mirror.check(glue_k.RmsArgs, sources.dsv41.glue, "TfRmsArgs");
        mirror.check(glue_k.EmbedArgs, sources.dsv41.glue, "TfEmbedArgs");
        mirror.check(glue_k.MeanArgs, sources.dsv41.glue, "TfMeanArgs");
        mirror.check(glue_k.RopeArgs, sources.dsv41.glue, "TfRopeArgs");
        mirror.check(glue_k.QuantArgs, sources.dsv41.glue, "TfQuantArgs");
        mirror.check(fold.NormArgs, sources.dsv41.fold, "TfONormArgs");
        mirror.check(fused.FhcArgs, sources.dsv41.fused, "TfFHcArgs");
        mirror.check(fused.NormArgs, sources.dsv41.fused, "TfFNormArgs");
        mirror.check(fused.KvArgs, sources.dsv41.fused, "TfFKvArgs");
        mirror.check(fused.SwiArgs, sources.dsv41.fused, "TfFSwiArgs");

        // ROUND-527: the payloads that used to be positional literals -- one named extern struct a twin, so
        // a reorder or a same-size retype on either side is a build error, not a silent argument shift.
        mirror.check(ops.DenseArgs, sources.deepseek_v41_ops, "Dense");
        mirror.check(ops.ActivationArgs, sources.deepseek_v41_ops, "Activation");
        mirror.check(ops.CopyArgs, sources.deepseek_v41_ops, "Copy");
        mirror.check(ops.CombineArgs, sources.deepseek_v41_ops, "Combine");
        mirror.check(glue.GRopeArgs, sources.dsv41.glue_tmp, "TfGRopeArgs");
        mirror.check(glue.GCandArgs, sources.dsv41.glue_tmp, "TfGCandArgs");
        mirror.check(glue.GHcArgs2, sources.dsv41.glue_tmp, "TfGHcArgs2");
        mirror.check(glue.GScatterArgs, sources.dsv41.glue_tmp, "TfGScatterArgs");
        mirror.check(glue.GEmbedArgs, sources.dsv41.glue_tmp, "TfGEmbedArgs");
        mirror.check(glue.GMeanArgs, sources.dsv41.glue_tmp, "TfGMeanArgs");
        mirror.check(glue.TopkArgs, sources.dsv41.topk512, "TfTopk512Args");
        mirror.check(cand.Args, sources.dsv41.cand, "TfCandArgs");
        mirror.check(fused.PostArgs, sources.dsv41.fused, "TfFPostArgs");
        mirror.check(fused.QuantArgs, sources.dsv41.fused, "TfFQuantArgs");
        mirror.check(compress.PoolArgs, sources.dsv41.compress, "TfCPoolArgs");
        mirror.check(compress.RtArgs, sources.dsv41.compress, "TfCRtArgs");
        mirror.check(compress.PackArgs, sources.dsv41.compress, "TfCPackArgs");
        mirror.check(fold.PostArgs, sources.dsv41.fold, "TfOPostArgs");
        mirror.check(engram.History, sources.deepseek_v41_engram, "EngramHistory");

        // The routed experts, the fp8 prompt paths and the narrow decode matvecs.
        mirror.check(experts.ExpertPtrs, sources.dsv41.experts_kimi, "ExpertPtrs");
        mirror.check(experts.PlanArgs, sources.dsv41.experts_kimi, "PlanArgs");
        mirror.check(experts.XpArgs, sources.dsv41.experts_kimi, "XpArgs");
        mirror.check(experts.UpArgs, sources.dsv41.experts, "DsArgs");
        mirror.check(experts_mma.XmArgs, sources.dsv41.xmma, "XmArgs");
        mirror.check(fp8_mma.Args, sources.dsv41.fp8_mma, "TfDsv41Fp8MmaArgs");
        mirror.check(fp8_rows.Args, sources.dsv41.fp8_rows, "TfDsv41Fp8Args");
        mirror.check(fp8_rows.WholeArgs, sources.dsv41.fp8_rows, "TfDsv41Fp8WArgs");
        mirror.check(mv.Args, sources.dsv41.mv, "TfDsv41MvArgs");
    }
}
