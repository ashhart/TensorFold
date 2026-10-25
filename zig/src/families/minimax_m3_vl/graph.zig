//! The MSA launch graph on device: the per-layer forward the receipts' boundary words describe,
//! running the TP2 rank split shard.zig draws. The quantized projections run through the mainline
//! EXL3 machinery the family already receipts — `cuda.exl3.Dense` decode-row GEMMs (mul1's odd
//! half steps and the head's 8 halves included) and the grouped routed experts
//! `cuda.exl3_experts.Experts` (the Stage-5 upload's fp32 epilogue, limit 7, act_mode 2 — the
//! OAI swiglu contract tf_m3_swigluoai's shape; the mode-1 silu path cannot express it); the MSA
//! indexer and the block-sparse attention run through the parity-proven `tf_m3_scores/topk/
//! decode/decode_combine` launches (`cuda_sparse.zig`); the activation rows run through
//! `tf_m3_swigluoai` (`cuda_serve.zig`). The between-the-kernels glue — the gemma norm rows, the
//! partial rope, the per-head fp16 stores, the noaux sigmoid+bias router's normalize-then-scale
//! weights, the small fp16 indexer projections, the residual adds — is host fp32 code that reuses
//! serve.zig's own published helpers (`normRow`, `ropeRow`, `route`, `f16r`), so the graded bar is
//! this graph's contract by construction, not a re-typed copy. Where the device path stores fp16
//! where the host engine's fp32 linears would not, that is the fork's own device contract (exl3
//! GEMM inputs are fp16-rounded by `rot_in`; the gate/up/down wave and the swigluoai epilogue
//! store fp16; the attention reads fp16 cache rows through fp32 softmax) — each graded point's
//! receipts live in TF-STAGE4-ATTACH-RECEIPTS.md and the receipts this stage writes.
//!
//! Decode-first minimal path, honestly sized: one row at a time, greedy only — no batching, no
//! streaming, no HTTP surface. The KV cache and the indexer side plane are a fixed page-backed
//! pool of `keys_cap` keys (16 pages of 256); a row past it refuses loudly rather than mis-serve.
//!
//! TP2 (shard.zig's world): rank 0 embeds, runs its layers, hands the residual row to rank 1 over
//! NCCL, which runs its layers and hands it back; rank 0 applies the final gemma norm, the lm_head
//! and greedy. The experts replicate per rank exactly as Stage-5 uploaded them; the only per-token
//! cross-rank traffic is the two hidden-row handoffs.

const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const serve = @import("serve.zig");
const shard = @import("shard.zig");
const boundary = @import("cuda_native.zig");

const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;
const exl3_upload = cuda.exl3_upload;
const sp = @import("cuda_sparse.zig"); // the tf_m3 indexer/attention launches
const glue = @import("cuda_serve.zig"); // the family's tf_m3_swigluoai rows

const exl3_load = @import("core").exl3_load; // the core reader the Uploader takes parts from

/// The fixed KV/plane pool this decode path serves: 16 pages of 256 keys = 4096 keys (the size is
/// said here; the topk-16 selection's whole window fits inside it at the smoke's lengths).
pub const keys_cap: usize = 4096;
/// The decode split count (the served envelope; the combine's shared arrays hold at most 8).
const nsplit: u32 = 4;

pub const Options = struct {
    /// World 2's handoff: the NCCL library and communicator; null = TP1 (the fixture rank).
    lib: ?*const cuda.nccl.Library = null,
    comm: ?cuda.nccl.Comm = null,
    /// Additive Stage-8 (the host-f64 grade): a TP1-only backbone cap — build and run only the
    /// first `layer_cap` owned layers (the cap is the real-pack layer stack that fits the host's
    /// RAM beside the f64 reference). TP2 refuses a cap (each rank owns its exact half); 0 = all.
    layer_cap: usize = 0,
};

/// One layer's device tensors and the host glue weights this rank runs it with.
const Layer = struct {
    lno: u32, // the checkpoint's layer id (the caches offset by the local index)
    in_norm: []f32 = &.{}, // [hidden], RAW bf16 passthrough values (serve.normRow applies its own (1+w))
    post_norm: []f32 = &.{},
    q_norm: []f32 = &.{}, // [head_dim]
    k_norm: []f32 = &.{},
    iq_norm: []f32 = &.{}, // [i_dim]
    ik_norm: []f32 = &.{},
    q: ?exl3.Weight = null,
    k: ?exl3.Weight = null,
    v: ?exl3.Weight = null,
    o: ?exl3.Weight = null,
    iq: ?[]f16 = null, // the fp16 passthrough indexer projections, host-GEMV-ed
    ik: ?[]f16 = null,
    mlp: Mlp = undefined,

    const Mlp = union(enum) {
        dense: struct { gate: exl3.Weight, up: exl3.Weight, down: exl3.Weight },
        moe: struct {
            router: []f32, // [E, hidden] fp32 raw (the router gate is not quantized)
            bias: []f32, // [E] fp32; empty when the pack names none
            routed: exl3e.Layer, // the grouped experts' device tensors
            sgate: exl3.Weight,
            sup: exl3.Weight,
            sdown: exl3.Weight,
        },
    };
};

/// The last row's trace: the smoke's integrity assertions read this.
pub const Trace = struct {
    fired: usize = 0, // experts fired by the last MoE wave
    wsum: f32 = 0, // the routing weights' sum (= routed_scale by the contract)
    sel_valid: usize = 0, // valid selected blocks, the last kv head of the last layer
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    c: config.Config,
    d: *const cuda.Driver,
    s: cuda.Stream,
    dense: exl3.Dense,
    experts: exl3e.Experts,
    mod: cuda.Module, // the minimax_m3 image the tf_m3 glue kernels resolve from
    ks: sp.Kernels,
    gk: glue.Kernels,
    up: exl3_upload.Uploader, // owns every device weight buffer AND this engine's runtime rows
    rank: u32,
    world: u32,
    lib: ?*const cuda.nccl.Library,
    comm: ?cuda.nccl.Comm,
    layers: []Layer, // this rank's owned layers, in run order
    first: u32, // layers[0]'s checkpoint id
    embed_tbl: ?cuda.DeviceBuffer = null, // rank 0: the bf16 [vocab, hidden] table
    final_norm: ?[]f32 = null, // rank 0: [hidden]
    head: ?exl3.Weight = null, // rank 0: the [hidden -> vocab] trellis
    len: u32 = 0,
    x: []f32, // [hidden], the residual stream (host fp32, serve's contract)
    xn: []f32,
    xn2: []f32,
    iq32: []f32, // [i_heads * i_dim]
    ik32: []f32, // [i_dim]
    q16: []f16, // [qk_rows]
    k16: []f16, // [kv_rows]
    v16: []f16,
    iq16: []f16, // [i_heads * i_dim]
    ik16: []f16, // [i_dim]
    projf: []f32, // [qk_rows + 2*kv_rows], the q/k/v download
    o16: []f16, // [max(hidden, inter)] — the wave outputs' download
    elogits: []f32, // [experts], the router's scratch
    logits: []f32, // [vocab] (rank 0); the last row's
    sel_e: []u32, // [topk], the last MoE wave's picks in selection order
    pickw: []f32, // [topk], the routing weights (.half() rounded)
    picks: []bool, // [E]
    sel_seen: []i32, // [kv_heads, topk], the last layer's selection (the trace)
    last: Trace = .{},
    // device rows and caches (all allocated through `up`, so build-error safety is one owner):
    x32: cuda.DeviceBuffer, // [hidden] f32, every gemm input
    proj32: cuda.DeviceBuffer, // [qk_rows + 2*kv_rows] f32, the q/k/v gemm outputs
    recv32: cuda.DeviceBuffer, // [hidden] f32, the peer handoff
    ctrl32: cuda.DeviceBuffer, // world 2: the [2] u32 row-control word (op + token or new length)
    qdev16: cuda.DeviceBuffer, // [qk_rows] f16, the attention's q
    iqdev16: cuda.DeviceBuffer, // [i_heads*i_dim] f16, the scorer's queries
    attn16: cuda.DeviceBuffer, // [qk_rows] f16, the attention output
    g16: cuda.DeviceBuffer, // [max_inter] f16, the gate rows (and the down outputs)
    u16: cuda.DeviceBuffer,
    a16: cuda.DeviceBuffer,
    moe32: cuda.DeviceBuffer, // [hidden] f32, the routed experts' weighted sum
    pick32: cuda.DeviceBuffer, // [topk] i32
    wts32: cuda.DeviceBuffer, // [topk] f32
    logits32: cuda.DeviceBuffer, // [vocab] f32 (rank 0; a stub elsewhere)
    kc: cuda.DeviceBuffer, // [nl * cap, kv_rows] f16 pages — physical (P, PAGE, Hkv, D)
    vc: cuda.DeviceBuffer,
    plane: cuda.DeviceBuffer, // [nl * cap, i_dim] f16 side-plane rows, per layer
    scores32: cuda.DeviceBuffer, // [kv_heads, nb_max] f32
    sel32: cuda.DeviceBuffer, // [kv_heads, topk] i32
    qpos64: cuda.DeviceBuffer,
    kpos64: cuda.DeviceBuffer, // [cap] i64, the content keys 0..cap-1
    seqlens: cuda.DeviceBuffer, // [1] i64
    table: cuda.DeviceBuffer, // [pool pages] i32, the identity page table
    part32: cuda.DeviceBuffer, // the decode splits' scratch
    exp_scratch: cuda.DeviceBuffer, // the grouped experts' scratch

    /// The gemm row shapes (usize forms of the config's counts).
    fn hidden(e: *const Engine) usize {
        return @intCast(e.c.hidden);
    }
    fn qkr(e: *const Engine) usize {
        return @as(usize, e.c.attn_heads) * @as(usize, e.c.head_dim);
    }
    fn kvr(e: *const Engine) usize {
        return @as(usize, e.c.kv_heads) * @as(usize, e.c.head_dim);
    }
    fn irow(e: *const Engine) usize {
        return @as(usize, e.c.i_dim);
    }
    fn iplanes(e: *const Engine) usize {
        return @as(usize, e.c.i_heads) * @as(usize, e.c.i_dim);
    }
    fn maxInter(e: *const Engine) usize {
        const c = &e.c;
        return @max(@as(usize, c.dense_inter), @max(@as(usize, c.moe_inter), @as(usize, c.shared_inter)));
    }

    pub fn deinit(e: *Engine) void {
        e.s.deinit();
        e.dense.unload();
        e.experts.unload();
        e.mod.unload();
        e.up.deinit(); // frees every device weight buffer
        if (e.embed_tbl != null) freeBuf(&e.embed_tbl.?);
        inline for ([_][]const u8{ "x32", "proj32", "recv32", "ctrl32", "qdev16", "iqdev16", "attn16", "g16", "u16", "a16", "moe32", "pick32", "wts32", "logits32", "kc", "vc", "plane", "scores32", "sel32", "qpos64", "kpos64", "seqlens", "table", "part32", "exp_scratch" }) |name| {
            freeBuf(&@field(e, name));
        }
        const gpa = e.gpa;
        for (e.layers) |*l| deinitLayer(gpa, l);
        gpa.free(e.layers);
        if (e.final_norm) |w| gpa.free(w);
        for ([_]*[]f32{ &e.x, &e.xn, &e.xn2, &e.iq32, &e.ik32, &e.elogits, &e.logits, &e.pickw }) |p| {
            if (p.len > 0) gpa.free(p.*);
        }
        if (e.sel_e.len > 0) gpa.free(e.sel_e); // u32: the picks' ids
        if (e.picks.len > 0) gpa.free(e.picks);
        if (e.sel_seen.len > 0) gpa.free(e.sel_seen);
        for ([_]*[]f16{ &e.q16, &e.k16, &e.v16, &e.iq16, &e.ik16, &e.o16 }) |p| {
            if (p.len > 0) gpa.free(p.*);
        }
        if (e.projf.len > 0) gpa.free(e.projf); // f32: the q/k/v gemm outputs
        e.* = undefined;
    }

    /// Build from the checkpoint dir: the admission, the whole Stage-2 plan, the rank's shard,
    /// the Stage-5 upload, then this engine's own rows and caches.
    pub fn build(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, rank: u32, world: u32, o: Options) !Engine {
        if (world < 1 or world > shard.max_world) return error.TooManyRanks;
        if (rank >= world) return error.NoRank;
        _ = try shard.checked(.{ .rank = rank, .world = world });
        var admitted = try boundary.admitDir(gpa, io, dir);
        defer admitted.deinit();
        const c = &admitted.c;
        if (c.i_topk != sp.topk or c.i_dim != sp.d or c.block != sp.block or c.head_dim != sp.d or c.i_heads != c.kv_heads) {
            return error.NotMiniMaxM3VL; // the tf_m3 kernels' fixed widths: a different MSA refuses
        }
        if (c.topk > 32 or c.experts > 4096) return error.NotMiniMaxM3VL;
        const load = &admitted.load;
        const groups = admitted.groups.groups;
        const r = shard.range(c.layers, world, rank);
        const dims: usize = @intCast(c.hidden);
        // the Stage-8 cap: TP1's first `layer_cap` layers only (the grade's real-pack boot).
        // Every walk below gates on `run_count`; the default 0 keeps the full range.
        var run_count: usize = r.count;
        if (o.layer_cap > 0) {
            if (world != 1) return error.LayerCapTP2; // the cap serves TP1 only: no partial rank
            if (o.layer_cap > r.count) return error.LayerCapTooWide;
            run_count = o.layer_cap;
        }
        const nl: usize = run_count;

        // the reader + the rank's plan walk, the Stage-5 harness's upload shape
        var scan = try exl3_load.scan(gpa, io, dir);
        defer scan.deinit();
        var idx = std.StringHashMapUnmanaged(usize).empty;
        defer idx.deinit(gpa);
        for (scan.groups, 0..) |g, i| try idx.put(gpa, g.prefix, i);

        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const a = arena_state.allocator();

        var max_buffers: usize = 64;
        for (load.items) |*item| {
            if (!shard.owns(item, c.layers, world, rank)) continue;
            max_buffers += item.bytes / (64 << 20) + @as(usize, switch (item.kind) {
                .group => 4, // trellis + two scales + bias
                .passthrough => 1,
            });
        }
        // ownership discipline for the error paths: everything below registers guarded errdefers
        // (`built == false`); once `e` is fully constructed the single `errdefer e.deinit()`
        // owns it all and the guards stand down.
        var built = false;
        var up: exl3_upload.Uploader = .{ .d = d, .gpa = gpa, .max_buffers = max_buffers };
        errdefer {
            if (!built) up.deinit();
        }

        // the routed experts' per-expert act width: the first owned routed gate's n
        var width: usize = 0;
        for (groups) |g| {
            if (g.class != .routed or g.proj != 0) continue;
            if (g.layer >= r.first and g.layer < r.first + run_count) {
                width = @intCast(g.n);
                break;
            }
        }

        var layers = try gpa.alloc(Layer, nl);
        errdefer {
            if (!built) {
                for (layers) |*l| deinitLayer(gpa, l);
                gpa.free(layers);
            }
        }
        for (layers, 0..) |*l, i| l.* = .{ .lno = r.first + @as(u32, @intCast(i)) };

        // build-time temp slots per class (assembled into the layers' mlps after the walk)
        var d_gate = try a.alloc(?exl3.Weight, nl);
        var d_up = try a.alloc(?exl3.Weight, nl);
        var d_down = try a.alloc(?exl3.Weight, nl);
        var s_gate = try a.alloc(?exl3.Weight, nl);
        var s_up = try a.alloc(?exl3.Weight, nl);
        var s_down = try a.alloc(?exl3.Weight, nl);
        var routers = try a.alloc(?[]f32, nl);
        var biases = try a.alloc(?[]f32, nl);
        for (0..nl) |i| {
            d_gate[i] = null;
            d_up[i] = null;
            d_down[i] = null;
            s_gate[i] = null;
            s_up[i] = null;
            s_down[i] = null;
            routers[i] = null;
            biases[i] = null;
        }

        // the routed experts collect per owned MoE layer, then upload as one layer() call each
        const Projs = std.ArrayListUnmanaged(exl3_upload.Proj);
        const LayerProjs = struct { gate: Projs, up: Projs, down: Projs };
        var exps = try a.alloc(?LayerProjs, c.layers);
        for (exps) |*ep| ep.* = null;

        var head_w: ?exl3.Weight = null;
        var embed: ?cuda.DeviceBuffer = null;
        var final_norm: ?[]f32 = null;
        var norm_fail = false; // a passthrough slot the layer's glue needs and the pack refuses

        for (load.items) |*item| {
            if (!shard.owns(item, c.layers, world, rank)) continue;
            switch (item.kind) {
                .passthrough => {
                    const t = try scan.ck.get(item.name);
                    if (std.mem.eql(u8, item.name, "language_model.model.embed_tokens.weight")) {
                        embed = try cuda.DeviceBuffer.fromHost(d, t.bytes);
                    } else if (std.mem.eql(u8, item.name, "language_model.model.norm.weight")) {
                        final_norm = try expandNorm(gpa, t.bytes, dims);
                        norm_fail = norm_fail or final_norm == null;
                    } else {
                        try readPassthrough(gpa, c, layers, r.first, item.name, t.bytes, routers, biases, &norm_fail);
                    }
                },
                .group => |gi| {
                    const fg = groups[gi];
                    // the cap's skip BEFORE any upload: layers past the first `run_count` are not
                    // this build's layers (the head, a past-layer group, is always built)
                    if (fg.class != .head and fg.layer >= r.first + run_count) continue;
                    const core_gi = idx.get(item.name) orelse return error.MissingTensor;
                    const parts = try scan.parts(a, core_gi);
                    const lidx: usize = fg.layer - r.first;
                    switch (fg.class) {
                        .routed => {
                            const slot = &exps[fg.layer];
                            if (slot.* == null) slot.* = .{ .gate = .empty, .up = .empty, .down = .empty };
                            const proj: *Projs = switch (fg.proj) {
                                0 => &slot.*.?.gate,
                                1 => &slot.*.?.up,
                                else => &slot.*.?.down,
                            };
                            try proj.append(a, .{
                                .trellis = parts.trellis.bytes,
                                .suh = parts.suh.bytes,
                                .svh = parts.svh.bytes,
                                .k2 = fg.k2(),
                                .signs = parts.suh_packed,
                            });
                        },
                        .head => head_w = try up.dense(parts, fg),
                        .attn => switch (fg.proj) {
                            0 => layers[lidx].q = try up.dense(parts, fg),
                            1 => layers[lidx].k = try up.dense(parts, fg),
                            2 => layers[lidx].v = try up.dense(parts, fg),
                            else => layers[lidx].o = try up.dense(parts, fg),
                        },
                        .dense_mlp => switch (fg.proj) {
                            0 => d_gate[lidx] = try up.dense(parts, fg),
                            1 => d_up[lidx] = try up.dense(parts, fg),
                            else => d_down[lidx] = try up.dense(parts, fg),
                        },
                        .shared => switch (fg.proj) {
                            0 => s_gate[lidx] = try up.dense(parts, fg),
                            1 => s_up[lidx] = try up.dense(parts, fg),
                            else => s_down[lidx] = try up.dense(parts, fg),
                        },
                    }
                },
            }
        }
        // each owned MoE layer's collected experts upload as one grouped layer, the shared wave
        // and the router beside it — then the mlp assembly
        for (exps, 0..) |*ep, dno| {
            if (ep.*) |*lp| {
                if (width == 0) return error.BadExperts;
                if (dno < r.first or dno >= r.first + run_count) continue;
                // act_mode 2: the OAI swiglu contract (tf_m3_swigluoai's) — the generic epilogue's
                // mode-1 silu path cannot express the alpha 1.702 sigmoid form
                const routed = try exl3_upload.layer(&up, a, dims, width, .mul1, c.swiglu_limit, 2, lp.gate.items, lp.up.items, lp.down.items);
                const lidx: usize = dno - r.first;
                const sg = s_gate[lidx] orelse return error.MissingTensor;
                const su = s_up[lidx] orelse return error.MissingTensor;
                const sd = s_down[lidx] orelse return error.MissingTensor;
                const rt = routers[lidx] orelse return error.MissingTensor;
                layers[lidx].mlp = .{ .moe = .{
                    .router = rt,
                    .bias = biases[lidx] orelse &.{},
                    .routed = routed,
                    .sgate = sg,
                    .sup = su,
                    .sdown = sd,
                } };
            }
        }
        for (layers, 0..) |*l, i| {
            if (l.lno < c.dense_layers) {
                l.mlp = .{ .dense = .{
                    .gate = d_gate[i] orelse return error.MissingTensor,
                    .up = d_up[i] orelse return error.MissingTensor,
                    .down = d_down[i] orelse return error.MissingTensor,
                } };
            } else switch (l.mlp) {
                .moe => {},
                else => return error.MissingTensor, // an owned MoE layer the walk never assembled
            }
        }
        if (norm_fail) return error.MissingTensor;

        var stream = try cuda.Stream.init(d, false);
        errdefer {
            if (!built) stream.deinit();
        }
        var dense = try exl3.Dense.load(d);
        errdefer {
            if (!built) dense.unload();
        }
        var experts = try exl3e.Experts.load(d);
        errdefer {
            if (!built) experts.unload();
        }
        var mod = try cuda.Module.load(d, cuda.kernels.minimax_m3);
        errdefer {
            if (!built) mod.unload();
        }
        const ks = try sp.Kernels.load(mod);
        const gk = try glue.Kernels.load(mod);

        const qk: usize = @as(usize, c.attn_heads) * @as(usize, c.head_dim);
        const kv: usize = @as(usize, c.kv_heads) * @as(usize, c.head_dim);
        const pages: usize = keys_cap / sp.page;
        const nb_max = sp.nbOf(keys_cap);
        const wide: usize = @max(dims, @max(@as(usize, c.dense_inter), @max(@as(usize, c.moe_inter), @as(usize, c.shared_inter))));
        const kvb: usize = nl * keys_cap * kv * 2;
        const plb: usize = nl * keys_cap * @as(usize, c.i_dim) * 2;

        // a shapes-only Layer for the grouped-experts scratch sizing (the real layers then size
        // the same pool: same dims/width/codebook for every MoE layer)
        var e = Engine{
            .gpa = gpa,
            .io = io,
            .c = c.*,
            .d = d,
            .s = stream,
            .dense = dense,
            .experts = experts,
            .mod = mod,
            .ks = ks,
            .gk = gk,
            .up = up,
            .rank = rank,
            .world = world,
            .lib = o.lib,
            .comm = o.comm,
            .layers = layers,
            .first = r.first,
            .embed_tbl = embed,
            .final_norm = final_norm,
            .head = head_w,
            .x = &.{},
            .xn = &.{},
            .xn2 = &.{},
            .iq32 = &.{},
            .ik32 = &.{},
            .q16 = &.{},
            .k16 = &.{},
            .v16 = &.{},
            .iq16 = &.{},
            .ik16 = &.{},
            .projf = &.{},
            .o16 = &.{},
            .elogits = &.{},
            .logits = &.{},
            .sel_e = &.{},
            .pickw = &.{},
            .picks = &.{},
            .sel_seen = &.{},
            .x32 = undefined,
            .proj32 = undefined,
            .recv32 = undefined,
            .ctrl32 = undefined,
            .qdev16 = undefined,
            .iqdev16 = undefined,
            .attn16 = undefined,
            .g16 = undefined,
            .u16 = undefined,
            .a16 = undefined,
            .moe32 = undefined,
            .pick32 = undefined,
            .wts32 = undefined,
            .logits32 = undefined,
            .kc = undefined,
            .vc = undefined,
            .plane = undefined,
            .scores32 = undefined,
            .sel32 = undefined,
            .qpos64 = undefined,
            .kpos64 = undefined,
            .seqlens = undefined,
            .table = undefined,
            .part32 = undefined,
            .exp_scratch = undefined,
        };
        // the runtime rows and caches, tracked for the error paths (freed once each: this list
        // below `built == true`, e.deinit() above it)
        var bufs: [25]cuda.DeviceBuffer = undefined;
        var nbufs: usize = 0;
        errdefer {
            if (!built) for (bufs[0..nbufs]) |*b| b.free();
        }
        e.x32 = try allocRow(&bufs, &nbufs, d, dims * 4);
        e.proj32 = try allocRow(&bufs, &nbufs, d, (qk + 2 * kv) * 4);
        e.recv32 = try allocRow(&bufs, &nbufs, d, dims * 4);
        e.ctrl32 = try allocRow(&bufs, &nbufs, d, 8); // the [2] u32 row-control word
        e.qdev16 = try allocRow(&bufs, &nbufs, d, qk * 2);
        e.iqdev16 = try allocRow(&bufs, &nbufs, d, @as(usize, c.i_heads) * @as(usize, c.i_dim) * 2);
        e.attn16 = try allocRow(&bufs, &nbufs, d, qk * 2);
        e.g16 = try allocRow(&bufs, &nbufs, d, wide * 2);
        e.u16 = try allocRow(&bufs, &nbufs, d, wide * 2);
        e.a16 = try allocRow(&bufs, &nbufs, d, wide * 2);
        e.moe32 = try allocRow(&bufs, &nbufs, d, dims * 4);
        e.pick32 = try allocRow(&bufs, &nbufs, d, @as(usize, c.topk) * 4);
        e.wts32 = try allocRow(&bufs, &nbufs, d, @as(usize, c.topk) * 4);
        e.logits32 = try allocRow(&bufs, &nbufs, d, if (rank == 0) @as(usize, c.vocab) * 4 else 16);
        e.kc = try allocRow(&bufs, &nbufs, d, kvb);
        e.vc = try allocRow(&bufs, &nbufs, d, kvb);
        e.plane = try allocRow(&bufs, &nbufs, d, plb);
        e.scores32 = try allocRow(&bufs, &nbufs, d, @as(usize, c.i_heads) * nb_max * 4); // scores: (B, i_heads, Sq, NB) fp32
        e.sel32 = try allocRow(&bufs, &nbufs, d, @as(usize, c.kv_heads) * @as(usize, c.i_topk) * 4); // sel: (B, Hkv, i_topk) i32 — the BLOCK selection width (16), NOT the router's topk
        e.qpos64 = try allocRow(&bufs, &nbufs, d, 8);
        e.kpos64 = try allocRow(&bufs, &nbufs, d, keys_cap * 8);
        e.seqlens = try allocRow(&bufs, &nbufs, d, 8);
        e.table = try allocRow(&bufs, &nbufs, d, pages * 4);
        e.part32 = try allocRow(&bufs, &nbufs, d, sp.scratchBytes(@as(usize, c.attn_heads), 1, nsplit));
        e.exp_scratch = blk: {
            const l: exl3e.Layer = .{
                .gate = 0,
                .up = 0,
                .down = 0,
                .gate_k2 = 0,
                .up_k2 = 0,
                .down_k2 = 0,
                .suh_g = 0,
                .suh_u = 0,
                .svh_g = 0,
                .svh_u = 0,
                .suh_d = 0,
                .svh_d = 0,
                .experts = 1,
                .dims = dims,
                .width = if (width > 0) width else 128,
                .codebook = .mul1,
            };
            break :blk try allocRow(&bufs, &nbufs, d, if (width > 0) try exl3e.scratchBytes(l, 1, c.topk) else 4096);
        };
        errdefer e.deinit();
        built = true;

        // the host state
        e.x = try gpa.alloc(f32, dims);
        e.xn = try gpa.alloc(f32, dims);
        e.xn2 = try gpa.alloc(f32, dims);
        e.iq32 = try gpa.alloc(f32, @as(usize, c.i_heads) * @as(usize, c.i_dim));
        e.ik32 = try gpa.alloc(f32, @as(usize, c.i_dim));
        e.q16 = try gpa.alloc(f16, qk);
        e.k16 = try gpa.alloc(f16, kv);
        e.v16 = try gpa.alloc(f16, kv);
        e.iq16 = try gpa.alloc(f16, e.iq32.len);
        e.ik16 = try gpa.alloc(f16, e.ik32.len);
        e.projf = try gpa.alloc(f32, qk + 2 * kv);
        e.o16 = try gpa.alloc(f16, wide);
        e.elogits = try gpa.alloc(f32, c.experts);
        e.logits = try gpa.alloc(f32, if (rank == 0) c.vocab else 1);
        e.sel_e = try gpa.alloc(u32, c.topk);
        e.pickw = try gpa.alloc(f32, c.topk);
        e.picks = try gpa.alloc(bool, c.experts);
        e.sel_seen = try gpa.alloc(i32, @as(usize, c.kv_heads) * @as(usize, c.i_topk)); // the trace mirrors sel32's (B, Hkv, i_topk) width
        @memset(e.picks, false);
        @memset(e.sel_seen, -1);
        // the identity page table and the content keys' mask row, once
        const tab = try a.alloc(i32, pages);
        for (tab, 0..) |*t, i| t.* = @intCast(i);
        try e.table.upload(0, std.mem.sliceAsBytes(tab));
        const kpos = try a.alloc(i64, keys_cap);
        for (kpos, 0..) |*t, i| t.* = @intCast(i);
        try e.kpos64.upload(0, std.mem.sliceAsBytes(kpos));
        if (rank == 0) {
            if (e.embed_tbl == null or e.final_norm == null or e.head == null) return error.MissingTensor;
        }
        return e;
    }

    /// One row: rank 0 embeds `tok`, every rank runs its own layers, the world-2 handoffs move
    /// the residual row between them, rank 0 closes with the final norm + lm_head. The row's
    /// fp32 logits land in `logits` (rank 0).
    pub fn put(e: *Engine, tok: u32) !void {
        const pos = e.len;
        if (pos >= keys_cap) return error.ContextCap;
        e.len += 1;
        const sk: usize = pos + 1;
        const nb = sp.nbOf(sk);
        const dims = e.hidden();
        if (e.world == 2) {
            const lib = e.lib orelse return error.NoRank;
            const comm = e.comm orelse return error.NoRank;
            if (e.rank == 1) {
                try lib.check(lib.api.ncclRecv(e.recv32.ptr, dims, .f32, 0, comm, e.s.handle), "ncclRecv");
                try e.s.synchronize();
                try e.recv32.download(0, std.mem.sliceAsBytes(e.x));
            } else {
                try e.embedRow(tok);
            }
        } else {
            try e.embedRow(tok);
        }
        for (e.layers, 0..) |*l, i| try e.layerRow(l, pos, sk, nb, i);
        if (e.world == 2) {
            const lib = e.lib orelse return error.NoRank;
            const comm = e.comm orelse return error.NoRank;
            if (e.rank == 0) {
                try e.x32.upload(0, std.mem.sliceAsBytes(e.x));
                try lib.check(lib.api.ncclSend(e.x32.ptr, dims, .f32, 1, comm, e.s.handle), "ncclSend");
                try lib.check(lib.api.ncclRecv(e.recv32.ptr, dims, .f32, 1, comm, e.s.handle), "ncclRecv");
                try e.s.synchronize();
                try e.recv32.download(0, std.mem.sliceAsBytes(e.x));
            } else {
                try e.x32.upload(0, std.mem.sliceAsBytes(e.x));
                try lib.check(lib.api.ncclSend(e.x32.ptr, dims, .f32, 0, comm, e.s.handle), "ncclSend");
                try e.s.synchronize();
            }
        }
        if (e.rank == 0) try e.headRow();
    }

    // ------- world 2's row-control wire (the HTTP server boot's mirror contract) -------
    //
    // The lockstep contract is row-count: every `put` is one paired NCCL row (send/recv each
    // way), the mirror's token value is rank 0's business. But the lane surface changes the
    // cache length OUTSIDE a put — a stream's release rolls it to 0, a prefix keep rolls it
    // back — and the mirror would otherwise key its pool rows at a stale length. So world 2's
    // surface rows carry one 8-byte control word ahead of the data row, and release/keep send
    // a control-only exchange. The math never changes: same kernel work, same collectives for
    // the row's values, one extra tiny u32 pair per row.

    /// The control word: op 1 = a data row follows (`tok` is rank 0's row token), op 0 = the
    /// mirror rolls its cache length to 0 (release), op 2 = the mirror rolls it to `tok`
    /// (a prefix keep).
    pub const ctrl_row: u32 = 1;
    pub const ctrl_reset: u32 = 0;
    pub const ctrl_keep: u32 = 2;

    fn ctrlSend(e: *Engine, op: u32, tok: u32) !void {
        const lib = e.lib orelse return error.NoRank;
        const comm = e.comm orelse return error.NoRank;
        var host = [2]u32{ op, tok };
        try e.ctrl32.upload(0, std.mem.sliceAsBytes(&host));
        try lib.check(lib.api.ncclSend(e.ctrl32.ptr, 2, .u32, 1, comm, e.s.handle), "ncclSend");
        try e.s.synchronize();
    }

    fn ctrlRecv(e: *Engine) !struct { op: u32, tok: u32 } {
        const lib = e.lib orelse return error.NoRank;
        const comm = e.comm orelse return error.NoRank;
        try lib.check(lib.api.ncclRecv(e.ctrl32.ptr, 2, .u32, 0, comm, e.s.handle), "ncclRecv");
        try e.s.synchronize();
        var host = [2]u32{ 0, 0 };
        try e.ctrl32.download(0, std.mem.sliceAsBytes(&host));
        return .{ .op = host[0], .tok = host[1] };
    }

    /// rank 0's lane-surface row under world 2: the control word first, then the paired row.
    pub fn leaderRow(e: *Engine, tok: u32) !void {
        if (e.world != 2) return e.put(tok);
        try e.ctrlSend(ctrl_row, tok);
        try e.put(tok);
    }

    /// rank 0's prefix keep under world 2: the mirror rolls its cache length to `kept_len`.
    pub fn leaderKeep(e: *Engine, kept_len: u32) !void {
        if (e.world != 2 or e.rank != 0) return;
        try e.ctrlSend(ctrl_keep, kept_len);
    }

    /// rank 0's stream release under world 2: the mirror rolls its cache length to 0.
    pub fn leaderReset(e: *Engine) !void {
        if (e.world != 2 or e.rank != 0) return;
        try e.ctrlSend(ctrl_reset, 0);
    }

    /// rank 1's mirror step: the control word, then its paired work (a `.row` runs the full
    /// put — its cache length advances exactly like rank 0's; its row's token value is rank
    /// 0's business, exactly the receipted lane-tp2 lockstep). An unknown control word
    /// refuses rather than desyncing the ranks.
    pub fn mirrorStep(e: *Engine) !void {
        if (e.world != 2) return;
        const c = try e.ctrlRecv();
        switch (c.op) {
            ctrl_row => try e.put(c.tok),
            ctrl_reset => e.len = 0,
            ctrl_keep => e.len = c.tok,
            else => return error.NoRank,
        }
    }

    /// Greedy continuation (rank 0): `steps` first-max argmax picks, each picked token fed back.
    /// The peer rank calls `put` the same number of times to stay in the collectives' lockstep.
    pub fn greedy(e: *Engine, steps: usize, out: []u32) !void {
        std.debug.assert(out.len >= steps);
        for (0..steps) |g| {
            var best: usize = 0;
            var bv = -std.math.inf(f32);
            for (e.logits, 0..) |v, i| if (v > bv) {
                bv = v;
                best = i;
            };
            out[g] = @intCast(best);
            if (g + 1 < steps) try e.put(@intCast(best));
        }
    }

    fn embedRow(e: *Engine, tok: u32) !void {
        const dims = e.hidden();
        const b = e.embed_tbl orelse return error.MissingTensor;
        const row = try e.gpa.alloc(u8, dims * 2);
        defer e.gpa.free(row);
        try b.download(@as(usize, tok) * dims * 2, row); // one bf16 [hidden] table row
        for (e.x, 0..) |*v, i| {
            const bits: u32 = @as(u32, std.mem.readInt(u16, row[i * 2 ..][0..2], .little)) << 16;
            v.* = @bitCast(bits);
        }
    }

    fn headRow(e: *Engine) !void {
        serve.normRow(e.x, e.final_norm.?, e.c.eps);
        try e.x32.upload(0, std.mem.sliceAsBytes(e.x));
        try e.dense.decode(e.s, e.head.?, e.x32.ptr, .f32, 1, e.logits32.ptr, .f32);
        try e.s.synchronize();
        try e.logits32.download(0, std.mem.sliceAsBytes(e.logits));
    }

    /// The dense-shaped wave: gate/up GEMMs, tf_m3_swigluoai, the down GEMM; the fp16 rows
    /// land on the host in `o16` (the caller's fp16 rounding point on the residual add).
    fn waveRow(e: *Engine, gate: exl3.Weight, up: exl3.Weight, down: exl3.Weight, inter: u32) !void {
        const dims = e.hidden();
        const n: usize = inter;
        try e.dense.decode(e.s, gate, e.x32.ptr, .f32, 1, e.g16.ptr, .f16);
        try e.dense.decode(e.s, up, e.x32.ptr, .f32, 1, e.u16.ptr, .f16);
        try glue.swigluoai(e.gk, e.s, e.g16.ptr, e.u16.ptr, e.a16.ptr, 1, n, e.c.swiglu_limit);
        try e.dense.decode(e.s, down, e.a16.ptr, .f16, 1, e.g16.ptr, .f16);
        try e.s.synchronize();
        try e.g16.download(0, std.mem.sliceAsBytes(e.o16[0..dims]));
    }

    /// One owned layer's row, attention half then MLP wave, serve.put's contracts line for line.
    fn layerRow(e: *Engine, l: *Layer, pos: usize, sk: usize, nb: usize, lidx: usize) !void {
        const c = &e.c;
        const dims = e.hidden();
        const qk = e.qkr();
        const kv = e.kvr();
        const hd: usize = c.head_dim;
        const qh: usize = c.attn_heads;
        const kvh: usize = c.kv_heads;
        const rot: usize = c.rotary_dim;
        const lbytes: usize = lidx * keys_cap; // this layer's first row in the caches/plane
        const pages: usize = keys_cap / sp.page;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));

        // -- the attention half: gemma norm -> q/k/v GEMMs -> per-head qk norm + partial rope ->
        //    fp16 stores -> the indexer's side plane -> selection -> block-sparse attention -> o
        @memcpy(e.xn, e.x);
        serve.normRow(e.xn, l.in_norm, c.eps);
        try e.x32.upload(0, std.mem.sliceAsBytes(e.xn));
        const q_w = l.q orelse return error.MissingTensor;
        const k_w = l.k orelse return error.MissingTensor;
        const v_w = l.v orelse return error.MissingTensor;
        const o_w = l.o orelse return error.MissingTensor;
        try e.dense.decode(e.s, q_w, e.x32.ptr, .f32, 1, e.proj32.ptr, .f32);
        try e.dense.decode(e.s, k_w, e.x32.ptr, .f32, 1, e.proj32.ptr + qk * 4, .f32);
        try e.dense.decode(e.s, v_w, e.x32.ptr, .f32, 1, e.proj32.ptr + (qk + kv) * 4, .f32);
        try e.s.synchronize();
        try e.proj32.download(0, std.mem.sliceAsBytes(e.projf));

        // q: per-head gemma norm + rope, one fp16 rounding; k: norm + rope into the cache row;
        // v: fp16 into the cache row (serve.put's stores, the fork's cache contract)
        const qs = e.projf[0..qk];
        const ks_ = e.projf[qk..][0..kv];
        const vs = e.projf[qk + kv ..][0..kv];
        for (0..qh) |h| {
            const row = qs[h * hd ..][0..hd];
            serve.normRow(row, l.q_norm, c.eps);
            serve.ropeRow(row, pos, rot);
            for (row, 0..) |v, i| e.q16[h * hd + i] = @floatCast(serve.f16r(v));
        }
        for (0..kvh) |h| {
            const row = ks_[h * hd ..][0..hd];
            serve.normRow(row, l.k_norm, c.eps);
            serve.ropeRow(row, pos, rot);
            for (row, 0..) |v, i| e.k16[h * hd + i] = @floatCast(serve.f16r(v));
            const vrow = vs[h * hd ..][0..hd];
            for (vrow, 0..) |v, i| e.v16[h * hd + i] = @floatCast(serve.f16r(v));
        }
        const co: usize = (lbytes + pos) * kv * 2;
        try e.kc.upload(co, std.mem.sliceAsBytes(e.k16));
        try e.vc.upload(co, std.mem.sliceAsBytes(e.v16));
        try e.qdev16.upload(0, std.mem.sliceAsBytes(e.q16));

        // the indexer's side path: the fp16 passthrough projections (host GEMV, fp32 math over
        // fp16 weights), per-head norm + rope, fp16 stores, the side-plane append
        const iq_w = l.iq orelse return error.MissingTensor;
        const ik_w = l.ik orelse return error.MissingTensor;
        try gemvF16(e.xn, iq_w, e.iq32);
        try gemvF16(e.xn, ik_w, e.ik32);
        const id: usize = c.i_dim;
        for (0..c.i_heads) |h| {
            const row = e.iq32[h * id ..][0..id];
            serve.normRow(row, l.iq_norm, c.eps);
            serve.ropeRow(row, pos, rot);
            for (row, 0..) |v, i| e.iq16[h * id + i] = @floatCast(serve.f16r(v));
        }
        serve.normRow(e.ik32, l.ik_norm, c.eps);
        serve.ropeRow(e.ik32, pos, rot);
        for (e.ik32, 0..) |v, i| e.ik16[i] = @floatCast(serve.f16r(v));
        try e.iqdev16.upload(0, std.mem.sliceAsBytes(e.iq16));
        try e.plane.upload((lbytes + pos) * id * 2, std.mem.sliceAsBytes(e.ik16));

        // the selection: tf_m3_scores over the fp16 rows (fp32 scoring, the own-block +inf boost)
        // then tf_m3_topk's fixed-width picks (descending, ties at the smaller id, -1 pad)
        try e.qpos64.upload(0, std.mem.sliceAsBytes(&[1]i64{@intCast(pos)}));
        try e.seqlens.upload(0, std.mem.sliceAsBytes(&[1]i64{@intCast(sk)}));
        try sp.scores(e.ks, e.s, e.iqdev16.ptr, e.plane.ptr + lbytes * id * 2, e.qpos64.ptr, e.kpos64.ptr, e.scores32.ptr, 1, nb, sk, c.i_heads, 1, 1, 0);
        try sp.topk16(e.ks, e.s, e.scores32.ptr, e.sel32.ptr, nb, kvh);
        // the trace: the last layer's selection (the smoke reads this)
        if (lidx + 1 == e.layers.len) {
            try e.s.synchronize();
            try e.sel32.download(0, std.mem.sliceAsBytes(e.sel_seen));
            var valid: usize = 0;
            for (e.sel_seen[e.sel_seen.len - @as(usize, c.i_topk) ..]) |v| {
                if (v >= 0) valid += 1;
            }
            e.last.sel_valid = valid;
        }
        // the block-sparse attention: the splits read the paged cache by the selection, the
        // combine merges them into the fp16 rows
        try sp.decode(e.ks, e.s, e.qdev16.ptr, e.kc.ptr + lbytes * kv * 2, e.vc.ptr + lbytes * kv * 2, e.table.ptr, e.sel32.ptr, e.seqlens.ptr, e.part32.ptr, qh, kvh, pages, scale, 1, nsplit);
        try sp.combine(e.ks, e.s, e.part32.ptr, e.attn16.ptr, qh, 1, nsplit);
        try e.dense.decode(e.s, o_w, e.attn16.ptr, .f16, 1, e.g16.ptr, .f16);
        try e.s.synchronize();
        try e.g16.download(0, std.mem.sliceAsBytes(e.o16[0..dims]));
        for (0..dims) |i| e.x[i] += serve.f16r(@as(f32, e.o16[i]));

        // -- the MLP wave: gemma norm over the residual, dense (layers 0-2) or routed + shared
        @memcpy(e.xn2, e.x);
        serve.normRow(e.xn2, l.post_norm, c.eps);
        try e.x32.upload(0, std.mem.sliceAsBytes(e.xn2));
        switch (l.mlp) {
            .dense => |m| {
                try e.waveRow(m.gate, m.up, m.down, c.dense_inter);
                for (0..dims) |i| e.x[i] += serve.f16r(@as(f32, e.o16[i]));
            },
            .moe => |m| {
                // the noaux sigmoid+bias router on the host, serve.route's exact fp32 contract:
                // select on sigmoid+bias (strict >, ties at the smaller id), weights = unbiased
                // sigmoid normalized over the picks, * routed_scale, each .half() (fp16-rounded
                // values carried as f32, the device epilogue's weight tensor's width)
                serve.route(e.xn2, m.router, m.bias, e.picks, e.elogits, e.sel_e, e.pickw, c);
                var fires: usize = 0;
                var wsum: f32 = 0;
                for (e.picks) |p| {
                    if (p) fires += 1;
                }
                for (e.pickw) |w| wsum += w;
                e.last.fired = fires;
                e.last.wsum = wsum;
                try e.pick32.upload(0, std.mem.sliceAsBytes(e.sel_e));
                try e.wts32.upload(0, std.mem.sliceAsBytes(e.pickw));
                const sc = try exl3e.scratch(e.exp_scratch.ptr, m.routed, 1, c.topk);
                try e.experts.run(e.s, m.routed, e.x32.ptr, dims, .f32, e.pick32.ptr, e.wts32.ptr, 1, e.moe32.ptr, sc);
                // the shared expert wave beside the routed one
                try e.waveRow(m.sgate, m.sup, m.sdown, c.shared_inter);
                try e.moe32.download(0, std.mem.sliceAsBytes(e.xn)); // xn reused as the routed download
                for (0..dims) |i| e.x[i] += serve.f16r(e.xn[i] + @as(f32, e.o16[i]));
            },
        }
    }
};

/// One tracked runtime-buffer allocation: appended to the caller's build-time list so a
/// mid-literal failure frees exactly what landed, and e.deinit() owns it after construction.
fn allocRow(bufs: *[25]cuda.DeviceBuffer, n: *usize, d: *const cuda.Driver, bytes: usize) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(d, bytes);
    bufs[n.*] = b;
    n.* += 1;
    return b;
}

fn freeBuf(b: *cuda.DeviceBuffer) void {
    if (b.len > 0) b.free();
    b.len = 0;
}

fn deinitLayer(gpa: std.mem.Allocator, l: *Layer) void {
    for ([_]*[]f32{ &l.in_norm, &l.post_norm, &l.q_norm, &l.k_norm, &l.iq_norm, &l.ik_norm }) |p| {
        if (p.len > 0) gpa.free(p.*);
    }
    if (l.iq) |w| gpa.free(w);
    if (l.ik) |w| gpa.free(w);
    switch (l.mlp) {
        .dense => {},
        .moe => |m| {
            if (m.router.len > 0) gpa.free(m.router);
            if (m.bias.len > 0) gpa.free(m.bias);
        },
    }
}

/// y = x @ W^T exactly as serve.linear accumulates (fp32 over fp16 weights widened per element).
fn gemvF16(x: []const f32, w: []const f16, y: []f32) !void {
    const n_in = x.len;
    if (w.len != y.len * n_in) return error.MissingTensor;
    for (0..y.len) |o| {
        var acc: f32 = 0;
        const row = w[o * n_in ..][0..n_in];
        for (0..n_in) |j| acc += x[j] * @as(f32, row[j]);
        y[o] = acc;
    }
}

/// bf16 -> f32 widening (exact) of one norm row.
fn expandNorm(gpa: std.mem.Allocator, bytes: []const u8, n: usize) ![]f32 {
    if (bytes.len != n * 2) return error.MissingTensor;
    const w = try gpa.alloc(f32, n);
    errdefer gpa.free(w);
    for (0..n) |i| {
        const bits: u32 = @as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16;
        w[i] = @bitCast(bits);
    }
    return w;
}

/// The checkpoint layer id a passthrough name carries ("language_model.model.layers.N."); a
/// foreign tower's or non-layer passthrough reads without a slot (the fixture's only one).
fn layerOf(name: []const u8) ?u32 {
    const mark = "layers.";
    const at = std.mem.indexOf(u8, name, mark) orelse return null;
    var i = at + mark.len;
    var v: u32 = 0;
    while (i < name.len and name[i] >= '0' and name[i] <= '9') : (i += 1) {
        v = v * 10 + (name[i] - '0');
    }
    if (i == at + mark.len) return null;
    return v;
}

fn ends(name: []const u8, suffix: []const u8) bool {
    return std.mem.endsWith(u8, name, suffix);
}

/// One passthrough tensor beside the plan: the norms expand to (1 + w) fp32 host rows, the
/// indexer's fp16 projections land on the host for the small GEMVs, the router and its bias stay
/// raw fp32 arrays (serve.route's inputs). The embed and the final norm the builder reads itself.
/// Missing-but-needed tensors set `fail` instead of erroring mid-walk (a per-tensor log first).
fn readPassthrough(
    gpa: std.mem.Allocator,
    c: *const config.Config,
    layers: []Layer,
    first: u32,
    name: []const u8,
    bytes: []const u8,
    routers: []?[]f32,
    biases: []?[]f32,
    fail: *bool,
) !void {
    const lno = layerOf(name) orelse return;
    if (lno < first) return;
    const lidx: usize = lno - first;
    if (lidx >= layers.len) return;
    const l = &layers[lidx];
    if (ends(name, "index_q_norm.weight")) {
        l.iq_norm = expNorm(fail, gpa, bytes, c.i_dim);
    } else if (ends(name, "index_k_norm.weight")) {
        l.ik_norm = expNorm(fail, gpa, bytes, c.i_dim);
    } else if (ends(name, "input_layernorm.weight")) {
        l.in_norm = expNorm(fail, gpa, bytes, c.hidden);
    } else if (ends(name, "post_attention_layernorm.weight")) {
        l.post_norm = expNorm(fail, gpa, bytes, c.hidden);
    } else if (ends(name, "q_norm.weight")) {
        l.q_norm = expNorm(fail, gpa, bytes, c.head_dim);
    } else if (ends(name, "k_norm.weight")) {
        l.k_norm = expNorm(fail, gpa, bytes, c.head_dim);
    } else if (ends(name, "index_q_proj.weight")) {
        l.iq = passF16(fail, gpa, bytes);
    } else if (ends(name, "index_k_proj.weight")) {
        l.ik = passF16(fail, gpa, bytes);
    } else if (ends(name, "block_sparse_moe.gate.weight")) {
        if (bytes.len != @as(usize, c.experts) * @as(usize, c.hidden) * 4) {
            fail.* = true;
        } else {
            const r = try gpa.alloc(f32, @as(usize, c.experts) * @as(usize, c.hidden));
            @memcpy(std.mem.sliceAsBytes(r), bytes);
            if (routers[lidx] != null) gpa.free(routers[lidx].?);
            routers[lidx] = r;
        }
    } else if (ends(name, "e_score_correction_bias")) {
        if (bytes.len != @as(usize, c.experts) * 4) {
            fail.* = true;
        } else {
            const r = try gpa.alloc(f32, c.experts);
            @memcpy(std.mem.sliceAsBytes(r), bytes);
            if (biases[lidx] != null) gpa.free(biases[lidx].?);
            biases[lidx] = r;
        }
    } else {
        return error.MissingTensor; // a passthrough the graph reads by no other spelling
    }
}

fn set(fail: *bool) error{MissingTensor} {
    fail.* = true;
    return error.MissingTensor;
}

/// The gemma (1 + w) expansion of one bf16 norm row, f32.
fn expNorm(fail: *bool, gpa: std.mem.Allocator, bytes: []const u8, n: usize) []f32 {
    return expandGemma(gpa, bytes, n) catch {
        fail.* = true;
        return &.{};
    };
}

/// The fp16 passthrough mat, []f16.
fn passF16(fail: *bool, gpa: std.mem.Allocator, bytes: []const u8) []f16 {
    return copyF16(gpa, bytes) catch {
        fail.* = true;
        return &.{};
    };
}

fn expandGemma(gpa: std.mem.Allocator, bytes: []const u8, n: usize) ![]f32 {
    if (bytes.len != n * 2) return error.MissingTensor;
    const w = try gpa.alloc(f32, n);
    errdefer gpa.free(w);
    for (0..n) |i| {
        const bits: u32 = @as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16;
        // The checkpoint gemma weight is ABSOLUTE (~1.0); serve.normRow applies its own
        // (1.0 + wi) — adding another 1 here made every layer's norm 1.5x the reference's.
        w[i] = @as(f32, @bitCast(bits));
    }
    return w;
}

fn copyF16(gpa: std.mem.Allocator, bytes: []const u8) ![]f16 {
    if (bytes.len % 2 != 0) return error.MissingTensor;
    const w = try gpa.alloc(f16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(w), bytes);
    return w;
}
