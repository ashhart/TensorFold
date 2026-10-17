//! Drama 3 text backbone on CUDA (Qwen3.5-MoE hybrid, 30 DeltaNet + 10 attention layers), one request's state.
const std = @import("std");
const cuda = @import("cuda");

pub const config = @import("backbone_config.zig");
pub const weights = @import("backbone_weights.zig");
pub const kernels = @import("backbone_kernels.zig");
pub const gdn = @import("backbone_gdn.zig");

const H = config.hidden;
const top = config.top_k;

pub const Options = struct {
    context: usize = 8192,
    max_rows: usize = 128,
};

/// Prefill is chunked GDN (bf16 q/k), decode the packed recurrence (fp32 q/k); callers choose it, never row count.
pub const Phase = enum { prefill, decode };

pub const Rows = struct {
    // Borrowed views, valid until next run/reset/deinit. BF16 row-major [rows, hidden].
    hidden: cuda.DeviceBuffer,
    // F32 [4097] for the LAST row only: EOS first, then semantic_start..semantic_end.
    semantic_logits: cuda.DeviceBuffer,
    rows: usize,
};

/// Diagnostic capture points of forward, BF16 [rows, hidden] each; tf-drama-backbone-trace is the only tracer.
pub const Stage = enum { attention_input, attention_residual, attention_output, mlp_input, mlp_residual, mlp_output };

/// Hashed by the trace driver so a capture names the exact forward it observed; unreferenced in production.
pub const trace_source = @embedFile("backbone.zig");

/// Diagnostic selector for Model.replayOperator: a layer's attention or moe; tf-drama-backbone-replay only.
pub const ReplayOperator = enum { attention, moe };

/// Bump offsets in one device allocation, 256-byte aligned.
const Bump = struct {
    used: usize = 0,

    fn take(b: *Bump, bytes: usize) usize {
        const at = std.mem.alignForward(usize, b.used, 256);
        b.used = at + bytes;
        return at;
    }
};

/// Per-call activations for up to `max_rows` rows (offsets until `place`, then device addresses).
const Scratch = struct {
    ids: u64 = 0, // i32 [R]
    x: u64 = 0, // bf16 [R][H] layer delta (embedding, mixer or MoE output)
    residual: u64 = 0, // bf16 [R][H]
    normed: u64 = 0, // bf16 [R][H]
    proj: u64 = 0, // bf16 [R][max(gdn_in_rows, attn_in_rows)]
    conv: u64 = 0, // bf16 [R][conv_dim]
    core: u64 = 0, // bf16 [R][value_dim] recurrence output
    gated: u64 = 0, // bf16 [R][4096] gated core (DeltaNet) or gated attention, the out projection's input
    q: u64 = 0, // bf16 [R][heads][head_dim]
    moe_in: u64 = 0, // bf16 [R][moe_in_rows]
    picks: u64 = 0, // i32 [P]
    routed: u64 = 0, // f32 [P]
    xq: u64 = 0, // e4m3 [R][H]
    xs: u64 = 0, // f32 [R][H / 128]
    members: u64 = 0, // i32 [P]
    items: u64 = 0, // i32 [3 * max items]
    counts: u64 = 0, // i32 [2]
    rank: u64 = 0, // i32 [P]
    hist: u64 = 0, // i32 [ceil(P / 1024) * experts]
    gu: u64 = 0, // bf16 [P][gu_rows]
    hq: u64 = 0, // e4m3 [P][expert_inner]
    hs: u64 = 0, // f32 [P][expert_inner / 128]
    down: u64 = 0, // bf16 [P][H]
    shared_h: u64 = 0, // bf16 [R][shared_inner]
    shared_out: u64 = 0, // bf16 [R][H]
    gdn: u64 = 0, // the DeltaNet prefill workspace (gdn.Workspace), shared by every layer

    fn layout(rows: usize) struct { Scratch, usize } {
        const pairs = rows * top;
        const proj_cols = @max(weights.gdn_in_rows, weights.attn_in_rows);
        const items = cuda.grouped.maxItems(pairs, config.experts, tile);
        const blocks = (pairs + 1023) / 1024;
        var b: Bump = .{};
        var s: Scratch = .{};
        s.ids = b.take(rows * 4);
        s.x = b.take(rows * H * 2);
        s.residual = b.take(rows * H * 2);
        s.normed = b.take(rows * H * 2);
        s.proj = b.take(rows * proj_cols * 2);
        s.conv = b.take(rows * config.conv_dim * 2);
        s.core = b.take(rows * config.value_dim * 2);
        s.gated = b.take(rows * @max(config.value_dim, config.query_heads * config.head_dim) * 2);
        s.q = b.take(rows * config.query_heads * config.head_dim * 2);
        s.moe_in = b.take(rows * weights.moe_in_rows * 2);
        s.picks = b.take(pairs * 4);
        s.routed = b.take(pairs * 4);
        s.xq = b.take(rows * H);
        s.xs = b.take(rows * (H / config.block) * 4);
        s.members = b.take(pairs * 4);
        s.items = b.take(3 * items * 4);
        s.counts = b.take(2 * 4);
        s.rank = b.take(pairs * 4);
        s.hist = b.take(blocks * config.experts * 4);
        s.gu = b.take(pairs * weights.gu_rows * 2);
        s.hq = b.take(pairs * config.expert_inner);
        s.hs = b.take(pairs * (config.expert_inner / config.block) * 4);
        s.down = b.take(pairs * H * 2);
        s.shared_h = b.take(rows * config.shared_inner * 2);
        s.shared_out = b.take(rows * H * 2);
        s.gdn = b.take(gdnWorkspace(rows)[1]);
        return .{ s, b.used };
    }

    fn place(s: *Scratch, base: u64) void {
        inline for (@typeInfo(Scratch).@"struct".field_names) |name| @field(s, name) += base;
    }
};

const tile = 16; // routed pairs an expert item takes (the plan's decode tile; the expert kernel's 16 mma rows)

const conv_state_bytes = (config.conv_taps - 1) * config.conv_dim * 2; // bf16 [3][conv_dim] a DeltaNet layer
const rec_state_bytes = config.linear_value_heads * config.linear_dim * config.linear_dim * 4; // f32 [32][128][128]
const gdn_qkv_cache_bytes = gdn.chunk_rows * config.conv_dim * 2; // bf16 [64][conv_dim] the unfinished chunk's rows
const gdn_ba_cache_bytes = gdn.chunk_rows * 2 * config.linear_value_heads * 2; // bf16 [64][b 32 | a 32]
const gdn_state_bytes = 2 * rec_state_bytes + gdn_qkv_cache_bytes + gdn_ba_cache_bytes; // a DeltaNet layer's gdn.State

fn gdnWorkspace(rows: usize) struct { gdn.Workspace, usize } {
    return gdn.Workspace.layout(rows, config.linear_key_heads, config.linear_value_heads);
}

/// DeltaNet layer `slot`'s recurrent state in the state buffer at `base`: two FP32 banks, then the row caches.
fn gdnState(base: u64, slot: usize) gdn.State {
    const at = base + config.linear_layers * conv_state_bytes + slot * gdn_state_bytes;
    return .{
        .banks = .{ at, at + rec_state_bytes },
        .qkv_cache = at + 2 * rec_state_bytes,
        .ba_cache = at + 2 * rec_state_bytes + gdn_qkv_cache_bytes,
    };
}

pub const Model = struct {
    hidden: usize,
    semantic_start: u32,
    semantic_end: u32,
    eos: u32,

    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    options: Options,
    k: kernels.Kernels,
    w: weights.Weights,
    scratch_buf: cuda.DeviceBuffer,
    state_buf: cuda.DeviceBuffer, // DeltaNet conv states, then per DeltaNet layer its gdn.State banks and row caches
    kv_buf: cuda.DeviceBuffer, // per attention layer: k cache then v cache, bf16 [context][kv_heads][head_dim]
    hidden_buf: cuda.DeviceBuffer, // bf16 [max_rows][H]
    logits_buf: cuda.DeviceBuffer, // f32 [4097], the last row's semantic logits
    tokens: cuda.HostBuffer, // pinned i32 [max_rows]
    s: Scratch,
    gdn_ws: gdn.Workspace, // placed in scratch_buf
    gdn_states: [config.linear_layers]gdn.State, // per DeltaNet layer: device addresses and chunk metadata
    pos: usize = 0,
    /// Set while a run's device work is in flight; a failed run leaves it set and only reset clears it.
    poisoned: bool = false,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *cuda.Context, checkpoint_dir: []const u8, kernel_dir: []const u8, options: Options) !Model {
        if (options.max_rows == 0 or options.context == 0 or options.context > 262144) return error.BadDramaOptions;
        if (try ctx.capability() < 89) return error.DramaNeedsFp8Mma; // the experts' e4m3 mma is sm_89+
        const d = ctx.d;
        const c = try config.Config.read(gpa, io, checkpoint_dir);
        var k = try kernels.Kernels.load(gpa, io, d, kernel_dir);
        errdefer k.unload();
        var w = blk: {
            var stream = try cuda.Stream.init(d, true);
            defer stream.deinit();
            break :blk try weights.load(gpa, io, d, stream, checkpoint_dir, c);
        };
        errdefer w.deinit();
        const R = options.max_rows;
        var s, const scratch_bytes = Scratch.layout(R);
        var scratch_buf = try cuda.DeviceBuffer.alloc(d, scratch_bytes);
        errdefer scratch_buf.free();
        try scratch_buf.fill8(0, null);
        s.place(scratch_buf.ptr);
        const gdn_ws = gdnWorkspace(R)[0].place(s.gdn);
        var state_buf = try cuda.DeviceBuffer.alloc(d, config.linear_layers * (conv_state_bytes + gdn_state_bytes));
        errdefer state_buf.free();
        try state_buf.fill8(0, null);
        var gdn_states: [config.linear_layers]gdn.State = undefined;
        for (&gdn_states, 0..) |*st, slot| st.* = gdnState(state_buf.ptr, slot);
        var kv_buf = try cuda.DeviceBuffer.alloc(d, config.full_layers * 2 * kvBytes(options.context));
        errdefer kv_buf.free();
        var hidden_buf = try cuda.DeviceBuffer.alloc(d, R * H * 2);
        errdefer hidden_buf.free();
        var logits_buf = try cuda.DeviceBuffer.alloc(d, weights.head_cols * 4);
        errdefer logits_buf.free();
        const tokens = try cuda.HostBuffer.alloc(d, R * 4);
        return .{
            .hidden = H,
            .semantic_start = c.semantic_start,
            .semantic_end = c.semantic_end,
            .eos = c.eos,
            .gpa = gpa,
            .d = d,
            .options = options,
            .k = k,
            .w = w,
            .scratch_buf = scratch_buf,
            .state_buf = state_buf,
            .kv_buf = kv_buf,
            .hidden_buf = hidden_buf,
            .logits_buf = logits_buf,
            .tokens = tokens,
            .s = s,
            .gdn_ws = gdn_ws,
            .gdn_states = gdn_states,
        };
    }

    pub fn deinit(self: *Model) void {
        self.tokens.free();
        self.logits_buf.free();
        self.hidden_buf.free();
        self.kv_buf.free();
        self.state_buf.free();
        self.scratch_buf.free();
        self.w.deinit();
        self.k.unload();
        self.* = undefined;
    }

    /// Clears conv and recurrent states, row caches and chunk metadata to position 0; the KV cache is overwritten.
    pub fn reset(self: *Model, stream: cuda.Stream) !void {
        try self.state_buf.fill8(0, stream.handle);
        try stream.synchronize();
        for (&self.gdn_states) |*st| st.resetMetadata();
        self.pos = 0;
        self.poisoned = false;
    }

    fn kvBytes(context: usize) usize {
        return context * config.kv_heads * config.head_dim * 2;
    }

    fn convState(self: *const Model, slot: usize) u64 {
        return self.state_buf.ptr + slot * conv_state_bytes;
    }

    fn kCache(self: *const Model, slot: usize) u64 {
        return self.kv_buf.ptr + (2 * slot) * kvBytes(self.options.context);
    }

    fn vCache(self: *const Model, slot: usize) u64 {
        return self.kv_buf.ptr + (2 * slot + 1) * kvBytes(self.options.context);
    }

    // Appends rows; semantic rows mix in extra (the codebook-1 embedding) times 1/sqrt(2); overflow is refused first.
    pub fn run(self: *Model, stream: cuda.Stream, phase: Phase, tokens: []const u32, extra: ?cuda.DeviceBuffer) !Rows {
        return self.runTraced(stream, phase, tokens, extra, {});
    }

    /// run with stage captures: trace is {} (compiled out) or a tracer reading each stage on stream before returning.
    pub fn runTraced(self: *Model, stream: cuda.Stream, phase: Phase, tokens: []const u32, extra: ?cuda.DeviceBuffer, trace: anytype) !Rows {
        if (self.poisoned) return error.DramaStateNeedsReset;
        const n = tokens.len;
        if (n == 0) return error.EmptyDramaRun;
        if (n > self.options.max_rows) return error.TooManyDramaRows;
        if (self.pos + n > self.options.context) return error.DramaContextOverflow;
        for (tokens) |t| if (t >= config.vocab) return error.DramaTokenOutOfRange;
        if (extra) |e| if (e.len < n * H * 2) return error.DramaExtraRowsShort;
        self.poisoned = true;
        errdefer stream.synchronize() catch {}; // queued kernels may still read tokens/extra; drain before returning
        const ids = self.tokens.slice(i32)[0..n];
        for (ids, tokens) |*o, t| o.* = @intCast(t);
        try self.d.check(self.d.api.cuMemcpyHtoDAsync_v2(self.s.ids, ids.ptr, n * 4, stream.handle), "cuMemcpyHtoDAsync");
        try self.forward(stream, phase, n, extra, trace);
        try stream.synchronize();
        self.pos += n;
        self.poisoned = false;
        return .{
            .hidden = .{ .d = self.d, .ptr = self.hidden_buf.ptr, .len = n * H * 2, .borrowed = true },
            .semantic_logits = .{ .d = self.d, .ptr = self.logits_buf.ptr, .len = weights.head_cols * 4, .borrowed = true },
            .rows = n,
        };
    }

    fn forward(self: *Model, s: cuda.Stream, phase: Phase, n: usize, extra: ?cuda.DeviceBuffer, trace: anytype) !void {
        const k = &self.k;
        const sc = &self.s;
        try k.embed.bf16(s, sc.ids, self.w.embed, sc.x, n, H);
        if (extra) |e| try k.semanticMix(s, sc.ids, sc.x, e.ptr, n, H, self.semantic_start, self.semantic_end);
        for (&self.w.layers, 0..) |*l, i| {
            try k.addNorm(s, sc.x, sc.residual, l.ln1, sc.normed, n, H, config.eps, i == 0);
            try self.stage(trace, s, i, .attention_input, sc.normed, n);
            try self.stage(trace, s, i, .attention_residual, sc.residual, n);
            if (config.linear(i)) try self.deltaNet(s, phase, l, config.slot(i), n) else try self.fullAttention(s, l, config.slot(i), n);
            try self.stage(trace, s, i, .attention_output, sc.x, n);
            try k.addNorm(s, sc.x, sc.residual, l.ln2, sc.normed, n, H, config.eps, false);
            try self.stage(trace, s, i, .mlp_input, sc.normed, n);
            try self.stage(trace, s, i, .mlp_residual, sc.residual, n);
            try self.moe(s, l, n);
            try self.stage(trace, s, i, .mlp_output, sc.x, n);
        }
        try k.addNorm(s, sc.x, sc.residual, self.w.final_norm, self.hidden_buf.ptr, n, H, config.eps, false);
        try k.head(s, self.hidden_buf.ptr + (n - 1) * H * 2, self.w.head, weights.head_cols, H, self.logits_buf.ptr);
    }

    /// One capture point; with a void trace the comptime-false branch is never analyzed and emits nothing.
    inline fn stage(self: *const Model, trace: anytype, s: cuda.Stream, layer: usize, comptime which: Stage, ptr: u64, n: usize) !void {
        if (@TypeOf(trace) != void) try trace.capture(s, layer, which, cuda.DeviceBuffer{ .d = self.d, .ptr = ptr, .len = n * H * 2, .borrowed = true }, self.pos);
    }

    /// Diagnostic: one layer's production operator on uploaded chunks after a fresh reset; leaves the model poisoned.
    pub fn replayOperator(self: *Model, stream: cuda.Stream, layer: usize, operator: ReplayOperator, chunks: []const usize, input: []const u8, output: []u8) !void {
        if (self.poisoned) return error.DramaStateNeedsReset;
        if (self.pos != 0) return error.DramaReplayNeedsReset;
        if (layer >= self.w.layers.len) return error.DramaReplayLayerOutOfRange;
        var rows: usize = 0;
        for (chunks) |n| {
            if (n == 0) return error.EmptyDramaRun;
            if (n > self.options.max_rows) return error.TooManyDramaRows;
            rows += n;
        }
        if (rows == 0) return error.EmptyDramaRun;
        if (rows > self.options.context) return error.DramaContextOverflow;
        if (input.len != rows * H * 2 or output.len != rows * H * 2) return error.DramaReplayRowsMismatch;
        self.poisoned = true; // never cleared here: the other layers' state no longer matches pos
        errdefer stream.synchronize() catch {}; // queued copies may still touch input/output; drain before returning
        const l = &self.w.layers[layer];
        var row0: usize = 0;
        for (chunks) |n| {
            const at = row0 * H * 2;
            const bytes = n * H * 2;
            const normed: cuda.DeviceBuffer = .{ .d = self.d, .ptr = self.s.normed, .len = bytes, .borrowed = true };
            try normed.uploadAsync(0, input[at..][0..bytes], stream.handle);
            self.pos = row0;
            switch (operator) {
                .attention => if (config.linear(layer))
                    try self.deltaNet(stream, .prefill, l, config.slot(layer), n)
                else
                    try self.fullAttention(stream, l, config.slot(layer), n),
                .moe => try self.moe(stream, l, n),
            }
            const x: cuda.DeviceBuffer = .{ .d = self.d, .ptr = self.s.x, .len = bytes, .borrowed = true };
            try x.downloadAsync(0, output[at..][0..bytes], stream.handle);
            row0 += n;
        }
        try stream.synchronize();
        self.pos = row0;
    }

    /// Qwen3.5 GatedDeltaNet: in projection, causal conv + silu, gated delta rule, RMSNormGated * silu(z), out_proj.
    fn deltaNet(self: *Model, s: cuda.Stream, phase: Phase, l: *const weights.Layer, slot: usize, n: usize) !void {
        const k = &self.k;
        const sc = &self.s;
        const ld = weights.gdn_in_rows;
        try k.gemm(s, sc.normed, H, n, l.gdn_in, ld, H, sc.proj, ld);
        const recurrence: gdn.Run = .{
            .qkv = sc.conv,
            .ld_qkv = config.conv_dim,
            .ba = sc.proj + weights.gdn_b_col * 2,
            .ld_ba = ld,
            .a_log = l.a_log,
            .dt_bias = l.dt_bias,
            .y = sc.core,
            .rows = n,
            .nk = config.linear_key_heads,
            .nv = config.linear_value_heads,
            .dv = config.linear_dim,
            .scale = @floatCast(1.0 / @sqrt(@as(f64, config.linear_dim))),
        };
        const state = &self.gdn_states[slot];
        switch (phase) {
            .prefill => {
                try k.convPrefill(s, sc.proj, ld, l.conv, self.convState(slot), sc.conv, n, config.conv_dim);
                try k.gdn.prefill(s, state, self.gdn_ws, recurrence);
            },
            .decode => {
                try k.conv(s, sc.proj, ld, l.conv, self.convState(slot), sc.conv, n, config.conv_dim);
                try k.gdn.decode(s, state, recurrence);
            },
        }
        try k.gatedRms(s, sc.core, config.value_dim, sc.proj + weights.gdn_z_col * 2, ld, l.gnorm, sc.gated, n, config.eps);
        try k.gemm(s, sc.gated, config.value_dim, n, l.gdn_out, H, config.value_dim, sc.x, H);
    }

    /// Gated full attention: q|gate, k, v projection; q/k Gemma norm + partial RoPE; causal GQA; sigmoid gate; o_proj.
    fn fullAttention(self: *Model, s: cuda.Stream, l: *const weights.Layer, slot: usize, n: usize) !void {
        const k = &self.k;
        const sc = &self.s;
        const ld = weights.attn_in_rows;
        const width = config.query_heads * config.head_dim;
        try k.gemm(s, sc.normed, H, n, l.attn_in, ld, H, sc.proj, ld);
        try k.attention(s, .{
            .qkv = sc.proj,
            .ld_qkv = ld,
            .q_norm = l.q_norm,
            .k_norm = l.k_norm,
            .inv_freq = self.w.inv_freq,
            .q = sc.q,
            .kc = self.kCache(slot),
            .vc = self.vCache(slot),
            .out = sc.gated,
            .pos0 = self.pos,
            .rows = n,
            .heads = config.query_heads,
            .kv_heads = config.kv_heads,
            .eps = config.eps,
            .scale = 1.0 / 16.0, // head_dim ** -0.5
        });
        try k.gemm(s, sc.gated, width, n, l.attn_out, H, width, sc.x, H);
    }

    /// Qwen2MoeSparseMoeBlock: top-8 routing, block-FP8 experts, SwiGLU, weighted down, sigmoid-gated shared expert.
    fn moe(self: *Model, s: cuda.Stream, l: *const weights.Layer, n: usize) !void {
        const k = &self.k;
        const sc = &self.s;
        const pairs = n * top;
        const ld = weights.moe_in_rows;
        const I = config.expert_inner;
        try k.gemm(s, sc.normed, H, n, l.moe_in, ld, H, sc.moe_in, ld);
        try k.topk(s, sc.moe_in + weights.router_col * 2, ld, sc.picks, sc.routed, n, top);
        try k.actQuant(s, sc.normed, H, sc.xq, sc.xs, n, H);
        const plan: cuda.grouped.Plan = .{ .members = sc.members, .items = sc.items, .counts = sc.counts, .rank = sc.rank, .hist = sc.hist };
        try k.router.route(s, sc.picks, pairs, config.experts, tile, plan);
        const items = cuda.grouped.maxItems(pairs, config.experts, tile);
        try k.expert(s, plan, pairs, .{
            .a = sc.xq,
            .a_div = top,
            .a_scales = sc.xs,
            .w = l.gu,
            .w_scales = l.gu_s,
            .n = weights.gu_rows,
            .kdim = H,
            .max_items = items,
            .routed = null,
            .y = sc.gu,
        });
        try k.siluMul(s, sc.gu, weights.gu_rows, I, null, sc.hq, sc.hs, pairs);
        try k.expert(s, plan, pairs, .{
            .a = sc.hq,
            .a_div = 1,
            .a_scales = sc.hs,
            .w = l.dn,
            .w_scales = l.dn_s,
            .n = H,
            .kdim = I,
            .max_items = items,
            .routed = sc.routed,
            .y = sc.down,
        });
        try k.siluMul(s, sc.moe_in, ld, config.shared_inner, sc.shared_h, null, null, n);
        try k.gemm(s, sc.shared_h, config.shared_inner, n, l.shared_down, H, config.shared_inner, sc.shared_out, H);
        try k.combine(s, sc.down, top, sc.normed, l.shared_gate, sc.shared_out, sc.x, n, H);
    }
};

test "scratch layout keeps every region 256-byte aligned and disjoint" {
    const s, const total = Scratch.layout(128);
    var last: u64 = 0;
    inline for (@typeInfo(Scratch).@"struct".field_names, 0..) |name, i| {
        const at = @field(s, name);
        try std.testing.expect(at % 256 == 0);
        if (i > 0) try std.testing.expect(at > last);
        last = at;
    }
    try std.testing.expect(total > last);
}
