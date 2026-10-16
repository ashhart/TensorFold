//! Drama 3's temporal codebook tower: four shared and nine residual layers whose caches keep every earlier row.
const std = @import("std");
const cuda = @import("cuda");
const checkpoint = @import("core").checkpoint;
const Stager = @import("stager.zig").Stager;
const Buffer = cuda.DeviceBuffer;
const Linear = cuda.cublaslt.Linear;
const Allocator = std.mem.Allocator;

pub const Frame = [11]u32;
pub const Options = struct { context: usize = 8192, max_rows: usize = 128 };
pub const Sampling = struct { temperature: f32 = 1, top_p: f32 = 0.9, top_k: u32 = 30 };
const hidden = 2048;
const intermediate = 6144;
const query_width = 4096;
const kv_width = 512;
const qkv_width = 9216;
const layer_count = 13;
const workspace_bytes = 8 << 20;
const sample_shared_bytes = 65536;
/// drama_tower.cu's attention_split: the CTAs of drama_attention_decode's cluster for one query head.
const attention_split = 4;
const projections = [_][2]usize{
    .{ hidden, 2 * hidden },       .{ qkv_width, hidden },    .{ hidden, query_width },
    .{ 2 * intermediate, hidden }, .{ hidden, intermediate }, .{ 4096, hidden },
};

/// Diagnostic capture points of Model.prefillTraced in execution order, BF16 [rows][stageWidth(stage)] each.
pub const Stage = enum {
    shared_input,
    joined,
    mixed,
    attention_input,
    qkv,
    query,
    key,
    value,
    attended,
    attention_output,
    mlp_input,
    mlp_residual,
    gate_up,
    activated,
    mlp_output,
    output,
};

pub fn stageWidth(stage: Stage) usize {
    return switch (stage) {
        .shared_input, .mixed, .attention_input, .attention_output, .mlp_input, .mlp_residual, .mlp_output, .output => hidden,
        .joined => 2 * hidden,
        .qkv => qkv_width,
        .query, .attended => query_width,
        .key, .value => kv_width,
        .gate_up => 2 * intermediate,
        .activated => intermediate,
    };
}

/// Hashed by the trace driver so a capture names the exact tower it observed; unreferenced in production.
pub const trace_source = @embedFile("tower.zig");

const Config = struct {
    model_type: []const u8,
    semantic_start_token_id: u32,
    semantic_end_token_id: u32,
    eos_token_id: u32,
    audio_decoder_config: struct {
        decoder_type: []const u8,
        dim: usize,
        intermediate_size: usize,
        n_head: usize,
        n_local_heads: usize,
        head_dim: usize,
        shared_enrichment_layers: usize,
        num_codebooks: usize,
        vocab_size: usize,
        vq_backbone_codebooks: usize,
        rotary_percent: f32,
        rotary_interleaved: bool,
        rope_base: f32,
        norm_eps: f32,
        attention_output_gate: bool,
        layernorm_zero_centered_gamma: bool,
    },
};

const Layer = struct {
    embedding: Buffer,
    enorm: Buffer,
    hnorm: Buffer,
    eh: Buffer,
    input_norm: Buffer,
    qkv: Buffer,
    qnorm: Buffer,
    knorm: Buffer,
    out: Buffer,
    post_norm: Buffer,
    gate_up: Buffer,
    down: Buffer,
    final_norm: Buffer,
};
const Kernels = struct {
    embeddings: cuda.Function,
    mix: cuda.Function,
    norm: cuda.Function,
    rope: cuda.Function,
    attention: cuda.Function,
    attention_decode: cuda.Function,
    silu: cuda.Function,
    sample: cuda.Function,
    fn resolve(m: cuda.Module) !Kernels {
        return .{
            .embeddings = try m.function("drama_frame_embeddings"),
            .mix = try m.function("drama_mix"),
            .norm = try m.function("drama_norm"),
            .rope = try m.function("drama_rope_cache"),
            .attention = try m.function("drama_attention"),
            .attention_decode = try m.function("drama_attention_decode"),
            .silu = try m.function("drama_silu_mul"),
            .sample = try m.function("drama_sample"),
        };
    }
};

const Impl = struct {
    gpa: Allocator,
    driver: *const cuda.Driver,
    options: Options,
    semantic_start: u32,
    semantic_end: u32,
    eos: u32,
    eps: f32,
    rope_base: f32,
    buffers: std.ArrayList(Buffer) = .empty,
    plans: std.ArrayList(Linear) = .empty,
    module: ?cuda.Module = null,
    library: cuda.cublaslt.Library = undefined,
    library_open: bool = false,
    kernels: Kernels = undefined,
    layers: [layer_count]Layer = undefined,
    heads: [9]Buffer = undefined,
    embeddings: Buffer = undefined,
    workspace: Buffer = undefined,
    cache: Buffer = undefined,
    input_frames: Buffer = undefined,
    labels: Buffer = undefined,
    extra: Buffer = undefined,
    shared: Buffer = undefined,
    joined: Buffer = undefined,
    mixed: Buffer = undefined,
    normalized: Buffer = undefined,
    qkv: Buffer = undefined,
    queries: Buffer = undefined,
    attended: Buffer = undefined,
    branch: Buffer = undefined,
    residual: Buffer = undefined,
    gate_up: Buffer = undefined,
    activated: Buffer = undefined,
    output_hidden: Buffer = undefined,
    head_logits: Buffer = undefined,
    result: Buffer = undefined,
    position: usize = 0,
    input_rows: usize = 0,
    step: u64 = 0,
    seed: u64 = 0,

    fn release(p: *Impl) void {
        for (p.plans.items) |*plan| plan.deinit();
        p.plans.deinit(p.gpa);
        if (p.library_open) p.library.close();
        for (p.buffers.items) |*buffer| buffer.free();
        p.buffers.deinit(p.gpa);
        if (p.module) |*module| module.unload();
        const gpa = p.gpa;
        gpa.destroy(p);
    }

    fn allocate(p: *Impl, bytes: usize) !Buffer {
        var owned = try Buffer.alloc(p.driver, bytes);
        errdefer owned.free();
        try p.buffers.append(p.gpa, owned);
        return view(owned, 0, bytes);
    }

    fn weight(p: *Impl, st: *Stager, ck: *checkpoint.Checkpoint, name: []const u8, shape: []const usize) !Buffer {
        const t = try ck.expect(name, .bf16, shape);
        const out = try p.allocate(t.bytes.len);
        try st.put(out.ptr, t.bytes);
        return out;
    }

    fn layerWeight(p: *Impl, st: *Stager, ck: *checkpoint.Checkpoint, prefix: []const u8, suffix: []const u8, shape: []const usize) !Buffer {
        var name: [256]u8 = undefined;
        return p.weight(st, ck, try std.fmt.bufPrint(&name, "{s}.{s}.weight", .{ prefix, suffix }), shape);
    }

    fn loadLayer(p: *Impl, st: *Stager, ck: *checkpoint.Checkpoint, i: usize) !Layer {
        var name: [128]u8 = undefined;
        const prefix = if (i < 4)
            try std.fmt.bufPrint(&name, "audio_decoder_mtp_shared.layers.{d}", .{i})
        else
            try std.fmt.bufPrint(&name, "audio_decoder_mtp.layers.{d}", .{i - 4});
        const gate_up = try p.allocate(2 * intermediate * hidden * 2);
        var key: [256]u8 = undefined;
        for ([_][]const u8{ "w1", "w3" }, 0..) |part, half| {
            const t = try ck.expect(try std.fmt.bufPrint(&key, "{s}.transformer_layer.feed_forward.{s}.weight", .{ prefix, part }), .bf16, &.{ intermediate, hidden });
            try st.put(gate_up.ptr + half * intermediate * hidden * 2, t.bytes);
        }
        return .{
            .embedding = try p.layerWeight(st, ck, prefix, if (i < 4) "semantic_embedding" else "codebook_embedding", &.{ 4096, hidden }),
            .enorm = try p.layerWeight(st, ck, prefix, "enorm", &.{hidden}),
            .hnorm = try p.layerWeight(st, ck, prefix, "hnorm", &.{hidden}),
            .eh = try p.layerWeight(st, ck, prefix, "eh_proj", &.{ hidden, 2 * hidden }),
            .input_norm = try p.layerWeight(st, ck, prefix, "transformer_layer.attention_norm", &.{hidden}),
            .qkv = try p.layerWeight(st, ck, prefix, "transformer_layer.attention.wqkv", &.{ qkv_width, hidden }),
            .qnorm = try p.layerWeight(st, ck, prefix, "transformer_layer.attention.q_norm", &.{256}),
            .knorm = try p.layerWeight(st, ck, prefix, "transformer_layer.attention.k_norm", &.{256}),
            .out = try p.layerWeight(st, ck, prefix, "transformer_layer.attention.wo", &.{ hidden, query_width }),
            .post_norm = try p.layerWeight(st, ck, prefix, "transformer_layer.ffn_norm", &.{hidden}),
            .gate_up = gate_up,
            .down = try p.layerWeight(st, ck, prefix, "transformer_layer.feed_forward.w2", &.{ hidden, intermediate }),
            .final_norm = try p.layerWeight(st, ck, prefix, "final_layernorm", &.{hidden}),
        };
    }

    fn project(p: *Impl, stream: cuda.Stream, rows: usize, kind: usize, x: Buffer, w: Buffer, out: Buffer) !void {
        const plan = &p.plans.items[(rows - 1) * projections.len + kind];
        try plan.run(x.ptr, w.ptr, out.ptr, p.workspace.ptr, p.workspace.len, stream.handle);
    }

    fn norm(p: *Impl, stream: cuda.Stream, rows: usize, x: Buffer, residual: ?Buffer, w: Buffer, out: Buffer, residual_out: ?Buffer) !void {
        var a: cuda.Args = .{};
        a.add(x.ptr);
        a.add(if (residual) |r| r.ptr else @as(u64, 0));
        a.add(w.ptr);
        a.add(out.ptr);
        a.add(if (residual_out) |r| r.ptr else @as(u64, 0));
        a.add(@as(i32, hidden));
        a.add(p.eps);
        try launch(p.kernels.norm, stream, rows, 256, 0, &a);
    }

    /// One layer. `trace` is `{}` in production: every `stage` call below compiles to nothing.
    fn layer(p: *Impl, stream: cuda.Stream, rows: usize, i: usize, input: Buffer, labels: Buffer, trace: anytype) !void {
        const w = p.layers[i];
        var a: cuda.Args = .{};
        a.add(input.ptr);
        a.add(w.embedding.ptr);
        a.add(labels.ptr);
        a.add(if (i == 0) p.shared.ptr else @as(u64, 0));
        a.add(w.enorm.ptr);
        a.add(w.hnorm.ptr);
        a.add(p.joined.ptr);
        a.add(@as(i32, hidden));
        a.add(@as(i32, @intCast(if (i < 4) 0 else i - 4)));
        a.add(@as(i32, @intCast(p.semantic_start)));
        a.add(@as(i32, @intCast(p.semantic_end)));
        a.add(p.eps);
        try launch(p.kernels.mix, stream, rows, 256, 0, &a);
        try p.stage(trace, stream, i, rows, .joined);
        try p.project(stream, rows, 0, p.joined, w.eh, p.mixed);
        try p.stage(trace, stream, i, rows, .mixed);
        try p.norm(stream, rows, p.mixed, null, w.input_norm, p.normalized, null);
        try p.stage(trace, stream, i, rows, .attention_input);
        try p.project(stream, rows, 1, p.normalized, w.qkv, p.qkv);
        try p.stage(trace, stream, i, rows, .qkv);
        const cache_bytes = p.options.context * kv_width * 2;
        const keys = view(p.cache, i * 2 * cache_bytes, cache_bytes);
        const values = view(p.cache, (i * 2 + 1) * cache_bytes, cache_bytes);
        a = .{};
        a.add(p.qkv.ptr);
        a.add(w.qnorm.ptr);
        a.add(w.knorm.ptr);
        a.add(p.queries.ptr);
        a.add(keys.ptr);
        a.add(values.ptr);
        a.add(@as(i32, @intCast(p.position)));
        a.add(@as(i32, 16));
        a.add(@as(i32, 2));
        a.add(@as(i32, 256));
        a.add(@as(i32, 64));
        a.add(p.rope_base);
        a.add(p.eps);
        try launch(p.kernels.rope, stream, rows * 18, 256, 0, &a);
        try p.stage(trace, stream, i, rows, .query);
        try p.stage(trace, stream, i, rows, .key);
        try p.stage(trace, stream, i, rows, .value);
        a = .{};
        a.add(p.queries.ptr);
        a.add(keys.ptr);
        a.add(values.ptr);
        a.add(p.qkv.ptr);
        a.add(p.attended.ptr);
        a.add(@as(i32, @intCast(p.position)));
        a.add(@as(i32, 16));
        a.add(@as(i32, 2));
        a.add(@as(i32, 256));
        if (rows == 1) try launch(p.kernels.attention_decode, stream, 16 * attention_split, 256, 0, &a) else try launch(p.kernels.attention, stream, rows * 16, 256, 0, &a);
        try p.stage(trace, stream, i, rows, .attended);
        try p.project(stream, rows, 2, p.attended, w.out, p.branch);
        try p.stage(trace, stream, i, rows, .attention_output);
        try p.norm(stream, rows, p.branch, p.mixed, w.post_norm, p.normalized, p.residual);
        try p.stage(trace, stream, i, rows, .mlp_input);
        try p.stage(trace, stream, i, rows, .mlp_residual);
        try p.project(stream, rows, 3, p.normalized, w.gate_up, p.gate_up);
        try p.stage(trace, stream, i, rows, .gate_up);
        a = .{};
        a.add(p.gate_up.ptr);
        a.add(p.activated.ptr);
        a.add(@as(i32, intermediate));
        a.add(@as(i32, @intCast(rows)));
        try launch(p.kernels.silu, stream, (rows * intermediate + 255) / 256, 256, 0, &a);
        try p.stage(trace, stream, i, rows, .activated);
        try p.project(stream, rows, 4, p.activated, w.down, p.branch);
        try p.stage(trace, stream, i, rows, .mlp_output);
        try p.norm(stream, rows, p.branch, p.residual, w.final_norm, p.output_hidden, null);
        try p.stage(trace, stream, i, rows, .output);
    }

    /// One capture point, compiled out for a void trace; the tracer must finish reading on stream before returning.
    inline fn stage(p: *const Impl, trace: anytype, stream: cuda.Stream, i: usize, rows: usize, comptime which: Stage) !void {
        if (@TypeOf(trace) != void) {
            const width = comptime stageWidth(which);
            const cache_bytes = p.options.context * kv_width * 2;
            const data = switch (which) {
                .shared_input => p.shared,
                .joined => p.joined,
                .mixed => p.mixed,
                .attention_input, .mlp_input => p.normalized,
                .qkv => p.qkv,
                .query => p.queries,
                .key => view(p.cache, i * 2 * cache_bytes + p.position * width * 2, cache_bytes - p.position * width * 2),
                .value => view(p.cache, (i * 2 + 1) * cache_bytes + p.position * width * 2, cache_bytes - p.position * width * 2),
                .attended => p.attended,
                .attention_output, .mlp_output => p.branch,
                .mlp_residual => p.residual,
                .gate_up => p.gate_up,
                .activated => p.activated,
                .output => p.output_hidden,
            };
            try trace.capture(stream, i, which, view(data, 0, rows * width * 2), p.position);
        }
    }

    fn sample(p: *Impl, stream: cuda.Stream, logits: Buffer, is_bf16: bool, output_index: usize, sampling: Sampling) !void {
        var a: cuda.Args = .{};
        a.add(logits.ptr);
        a.add(@as(i32, @intFromBool(is_bf16)));
        a.add(@as(i32, if (output_index == 0) 4097 else 4096));
        a.add(sampling.temperature);
        a.add(@as(i32, @intCast(sampling.top_k)));
        a.add(sampling.top_p);
        a.add(p.seed);
        a.add(p.step);
        a.add(@as(i32, @intCast(output_index)));
        a.add(@as(i32, @intCast(p.semantic_start)));
        a.add(@as(i32, @intCast(p.eos)));
        a.add(p.result.ptr + 48);
        a.add(p.result.ptr);
        a.add(p.result.ptr + 44);
        try launch(p.kernels.sample, stream, 1, 1024, sample_shared_bytes, &a);
    }
};

fn view(buffer: Buffer, offset: usize, bytes: usize) Buffer {
    std.debug.assert(offset <= buffer.len and bytes <= buffer.len - offset);
    return .{ .d = buffer.d, .ptr = buffer.ptr + offset, .len = bytes, .borrowed = true };
}
fn launch(function: cuda.Function, stream: cuda.Stream, blocks: usize, threads: u32, shared: u32, args: *cuda.Args) !void {
    try cuda.launch.launch(function, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 }, .shared = shared }, stream, args);
}

pub const Model = struct {
    impl: *Impl,

    pub fn init(gpa: Allocator, io: std.Io, ctx: *cuda.Context, dir: []const u8, kernel_dir: []const u8, options: Options) !Model {
        if (options.context == 0 or options.context > 8192 or options.max_rows == 0 or options.max_rows > 128) return error.BadTowerCapacity;
        const config_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(config_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, config_path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(Config, gpa, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const c = parsed.value.audio_decoder_config;
        if (!std.mem.eql(u8, parsed.value.model_type, "fish_omni") or !std.mem.eql(u8, c.decoder_type, "codebook_mtp_shared") or
            c.dim != hidden or c.intermediate_size != intermediate or c.n_head != 16 or c.n_local_heads != 2 or c.head_dim != 256 or
            c.shared_enrichment_layers != 4 or c.num_codebooks != 10 or c.vocab_size != 4096 or c.vq_backbone_codebooks != 1 or
            c.rotary_percent != 0.25 or !c.rotary_interleaved or !c.attention_output_gate or !c.layernorm_zero_centered_gamma or
            c.rope_base != 10000000 or c.norm_eps != 1e-6 or parsed.value.semantic_end_token_id -| parsed.value.semantic_start_token_id != 4095 or
            parsed.value.eos_token_id >= parsed.value.semantic_start_token_id) return error.UnsupportedTowerConfig;
        const p = try gpa.create(Impl);
        p.* = .{ .gpa = gpa, .driver = ctx.d, .options = options, .semantic_start = parsed.value.semantic_start_token_id, .semantic_end = parsed.value.semantic_end_token_id, .eos = parsed.value.eos_token_id, .eps = c.norm_eps, .rope_base = c.rope_base };
        errdefer p.release();
        p.module = if (kernel_dir.len == 0) try cuda.Module.load(ctx.d, cuda.kernels.drama_tower) else blk: {
            const kernel_path = try std.fs.path.join(gpa, &.{ kernel_dir, "drama_tower.fatbin" });
            defer gpa.free(kernel_path);
            const image = try std.Io.Dir.cwd().readFileAllocOptions(io, kernel_path, gpa, .limited(1 << 26), .@"16", null);
            defer gpa.free(image);
            break :blk try cuda.Module.load(ctx.d, image);
        };
        p.kernels = try Kernels.resolve(p.module.?);
        try p.kernels.sample.allowDynamicShared(sample_shared_bytes);
        p.library = try cuda.cublaslt.Library.open();
        p.library_open = true;
        try p.plans.ensureTotalCapacityPrecise(gpa, options.max_rows * projections.len);
        for (1..options.max_rows + 1) |rows| for (projections) |shape| {
            var plan = try Linear.init(&p.library, rows, shape[0], shape[1], .bf16, workspace_bytes);
            errdefer plan.deinit();
            try p.plans.append(gpa, plan);
        };
        var ck = try checkpoint.Checkpoint.openModelPrefix(gpa, io, dir, "audio_decoder");
        defer ck.close();
        var stream = try cuda.Stream.init(ctx.d, true);
        defer stream.deinit();
        var st: Stager = .init(gpa, ctx.d, stream);
        defer st.deinit();
        for (ck.files.items) |*f| try st.addFile(f.path, f.map.memory);
        p.embeddings = try p.weight(&st, &ck, "audio_decoder.codebook_embeddings.weight", &.{ 40960, hidden });
        for (0..layer_count) |i| p.layers[i] = try p.loadLayer(&st, &ck, i);
        for (&p.heads, 0..) |*head, i| {
            var name: [128]u8 = undefined;
            head.* = try p.weight(&st, &ck, try std.fmt.bufPrint(&name, "audio_decoder_mtp.heads.{d}.weight", .{i}), &.{ 4096, hidden });
        }
        if (ck.unused() != 0) return error.UnusedTowerWeights;
        try st.finish();
        const rows = options.max_rows;
        p.workspace = try p.allocate(workspace_bytes);
        p.cache = try p.allocate(layer_count * 2 * options.context * kv_width * 2);
        p.input_frames = try p.allocate(rows * @sizeOf(Frame));
        p.labels = try p.allocate(rows * @sizeOf(Frame));
        p.extra = try p.allocate(rows * hidden * 2);
        p.shared = try p.allocate(rows * hidden * 2);
        p.joined = try p.allocate(rows * 2 * hidden * 2);
        p.mixed = try p.allocate(rows * hidden * 2);
        p.normalized = try p.allocate(rows * hidden * 2);
        p.qkv = try p.allocate(rows * qkv_width * 2);
        p.queries = try p.allocate(rows * query_width * 2);
        p.attended = try p.allocate(rows * query_width * 2);
        p.branch = try p.allocate(rows * hidden * 2);
        p.residual = try p.allocate(rows * hidden * 2);
        p.gate_up = try p.allocate(rows * 2 * intermediate * 2);
        p.activated = try p.allocate(rows * intermediate * 2);
        p.output_hidden = try p.allocate(rows * hidden * 2);
        p.head_logits = try p.allocate(4096 * 2);
        p.result = try p.allocate(48 + 16 * 4);
        return .{ .impl = p };
    }

    pub fn deinit(self: *Model) void {
        self.impl.release();
        self.* = undefined;
    }

    pub fn reset(self: *Model, stream: cuda.Stream, prompt_tail: []const u32, seed: u64) !void {
        const p = self.impl;
        try stream.synchronize();
        try p.result.fill8(0, null);
        var history: [16]u32 = @splat(std.math.maxInt(u32));
        const n = @min(prompt_tail.len, history.len);
        @memcpy(history[history.len - n ..], prompt_tail[prompt_tail.len - n ..]);
        try p.result.upload(48, std.mem.sliceAsBytes(&history));
        p.position = 0;
        p.input_rows = 0;
        p.step = 0;
        p.seed = seed;
        // Every KV position read later is rewritten first, so the cache is not cleared.
    }

    /// Stages input frames before the backbone reads the rows; the codebook-1 embedding is borrowed until the next one.
    pub fn setInput(self: *Model, stream: cuda.Stream, frames: []const Frame) !?Buffer {
        const p = self.impl;
        if (frames.len == 0 or frames.len > p.options.max_rows) return error.BadTowerRows;
        var has_audio = false;
        for (frames) |frame| {
            if (frame[0] >= p.semantic_start and frame[0] <= p.semantic_end) has_audio = true;
            for (frame[1..]) |v| if (v >= 4096) return error.InvalidCodebookIndex;
        }
        p.input_rows = frames.len;
        if (!has_audio) {
            try view(p.shared, 0, frames.len * hidden * 2).fill8(0, stream.handle);
            return null;
        }
        try p.input_frames.upload(0, std.mem.sliceAsBytes(frames));
        var a: cuda.Args = .{};
        a.add(p.input_frames.ptr);
        a.add(p.embeddings.ptr);
        a.add(p.extra.ptr);
        a.add(p.shared.ptr);
        a.add(@as(i32, hidden));
        a.add(@as(i32, @intCast(p.semantic_start)));
        a.add(@as(i32, @intCast(p.semantic_end)));
        try launch(p.kernels.embeddings, stream, frames.len, 256, 0, &a);
        return view(p.extra, 0, frames.len * hidden * 2);
    }

    /// Teacher-forces every prompt row but the last, whose tower pass uses generate()'s first sampled token.
    pub fn prefill(self: *Model, stream: cuda.Stream, hidden_rows: Buffer, next_frames: []const Frame) !void {
        return self.prefillTraced(stream, hidden_rows, next_frames, {});
    }

    /// prefill with stage captures: trace is {} (compiled out) or a tracer reading each stage on stream before return.
    pub fn prefillTraced(self: *Model, stream: cuda.Stream, hidden_rows: Buffer, next_frames: []const Frame, trace: anytype) !void {
        const p = self.impl;
        const n = next_frames.len;
        if (n == 0 or n > p.input_rows or n > p.options.max_rows or p.position + n > p.options.context or hidden_rows.len < n * hidden * 2) return error.BadTowerRows;
        try p.labels.upload(0, std.mem.sliceAsBytes(next_frames));
        try p.stage(trace, stream, 0, n, .shared_input);
        for (0..layer_count) |i| try p.layer(stream, n, i, if (i == 0) hidden_rows else p.output_hidden, p.labels, trace);
        try stream.synchronize();
        p.position += n;
        p.input_rows = 0;
    }

    pub fn generate(self: *Model, stream: cuda.Stream, semantic_logits: Buffer, hidden_row: Buffer, sampling: Sampling) !Frame {
        const p = self.impl;
        if (p.input_rows != 1 or p.position >= p.options.context or hidden_row.len < hidden * 2 or semantic_logits.len < 4097 * 4) return error.BadTowerRows;
        if (!std.math.isFinite(sampling.temperature) or sampling.temperature < 0 or !std.math.isFinite(sampling.top_p) or
            sampling.top_p <= 0 or sampling.top_p > 1 or sampling.top_k == 0 or sampling.top_k > std.math.maxInt(i32)) return error.BadSpeechSampling;
        try p.sample(stream, semantic_logits, false, 0, sampling);
        for (0..layer_count) |i| {
            try p.layer(stream, 1, i, if (i == 0) hidden_row else p.output_hidden, p.result, {});
            if (i >= 4) {
                try p.project(stream, 1, 5, p.output_hidden, p.heads[i - 4], p.head_logits);
                try p.sample(stream, p.head_logits, true, i - 2, sampling);
            }
        }
        try stream.synchronize();
        var result: [12]u32 = undefined;
        try p.result.download(0, std.mem.sliceAsBytes(&result));
        if (result[11] != 0) return error.NonFiniteSpeechLogits;
        p.position += 1;
        p.step += 1;
        p.input_rows = 0;
        return result[0..11].*;
    }
};
