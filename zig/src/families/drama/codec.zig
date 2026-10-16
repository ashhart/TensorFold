//! Drama 3 codec on CUDA: ten codebooks a frame to 44.1 kHz mono PCM; decode is one pass, begin/advance chunked.

const std = @import("std");
const cuda = @import("cuda");
const pth = @import("codec_pth.zig");
const kern = @import("codec_kernels.zig");
const load = @import("codec_load.zig");

const Allocator = std.mem.Allocator;

pub const Options = struct { max_frames: usize = 1024 };

const latent = load.latent;
const codebooks = load.codebooks;
const layers = load.layers;
const heads = load.heads;
const head_dim = load.head_dim;
const window = load.window;
const ffn = load.ffn;
const rms_eps = load.rms_eps;
const layer_norm_eps = load.layer_norm_eps;
const convnext_hidden = load.convnext_hidden;
const upsample_stages = load.upsample_stages;
const decoder_dim = load.decoder_dim;
const rates = load.rates;
const dilations = load.dilations;
const latent_rows_per_frame = load.latent_rows_per_frame;
const samples_per_frame = load.samples_per_frame;
const out_channels = load.out_channels;

/// FP16 elements a frame each of the three activation buffers holds: the widest tensor of the decode.
const act_halves_per_frame = blk: {
    var most: usize = latent_rows_per_frame * convnext_hidden; // ConvNeXt hidden at 4T rows
    var rows: usize = latent_rows_per_frame;
    var ch: usize = decoder_dim;
    most = @max(most, rows * ch);
    for (rates) |r| {
        rows *= r;
        ch /= 2;
        most = @max(most, rows * ch);
    }
    break :blk most;
};

/// One causal consumer's kept rows: the last `rows` rows of `width` FP16 channels, right-aligned at `offset` in carry.
const Carry = struct { offset: usize, rows: usize, width: usize };

/// Every consumer that reads rows before a chunk's first: depthwise convs, conv0, transposed convs, conv7s, output.
const Carries = struct {
    dwconv: [upsample_stages]Carry,
    conv0: Carry,
    blocks: [rates.len]struct { transconv: Carry, units: [dilations.len]Carry },
    out: Carry,
    bytes: usize,
};

const carries: Carries = blk: {
    var c: Carries = undefined;
    var at: usize = 0;
    for (&c.dwconv) |*d| d.* = keep(&at, 6, latent);
    c.conv0 = keep(&at, 6, latent);
    var ch: usize = decoder_dim;
    for (&c.blocks) |*b| {
        b.transconv = keep(&at, 1, ch);
        ch /= 2;
        for (&b.units, dilations) |*u, d| u.* = keep(&at, 6 * d, ch);
    }
    c.out = keep(&at, 6, out_channels);
    c.bytes = at;
    break :blk c;
};

fn keep(at: *usize, rows: usize, width: usize) Carry {
    defer at.* += rows * width * 2;
    return .{ .offset = at.*, .rows = rows, .width = width };
}

comptime {
    std.debug.assert(samples_per_frame == 2048);
    std.debug.assert(out_channels == 96);
    // the transformer's scratch (2T rows of 1024 in one buffer, 9T in another) fits the same buffers
    std.debug.assert(act_halves_per_frame >= 9 * latent);
    std.debug.assert(carries.bytes == 268416);
}

pub const Model = struct {
    sample_rate: u32 = 44100,
    frame_length: usize = samples_per_frame,
    num_codebooks: usize = codebooks,
    max_frames: usize,
    ctx: *cuda.Context,
    kernels: kern.Kernels,
    weights: cuda.DeviceBuffer,
    w: load.Weights,
    acts: [3]cuda.DeviceBuffer,
    codes_dev: cuda.DeviceBuffer,
    audio_dev: cuda.DeviceBuffer,
    codes_host: cuda.HostBuffer,
    audio_host: cuda.HostBuffer,
    /// Each transformer layer's qkv rows [max_frames][3 * latent]: attention reads earlier chunks' keys here.
    kv: cuda.DeviceBuffer,
    /// The rows each causal consumer reads before a chunk (`carries`).
    carry: cuda.DeviceBuffer,
    /// Frames of the utterance `advance` has decoded.
    decoded: usize = 0,

    pub fn init(gpa: Allocator, io: std.Io, ctx: *cuda.Context, checkpoint_dir: []const u8, kernel_dir: []const u8, options: Options) !Model {
        if (options.max_frames == 0) return error.Invalid;
        try ctx.makeCurrent();
        const d = ctx.d;
        var kernels = try kern.Kernels.load(gpa, io, d, kernel_dir);
        errdefer kernels.unload();

        const path = try std.fs.path.join(gpa, &.{ checkpoint_dir, "codec.pth" });
        defer gpa.free(path);
        var archive = try pth.Archive.open(gpa, io, path);
        defer archive.close();
        var names = std.heap.ArenaAllocator.init(gpa);
        defer names.deinit();
        var stage: load.Stage = .{ .gpa = gpa };
        defer stage.bytes.deinit(gpa);
        var loader: load.Loader = .{ .gpa = gpa, .names = names.allocator(), .archive = &archive, .stage = &stage };
        const w = try loader.weights(options.max_frames);
        var weights = try cuda.DeviceBuffer.fromHost(d, stage.bytes.items);
        errdefer weights.free();

        var acts: [3]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (acts[0..made]) |*b| b.free();
        for (&acts) |*b| {
            b.* = try cuda.DeviceBuffer.alloc(d, act_halves_per_frame * options.max_frames * 2);
            made += 1;
        }
        var codes_dev = try cuda.DeviceBuffer.alloc(d, options.max_frames * codebooks * @sizeOf(u32));
        errdefer codes_dev.free();
        var audio_dev = try cuda.DeviceBuffer.alloc(d, options.max_frames * samples_per_frame * @sizeOf(f32));
        errdefer audio_dev.free();
        var codes_host = try cuda.HostBuffer.alloc(d, options.max_frames * codebooks * @sizeOf(u32));
        errdefer codes_host.free();
        var audio_host = try cuda.HostBuffer.alloc(d, options.max_frames * samples_per_frame * @sizeOf(f32));
        errdefer audio_host.free();
        var kv = try cuda.DeviceBuffer.alloc(d, layers * options.max_frames * 3 * latent * 2);
        errdefer kv.free();
        const carry = try cuda.DeviceBuffer.alloc(d, carries.bytes);
        return .{
            .max_frames = options.max_frames,
            .ctx = ctx,
            .kernels = kernels,
            .weights = weights,
            .w = w,
            .acts = acts,
            .codes_dev = codes_dev,
            .audio_dev = audio_dev,
            .codes_host = codes_host,
            .audio_host = audio_host,
            .kv = kv,
            .carry = carry,
        };
    }

    pub fn deinit(self: *Model) void {
        self.ctx.makeCurrent() catch {};
        self.carry.free();
        self.kv.free();
        self.audio_host.free();
        self.codes_host.free();
        self.audio_dev.free();
        self.codes_dev.free();
        for (&self.acts) |*b| b.free();
        self.weights.free();
        self.kernels.unload();
        self.* = undefined;
    }

    /// Starts an utterance for `advance`; `decode` and a failed `advance` also end one (the next starts at frame 0).
    pub fn begin(self: *Model) void {
        self.decoded = 0;
    }

    /// The frames after those decoded into the same rows of `output`, bit-identical to `decode(codes)` there.
    pub fn advance(self: *Model, stream: cuda.Stream, codes: []const [10]u32, output: []f32) !usize {
        const first = self.decoded;
        if (codes.len <= first) return error.NoFrames;
        self.decoded = 0;
        const samples = try self.run(stream, codes, first, true, output);
        self.decoded = codes.len;
        return samples;
    }

    /// Decodes frame-major codes (clamped as rvq.py clamps them) to codes.len * 2048 FP32 samples, device work done.
    pub fn decode(self: *Model, stream: cuda.Stream, codes: []const [10]u32, output: []f32) !usize {
        self.decoded = 0;
        return self.run(stream, codes, 0, false, output);
    }

    /// Frames [first, codes.len) into output[first * 2048 ..]; `keep_rows`: save the rows the next chunk reads.
    fn run(self: *Model, stream: cuda.Stream, codes: []const [10]u32, first: usize, keep_rows: bool, output: []f32) !usize {
        const frames = codes.len;
        if (frames == 0) return error.NoFrames;
        if (frames > self.max_frames) return error.TooManyFrames;
        const samples = frames * samples_per_frame;
        if (output.len < samples) return error.OutputTooSmall;
        try self.ctx.makeCurrent();
        const code_at = first * codebooks * @sizeOf(u32);
        const code_bytes = std.mem.sliceAsBytes(codes[first..]);
        const staged = self.codes_host.bytes[code_at..][0..code_bytes.len];
        @memcpy(staged, code_bytes);
        try self.codes_dev.uploadAsync(code_at, staged, stream.handle);
        try self.forward(stream, first, frames, keep_rows);
        const audio = self.audio_host.slice(f32)[first * samples_per_frame .. samples];
        try self.audio_dev.downloadAsync(first * samples_per_frame * @sizeOf(f32), std.mem.sliceAsBytes(audio), stream.handle);
        try stream.synchronize();
        @memcpy(output[first * samples_per_frame .. samples], audio);
        return samples;
    }

    fn wp(self: *const Model, offset: u64) u64 {
        return self.weights.ptr + offset;
    }

    /// Before a consumer reads rows [at, end) of `tensor`: restore the kept rows before `at`, then keep the last ones.
    fn history(self: *Model, s: cuda.Stream, c: Carry, tensor: u64, at: usize, end: usize, keep_rows: bool) !void {
        const row = c.width * 2;
        const back = @min(c.rows, at);
        if (back > 0) {
            const into: cuda.DeviceBuffer = .{ .d = self.carry.d, .ptr = tensor + (at - back) * row, .len = back * row, .borrowed = true };
            try into.copyFrom(0, self.carry.ptr + c.offset + (c.rows - back) * row, back * row, s.handle);
        }
        if (!keep_rows) return;
        const n = @min(c.rows, end);
        try self.carry.copyFrom(c.offset + (c.rows - n) * row, tensor + (end - n) * row, n * row, s.handle);
    }

    /// Frames [first, end): every stage computes only its new rows, at the absolute rows a whole decode uses.
    fn forward(self: *Model, s: cuda.Stream, first: usize, end: usize, keep_rows: bool) !void {
        const k = &self.kernels;
        const w = &self.w;
        const h = 2; // bytes an FP16 element
        const p = [3]u64{ self.acts[0].ptr, self.acts[1].ptr, self.acts[2].ptr };
        const frames = end - first;

        // rvq.py decode: z = semantic + residual codebooks through out_proj
        try k.rvqLatent(s, self.codes_dev.ptr + first * codebooks * @sizeOf(u32), self.wp(w.semantic_book), self.wp(w.residual_books), self.wp(w.out_w), self.wp(w.out_b), p[0] + first * latent * h, frames);

        // post_module (WindowLimitedTransformer): x in p[0], scratch in p[1] and p[2], each layer's qkv in kv
        const x = p[0];
        const xn = p[1];
        const att = p[1] + end * latent * h;
        const gu = p[2];
        const act = p[2] + end * 2 * ffn * h;
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        for (&w.layers, 0..) |*l, i| {
            const qkv = self.kv.ptr + i * self.max_frames * 3 * latent * h;
            try k.rmsNorm(s, x + first * latent * h, self.wp(l.attention_norm), xn + first * latent * h, frames, latent, rms_eps);
            try k.gemm(s, .{ .x = xn, .w = self.wp(l.wqkv), .y = qkv, .first = first, .m = end, .n = 3 * latent, .cin = latent, .ldx = latent, .ldy = 3 * latent });
            try k.ropeQk(s, qkv + first * 3 * latent * h, self.wp(w.rope) + first * head_dim * h, frames, latent);
            try k.windowAttention(s, qkv, att, first, end, heads, window, scale);
            // h = x + attention_layer_scale(wo(y)), in place over x
            try k.gemm(s, .{ .x = att, .w = self.wp(l.wo), .y = x, .first = first, .m = end, .n = latent, .cin = latent, .ldx = latent, .ldy = latent, .scale = self.wp(l.attention_scale), .residual = x, .ldr = latent });
            try k.rmsNorm(s, x + first * latent * h, self.wp(l.ffn_norm), xn + first * latent * h, frames, latent, rms_eps);
            try k.gemm(s, .{ .x = xn, .w = self.wp(l.w13), .y = gu, .first = first, .m = end, .n = 2 * ffn, .cin = latent, .ldx = latent, .ldy = 2 * ffn });
            try k.siluMul(s, gu + first * 2 * ffn * h, act + first * ffn * h, frames, ffn);
            try k.gemm(s, .{ .x = act, .w = self.wp(l.w2), .y = x, .first = first, .m = end, .n = latent, .cin = ffn, .ldx = ffn, .ldy = latent, .scale = self.wp(l.ffn_scale), .residual = x, .ldr = latent });
        }
        try k.rmsNorm(s, x + first * latent * h, self.wp(w.norm), p[1] + first * latent * h, frames, latent, rms_eps);

        // upsample: (ConvTranspose1d k2 s2, ConvNeXtBlock) twice; rows [first, end) -> [4 first, 4 end)
        var input = p[1];
        var spent = p[0];
        var from = first;
        var to = end;
        for (&w.upsample, carries.dwconv) |*u, carry| {
            const y = spent;
            try k.gemm(s, .{ .x = input, .w = self.wp(u.conv), .y = y, .first = from, .m = to, .n = 2 * latent, .cin = latent, .ldx = latent, .ldy = 2 * latent, .bias = self.wp(u.conv_b), .bias_mode = .separate, .period = latent });
            from *= 2;
            to *= 2;
            const dw = p[2];
            const normed = p[2] + to * latent * h;
            try self.history(s, carry, y, from, to, keep_rows);
            try k.dwconv(s, y, self.wp(u.dw), self.wp(u.dw_b), dw, from, to, latent);
            try k.layerNorm(s, dw + from * latent * h, self.wp(u.norm_w), self.wp(u.norm_b), normed + from * latent * h, to - from, latent, layer_norm_eps);
            const hidden = input; // the stage input is spent
            try k.gemm(s, .{ .x = normed, .w = self.wp(u.pw1), .y = hidden, .first = from, .m = to, .n = convnext_hidden, .cin = latent, .ldx = latent, .ldy = convnext_hidden, .bias = self.wp(u.pw1_b), .bias_mode = .fused, .act = .gelu });
            // y = input + gamma * pwconv2(...), in place over y's new rows
            try k.gemm(s, .{ .x = hidden, .w = self.wp(u.pw2), .y = y, .first = from, .m = to, .n = latent, .cin = convnext_hidden, .ldx = convnext_hidden, .ldy = latent, .bias = self.wp(u.pw2_b), .bias_mode = .fused, .scale = self.wp(u.gamma), .residual = y, .ldr = latent });
            input = y;
            spent = hidden;
        }

        // decoder.model: conv 1024 -> 1536 (kernel 7, causal); its kept rows replace the depthwise conv's in `input`
        var cur = spent;
        try self.history(s, carries.conv0, input, from, to, keep_rows);
        try k.gemm(s, .{ .x = input, .w = self.wp(w.conv0), .y = cur, .first = from, .m = to, .n = decoder_dim, .cin = latent, .ldx = latent, .ldy = decoder_dim, .taps = 7, .bias = self.wp(w.conv0_b), .bias_mode = .separate });
        var spare = [2]u64{ input, p[2] };
        var ch: usize = decoder_dim;
        for (&w.blocks, rates, carries.blocks) |*b, rate, kept| {
            const out_ch = ch / 2;
            // DecoderBlock: Snake, then ConvTranspose1d(kernel 2 * rate, stride rate) trimmed causally
            try k.snake(s, cur + from * ch * h, self.wp(b.snake.alpha), self.wp(b.snake.inv), spare[0] + from * ch * h, to - from, ch);
            try self.history(s, kept.transconv, spare[0], from, to, keep_rows);
            try k.gemm(s, .{ .x = spare[0], .w = self.wp(b.conv), .y = spare[1], .first = from, .m = to, .n = rate * out_ch, .cin = ch, .ldx = ch, .ldy = rate * out_ch, .taps = 2, .bias = self.wp(b.conv_b), .bias_mode = .separate, .period = out_ch });
            from *= rate;
            to *= rate;
            ch = out_ch;
            // ResidualUnits: x + conv1(snake(conv7(snake(x)))), rotating three buffers
            var xb = spare[1];
            var sb = cur;
            var tb = spare[0];
            for (&b.units, dilations, kept.units) |*u, dilation, carry| {
                try k.snake(s, xb + from * ch * h, self.wp(u.snake1.alpha), self.wp(u.snake1.inv), sb + from * ch * h, to - from, ch);
                try self.history(s, carry, sb, from, to, keep_rows);
                try k.gemm(s, .{ .x = sb, .w = self.wp(u.conv7), .y = tb, .first = from, .m = to, .n = ch, .cin = ch, .ldx = ch, .ldy = ch, .taps = 7, .dilation = dilation, .bias = self.wp(u.conv7_b), .bias_mode = .separate });
                try k.snake(s, tb + from * ch * h, self.wp(u.snake2.alpha), self.wp(u.snake2.inv), tb + from * ch * h, to - from, ch);
                try k.gemm(s, .{ .x = tb, .w = self.wp(u.conv1), .y = sb, .first = from, .m = to, .n = ch, .cin = ch, .ldx = ch, .ldy = ch, .bias = self.wp(u.conv1_b), .bias_mode = .separate, .residual = xb, .ldr = ch });
                const unit_out = sb;
                sb = tb;
                tb = xb;
                xb = unit_out;
            }
            cur = xb;
            spare = .{ sb, tb };
        }
        // Snake, conv 96 -> 1 (kernel 7, causal), tanh
        try k.snake(s, cur + from * ch * h, self.wp(w.out_snake.alpha), self.wp(w.out_snake.inv), cur + from * ch * h, to - from, ch);
        try self.history(s, carries.out, cur, from, to, keep_rows);
        try k.convOutTanh(s, cur, self.wp(w.out_conv), self.wp(w.out_conv_b), self.audio_dev.ptr, from, to, ch);
    }
};
