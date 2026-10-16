//! Diagnostic: replays a production codec decode op by op with stage captures; the replay must equal production.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

const Allocator = std.mem.Allocator;
const Model = drama.codec.Model;
const Weights = @FieldType(Model, "w");

// codec.zig's private geometry (modded_dac_vq.yaml), restated; the asserts below tie it to the model's types.
const latent = 1024;
const codebook_dim = 8;
const semantic_size = 4096;
const residual_size = 1024;
const residual_books = 9;
const codebooks = 1 + residual_books;
const heads = 16;
const head_dim = 64;
const ffn = 3072;
const window = 128;
const rms_eps: f32 = 1e-5;
const layer_norm_eps: f32 = 1e-6;
const convnext_hidden = 4 * latent;
const decoder_dim = 1536;
const rates = [_]usize{ 8, 8, 4, 2 };
const dilations = [_]usize{ 1, 3, 9 };
const samples_per_frame = 2048;
const out_channels = decoder_dim >> rates.len;
const max_frames = 1024;

comptime {
    std.debug.assert(@typeInfo(@FieldType(Weights, "layers")).array.len == 8);
    std.debug.assert(@typeInfo(@FieldType(Weights, "upsample")).array.len == 2);
    std.debug.assert(@typeInfo(@FieldType(Weights, "blocks")).array.len == rates.len);
    std.debug.assert(@typeInfo(@FieldType(@typeInfo(@FieldType(Weights, "blocks")).array.child, "units")).array.len == dilations.len);
    std.debug.assert(out_channels == 96);
}

const Dtype = enum {
    f16,
    f32,

    fn size(d: Dtype) usize {
        return switch (d) {
            .f16 => 2,
            .f32 => 4,
        };
    }

    fn numpy(d: Dtype) []const u8 {
        return switch (d) {
            .f16 => "float16",
            .f32 => "float32",
        };
    }
};

const Dumped = struct { sha: [64]u8, equal: bool };

const Tracer = struct {
    gpa: Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    model: *Model,
    stream: cuda.Stream,
    frames: usize,
    /// production.f32's bytes: the replayed PCM must equal them.
    production: []const u8,
    captures: std.ArrayList(u8) = .empty,
    count: usize = 0,
    written: usize = 0,
    replay_equal: bool = false,

    fn wp(t: *const Tracer, offset: u64) u64 {
        return t.model.weights.ptr + offset;
    }

    /// The model-owned device buffer holding `n` bytes at `ptr`.
    fn source(t: *const Tracer, ptr: u64, n: usize) !struct { buf: cuda.DeviceBuffer, offset: usize } {
        const m = t.model;
        const owned = [_]cuda.DeviceBuffer{ m.acts[0], m.acts[1], m.acts[2], m.audio_dev, m.weights };
        for (owned) |b| {
            if (ptr >= b.ptr and ptr - b.ptr <= b.len and n <= b.len - (ptr - b.ptr)) return .{ .buf = b, .offset = ptr - b.ptr };
        }
        return error.CaptureOutsideModelBuffers;
    }

    /// Copies n device bytes into a new file through the pinned staging buffer, hashing them and comparing to expect.
    fn dump(t: *Tracer, file_name: []const u8, ptr: u64, n: usize, expect: ?[]const u8) !Dumped {
        const src = try t.source(ptr, n);
        const staging = t.model.audio_host.bytes;
        const file = try t.dir.createFile(t.io, file_name, .{ .exclusive = true });
        defer file.close(t.io);
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        var equal = if (expect) |e| e.len == n else false;
        var done: usize = 0;
        while (done < n) {
            const len = @min(staging.len, n - done);
            const chunk = staging[0..len];
            try src.buf.downloadAsync(src.offset + done, chunk, t.stream.handle);
            try t.stream.synchronize();
            hash.update(chunk);
            if (expect) |e| {
                if (equal and !std.mem.eql(u8, chunk, e[done..][0..len])) equal = false;
            }
            try file.writePositionalAll(t.io, chunk, done);
            done += len;
        }
        t.written += n;
        return .{ .sha = std.fmt.bytesToHex(hash.finalResult(), .lower), .equal = equal };
    }

    /// One stage output [rows][cols], synchronized and written before the next launch can reuse its buffer.
    fn capture(t: *Tracer, name: []const u8, ptr: u64, dtype: Dtype, rows: usize, cols: usize, op: []const u8) !void {
        _ = try t.captureCompare(name, ptr, dtype, rows, cols, op, null);
    }

    fn captureCompare(t: *Tracer, name: []const u8, ptr: u64, dtype: Dtype, rows: usize, cols: usize, op: []const u8, expect: ?[]const u8) !bool {
        if (rows % t.frames != 0) return error.Invalid;
        var file_buffer: [96]u8 = undefined;
        const file_name = try std.fmt.bufPrint(&file_buffer, "{s}.{s}", .{ name, @tagName(dtype) });
        const n = rows * cols * dtype.size();
        const d = try t.dump(file_name, ptr, n, expect);
        try t.captures.print(t.gpa, "{s}\n    {{\"index\": {d}, \"name\": \"{s}\", \"file\": \"{s}\", \"dtype\": \"{s}\", \"byte_order\": \"little\", \"shape\": [{d}, {d}], \"rows_per_frame\": {d}, \"layout\": \"row-major [rows][cols]; rows time-major, cols channels\", \"bytes\": {d}, \"sha256\": \"{s}\", \"op\": \"{s}\"}}", .{
            if (t.count == 0) "" else ",", t.count, name, file_name, dtype.numpy(), rows, cols, rows / t.frames, n, d.sha[0..], op,
        });
        t.count += 1;
        return d.equal;
    }

    /// codec.zig Model.forward, op for op, with captures between launches.
    fn replay(t: *Tracer) !void {
        const m = t.model;
        const k = &m.kernels;
        const w = &m.w;
        const s = t.stream;
        const frames = t.frames;
        const h = 2; // bytes an FP16 element
        const p = [3]u64{ m.acts[0].ptr, m.acts[1].ptr, m.acts[2].ptr };
        var nb: [64]u8 = undefined;

        try k.rvqLatent(s, m.codes_dev.ptr, t.wp(w.semantic_book), t.wp(w.residual_books), t.wp(w.out_w), t.wp(w.out_b), p[0], frames);
        try t.capture("rvq_latent", p[0], .f16, frames, latent, "semantic + nine residual codebook rows through folded out_proj, summed");

        // post_module
        const x = p[0];
        const xn = p[1];
        const qkv = p[1] + frames * latent * h;
        const att = p[1] + 4 * frames * latent * h;
        const gu = p[2];
        const act = p[2] + frames * 2 * ffn * h;
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        for (&w.layers, 0..) |*l, i| {
            try k.rmsNorm(s, x, t.wp(l.attention_norm), xn, frames, latent, rms_eps);
            try t.capture(try std.fmt.bufPrint(&nb, "layer{d}.attention_norm", .{i}), xn, .f16, frames, latent, "attention_norm(x)");
            try k.gemm(s, .{ .x = xn, .w = t.wp(l.wqkv), .y = qkv, .m = frames, .n = 3 * latent, .cin = latent, .ldx = latent, .ldy = 3 * latent });
            try k.ropeQk(s, qkv, t.wp(w.rope), frames, latent);
            try k.windowAttention(s, qkv, att, frames, heads, window, scale);
            try t.capture(try std.fmt.bufPrint(&nb, "layer{d}.attention_core", .{i}), att, .f16, frames, latent, "windowed attention output, wo input");
            try k.gemm(s, .{ .x = att, .w = t.wp(l.wo), .y = x, .m = frames, .n = latent, .cin = latent, .ldx = latent, .ldy = latent, .scale = t.wp(l.attention_scale), .residual = x, .ldr = latent });
            try t.capture(try std.fmt.bufPrint(&nb, "layer{d}.attention_residual", .{i}), x, .f16, frames, latent, "h = x + attention_layer_scale(wo(core)), ffn_norm input");
            try k.rmsNorm(s, x, t.wp(l.ffn_norm), xn, frames, latent, rms_eps);
            try k.gemm(s, .{ .x = xn, .w = t.wp(l.w13), .y = gu, .m = frames, .n = 2 * ffn, .cin = latent, .ldx = latent, .ldy = 2 * ffn });
            try k.siluMul(s, gu, act, frames, ffn);
            try t.capture(try std.fmt.bufPrint(&nb, "layer{d}.ffn_act", .{i}), act, .f16, frames, ffn, "silu(w1(n)) * w3(n), w2 input");
            try k.gemm(s, .{ .x = act, .w = t.wp(l.w2), .y = x, .m = frames, .n = latent, .cin = ffn, .ldx = ffn, .ldy = latent, .scale = t.wp(l.ffn_scale), .residual = x, .ldr = latent });
            try t.capture(try std.fmt.bufPrint(&nb, "layer{d}.out", .{i}), x, .f16, frames, latent, "transformer block output");
        }
        try k.rmsNorm(s, x, t.wp(w.norm), p[1], frames, latent, rms_eps);
        try t.capture("post_norm", p[1], .f16, frames, latent, "post_module final RMSNorm");

        // upsample
        var input = p[1];
        var spent = p[0];
        var rows = frames;
        for (&w.upsample, 0..) |*u, i| {
            const y = spent;
            try k.gemm(s, .{ .x = input, .w = t.wp(u.conv), .y = y, .m = rows, .n = 2 * latent, .cin = latent, .ldx = latent, .ldy = 2 * latent, .bias = t.wp(u.conv_b), .bias_mode = .separate, .period = latent });
            rows *= 2;
            try t.capture(try std.fmt.bufPrint(&nb, "upsample{d}.transconv", .{i}), y, .f16, rows, latent, "ConvTranspose1d k2 s2 + bias");
            const dw = p[2];
            const normed = p[2] + rows * latent * h;
            try k.dwconv(s, y, t.wp(u.dw), t.wp(u.dw_b), dw, rows, latent);
            try t.capture(try std.fmt.bufPrint(&nb, "upsample{d}.dwconv", .{i}), dw, .f16, rows, latent, "ConvNeXt causal depthwise conv7 + bias");
            try k.layerNorm(s, dw, t.wp(u.norm_w), t.wp(u.norm_b), normed, rows, latent, layer_norm_eps);
            try t.capture(try std.fmt.bufPrint(&nb, "upsample{d}.layer_norm", .{i}), normed, .f16, rows, latent, "ConvNeXt LayerNorm");
            const hidden = input;
            try k.gemm(s, .{ .x = normed, .w = t.wp(u.pw1), .y = hidden, .m = rows, .n = convnext_hidden, .cin = latent, .ldx = latent, .ldy = convnext_hidden, .bias = t.wp(u.pw1_b), .bias_mode = .fused, .act = .gelu });
            try t.capture(try std.fmt.bufPrint(&nb, "upsample{d}.gelu", .{i}), hidden, .f16, rows, convnext_hidden, "GELU(pwconv1(n))");
            try k.gemm(s, .{ .x = hidden, .w = t.wp(u.pw2), .y = y, .m = rows, .n = latent, .cin = convnext_hidden, .ldx = convnext_hidden, .ldy = latent, .bias = t.wp(u.pw2_b), .bias_mode = .fused, .scale = t.wp(u.gamma), .residual = y, .ldr = latent });
            try t.capture(try std.fmt.bufPrint(&nb, "upsample{d}.out", .{i}), y, .f16, rows, latent, "upsample stage output: transconv + gamma * pwconv2(...)");
            input = y;
            spent = hidden;
        }

        // decoder
        var cur = spent;
        try k.gemm(s, .{ .x = input, .w = t.wp(w.conv0), .y = cur, .m = rows, .n = decoder_dim, .cin = latent, .ldx = latent, .ldy = decoder_dim, .taps = 7, .bias = t.wp(w.conv0_b), .bias_mode = .separate });
        try t.capture("decoder.conv0", cur, .f16, rows, decoder_dim, "decoder.model.0 causal conv7 1024 -> 1536");
        var spare = [2]u64{ input, p[2] };
        var ch: usize = decoder_dim;
        for (&w.blocks, rates, 0..) |*b, rate, bi| {
            const out_ch = ch / 2;
            try k.snake(s, cur, t.wp(b.snake.alpha), t.wp(b.snake.inv), spare[0], rows, ch);
            try k.gemm(s, .{ .x = spare[0], .w = t.wp(b.conv), .y = spare[1], .m = rows, .n = rate * out_ch, .cin = ch, .ldx = ch, .ldy = rate * out_ch, .taps = 2, .bias = t.wp(b.conv_b), .bias_mode = .separate, .period = out_ch });
            rows *= rate;
            ch = out_ch;
            var xb = spare[1];
            var sb = cur;
            var tb = spare[0];
            for (&b.units, dilations) |*u, dilation| {
                try k.snake(s, xb, t.wp(u.snake1.alpha), t.wp(u.snake1.inv), sb, rows, ch);
                try k.gemm(s, .{ .x = sb, .w = t.wp(u.conv7), .y = tb, .m = rows, .n = ch, .cin = ch, .ldx = ch, .ldy = ch, .taps = 7, .dilation = dilation, .bias = t.wp(u.conv7_b), .bias_mode = .separate });
                try k.snake(s, tb, t.wp(u.snake2.alpha), t.wp(u.snake2.inv), tb, rows, ch);
                try k.gemm(s, .{ .x = tb, .w = t.wp(u.conv1), .y = sb, .m = rows, .n = ch, .cin = ch, .ldx = ch, .ldy = ch, .bias = t.wp(u.conv1_b), .bias_mode = .separate, .residual = xb, .ldr = ch });
                const unit_out = sb;
                sb = tb;
                tb = xb;
                xb = unit_out;
            }
            cur = xb;
            spare = .{ sb, tb };
            try t.capture(try std.fmt.bufPrint(&nb, "decoder.block{d}.out", .{bi}), cur, .f16, rows, ch, "DecoderBlock: snake, causal transposed conv, residual units d1 d3 d9");
        }
        try k.snake(s, cur, t.wp(w.out_snake.alpha), t.wp(w.out_snake.inv), cur, rows, ch);
        try t.capture("decoder.out_snake", cur, .f16, rows, ch, "decoder.model.5 Snake1d");
        try k.convOutTanh(s, cur, t.wp(w.out_conv), t.wp(w.out_conv_b), m.audio_dev.ptr, rows, ch);
        if (rows != t.frames * samples_per_frame) return error.UnexpectedSampleCount;
        t.replay_equal = try t.captureCompare("pcm", m.audio_dev.ptr, .f32, rows, 1, "decoder.model.6 causal conv7 96 -> 1, tanh (model.7)", t.production);
    }
};

/// The weight image's tensors: offset, dtype, native shape and the transform from the reference tensor(s).
const Entries = struct {
    gpa: Allocator,
    names: Allocator,
    image_len: usize,
    list: std.ArrayList(u8) = .empty,
    count: usize = 0,

    const identity = "{\"kind\": \"identity\"}";
    const stack = "{\"kind\": \"stack\"}";
    const concat_rows = "{\"kind\": \"concat_rows\"}";
    const snake_alpha = "{\"kind\": \"snake_alpha\"}";
    const snake_inverse = "{\"kind\": \"snake_inverse_alpha\"}";

    fn fmt(e: *Entries, comptime f: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(e.names, f, args);
    }

    fn convTaps(e: *Entries, cout: usize, cin: usize, kernel: usize) ![]const u8 {
        return e.fmt("{{\"kind\": \"conv_taps\", \"cout\": {d}, \"cin\": {d}, \"kernel\": {d}}}", .{ cout, cin, kernel });
    }

    fn transposedTaps(e: *Entries, cin: usize, cout: usize, stride: usize, kernel: usize) ![]const u8 {
        return e.fmt("{{\"kind\": \"transposed_taps\", \"cin\": {d}, \"cout\": {d}, \"stride\": {d}, \"kernel\": {d}}}", .{ cin, cout, stride, kernel });
    }

    fn add(e: *Entries, name: []const u8, offset: u64, dtype: Dtype, shape: []const usize, layout: []const u8, sources: []const []const u8) !void {
        var n: usize = dtype.size();
        for (shape) |d| n *= d;
        if (offset % 256 != 0 or offset > e.image_len or n > e.image_len - offset) return error.WeightOutsideImage;
        try e.list.print(e.gpa, "{s}\n    {{\"name\": \"{s}\", \"offset\": {d}, \"bytes\": {d}, \"dtype\": \"{s}\", \"shape\": [", .{
            if (e.count == 0) "" else ",", name, offset, n, dtype.numpy(),
        });
        for (shape, 0..) |d, i| try e.list.print(e.gpa, "{s}{d}", .{ if (i == 0) "" else ", ", d });
        try e.list.print(e.gpa, "], \"layout\": {s}, \"sources\": [", .{layout});
        for (sources, 0..) |src, i| try e.list.print(e.gpa, "{s}\"{s}\"", .{ if (i == 0) "" else ", ", src });
        try e.list.appendSlice(e.gpa, "]}");
        e.count += 1;
    }

    fn one(e: *Entries, name: []const u8, offset: u64, dtype: Dtype, shape: []const usize, layout: []const u8, src: []const u8) !void {
        try e.add(name, offset, dtype, shape, layout, &[_][]const u8{src});
    }

    fn snake(e: *Entries, name: []const u8, sn: anytype, module: []const u8, channels: usize) !void {
        try e.one(try e.fmt("{s}.alpha", .{name}), sn.alpha, .f32, &.{channels}, snake_alpha, module);
        try e.one(try e.fmt("{s}.inverse_alpha", .{name}), sn.inv, .f32, &.{channels}, snake_inverse, module);
    }

    fn all(e: *Entries, w: *const Weights) !void {
        try e.one("semantic_book", w.semantic_book, .f16, &.{ semantic_size, codebook_dim }, identity, "quantizer.semantic_quantizer.quantizers.0.codebook.weight");
        {
            var src: [residual_books][]const u8 = undefined;
            for (&src, 0..) |*v, i| v.* = try e.fmt("quantizer.quantizer.quantizers.{d}.codebook.weight", .{i});
            try e.add("residual_books", w.residual_books, .f16, &.{ residual_books, residual_size, codebook_dim }, stack, &src);
        }
        {
            var ws: [codebooks][]const u8 = undefined;
            var bs: [codebooks][]const u8 = undefined;
            for (&ws, &bs, 0..) |*wv, *bv, i| {
                const prefix = if (i == 0) "quantizer.semantic_quantizer.quantizers.0" else try e.fmt("quantizer.quantizer.quantizers.{d}", .{i - 1});
                wv.* = try e.fmt("{s}.out_proj.weight", .{prefix});
                bv.* = try e.fmt("{s}.out_proj.bias", .{prefix});
            }
            try e.add("out_proj.weight", w.out_w, .f16, &.{ codebooks, latent, codebook_dim }, stack, &ws);
            try e.add("out_proj.bias", w.out_b, .f16, &.{ codebooks, latent }, stack, &bs);
        }
        for (&w.layers, 0..) |*l, i| {
            const p = try e.fmt("quantizer.post_module.layers.{d}", .{i});
            try e.one(try e.fmt("layer{d}.wqkv", .{i}), l.wqkv, .f16, &.{ 3 * latent, latent }, identity, try e.fmt("{s}.attention.wqkv.weight", .{p}));
            try e.one(try e.fmt("layer{d}.wo", .{i}), l.wo, .f16, &.{ latent, latent }, identity, try e.fmt("{s}.attention.wo.weight", .{p}));
            try e.add(try e.fmt("layer{d}.w13", .{i}), l.w13, .f16, &.{ 2 * ffn, latent }, concat_rows, &[_][]const u8{
                try e.fmt("{s}.feed_forward.w1.weight", .{p}), try e.fmt("{s}.feed_forward.w3.weight", .{p}),
            });
            try e.one(try e.fmt("layer{d}.w2", .{i}), l.w2, .f16, &.{ latent, ffn }, identity, try e.fmt("{s}.feed_forward.w2.weight", .{p}));
            try e.one(try e.fmt("layer{d}.attention_norm", .{i}), l.attention_norm, .f16, &.{latent}, identity, try e.fmt("{s}.attention_norm.weight", .{p}));
            try e.one(try e.fmt("layer{d}.ffn_norm", .{i}), l.ffn_norm, .f16, &.{latent}, identity, try e.fmt("{s}.ffn_norm.weight", .{p}));
            try e.one(try e.fmt("layer{d}.attention_scale", .{i}), l.attention_scale, .f16, &.{latent}, identity, try e.fmt("{s}.attention_layer_scale.gamma", .{p}));
            try e.one(try e.fmt("layer{d}.ffn_scale", .{i}), l.ffn_scale, .f16, &.{latent}, identity, try e.fmt("{s}.ffn_layer_scale.gamma", .{p}));
        }
        try e.one("post_norm", w.norm, .f16, &.{latent}, identity, "quantizer.post_module.norm.weight");
        for (&w.upsample, 0..) |*u, i| {
            const p = try e.fmt("quantizer.upsample.{d}", .{i});
            try e.one(try e.fmt("upsample{d}.conv", .{i}), u.conv, .f16, &.{ 2 * latent, latent }, try e.transposedTaps(latent, latent, 2, 2), try e.fmt("{s}.0.conv.weight", .{p}));
            try e.one(try e.fmt("upsample{d}.conv_bias", .{i}), u.conv_b, .f16, &.{latent}, identity, try e.fmt("{s}.0.conv.bias", .{p}));
            try e.one(try e.fmt("upsample{d}.dwconv", .{i}), u.dw, .f16, &.{ latent, 1, 7 }, identity, try e.fmt("{s}.1.dwconv.conv.weight", .{p}));
            try e.one(try e.fmt("upsample{d}.dwconv_bias", .{i}), u.dw_b, .f16, &.{latent}, identity, try e.fmt("{s}.1.dwconv.conv.bias", .{p}));
            try e.one(try e.fmt("upsample{d}.norm_weight", .{i}), u.norm_w, .f16, &.{latent}, identity, try e.fmt("{s}.1.norm.weight", .{p}));
            try e.one(try e.fmt("upsample{d}.norm_bias", .{i}), u.norm_b, .f16, &.{latent}, identity, try e.fmt("{s}.1.norm.bias", .{p}));
            try e.one(try e.fmt("upsample{d}.pwconv1", .{i}), u.pw1, .f16, &.{ convnext_hidden, latent }, identity, try e.fmt("{s}.1.pwconv1.weight", .{p}));
            try e.one(try e.fmt("upsample{d}.pwconv1_bias", .{i}), u.pw1_b, .f16, &.{convnext_hidden}, identity, try e.fmt("{s}.1.pwconv1.bias", .{p}));
            try e.one(try e.fmt("upsample{d}.pwconv2", .{i}), u.pw2, .f16, &.{ latent, convnext_hidden }, identity, try e.fmt("{s}.1.pwconv2.weight", .{p}));
            try e.one(try e.fmt("upsample{d}.pwconv2_bias", .{i}), u.pw2_b, .f16, &.{latent}, identity, try e.fmt("{s}.1.pwconv2.bias", .{p}));
            try e.one(try e.fmt("upsample{d}.gamma", .{i}), u.gamma, .f16, &.{latent}, identity, try e.fmt("{s}.1.gamma", .{p}));
        }
        try e.one("decoder.conv0", w.conv0, .f16, &.{ decoder_dim, 7, latent }, try e.convTaps(decoder_dim, latent, 7), "decoder.model.0.conv.weight");
        try e.one("decoder.conv0_bias", w.conv0_b, .f16, &.{decoder_dim}, identity, "decoder.model.0.conv.bias");
        var ch: usize = decoder_dim;
        for (&w.blocks, rates, 0..) |*b, rate, bi| {
            const out_ch = ch / 2;
            const p = try e.fmt("decoder.model.{d}.block", .{bi + 1});
            try e.snake(try e.fmt("block{d}.snake", .{bi}), b.snake, try e.fmt("{s}.0", .{p}), ch);
            try e.one(try e.fmt("block{d}.transconv", .{bi}), b.conv, .f16, &.{ rate * out_ch, 2 * ch }, try e.transposedTaps(ch, out_ch, rate, 2 * rate), try e.fmt("{s}.1.conv.weight", .{p}));
            try e.one(try e.fmt("block{d}.transconv_bias", .{bi}), b.conv_b, .f16, &.{out_ch}, identity, try e.fmt("{s}.1.conv.bias", .{p}));
            for (&b.units, 0..) |*u, j| {
                const q = try e.fmt("{s}.{d}.block", .{ p, j + 2 });
                try e.snake(try e.fmt("block{d}.unit{d}.snake1", .{ bi, j }), u.snake1, try e.fmt("{s}.0", .{q}), out_ch);
                try e.one(try e.fmt("block{d}.unit{d}.conv7", .{ bi, j }), u.conv7, .f16, &.{ out_ch, 7, out_ch }, try e.convTaps(out_ch, out_ch, 7), try e.fmt("{s}.1.conv.weight", .{q}));
                try e.one(try e.fmt("block{d}.unit{d}.conv7_bias", .{ bi, j }), u.conv7_b, .f16, &.{out_ch}, identity, try e.fmt("{s}.1.conv.bias", .{q}));
                try e.snake(try e.fmt("block{d}.unit{d}.snake2", .{ bi, j }), u.snake2, try e.fmt("{s}.2", .{q}), out_ch);
                try e.one(try e.fmt("block{d}.unit{d}.conv1", .{ bi, j }), u.conv1, .f16, &.{ out_ch, 1, out_ch }, try e.convTaps(out_ch, out_ch, 1), try e.fmt("{s}.3.conv.weight", .{q}));
                try e.one(try e.fmt("block{d}.unit{d}.conv1_bias", .{ bi, j }), u.conv1_b, .f16, &.{out_ch}, identity, try e.fmt("{s}.3.conv.bias", .{q}));
            }
            ch = out_ch;
        }
        try e.snake("out_snake", w.out_snake, "decoder.model.5", out_channels);
        try e.one("out_conv", w.out_conv, .f16, &.{ 7, out_channels }, try e.convTaps(1, out_channels, 7), "decoder.model.6.conv.weight");
        try e.one("out_conv_bias", w.out_conv_b, .f16, &.{1}, identity, "decoder.model.6.conv.bias");
        try e.one("rope", w.rope, .f16, &.{ max_frames, head_dim / 2, 2 }, try e.fmt("{{\"kind\": \"rope_table\", \"positions\": {d}}}", .{max_frames}), "quantizer.post_module.freqs_cis");
    }
};

fn jsonString(list: *std.ArrayList(u8), gpa: Allocator, s: []const u8) !void {
    try list.append(gpa, '"');
    for (s) |c| switch (c) {
        '"', '\\' => {
            try list.append(gpa, '\\');
            try list.append(gpa, c);
        },
        0...0x1f => try list.print(gpa, "\\u{x:0>4}", .{c}),
        else => try list.append(gpa, c),
    };
    try list.append(gpa, '"');
}

fn sha256Hex(bytes: []const u8) [64]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const usage = "usage: tf-drama-codec-trace CHECKPOINT CODES_U32LE OUTPUT_DIRECTORY FRAMES [--no-weights]\n";
    if ((args.len != 5 and args.len != 6) or (args.len == 6 and !std.mem.eql(u8, args[5], "--no-weights"))) {
        std.debug.print(usage, .{});
        return 2;
    }
    const with_weights = args.len == 5;
    const frames = std.fmt.parseInt(usize, args[4], 10) catch 0;
    if (frames < 1 or frames > max_frames) {
        std.debug.print("FRAMES must be 1..{d}\n" ++ usage, .{max_frames});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;

    var bytes: [max_frames * codebooks * 4 + 1]u8 align(@alignOf(u32)) = undefined;
    const input = try std.Io.Dir.cwd().openFile(io, args[2], .{});
    defer input.close(io);
    const count = try input.readPositionalAll(io, &bytes, 0);
    if (count != max_frames * codebooks * 4) return error.Expected1024Frames;
    const codes_sha = sha256Hex(bytes[0..count]);
    const codes = std.mem.bytesAsSlice([codebooks]u32, @as([]align(@alignOf(u32)) u8, bytes[0..count]));
    const dir = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer dir.close(io);

    const samples = frames * samples_per_frame;
    const production = try gpa.alloc(f32, samples);
    defer gpa.free(production);

    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var codec = try Model.init(gpa, io, &ctx, args[1], "", .{ .max_frames = max_frames });
    defer codec.deinit();
    if (codec.frame_length != samples_per_frame or codec.num_codebooks != codebooks or codec.max_frames != max_frames) return error.UnexpectedCodecGeometry;

    // 1. production decode, untouched
    if (try codec.decode(stream, codes[0..frames], production) != samples) return error.UnexpectedSampleCount;
    for (production) |v| if (!std.math.isFinite(v)) return error.NonFiniteAudio;
    const production_bytes = std.mem.sliceAsBytes(production);
    const production_sha = sha256Hex(production_bytes);
    try dir.writeFile(io, .{ .sub_path = "production.f32", .data = production_bytes, .flags = .{ .exclusive = true } });

    // 2. replay with captures: the same code upload decode performs, then the forward
    try ctx.makeCurrent();
    const code_bytes = std.mem.sliceAsBytes(codes[0..frames]);
    @memcpy(codec.codes_host.bytes[0..code_bytes.len], code_bytes);
    try codec.codes_dev.uploadAsync(0, codec.codes_host.bytes[0..code_bytes.len], stream.handle);
    var tracer: Tracer = .{ .gpa = gpa, .io = io, .dir = dir, .model = &codec, .stream = stream, .frames = frames, .production = production_bytes };
    defer tracer.captures.deinit(gpa);
    try tracer.replay();

    // 3. the packed weight image and its tensor map
    var weights_json: std.ArrayList(u8) = .empty;
    defer weights_json.deinit(gpa);
    if (with_weights) {
        const d = try tracer.dump("weights.bin", codec.weights.ptr, codec.weights.len, null);
        var entries: Entries = .{ .gpa = gpa, .names = arena, .image_len = codec.weights.len };
        defer entries.list.deinit(gpa);
        try entries.all(&codec.w);
        try weights_json.print(gpa, "{{\"file\": \"weights.bin\", \"bytes\": {d}, \"sha256\": \"{s}\", \"alignment\": 256, \"tensors\": [{s}\n  ]}}", .{ codec.weights.len, d.sha[0..], entries.list.items });
    } else {
        try weights_json.appendSlice(gpa, "null");
    }

    var manifest: std.ArrayList(u8) = .empty;
    defer manifest.deinit(gpa);
    try manifest.appendSlice(gpa, "{\n  \"format\": \"drama-codec-trace/1\",\n  \"checkpoint\": ");
    try jsonString(&manifest, gpa, args[1]);
    try manifest.appendSlice(gpa, ",\n  \"codes_file\": ");
    try jsonString(&manifest, gpa, args[2]);
    try manifest.print(gpa, ",\n  \"codes_sha256\": \"{s}\",\n  \"frames\": {d},\n  \"max_frames\": {d},\n  \"samples\": {d},\n  \"samples_per_frame\": {d}," ++
        "\n  \"production_pcm\": {{\"file\": \"production.f32\", \"dtype\": \"float32\", \"shape\": [{d}], \"sha256\": \"{s}\"}}," ++
        "\n  \"replay_pcm_equal\": {s},\n  \"captures\": [{s}\n  ],\n  \"weights\": {s}\n}}\n", .{
        codes_sha[0..],                               frames,                max_frames,         samples, samples_per_frame, samples, production_sha[0..],
        if (tracer.replay_equal) "true" else "false", tracer.captures.items, weights_json.items,
    });
    try dir.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest.items, .flags = .{ .exclusive = true } });

    const used = cuda.usage(false);
    std.debug.print("frames={d} captures={d} bytes_written={d} replay_pcm_equal={s} cuda_bytes={d} pinned_bytes={d}\n", .{
        frames, tracer.count, tracer.written + production_bytes.len + manifest.items.len, if (tracer.replay_equal) "true" else "false", used.device, used.host,
    });
    return if (tracer.replay_equal) 0 else 3;
}
