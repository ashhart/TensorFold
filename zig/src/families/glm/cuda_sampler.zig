//! Sampled draws on upstream's sample.cu (#542 option C): every rank runs tf_draw.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Span = @import("cuda_shard.zig").Span;
const Exchange = @import("cuda_forward.zig").Exchange;

pub const Sampling = lanes.Sampling;

/// The request's rule as the engine runs it: null (or temperature 0) decodes greedily.
pub fn check(s: ?Sampling) ?Sampling {
    const x = s orelse return null;
    return if (x.temperature > 0) x else null;
}

/// sample.cu's Rule: the seed, 1 / T, top_p, the candidates' near window, ln(min_p) and top_k (0: off).
pub const Rule = extern struct { seed: u64, inv_t: f32, top_p: f32, near: f32, min_log: f32, top_k: u32, pad: u32 = 0 };

/// The rule as nemotron/cuda_sampler.zig fills it.
pub fn rule(s: Sampling) Rule {
    return .{ .seed = s.seed, .inv_t = @floatCast(1.0 / @max(s.temperature, 1e-6)), .top_p = @floatCast(s.top_p), .near = 20.0, .min_log = @floatCast(s.minLog()), .top_k = s.top_k };
}

/// A rank's logits rows: bf16 [rows][its vocabulary span].
pub const Local = struct { rank: usize, logits: u64 };

pub const Sampler = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    tf_draw: cuda.Function,
    tf_logprob_rows: cuda.Function,
    tf_grammar_mask: cuda.Function,
    vocab: usize,
    rows: usize,
    full: cuda.DeviceBuffer, // bf16 [rows][vocab]: every rank's columns in place
    stage: cuda.DeviceBuffer, // bf16, rows * vocab: every rank's columns as one block a rank (gather)
    rule: cuda.DeviceBuffer,
    meta: cuda.DeviceBuffer,
    tok: cuda.DeviceBuffer,
    prob: cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver, vocab: usize, rows: usize) !Sampler {
        var m = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer m.unload();
        const f = try m.function("tf_draw");
        const lp = try m.function("tf_logprob_rows");
        const gm = try m.function("tf_grammar_mask");
        var full = try cuda.DeviceBuffer.alloc(d, rows * vocab * 2);
        errdefer full.free();
        var stage = try cuda.DeviceBuffer.alloc(d, rows * vocab * 2);
        errdefer stage.free();
        var r = try cuda.DeviceBuffer.alloc(d, @sizeOf(Rule));
        errdefer r.free();
        var meta = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer meta.free();
        var tok = try cuda.DeviceBuffer.alloc(d, rows * 4);
        errdefer tok.free();
        const prob = try cuda.DeviceBuffer.alloc(d, rows * 4);
        return .{ .d = d, .module = m, .tf_draw = f, .tf_logprob_rows = lp, .tf_grammar_mask = gm, .vocab = vocab, .rows = rows, .full = full, .stage = stage, .rule = r, .meta = meta, .tok = tok, .prob = prob };
    }

    pub fn deinit(s: *Sampler) void {
        s.prob.free();
        s.tok.free();
        s.meta.free();
        s.rule.free();
        s.stage.free();
        s.full.free();
        s.module.unload();
    }

    /// `rows` rows of `mine` -> `out`, tokens at positions first, first + 1, ...; `spans`: every rank's, rank order.
    pub fn draw(s: *Sampler, stream: cuda.Stream, exchange: Exchange, spans: []const Span, mine: []const Local, rows: usize, first: u64, r: Rule, out: []u32) !void {
        try s.gather(stream, exchange, spans, mine, rows, s.stage.ptr, s.full.ptr);
        try s.drawFull(stream, rows, first, r, out);
    }

    /// `rows` whole rows already in `full` -> `out` under `r`, tokens at positions first, first + 1, ...
    pub fn drawFull(s: *Sampler, stream: cuda.Stream, rows: usize, first: u64, r: Rule, out: []u32) !void {
        std.debug.assert(rows <= s.rows and out.len >= rows and first >= 1);
        const V = s.vocab;
        try s.rule.uploadAsync(0, std.mem.asBytes(&r), stream.handle);
        const meta = [4]i32{ @intCast(first - 1), 0, 0, 0 }; // row r keys meta[0] + r + 1
        try s.meta.uploadAsync(0, std.mem.asBytes(&meta), stream.handle);
        var args: cuda.Args = .{};
        args.add(s.full.ptr);
        args.add(@as(u32, @intCast(V)));
        args.add(s.rule.ptr);
        args.add(s.meta.ptr);
        args.add(@as(i32, 0));
        args.add(s.tok.ptr);
        args.add(s.prob.ptr);
        try cuda.launch.launch(s.tf_draw, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &args);
        try stream.synchronize();
        try s.tok.download(0, std.mem.sliceAsBytes(out[0..rows]));
    }

    /// `full` (bf16 [rows][vocab]) whole on every rank: one broadcast a rank into `stage`, one strided copy a rank.
    pub fn gather(s: *const Sampler, stream: cuda.Stream, exchange: Exchange, spans: []const Span, mine: []const Local, rows: usize, stage: u64, full: u64) !void {
        switch (exchange) {
            .local => for (mine) |l| try s.place(stream, full, l.logits, spans[l.rank], rows),
            .nccl => |x| {
                try x.lib.group(Spread{ .lib = x.lib, .comm = x.comm, .stream = stream, .stage = stage, .spans = spans, .mine = mine, .rows = rows }, Spread.post);
                for (spans) |span| try s.place(stream, full, stage + rows * span.lo * 2, span, rows);
            },
        }
    }

    /// One rank's columns (bf16 [rows][span], packed) into their columns of `full`.
    fn place(s: *const Sampler, stream: cuda.Stream, full: u64, src: u64, span: Span, rows: usize) !void {
        const p: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = span.len() * 2, .dst = full + span.lo * 2, .dst_pitch = s.vocab * 2, .width = span.len() * 2, .height = rows };
        try s.d.check(s.d.api.cuMemcpy2DAsync_v2(&p, stream.handle), "cuMemcpy2DAsync");
    }

    /// Structured output: each of `rows` whole rows in `full` keeps its allowed tokens (`words` u32 a row at `bits`).
    pub fn mask(s: *const Sampler, stream: cuda.Stream, rows: usize, bits: u64, words: usize) !void {
        var args: cuda.Args = .{};
        args.add(s.full.ptr);
        args.add(@as(u32, @intCast(s.vocab)));
        args.add(bits);
        args.add(@as(u32, @intCast(words)));
        try cuda.launch.launch(s.tf_grammar_mask, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
    }

    /// sample.cu's logprob words of `rows` whole rows at `full`: each row's pick in `picks`, its `k` best, its rank.
    pub fn logprobRows(s: *const Sampler, stream: cuda.Stream, full: u64, rows: usize, picks: u64, k: u8, out: u64) !void {
        var args: cuda.Args = .{};
        args.add(full);
        args.add(@as(u32, @intCast(s.vocab)));
        args.add(picks);
        args.add(@as(u32, k));
        args.add(@as(u32, 1)); // the rank too: vLLM's prompt_logprobs give it
        args.add(out);
        try cuda.launch.launch(s.tf_logprob_rows, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &args);
    }

    /// The whole rows a draw read (bf16 [rows][vocab]), for checks against the host rule.
    pub fn rowsRead(s: *const Sampler, rows: usize, host: []u16) !void {
        try s.full.download(0, std.mem.sliceAsBytes(host[0 .. rows * s.vocab]));
    }

    fn copy(s: *const Sampler, stream: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try s.d.check(s.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle), "cuMemcpyDtoDAsync");
    }
};

test "rules as nemotron/cuda_sampler fills them; temperature 0 is greedy" {
    const s: Sampling = .{ .seed = 1234, .temperature = 0.5, .top_k = 20, .top_p = 0.95 };
    const r = rule(check(s).?);
    try std.testing.expectEqual(@as(f32, 2.0), r.inv_t);
    try std.testing.expectEqual(@as(u32, 20), r.top_k);
    try std.testing.expect(std.math.isInf(r.min_log));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Rule));
    try std.testing.expectEqual(@as(?Sampling, null), check(.{ .seed = 0, .temperature = 0 }));
}

/// Each rank's columns as one block from that rank (spans may differ in width: TP=3), in one group: world broadcasts.
const Spread = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    stage: u64,
    spans: []const Span,
    mine: []const Local,
    rows: usize,

    fn post(t: Spread) anyerror!void {
        for (t.spans, 0..) |span, q| {
            const at = t.stage + t.rows * span.lo * 2;
            var send = at;
            for (t.mine) |l| if (l.rank == q) {
                send = l.logits;
            };
            try t.lib.check(t.lib.api.ncclBroadcast(send, at, t.rows * span.len(), .bf16, @intCast(q), t.comm, t.stream.handle), "ncclBroadcast");
        }
    }
};
