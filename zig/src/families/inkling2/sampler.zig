//! Sampled draws on main's sample.cu: each rank's bf16 columns broadcast into whole rows, every rank draws the same.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Span = @import("shard.zig").Span;
const Exchange = @import("forward.zig").Exchange;

pub const Sampling = lanes.Sampling;

/// The request's rule as the engine runs it: null (or temperature 0) decodes greedily.
pub fn check(s: ?Sampling) ?Sampling {
    const x = s orelse return null;
    return if (x.temperature > 0) x else null;
}

pub const Rule = cuda.sample.Rule;

pub fn rule(s: Sampling) Rule {
    return cuda.sample.rule(s);
}

pub const Sampler = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    tf_draw: cuda.Function,
    vocab: usize, // a whole row's columns: every rank's padded span
    rows: usize,
    full: cuda.DeviceBuffer, // bf16 [rows][vocab]: every rank's columns in place
    rule: cuda.DeviceBuffer,
    meta: cuda.DeviceBuffer,
    tok: cuda.DeviceBuffer,
    prob: cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver, vocab: usize, rows: usize) !Sampler {
        var m = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer m.unload();
        const f = try m.function("tf_draw");
        var full = try cuda.DeviceBuffer.alloc(d, rows * vocab * 2);
        errdefer full.free();
        var r = try cuda.DeviceBuffer.alloc(d, @sizeOf(Rule));
        errdefer r.free();
        var meta = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer meta.free();
        var tok = try cuda.DeviceBuffer.alloc(d, rows * 4);
        errdefer tok.free();
        const prob = try cuda.DeviceBuffer.alloc(d, rows * 4);
        return .{ .d = d, .module = m, .tf_draw = f, .vocab = vocab, .rows = rows, .full = full, .rule = r, .meta = meta, .tok = tok, .prob = prob };
    }

    pub fn deinit(s: *Sampler) void {
        s.prob.free();
        s.tok.free();
        s.meta.free();
        s.rule.free();
        s.full.free();
        s.module.unload();
    }

    /// `rows` rows of `full` (this rank's columns written) made whole, then drawn: tokens at first, first + 1, ...
    pub fn draw(s: *Sampler, stream: cuda.Stream, exchange: Exchange, spans: []const Span, rows: usize, first: u64, r: Rule, out: []u32) !void {
        std.debug.assert(rows <= s.rows and out.len >= rows and first >= 1);
        switch (exchange) {
            .local => {},
            .nccl => |x| try x.lib.group(Spread{ .lib = x.lib, .comm = x.comm, .stream = stream, .full = s.full.ptr, .vocab = s.vocab, .spans = spans, .rows = rows }, Spread.post),
            .pair => |x| for (0..rows) |row| { // the peer's columns of each row, as Spread broadcasts them
                const sp = spans[1 - x.rank];
                try x.p.pull(x.rank, stream, s.full.ptr, &.{.{ .off = (row * s.vocab + sp.lo) * 2, .len = sp.len() * 2 }});
            },
        }
        try s.rule.uploadAsync(0, std.mem.asBytes(&r), stream.handle);
        const meta = [4]i32{ @intCast(first - 1), 0, 0, 0 }; // row r keys meta[0] + r + 1
        try s.meta.uploadAsync(0, std.mem.asBytes(&meta), stream.handle);
        var args: cuda.Args = .{};
        args.add(s.full.ptr);
        args.add(@as(u32, @intCast(s.vocab)));
        args.add(s.rule.ptr);
        args.add(s.meta.ptr);
        args.add(@as(i32, 0));
        args.add(s.tok.ptr);
        args.add(s.prob.ptr);
        try cuda.launch.launch(s.tf_draw, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &args);
        try stream.synchronize();
        try s.tok.download(0, std.mem.sliceAsBytes(out[0..rows]));
    }
};

/// Each rank's columns of each row from that rank, in one NCCL group.
const Spread = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    full: u64,
    vocab: usize,
    spans: []const Span,
    rows: usize,

    fn post(t: Spread) anyerror!void {
        for (0..t.rows) |row| for (t.spans, 0..) |span, q| {
            const at = t.full + (row * t.vocab + span.lo) * 2;
            try t.lib.check(t.lib.api.ncclBroadcast(at, at, span.len(), .bf16, @intCast(q), t.comm, t.stream.handle), "ncclBroadcast");
        };
    }
};

test "sampler: temperature 0 is greedy, a sampled rule keeps its settings" {
    const s: Sampling = .{ .seed = 1234, .temperature = 0.5, .top_k = 20, .top_p = 0.95 };
    const r = rule(check(s).?);
    try std.testing.expectEqual(@as(f32, 2.0), r.inv_t);
    try std.testing.expectEqual(@as(u32, 20), r.top_k);
    try std.testing.expectEqual(@as(?Sampling, null), check(.{ .seed = 0, .temperature = 0 }));
}
