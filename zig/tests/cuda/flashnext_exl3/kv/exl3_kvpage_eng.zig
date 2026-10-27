//! exl3-kvpage engine: the family's Engine with a paged sequence, and the id and list readers.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
const flash = @import("flashnext");
const Gpu = check.Gpu;
const prompt = flash.prompt;
const xpass = flash.exl3.pass;
const xpre = flash.exl3.prefill;
const vocab = prompt.vocabulary;

pub const Eng = struct {
    gpu: Gpu,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    attached: xpass.Attached,

    pub fn init(gpu: Gpu, dir: []const u8, cap: usize, deep_rows: usize) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.stream = try cuda.Stream.init(gpu.d, true);
        g.glue = try flash.Glue.load(gpu.d);
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, 1, 0);
        _ = try g.e.reserve(@min(cap, g.e.chunk));
        _ = try g.e.reserveScores(cap, @min(cap, g.e.chunk));
        g.seq = try g.e.newSeq(cap);
        if (deep_rows > 0) try xpre.reserve(gpu.gpa, gpu.d, g.e, deep_rows);
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, try xpass.budgetFromEnv(0), null);
        var free: usize = 0;
        var total: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        std.debug.print("engine: loaded in {d:.1} s; cap {d}; pool {d} lines a layer; {d:.2} GiB free\n", .{ @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9, cap, g.attached.lines, @as(f64, @floatFromInt(free)) / (1 << 30) });
        return g;
    }

    pub fn deinit(g: *Eng) void {
        xpre.release();
        g.seq.deinit(g.gpu.gpa);
        g.e.deinit();
        g.glue.deinit();
        g.stream.deinit();
        g.gpu.gpa.destroy(g);
    }

    fn tri(g: *Eng) flash.Tri {
        return .{ .own = &g.glue, .s = g.stream };
    }

    fn reset(g: *Eng) !void {
        const s = &g.seq;
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| for (bs) |b| try b.fill8(0, g.stream.handle);
        try s.tail.fill8(0, g.stream.handle);
        try g.stream.synchronize();
        s.fed = 0;
        s.history = .{ g.e.eos, g.e.eos };
    }

    pub fn prefill(g: *Eng, ids: []const u32) !f64 {
        try g.reset();
        const t0 = check.now(g.gpu.io);
        try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, .{ .logits = .last });
        return @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6;
    }

    /// `n` greedy rounds from the prefill's logits; the tokens into `out`; ms a round.
    pub fn decode(g: *Eng, out: []u32) !f64 {
        var dev = try cuda.DeviceBuffer.alloc(g.gpu.d, vocab * 2);
        defer dev.free();
        var tok = prompt.argmax(g.seq.logits);
        const t0 = check.now(g.gpu.io);
        for (out) |*o| {
            o.* = tok;
            const q = [_]*prompt.Seq{&g.seq};
            try g.e.round(flash.Tri, g.tri(), &q, (&tok)[0..1], dev.ptr);
            tok = prompt.argmax(g.seq.logits);
        }
        return @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6 / @as(f64, @floatFromInt(@max(out.len, 1)));
    }
};

pub fn readIds(gpu: Gpu, path: []const u8) ![]u32 {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 28));
    defer a.free(text);
    const ids = try a.alloc(u32, text.len / 4);
    for (ids, 0..) |*x, i| x.* = std.mem.readInt(u32, text[4 * i ..][0..4], .little);
    return ids;
}

pub fn parseList(s: []const u8, out: []usize) !usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |t| {
        out[n] = try std.fmt.parseInt(usize, t, 10);
        n += 1;
    }
    return n;
}
