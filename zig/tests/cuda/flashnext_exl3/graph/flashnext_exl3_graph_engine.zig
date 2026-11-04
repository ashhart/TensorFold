//! The EXL3 graph tests' engine harness: one rank's engine with its sequence, TP link and output buffer.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const xpass = X.pass;
const dplan = X.dplan;
const fpool = flash.exl3_pool;
const wire = flash.rank_wire;
const Collective = flash.collective.Collective;
const vocab = prompt.vocabulary;

pub const cap_default: usize = 32768;
pub const out_rows: usize = 64;

pub fn ms(t0: i96, t1: i96) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

pub fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

// -- the engine at tp=1/2 -------------------------------------------------------------------------------------------

pub const Ranks = struct {
    rank: usize,
    link: wire.Link,
    collective: Collective,
    seq: u64 = 10,

    pub fn open(gpu: Gpu, rank: usize, port: u16) !*Ranks {
        const r = try gpu.gpa.create(Ranks);
        errdefer gpu.gpa.destroy(r);
        const link = try wire.Link.open(gpu.io, "127.0.0.1", port, rank, 600_000);
        errdefer link.close();
        r.* = .{ .rank = rank, .link = link, .collective = try Collective.init(link, rank, 3_600_000) };
        return r;
    }

    pub fn close(r: *Ranks, gpa: std.mem.Allocator) void {
        r.collective.deinit();
        r.link.close();
        gpa.destroy(r);
    }

    pub fn swap(r: *Ranks, value: u64) !u64 {
        r.seq += 1;
        const until = wire.deadline(3_600_000);
        if (r.rank == 0) {
            const got = try r.link.receiveHeader(until);
            if (got.kind != .ready or got.sequence != r.seq) return error.RankOutOfStep;
            try r.link.sendHeader(.{ .kind = .ready, .sequence = r.seq, .arg = value }, until);
            return got.arg;
        }
        try r.link.sendHeader(.{ .kind = .ready, .sequence = r.seq, .arg = value }, until);
        const got = try r.link.receiveHeader(until);
        if (got.kind != .ready or got.sequence != r.seq) return error.RankOutOfStep;
        return got.arg;
    }
};

pub const Eng = struct {
    gpu: Gpu,
    rank: usize,
    world: usize,
    ranks: ?*Ranks,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    out: cuda.DeviceBuffer,
    attached: xpass.Attached,

    pub fn init(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.rank = rank;
        g.world = world;
        g.ranks = if (world == 2) try Ranks.open(gpu, rank, port) else null;
        errdefer if (g.ranks) |ranks| ranks.close(gpu.gpa);
        g.stream = try cuda.Stream.init(gpu.d, true);
        errdefer g.stream.deinit();
        g.glue = try flash.Glue.load(gpu.d);
        errdefer g.glue.deinit();
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, world, rank);
        errdefer g.e.deinit();
        if (g.ranks) |r| g.e.collective = &r.collective;
        _ = try g.e.reserve(@min(cap_default, g.e.chunk));
        _ = try g.e.reserveScores(cap_default, @min(cap_default, g.e.chunk));
        g.seq = try g.e.newSeq(cap_default);
        errdefer g.seq.deinit(gpu.gpa);
        g.out = try cuda.DeviceBuffer.alloc(gpu.d, out_rows * vocab * 2);
        errdefer g.out.free();
        X.prefill.reserveFor(g.e, cap_default, world);
        const budget = try xpass.budgetFromEnv(0);
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, &g.e.exl3_prefill, g.e.layers[0..g.e.n_layers], null, g.e.chunk, budget, null);
        std.debug.print("RESULT rank {d}/{d} load: {d:.1} s; {d} lines a layer x {d} layers ({d:.2} GiB of pool)\n", .{ rank, world, ms(t0, check.now(gpu.io)) / 1e3, g.attached.lines, g.attached.layers, gib(g.attached.device_bytes) });
        return g;
    }

    pub fn deinit(g: *Eng) void {
        g.out.free();
        g.seq.deinit(g.gpu.gpa);
        g.e.deinit();
        g.glue.deinit();
        g.stream.deinit();
        if (g.ranks) |r| r.close(g.gpu.gpa);
        g.gpu.gpa.destroy(g);
    }

    pub fn tri(g: *Eng) flash.Tri {
        return .{ .own = &g.glue, .s = g.stream };
    }

    pub fn reset(g: *Eng) !void {
        const s = &g.seq;
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| for (bs) |b| try b.fill8(0, g.stream.handle);
        try s.tail.fill8(0, g.stream.handle);
        try g.stream.synchronize();
        s.fed = 0;
        s.history = .{ g.e.eos, g.e.eos };
    }

    pub fn swap(g: *Eng, v: u64) !u64 {
        return if (g.ranks) |r| r.swap(v) else v;
    }

    pub fn forward(g: *Eng, ids: []const u32, hooks: prompt.batch.Hooks) !void {
        try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, hooks);
        if (g.ranks) |r| try r.collective.finish(g.stream) else try g.stream.synchronize();
    }

    pub fn round(g: *Eng, tok: u32) !u32 {
        const q = [_]*prompt.Seq{&g.seq};
        var t = tok;
        try g.e.round(flash.Tri, g.tri(), &q, (&t)[0..1], g.out.ptr);
        if (g.ranks) |r| try r.collective.finish(g.stream);
        return prompt.argmax(g.seq.logits);
    }

    pub fn stats(g: *Eng) !fpool.Stats {
        try dplan.toHost(&g.e.exl3_dplan);
        return xpass.poolStats(g.e.layers[0..g.e.n_layers], null);
    }
};

pub fn ints(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const items = v.array.items;
    const out = try a.alloc(u32, items.len);
    for (items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}
