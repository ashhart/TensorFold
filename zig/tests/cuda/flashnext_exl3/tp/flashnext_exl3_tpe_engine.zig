//! flashnext-exl3-tpe: the ranks' link, the engine wrapper and the golden reader the modes share.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");

const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const xpass = X.pass;
const tp = X.tp;
const fpool = flash.exl3_pool;
const wire = flash.rank_wire;
const Collective = flash.collective.Collective;
const vocab = prompt.vocabulary;
pub const cap_default: usize = 32768;
pub const out_rows: usize = 64;

pub fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn ms(t0: i96, t1: i96) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

pub fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

// -- the ranks' link ------------------------------------------------------------------------------------------------

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

    /// Both ranks here; returns the other rank's `value`.
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

// -- the engine -----------------------------------------------------------------------------------------------------

pub const Eng = struct {
    gpu: Gpu,
    rank: usize,
    world: usize,
    ranks: ?*Ranks,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    seq_live: bool,
    out: cuda.DeviceBuffer,
    attached: xpass.Attached,

    pub fn init(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.rank = rank;
        g.world = world;
        g.ranks = if (world == 2) try Ranks.open(gpu, rank, port) else null;
        g.stream = try cuda.Stream.init(gpu.d, true);
        g.glue = try flash.Glue.load(gpu.d);
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, world, rank);
        if (g.ranks) |r| g.e.collective = &r.collective;
        const t1 = check.now(gpu.io);
        _ = try g.e.reserve(@min(cap_default, g.e.chunk));
        _ = try g.e.reserveScores(cap_default, @min(cap_default, g.e.chunk));
        g.seq = try g.e.newSeq(cap_default);
        g.seq_live = true;
        g.out = try cuda.DeviceBuffer.alloc(gpu.d, out_rows * vocab * 2);
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, try xpass.budgetFromEnv(0), null);
        const t2 = check.now(gpu.io);
        var free: usize = 0;
        var total: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        const head_n = g.e.head_q.n;
        std.debug.print("RESULT rank {d}/{d} load: {d:.1} s (initRank {d:.1} s, pinned bank {d:.2} GiB copied in {d:.1} s); {d} lines a layer x {d} layers of {d:.2} MiB = {d:.2} GiB of pool; {d:.2} GiB free before the pools, {d:.2} GiB free after (card {d:.2} GiB); head columns {d}; chunk {d}\n", .{ rank, world, ms(t0, t2) / 1e3, ms(t0, t1) / 1e3, gib(g.attached.bank_bytes), @as(f64, @floatFromInt(g.attached.bank_ns)) / 1e9, g.attached.lines, g.attached.layers, @as(f64, @floatFromInt(g.attached.line_bytes)) / (1 << 20), gib(g.attached.device_bytes), gib(g.attached.free_before), gib(free), gib(total), head_n, g.e.chunk });
        return g;
    }

    pub fn deinit(g: *Eng) void {
        g.out.free();
        if (g.seq_live) g.seq.deinit(g.gpu.gpa);
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

    pub fn stats(g: *Eng) fpool.Stats {
        return xpass.poolStats(g.e.layers[0..g.e.n_layers], null);
    }

    /// Both ranks reach this point; the other rank's value (or `v` at tp=1).
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
};

pub fn printPool(g: *Eng, what: []const u8) void {
    const s = g.stats();
    const routed = s.hit_slots + s.miss_slots;
    std.debug.print("RESULT rank {d}/{d} pool {s}: {d} rounds, {d} passes, hit rate {d:.1}% ({d}/{d} routed slots of this rank), {d} copies, {d} evictions; host cut {d:.1} ms over {d} blocks, plan {d:.1} ms\n", .{ g.rank, g.world, what, s.rounds, s.passes, 100 * @as(f64, @floatFromInt(s.hit_slots)) / @as(f64, @floatFromInt(@max(routed, 1))), s.hit_slots, routed, s.misses, s.evictions, @as(f64, @floatFromInt(xpass.pool_time.cut_ns)) / 1e6, xpass.pool_time.blocks, @as(f64, @floatFromInt(xpass.pool_time.plan_ns)) / 1e6 });
}

// -- golden ---------------------------------------------------------------------------------------------------------

pub const Golden = struct {
    parsed: std.json.Parsed(std.json.Value),

    pub fn open(gpu: Gpu, path: []const u8) !Golden {
        const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 26));
        defer gpu.gpa.free(text);
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, gpu.gpa, text, .{}) };
    }

    pub fn prompts(g: *const Golden) []std.json.Value {
        return g.parsed.value.object.get("prompts").?.array.items;
    }
};

pub fn ints(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const items = v.array.items;
    const out = try a.alloc(u32, items.len);
    for (items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

pub fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
}

pub fn top1(row: []const u16) u32 {
    var best: usize = 0;
    for (row, 0..) |b, i| if (promote(b) > promote(row[best])) {
        best = i;
    };
    return @intCast(best);
}

// -- modes ----------------------------------------------------------------------------------------------------------

pub fn margin(row: []const u16) f32 {
    const b = top1(row);
    var second: f32 = -std.math.inf(f32);
    for (row, 0..) |v, i| if (i != b) {
        second = @max(second, promote(v));
    };
    return promote(row[b]) - second;
}
