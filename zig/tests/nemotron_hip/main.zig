//! Nemotron-H on HIP: admits and loads the checkpoint on one GPU, checks its bytes, runs what the skeleton can.
const std = @import("std");
const hip = @import("hip");
const nem = @import("nemotron");

const gen = @import("gen.zig");
const replay = @import("replay.zig");

const Tensor = hip.ops.Tensor;

/// No per-thread signal stack: its 256 KiB of static TLS leaves RCCL's host threads too little stack to start.
pub const std_options: std.Options = .{ .signal_stack_size = null };

const usage = "usage: tf-nemotron-hip-test load <model dir> [--capacity N] [--rows N] [--no-mtp] [--tp W " ++
    "--rank R]\n" ++ gen.usage ++ replay.usage;

const Args = struct {
    dir: []const u8 = "",
    capacity: usize = 4096,
    rows: usize = 128,
    mtp: bool = true,
    world: usize = 1,
    rank: usize = 0,
};

fn parse(args: []const [:0]const u8) !Args {
    var a: Args = .{ .dir = args[0] };
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const name = args[i];
        if (std.mem.eql(u8, name, "--no-mtp")) {
            a.mtp = false;
            continue;
        }
        i += 1;
        if (i >= args.len) return error.MissingArgument;
        const v = try std.fmt.parseInt(usize, args[i], 10);
        if (std.mem.eql(
            u8,
            name,
            "--capacity",
        )) a.capacity = v else if (std.mem.eql(
            u8,
            name,
            "--rows",
        )) a.rows = v else if (std.mem.eql(u8, name, "--tp")) a.world = v else if (std.mem.eql(
            u8,
            name,
            "--rank",
        )) a.rank = v else return error.UnknownOption;
    }
    if (a.rank >= a.world) return error.BadRank;
    if (a.rows < 2 or a.rows % 2 != 0) return error.OddRows;
    return a;
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// `n` values of `t` read back as fp32, after the stream's work.
fn readBack(e: *nem.engine.Engine, a: std.mem.Allocator, t: Tensor, n: usize) ![]f32 {
    const bytes = try a.alloc(u8, n * t.kind.size());
    try e.stream.synchronize();
    try hip.runtime.check(e.driver.api.hipMemcpyDtoH(bytes.ptr, @ptrFromInt(t.ptr), bytes.len));
    const out = try a.alloc(f32, n);
    for (out, 0..) |*v, i| v.* = switch (t.kind) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)))),
    };
    return out;
}

/// The first attention layer over the embedded `tokens` on a fresh stream: whole, or in halves (the second's rows).
fn attentionRows(e: *nem.engine.Engine, tokens: []const u32, halves: bool) !Tensor {
    const m = e.model_();
    const d = m.dims;
    var s = try e.openStream();
    defer e.closeStream(&s);
    try s.cover(&e.pool, tokens.len);
    const o = e.ops(&e.arena);
    const ids = try e.ids(o, tokens);
    const n = tokens.len;
    const x = try nem.forward.take(o, m.act, n * d.hidden);
    try o.embedRows(m.embed, ids, n, x);
    const at = for (m.blocks, 0..) |b, i| {
        if (b.mixer == .attention) break i;
    } else return error.NoAttention;
    const b = m.blocks[at];
    const normed = try nem.forward.take(o, m.act, n * d.hidden);
    try o.rms(x, b.norm, normed, n, d.hidden, d.eps);
    const c = e.pool.paged(0, s.table());
    var p = o;
    p.prefill = true;
    if (!halves) return nem.forward.attention(
        p,
        m.act,
        d,
        b.mixer.attention,
        .{ .paged = .{ .c = c, .pos = 0 } },
        normed,
        n,
    );
    const h = n / 2;
    _ = try nem.forward.attention(p, m.act, d, b.mixer.attention, .{ .paged = .{ .c = c, .pos = 0 } }, normed, h);
    return nem.forward.attention(
        p,
        m.act,
        d,
        b.mixer.attention,
        .{ .paged = .{ .c = c, .pos = h } },
        nem.forward.at(normed, h * d.hidden),
        h,
    );
}

fn check(gpa: std.mem.Allocator, io: std.Io, a: Args) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const al = arena.allocator();
    var problem: []const u8 = "";
    var facts = nem.native.factsOf(al, io, a.dir, &problem) catch {
        std.debug.print("refused: {s}\n", .{problem});
        return error.Refused;
    };
    if (!a.mtp) facts.mtp = null;
    const room = try nem.native.cardRoom(gpa, io, 0);
    const ask: nem.admit.Ask = .{ .window = a.capacity, .room = room };
    const v = nem.admit.admit(al, facts, ask, &problem) catch {
        std.debug.print("refused: {s}\n", .{problem});
        return error.Refused;
    };
    std.debug.print(
        "admitted: {d} layers, MTP head {s}; plan {d} bytes ({d:.2} GiB) a rank, card grants {d:.2} GiB, at " ++
            "most {d} positions a stream\n",
        .{ v.config.layers, if (v.mtp) "yes" else "no", v.weights, gibs(v.weights), gibs(room), v.most },
    );
    for ([_]u32{ 1, 2 }) |world| {
        const w = try nem.plan.weights(
            &v.config,
            world,
            .device,
            .{ .mtp = v.mtp, .drafts = if (v.mtp) nem.admit.draft_rows else 0 },
        );
        std.debug.print(
            "plan tp {d}: rank 0 {d} bytes, rank {d} {d} bytes; {d} bytes a token, {d} bytes a stream's Mamba state\n",
            .{
                world,
                w.device(0),
                world - 1,
                w.device(world - 1),
                try nem.plan.tokenBytes(&v.config, world, 2, v.mtp),
                try nem.plan.streamBytes(&v.config, world, 2),
            },
        );
    }
    if (a.world > 1) return loadRank(gpa, io, a, v);
    const t0 = nowNs(io);
    const e = try nem.engine.Engine.open(gpa, io, a.dir, .{ .capacity = a.capacity, .streams = 2, .mtp = a.mtp });
    defer e.deinit();
    const loaded = e.weights.bytes();
    std.debug.print(
        "loaded in {d:.1} s: {d} weight bytes in {d} buffers ({d:.2} GiB), {d:.2} GiB on the device in all, {t} rows\n",
        .{
            @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9,
            loaded,
            e.weights.buffers.items.len,
            gibs(loaded),
            gibs(hip.usage(false).device),
            e.act,
        },
    );
    if (loaded != v.weights) {
        std.debug.print("FAIL the plan says {d} bytes, the load made {d}\n", .{ v.weights, loaded });
        return error.PlanMismatch;
    }
    std.debug.print("weights match the plan\n", .{});
    const m = e.model_();
    if (m.mtp) |h| std.debug.print("MTP head loaded: {d} draft rows of the head\n", .{h.draft_count});
    // the whole forward runs once the Mamba mixer and the MoE layer exist
    const tokens = try al.alloc(u32, a.rows);
    for (tokens, 0..) |*t, i| t.* = @intCast(1000 + 37 * i);
    var s = try e.openStream();
    defer e.closeStream(&s);
    if (e.prompt(&s, tokens)) |_| {
        std.debug.print("forward: a {d}-token prompt ran through every block\n", .{a.rows});
    } else |err| std.debug.print("forward: stops at block 0 ({s}): {t}\n", .{ @tagName(m.blocks[0].mixer), err });
    // the paged attention path: one call of `rows` rows, then two halves on a fresh stream, the same bits
    try e.fresh();
    const whole = try readBack(e, al, try attentionRows(e, tokens, false), a.rows * m.dims.hidden);
    try e.fresh();
    const tail = (a.rows / 2) * m.dims.hidden;
    const split = try readBack(e, al, try attentionRows(e, tokens, true), tail);
    var worst: f32 = 0;
    var finite = true;
    for (whole[tail..], split) |x, y| {
        worst = @max(worst, @abs(x - y));
        finite = finite and std.math.isFinite(x);
    }
    std.debug.print(
        "attention: {d} rows whole and in halves, the second half's largest difference {e}, finite {}\n",
        .{ a.rows, worst, finite },
    );
    // the head on the attention rows' final norm: the logits of the last row
    try e.fresh();
    const o = e.ops(&e.arena);
    const y = try attentionRows(e, tokens, false);
    const last = try nem.forward.take(o, m.act, m.dims.hidden);
    try o.rms(nem.forward.at(y, (a.rows - 1) * m.dims.hidden), m.norm_f, last, 1, m.dims.hidden, m.dims.eps);
    const logits = try readBack(e, al, try e.logits(o, last, 1), m.dims.vocab);
    var best: usize = 0;
    for (logits, 0..) |x, i| if (x > logits[best]) {
        best = i;
    };
    std.debug.print("head: {d} logits, the largest {d:.3} at token {d}\n", .{ logits.len, logits[best], best });
    if (!finite or worst != 0) return error.AttentionMismatch;
}

/// Rank `a.rank` of `a.world`'s weights alone on this GPU, checked against the plan's share.
fn loadRank(gpa: std.mem.Allocator, io: std.Io, a: Args, v: nem.admit.Verdict) !void {
    const world: u32 = @intCast(a.world);
    const rank: u32 = @intCast(a.rank);
    const plan = try nem.plan.weights(
        &v.config,
        world,
        .device,
        .{ .mtp = v.mtp, .drafts = if (v.mtp) nem.admit.draft_rows else 0 },
    );
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    const t0 = nowNs(io);
    var w = try nem.load.load(gpa, io, &r, a.dir, &v.config, .{ .world = a.world, .rank = a.rank, .mtp = v.mtp });
    defer w.deinit();
    const d = w.model.dims;
    std.debug.print(
        "rank {d} of {d} loaded in {d:.1} s: {d} weight bytes ({d:.2} GiB); {d} heads, {d} kv, {d} Mamba " ++
            "heads, experts {d}..{d}, {d} vocabulary rows\n",
        .{
            rank,
            world,
            @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9,
            w.bytes(),
            gibs(w.bytes()),
            d.heads,
            d.kv_heads,
            d.mamba_heads,
            d.first_expert,
            d.first_expert + d.rank_experts,
            d.vocab_rows,
        },
    );
    if (w.bytes() != plan.device(rank)) {
        std.debug.print(
            "FAIL the plan says {d} bytes for rank {d}, the load made {d}\n",
            .{ plan.device(rank), rank, w.bytes() },
        );
        return error.PlanMismatch;
    }
    std.debug.print("rank {d}'s weights match the plan\n", .{rank});
}

fn run(f: anytype, init: std.process.Init, args: []const [:0]const u8) u8 {
    f(init.gpa, init.io, args) catch |err| {
        std.debug.print("FAIL {t}\n", .{err});
        return 1;
    };
    return 0;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 3 and std.mem.eql(u8, args[1], "gen")) return run(gen.gen, init, args[2..]);
    if (args.len >= 3 and std.mem.eql(u8, args[1], "teacher")) return run(gen.teacher, init, args[2..]);
    if (args.len >= 3 and std.mem.eql(u8, args[1], "window")) return run(gen.window, init, args[2..]);
    if (args.len >= 3 and std.mem.eql(u8, args[1], "rounds")) return run(gen.rounds, init, args[2..]);
    if (args.len >= 3 and std.mem.eql(u8, args[1], "trace")) return run(gen.trace, init, args[2..]);
    if (args.len >= 3 and std.mem.eql(u8, args[1], "replay")) return run(replay.replay, init, args[2..]);
    if (args.len < 3 or !std.mem.eql(u8, args[1], "load")) {
        std.debug.print(usage, .{});
        return 2;
    }
    const a = parse(args[2..]) catch {
        std.debug.print(usage, .{});
        return 2;
    };
    check(init.gpa, init.io, a) catch |err| {
        std.debug.print("FAIL {t}\n", .{err});
        return 1;
    };
    return 0;
}
