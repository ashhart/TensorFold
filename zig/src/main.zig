//! `tensorfold`: the native engine's command line (see `usage`).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const cluster = @import("cluster");

const lanes = tf.lanes;
const nemotron = tf.nemotron;
const checkpoint_cli = @import("cli/cli.zig");

const usage =
    \\usage: tensorfold run MODEL --tokens ID,... --max-tokens N [--temperature T --seed S [--top-p P] [--top-k K]
    \\                      [--min-p P]] [--no-drafts] [--warmup] [--prefill-chunk N] [--chunks S,...] [--streams N] [--lanes auto|N|0 [--lane-depth D] [--table-prices]]
    \\                      [--serial-encoder] [--gpu-round D|0 [--fixed-depth] [--stop-power P] [--no-stop] [--ahead N] [--hand-over N]]
    \\                      [--profile N] [--report PATH]
    \\
;

/// `--chunks`: every stream's prompt chunk starts (set once from the options).
var cli_chunks: []const u32 = &.{};

/// `--lanes auto`: the planner may take up to this many lanes a round.
const auto_lanes = 32;

const Options = struct {
    model: []const u8 = "",
    tokens: []u32 = &.{},
    max_tokens: u32 = 256,
    temperature: f64 = 0,
    seed: ?u64 = null,
    top_p: f64 = 1.0,
    top_k: u32 = 0,
    min_p: f64 = 0.0,
    drafts: bool = true,
    warmup: bool = false,
    chunk: usize = 2048,
    chunks: []const u32 = &.{}, // the prompt's chunk starts after 0 (a server's PrefillPlan), else every `chunk` rows
    report: ?[]const u8 = null,
    profile: usize = 0,
    streams: usize = 1,
    lanes: u32 = 0, // a lone stream's lanes at most, the planner picking each round's (0: the head's chain by its depth rule)
    lane_depth: u32 = 16, // the trees' depth at most
    table_prices: bool = false, // the planner prices picks from the load table alone (not the stream's own rounds)
    serial: bool = false,
    gpu_round: usize = 8, // a lone greedy stream's rounds on the GPU, the head at most D levels (0: the lane core's rounds)
    round: nemotron.gpu_round.Options = .{ .depth = 8 },
    hand_over: usize = 0, // also: the GPU round hands its stream to the lane core after N commits, a second stream joining
    block_lanes: usize = 0, // the head drafts in one pass: its first level plus this many placeholder lanes (random rows)
    block_weights: ?[]const u8 = null, // the placeholder rows from a trained block (TFBLOCK1 file)
};

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("tensorfold: " ++ fmt ++ "\n", args);
    std.process.exit(2);
}

fn parseArgs(arena: std.mem.Allocator, args: []const [:0]const u8) !Options {
    if (args.len < 3 or !std.mem.eql(u8, args[1], "run")) {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    var o = Options{ .model = args[2] };
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const value = struct {
            fn get(all: []const [:0]const u8, at: *usize) []const u8 {
                at.* += 1;
                if (at.* >= all.len) fail("{s} needs a value", .{all[at.* - 1]});
                return all[at.*];
            }
        }.get;
        if (std.mem.eql(u8, a, "--tokens")) {
            var list: std.ArrayList(u32) = .empty;
            var it = std.mem.tokenizeScalar(u8, value(args, &i), ',');
            while (it.next()) |t| try list.append(arena, try std.fmt.parseInt(u32, t, 10));
            o.tokens = list.items;
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            o.max_tokens = try std.fmt.parseInt(u32, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--temperature")) {
            o.temperature = try std.fmt.parseFloat(f64, value(args, &i));
        } else if (std.mem.eql(u8, a, "--seed")) {
            o.seed = try std.fmt.parseInt(u64, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--top-p")) {
            o.top_p = try std.fmt.parseFloat(f64, value(args, &i));
        } else if (std.mem.eql(u8, a, "--top-k")) {
            o.top_k = try std.fmt.parseInt(u32, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--min-p")) {
            o.min_p = try std.fmt.parseFloat(f64, value(args, &i));
        } else if (std.mem.eql(u8, a, "--chunks")) {
            var list: std.ArrayList(u32) = .empty;
            var it = std.mem.tokenizeScalar(u8, value(args, &i), ',');
            while (it.next()) |t| try list.append(arena, try std.fmt.parseInt(u32, t, 10));
            o.chunks = list.items;
        } else if (std.mem.eql(u8, a, "--prefill-chunk")) {
            o.chunk = try std.fmt.parseInt(usize, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--report")) {
            o.report = value(args, &i);
        } else if (std.mem.eql(u8, a, "--profile")) {
            o.profile = try std.fmt.parseInt(usize, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--streams")) {
            o.streams = try std.fmt.parseInt(usize, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--lanes")) {
            const v = value(args, &i);
            o.lanes = if (std.mem.eql(u8, v, "auto")) auto_lanes else try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, a, "--table-prices")) {
            o.table_prices = true;
        } else if (std.mem.eql(u8, a, "--lane-depth")) {
            o.lane_depth = try std.fmt.parseInt(u32, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--gpu-round")) {
            o.gpu_round = try std.fmt.parseInt(usize, value(args, &i), 10);
            o.round.depth = o.gpu_round;
        } else if (std.mem.eql(u8, a, "--stop-power")) {
            o.round.power = try std.fmt.parseFloat(f32, value(args, &i));
        } else if (std.mem.eql(u8, a, "--no-stop")) {
            o.round.stop = false;
        } else if (std.mem.eql(u8, a, "--stop-bar")) {
            o.round.bar = try std.fmt.parseFloat(f32, value(args, &i));
        } else if (std.mem.eql(u8, a, "--fixed-depth")) {
            o.round.fixed = true;
        } else if (std.mem.eql(u8, a, "--level-bench")) {
            o.round.level_bench = true;
        } else if (std.mem.eql(u8, a, "--min-match")) {
            o.round.min_match = try std.fmt.parseInt(u32, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--sibling-bar")) {
            o.round.sib_bar = try std.fmt.parseFloat(f64, value(args, &i));
            o.round.siblings = true;
        } else if (std.mem.eql(u8, a, "--siblings")) {
            o.round.siblings = true;
        } else if (std.mem.eql(u8, a, "--no-trees")) {
            o.round.trees = false;
        } else if (std.mem.eql(u8, a, "--no-tail")) {
            o.round.tail = false;
        } else if (std.mem.eql(u8, a, "--no-copy")) {
            o.round.copy = false;
        } else if (std.mem.eql(u8, a, "--copy-max")) {
            o.round.copy_max = try std.fmt.parseInt(usize, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--row-bar")) {
            o.round.head_bar = false;
        } else if (std.mem.eql(u8, a, "--round-profile")) {
            o.round.profile = try std.fmt.parseInt(usize, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--round-split")) {
            o.round.split = true;
        } else if (std.mem.eql(u8, a, "--ahead")) {
            o.round.ahead = try std.fmt.parseInt(usize, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--hand-over")) {
            o.hand_over = try std.fmt.parseInt(usize, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--serial-encoder")) {
            o.serial = true;
        } else if (std.mem.eql(u8, a, "--block-lanes")) {
            o.block_lanes = try std.fmt.parseInt(usize, value(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--block-weights")) {
            o.block_weights = value(args, &i);
        } else if (std.mem.eql(u8, a, "--no-drafts")) {
            o.drafts = false;
        } else if (std.mem.eql(u8, a, "--warmup")) {
            o.warmup = true;
        } else fail("unknown option {s}", .{a});
    }
    if (o.tokens.len == 0) fail("--tokens is required (prompt token ids)", .{});
    if (o.temperature < 0) fail("--temperature takes 0 (greedy) or more", .{});
    if (o.chunk < 1) fail("--prefill-chunk takes 1 row or more", .{});
    if (o.streams < 1 or o.streams > 16) fail("--streams takes 1 to 16", .{});
    return o;
}

const Run = struct {
    tokens: [][]const u32, // each stream's reply
    prefill_s: f64,
    decode_s: f64,
    rounds: u64,
    accepted: u64,
    branch_rows: u64 = 0,
    branch_accepted: u64 = 0,
    landed: [lanes.shape.max_depth][lanes.shape.ranks][2]u32 = @splat(@splat(.{ 0, 0 })), // tree lanes seen, taken
    picks: [lanes.plan_lanes.sizes.len + 1]u32 = @splat(0), // the planner's picks by tree size, then chain + grafts
};

/// `n` streams of the same prompt through the lane core on the Metal backend (n > 1: shared rounds).
fn generate(gpa: std.mem.Allocator, io: std.Io, metal: *nemotron.backend.Metal, cfg: *const lanes.Config, prompt: []const u32, max_new: u32, eos: []const u32, drafts: bool, sampling: ?lanes.Sampling, n: usize) !Run {
    var clock = nemotron.timing.RoundClock{ .b = metal, .wall = .{ .io = io } };
    var engine = lanes.Engine.init(gpa, cfg, metal.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(lanes.Stream, n);
    defer gpa.free(streams);
    var names: [64][8]u8 = undefined;
    for (streams, 0..) |*s, i| {
        const id = try std.fmt.bufPrint(&names[i], "s{d}", .{i});
        s.* = try lanes.Stream.init(gpa, .{ .id = id, .prompt = prompt, .max_new = max_new, .eos = eos, .drafts = drafts, .sampling = sampling, .chunks = cli_chunks });
    }
    defer for (streams) |*s| s.deinit(gpa);
    metal.timing = .{};
    const t0 = mtl.clock.seconds();
    for (streams) |*s| try engine.addStream(s);
    const t1 = mtl.clock.seconds();
    while (engine.activeCount() > 0) try engine.step();
    const t2 = mtl.clock.seconds();
    try metal.drain();
    const out = try gpa.alloc([]const u32, n);
    for (streams, out) |*s, *o| o.* = try gpa.dupe(u32, s.emitted());
    return .{ .tokens = out, .prefill_s = t1 - t0, .decode_s = t2 - t1, .rounds = streams[0].rounds, .accepted = streams[0].accepted, .branch_rows = streams[0].branch_rows, .branch_accepted = streams[0].branch_accepted, .landed = streams[0].landed, .picks = streams[0].picks };
}

/// One greedy stream's rounds on the GPU (gpu_round.zig), with its verify rows by round.
fn generateGpu(gpa: std.mem.Allocator, metal: *nemotron.backend.Metal, prompt: []const u32, max_new: u32, eos: []const u32, o: nemotron.gpu_round.Options) !Run {
    var s = try lanes.Stream.init(gpa, .{ .id = "s0", .prompt = prompt, .max_new = max_new, .eos = eos, .drafts = true, .sampling = null, .chunks = cli_chunks });
    defer s.deinit(gpa);
    metal.timing = .{};
    const g = try nemotron.gpu_round.run(gpa, metal, &s, o, .{});
    const out = try gpa.alloc([]const u32, 1);
    out[0] = try gpa.dupe(u32, s.emitted());
    const r = Run{ .tokens = out, .prefill_s = g.prefill_s, .decode_s = g.decode_s, .rounds = g.rounds, .accepted = g.accepted };
    std.debug.print("gpu round, {d} levels, {d} copied windows, second choices {d} kept of {d}, guesses {d} of {d}: verify rows (rounds):", .{ o.depth, g.copied, g.siblings_kept, g.siblings, g.guesses_kept, g.guesses });
    for (g.by_rows, 0..) |k, n| if (k > 0) std.debug.print(" {d}: {d}", .{ n, k });
    std.debug.print("\nhead depth the rule chose (rounds):", .{});
    for (g.by_cap, 0..) |k, n| if (k > 0) std.debug.print(" {d}: {d}", .{ n, k });
    std.debug.print("\nhead windows, tokens kept (rounds) by verify rows:", .{});
    for (g.head_kept, 0..) |by_kept, n| if (std.mem.indexOfNone(u32, &by_kept, &.{0}) != null) {
        std.debug.print(" | {d} rows", .{n});
        for (by_kept, 0..) |k, kept| if (k > 0) std.debug.print(" {d}:{d}", .{ kept, k });
    };
    std.debug.print("\n", .{});
    return r;
}

/// A GPU round handing its stream to the lane core after `after` commits, as a server does when a request arrives; a second stream joins.
fn generateHanded(gpa: std.mem.Allocator, io: std.Io, metal: *nemotron.backend.Metal, cfg: *const lanes.Config, prompt: []const u32, max_new: u32, eos: []const u32, o: nemotron.gpu_round.Options, after: usize) ![2][]const u32 {
    var clock = nemotron.timing.RoundClock{ .b = metal, .wall = .{ .io = io } };
    var engine = lanes.Engine.init(gpa, cfg, metal.backend(), clock.clock());
    defer engine.deinit();
    var s: [2]lanes.Stream = undefined;
    for (&s, [_][]const u8{ "s0", "s1" }) |*x, id| x.* = try lanes.Stream.init(gpa, .{ .id = id, .prompt = prompt, .max_new = max_new, .eos = eos, .drafts = true, .sampling = null, .chunks = cli_chunks });
    defer for (&s) |*x| x.deinit(gpa);
    const Count = struct {
        n: usize = 0,
        after: usize,
        fn committed(ctx: *anyopaque) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.n += 1;
        }
        fn yield(ctx: *anyopaque) bool {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            return c.n > c.after;
        }
    };
    var count = Count{ .after = after };
    const g = try nemotron.gpu_round.run(gpa, metal, &s[0], o, .{ .ctx = &count, .committed = Count.committed, .yield = Count.yield });
    std.debug.print("hand-over after {d} commits: {s} at token {d}\n", .{ after, if (g.paused) "handed to the lane core" else "finished first", s[0].emitted().len });
    if (g.paused) try engine.adopt(&s[0]);
    try engine.addStream(&s[1]);
    while (engine.activeCount() > 0) try engine.step();
    try metal.drain();
    return .{ try gpa.dupe(u32, s[0].emitted()), try gpa.dupe(u32, s[1].emitted()) };
}

/// Prefill one stream, then time `steps` one-row steps with every dispatch alone: GPU ms by kernel.
fn profileSteps(gpa: std.mem.Allocator, io: std.Io, metal: *nemotron.backend.Metal, cfg: *const lanes.Config, prompt: []const u32, eos: []const u32, steps: usize) !void {
    var clock = nemotron.timing.RoundClock{ .b = metal, .wall = .{ .io = io } };
    var engine = lanes.Engine.init(gpa, cfg, metal.backend(), clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "profile", .prompt = prompt, .max_new = 4, .eos = &.{}, .drafts = false, .sampling = null });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (!metal.caches.contains(&s)) try engine.step();
    _ = eos;
    try nemotron.timing.profile(metal, try metal.cacheOf(&s), steps);
}

fn free(gpa: std.mem.Allocator, r: Run) void {
    for (r.tokens) |t| gpa.free(t);
    gpa.free(r.tokens);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (cluster.cli.wants(@ptrCast(argv[1..]))) std.process.exit(try cluster.cli.main(init, argv[1..]));
    if (checkpoint_cli.wants(@ptrCast(argv[1..]))) std.process.exit(try checkpoint_cli.main(init, argv[1..]));
    const o = try parseArgs(arena, argv);
    cli_chunks = o.chunks;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();

    const m = try nemotron.Model.load(gpa, init.io, o.model, o.drafts);
    defer m.deinit();
    const c = m.config;
    std.debug.print("loaded {s} in {d:.2} s (kernels compiled in {d:.2} s)\n", .{ o.model, m.load_seconds, m.compile_seconds });

    const metal = try nemotron.backend.Metal.init(gpa, m, .{
        .capacity = o.tokens.len + o.max_tokens + 64 + o.profile,
        .chunk = o.chunk,
        .drafts = o.drafts,
        .streams = @max(o.streams, 2),
        .batch_rows = 32,
        .block = if (o.block_weights) |path| try nemotron.head_block.Weights.load(m.device, init.io, gpa, path, c.hidden) else if (o.block_lanes > 0) try nemotron.head_block.Weights.random(m.device, c.hidden, @min(o.block_lanes, 15), 20261004) else null,
    });
    defer metal.deinit();
    metal.concurrent = !o.serial;
    if (metal.head != null) {
        try nemotron.timing.measure(metal);
        const k = metal.costs;
        std.debug.print("windows of 1 to {d} rows (ms):", .{k.windows});
        for (k.window[0..k.windows]) |w| std.debug.print(" {d:.2}", .{w.ms});
        std.debug.print("; shared rounds of 2-row windows (rows: ms):", .{});
        for (k.shared[0..k.shareds]) |w| std.debug.print(" {d}: {d:.2}", .{ w.width, w.ms });
        std.debug.print("; head step {d:.3} ms\n", .{k.head_ms});
    }
    const rows: u32 = if (metal.head != null) nemotron.backend.max_window else 1;
    var cfg = try lanes.Config.init(gpa, metal.facts(), rows, rows - 1);
    defer cfg.deinit(gpa);
    cfg.head_lanes = o.lanes;
    cfg.head_depth = o.lane_depth;
    cfg.own_prices = !o.table_prices;
    metal.ranks_always = o.lanes > 0 and o.streams > 1;
    cfg.fill_lanes = if (o.lanes > 0) @min(o.lanes + 1, nemotron.backend.max_lanes) else 0;

    const eos = c.eos[0..c.eos_count];

    // greedy at temperature 0; else the keyed sampler, its seed given or Python's seed_for(prompt ids, salt 0)
    const sampling: ?lanes.Sampling = if (o.temperature == 0) null else .{
        .seed = o.seed orelse lanes.sampling.seedFor(o.tokens, 0),
        .temperature = o.temperature,
        .top_p = o.top_p,
        .top_k = o.top_k,
        .min_p = o.min_p,
    };
    const gpu = o.gpu_round > 0 and o.lanes == 0 and o.drafts and sampling == null;
    if (o.warmup) free(gpa, if (gpu) try generateGpu(gpa, metal, o.tokens, @min(o.max_tokens, 8), eos, o.round) else try generate(gpa, init.io, metal, &cfg, o.tokens, @min(o.max_tokens, 8), eos, o.drafts, sampling, 1));
    const r = if (gpu) try generateGpu(gpa, metal, o.tokens, o.max_tokens, eos, o.round) else try generate(gpa, init.io, metal, &cfg, o.tokens, o.max_tokens, eos, o.drafts, sampling, 1);
    defer free(gpa, r);

    const t = metal.timing;
    const steps = if (r.tokens[0].len > 1) r.tokens[0].len - 1 else 1;
    const token_wall = r.decode_s * 1e3 / @as(f64, @floatFromInt(steps));
    var hash: [32]u8 = undefined;
    const ids_json = try tf.ids_json.write(arena, r.tokens[0]);
    std.crypto.hash.sha2.Sha256.hash(ids_json, &hash, .{});
    if (gpu) std.debug.print("gpu round ({d} levels{s}): ", .{ o.round.depth, if (o.round.stop) ", confidence stop" else "" }) else if (o.lanes == 0) std.debug.print("lanes 0 (the head's chain): ", .{}) else std.debug.print("lanes auto (at most {d}): ", .{o.lanes});
    std.debug.print("prefill {d} tokens in {d:.3} s; {d} tokens, decode {d:.1} tok/s ({d:.3} ms a token), {d} rounds, {d} drafts accepted, {d} branch rows {d} kept; GPU ms: prefill {d:.3}, step {d:.3}, verify {d:.3}, draft {d:.3}; sha {x}\n", .{ o.tokens.len, r.prefill_s, r.tokens[0].len, @as(f64, @floatFromInt(steps)) / r.decode_s, token_wall, r.rounds, r.accepted, r.branch_rows, r.branch_accepted, t.gpu(.prefill), t.gpu(.step), t.gpu(.verify), t.gpu(.draft), hash[0..6] });
    if (o.lanes > 0) {
        std.debug.print("lanes the planner picked (rounds):", .{});
        for (lanes.plan_lanes.sizes, r.picks[0..lanes.plan_lanes.sizes.len]) |n, k| if (k > 0) std.debug.print(" {d}: {d}", .{ n, k });
        std.debug.print(" chain + grafts: {d}\n", .{r.picks[lanes.plan_lanes.chain_grafts]});
        std.debug.print("tree lanes taken / verified by depth (ranks 1-4):", .{});
        for (r.landed, 0..) |by_rank, d| {
            if (by_rank[0][0] == 0) continue;
            std.debug.print(" d{d}", .{d});
            for (by_rank) |x| std.debug.print(" {d}/{d}", .{ x[1], x[0] });
            std.debug.print(";", .{});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("host encoding ms a command buffer: step {d:.3}, verify {d:.3}, draft {d:.3}\n", .{ t.encode(.step), t.encode(.verify), t.encode(.draft) });

    // streams == solo: the same prompt on `--streams` streams at once, every reply equal to the lone one
    var shared_ok = true;
    if (o.streams > 1) {
        const many = try generate(gpa, init.io, metal, &cfg, o.tokens, o.max_tokens, eos, o.drafts, sampling, o.streams);
        defer free(gpa, many);
        for (many.tokens, 0..) |toks, i| {
            const same = std.mem.eql(u32, toks, r.tokens[0]);
            shared_ok = shared_ok and same;
            std.debug.print("stream {d} of {d}: {s} the lone stream's {d} tokens\n", .{ i, o.streams, if (same) "equals" else "DIFFERS from", r.tokens[0].len });
        }
        const total: usize = blk: {
            var n: usize = 0;
            for (many.tokens) |toks| n += toks.len - 1;
            break :blk n;
        };
        std.debug.print("{d} streams: {d:.1} tok/s together, {d:.1} each\n", .{ o.streams, @as(f64, @floatFromInt(total)) / many.decode_s, @as(f64, @floatFromInt(total)) / many.decode_s / @as(f64, @floatFromInt(o.streams)) });
    }

    if (o.hand_over > 0 and gpu) {
        const handed = try generateHanded(gpa, init.io, metal, &cfg, o.tokens, o.max_tokens, eos, o.round, o.hand_over);
        defer for (handed) |x| gpa.free(x);
        for (handed, [_][]const u8{ "the handed stream", "the stream that joined" }) |x, who| {
            const same = std.mem.eql(u32, x, r.tokens[0]);
            shared_ok = shared_ok and same;
            std.debug.print("{s}: {s} the lone GPU round's {d} tokens\n", .{ who, if (same) "equals" else "DIFFERS from", r.tokens[0].len });
        }
    }

    if (o.profile > 0) try profileSteps(gpa, init.io, metal, &cfg, o.tokens, eos, o.profile);

    if (o.report) |path| {
        var out: std.ArrayList(u8) = .empty;
        try out.print(arena, "{{\"tokens\": {s}, \"decode_seconds\": {d}, \"prefill_seconds\": {d}, \"rounds\": {d}, \"accepted_drafts\": {d}, \"branch_rows\": {d}, \"branch_accepted\": {d}, \"lanes\": {s}, \"picks\": {s}, \"token_ms_wall\": {d}, \"step_ms_gpu\": {d}, \"verify_ms_gpu\": {d}, \"draft_ms_gpu\": {d}, \"load_seconds\": {d}, \"streams_equal\": {}, \"engine\": \"zig-metal\"}}\n", .{ ids_json, r.decode_s, r.prefill_s, r.rounds, r.accepted, r.branch_rows, r.branch_accepted, try landedJson(arena, r), try picksJson(arena, r), token_wall, t.gpu(.step), t.gpu(.verify), t.gpu(.draft), m.load_seconds, shared_ok });
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = out.items });
    }
    if (!shared_ok) std.process.exit(1);
}

/// Tree lanes as [depth, rank, verified, taken] rows (verified only).
fn landedJson(arena: std.mem.Allocator, r: Run) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(arena, '[');
    var first = true;
    for (r.landed, 0..) |by_rank, d| for (by_rank, 0..) |x, k| {
        if (x[0] == 0) continue;
        try out.print(arena, "{s}[{d}, {d}, {d}, {d}]", .{ if (first) "" else ", ", d, k, x[0], x[1] });
        first = false;
    };
    try out.append(arena, ']');
    return out.items;
}

/// The planner's picks as [lanes, rounds] rows (0 lanes: chain + grafts).
fn picksJson(arena: std.mem.Allocator, r: Run) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(arena, '[');
    for (r.picks, 0..) |k, i| {
        const n: u32 = if (i < lanes.plan_lanes.sizes.len) lanes.plan_lanes.sizes[i] else 0;
        try out.print(arena, "{s}[{d}, {d}]", .{ if (i > 0) ", " else "", n, k });
    }
    try out.append(arena, ']');
    return out.items;
}
