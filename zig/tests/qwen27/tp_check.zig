//! The 27B TP gate on one Mac, no wire: ranks = 1 parity bit for bit, then two shards joined through `pair.Loopback`.
const std = @import("std");
const q = @import("qwen27");
const mtl = @import("metal");
const pair = @import("core").pair;
const rep = @import("tp_report.zig");

const Runner = q.decode_round.Runner;
const Session = q.session.Session;
const TopTwo = q.session.TopTwo;
const Model = q.model.Model;

const help =
    \\tf-qwen27-tp-check --model DIR [--steps N] [--ids N] [--chunk N] [--decode-prompt] [--stage 1|2|3] [--mode full|mlp]
    \\  stage 1: ranks = 1 through loadShard against the single-Mac load: logits bytes, state digest, greedy ids.
    \\  stage 2: two rank-local shards, every layer seam through pair.Loopback (host memcpy link), ids vs stage 1.
    \\  stage 3 (the default) runs both.
    \\--chunk N feeds the prompt in N-row rounds; 1 (the default) is a row a round, and N may reach 512,
    \\  model.promptRows(generation). A Mac whose prompt side fuses takes all 512 rows a round; elsewhere a round is 128
    \\  rows, and a wider `--chunk` is fed in those rounds. Greedy only, no drafter.
    \\--decode-prompt feeds those rounds through decodeChunk, an earlier reply's arithmetic rather than the prompt
    \\  kernels (the merge path a served chat takes), in the same rounds. The pair's stages keep 128, the rows the
    \\  seam window holds.
    \\--prompt-file F takes the prompt ids from F; --forced-file F feeds F's ids instead of each side's own picks.
    \\--taps A,B,C,D,E hashes five layers' post-prompt hidden rows on every side (stage 2 divergence localisation).
    \\Synthetic ids 1000.., no tokenizer, no weights written; JSON receipt on stdout, errors on stderr.
    \\Run with no --model DIR to print SKIP and exit 0: the suite does that on hosts without the checkpoint.
;

var dump = false;
/// Feed the prompt through `decodeChunk` in `chunk`-row rounds, not `prefill`'s prompt kernels (`--decode-prompt`).
var decode_prompt = false;
/// Five layers whose post-block residuals the tap planes keep, for stage 2 divergence localisation (`--taps`).
var tap_ids: ?[5]usize = null;
/// `--taps 0,1,2,3,4`: five layer indices, ascending, whose post-block residuals stage 2 keeps for comparison.
fn readTaps(text: []const u8) ![5]usize {
    var out: [5]usize = undefined;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    var n: usize = 0;
    while (it.next()) |token| {
        if (n == out.len) return error.BadTapList;
        out[n] = try std.fmt.parseInt(usize, token, 10);
        n += 1;
    }
    if (n != out.len) return error.BadTapList;
    return out;
}

/// The prompt as the arithmetic that feeds it: `prefill`'s prompt kernels, or `--decode-prompt`'s `decodeChunk` rounds.
fn feedPrompt(s: Session, ids: []const u32, chunk: usize, decode: bool) !void {
    // the rounds a served chat gives an earlier reply: `chunk` rows asked, `model.promptRows(generation)` granted
    if (!decode) return s.prefill(ids, chunk);
    var first: usize = 0;
    const row = s.runner.promptRows(@intCast(chunk));
    while (first < ids.len) {
        const end = @min(ids.len, first + row);
        try s.decodeChunk(ids[first..end], end == ids.len);
        first = end;
    }
}

const Local = struct {
    tokens: []u32,
    top2: []TopTwo,
    logits: [][32]u8,
    state: [32]u8,
    /// Post-block residuals of the prompt round at `tap_ids`, hashed on this side's own hidden state.
    taps: [5][32]u8 = undefined,
    /// `MTLDevice.currentAllocatedSize` with this one model resident (the engine's own memory accounting).
    resident: u64,
};

/// A whole stream on one Mac: `shard1` takes `loadShard(..., rank = 0, ranks = 1, .full)`, else `Model.load`.
fn localRun(a: std.mem.Allocator, io: std.Io, dir: []const u8, ids: []const u32, steps: usize, chunk: usize, decode: bool, force: ?[]const u32, shard1: bool) !Local {
    const m = if (shard1) try Model.loadShard(a, io, dir, 128, 0, 1, .full, null) else try Model.load(a, io, dir, 128, null);
    defer m.deinit();
    var runner = try Runner.init(a, m, 1, rowsFor(ids.len, steps));
    defer runner.deinit();
    const s = Session{ .runner = &runner };
    if (tap_ids) |layers| {
        runner.capture_taps = true;
        runner.taps.ids = layers;
    }
    try feedPrompt(s, ids, chunk, decode);
    const taps = try rep.tapDigests(&runner);
    runner.capture_taps = false;
    const state = try s.fingerprint();
    rep.dumpRow("single", m, &runner, dump);
    const resident = m.device.allocated();
    const tokens = try a.alloc(u32, steps);
    errdefer a.free(tokens);
    const top2 = try a.alloc(TopTwo, steps);
    errdefer a.free(top2);
    const logits = try a.alloc([32]u8, steps);
    errdefer a.free(logits);
    for (0..steps) |i| {
        tokens[i] = try s.greedy();
        top2[i] = try q.session.topTwo(m.frame.get(.logits).buffer.slice(u16, m.config.vocab));
        logits[i] = rep.shaLogits(m);
        // a forced replay feeds the recorded history, so both sides stand at the same prefix when they compare
        if (i + 1 < steps) try s.step(if (force) |f| f[i] else tokens[i]);
    }
    return .{ .tokens = tokens, .top2 = top2, .logits = logits, .state = state, .taps = taps, .resident = resident };
}

/// One rank of the pair: its shard, its seam transport, and the stream it produced.
const Side = struct {
    rank: u32,
    ids: []const u32,
    steps: usize,
    chunk: usize,
    /// The prompt through `decodeChunk` rounds, as a served chat feeds an earlier reply (see `--decode-prompt`).
    decode: bool = false,
    /// A recorded history both ranks step through instead of their own picks (the single Mac's reply, say).
    force: ?[]const u32 = null,
    m: *Model = undefined,
    lb: pair.Loopback = undefined,
    picks: mtl.Buffer = undefined,
    runner: Runner = undefined,
    tokens: []u32 = &.{},
    top2: []TopTwo = &.{},
    logits: [][32]u8 = &.{},
    /// When dumping, each rank copies its span of every step's logits here, so the whole row can be hashed.
    whole: ?[]u8 = null,
    state: [32]u8 = undefined,
    /// Post-block residuals of the prompt round at `tap_ids`, hashed on this rank's own hidden state.
    taps: [5][32]u8 = undefined,
    err: ?anyerror = null,

    fn open(s: *Side, a: std.mem.Allocator, io: std.Io, dir: []const u8, mode: q.config.Mode, bytes: usize) !void {
        s.m = try Model.loadShard(a, io, dir, 128, s.rank, 2, mode, null);
        errdefer s.m.deinit();
        s.runner = try Runner.init(a, s.m, 1, rowsFor(s.ids.len, s.steps));
        errdefer s.runner.deinit();
        // one `pickBytes` a row, and a prompt round writes `chunk` rows before the decodes write one each
        s.picks = try s.m.device.buffer(@max(s.steps, s.chunk) * 2 * @sizeOf(u32), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer s.picks.deinit();
        s.lb = try pair.Loopback.init(s.m.device, bytes, s.rank);
        errdefer s.lb.deinit();
        s.runner.reduce = .{ .ptr = &s.lb, .part = loopPart, .stage = loopStage, .gather = loopGather, .failed = loopFailed, .argmax = loopArgmax };
        s.runner.picks = .{ .buffer = s.picks, .offset = 0 };
        s.tokens = try a.alloc(u32, s.steps);
        s.top2 = try a.alloc(TopTwo, s.steps);
        s.logits = try a.alloc([32]u8, s.steps);
    }

    fn close(s: *Side) void {
        s.runner.deinit();
        s.picks.deinit();
        s.lb.deinit();
        s.m.deinit();
    }

    fn run(s: *Side) void {
        s.stream() catch |err| {
            s.err = err;
            // a rank that stops must not leave its peer's seam waiting: the sentinel is what a failed link does
            s.lb.satisfy();
        };
    }

    fn stream(s: *Side) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const ses = Session{ .runner = &s.runner };
        if (tap_ids) |layers| {
            s.runner.capture_taps = true;
            s.runner.taps.ids = layers;
        }
        try feedPrompt(ses, s.ids, s.chunk, s.decode);
        s.taps = try rep.tapDigests(&s.runner);
        s.runner.capture_taps = false;
        s.state = try ses.fingerprint();
        trace(s, "prefill");
        rep.dumpRow(if (s.rank == 0) "pair-rank0" else "pair-rank1", s.m, &s.runner, dump);
        for (0..s.steps) |i| {
            const values = s.m.frame.get(.logits).buffer.slice(u16, s.m.config.vocab);
            s.tokens[i] = try ses.greedy();
            s.top2[i] = try q.session.topTwo(values);
            s.logits[i] = rep.shaLogits(s.m);
            if (s.whole) |w| {
                const span = s.m.shard.vocabRows();
                const row = i * @as(usize, s.m.shard.global.vocab) * 2;
                @memcpy(w[row + span.from * 2 ..][0 .. span.len * 2], std.mem.sliceAsBytes(values));
            }
            trace(s, "step");
            if (i + 1 < s.steps) try ses.step(if (s.force) |f| f[i] else s.tokens[i]);
        }
    }
};

/// The seam's exchange counter after every round: both ranks must have encoded the same number of exchanges.
fn trace(s: *const Side, where: []const u8) void {
    if (!dump) return;
    std.debug.print("[x] rank={d} after={s} x={d} token={d}\n", .{ s.rank, where, s.lb.x, s.tokens[0] });
}

fn loopPart(p: *anyopaque) q.forward.Seam {
    const lb: *pair.Loopback = @ptrCast(@alignCast(p));
    const seam = lb.part();
    return .{ .x = seam.x, .ref = .{ .buffer = seam.ref.buf, .offset = seam.ref.off } };
}

fn loopStage(p: *anyopaque, e: *mtl.ComputeEncoder, x: u32, src: q.forward.Ref, rows: u32, width: u32, stride: u32) void {
    const lb: *pair.Loopback = @ptrCast(@alignCast(p));
    lb.stage(e, x, .{ .buf = src.buffer, .off = src.offset }, rows, width, stride);
}

fn loopGather(p: *anyopaque, e: *mtl.ComputeEncoder, x: u32, mine: q.forward.Ref, out: q.forward.Ref, rows: u32, width: u32, out_stride: u32) void {
    const lb: *pair.Loopback = @ptrCast(@alignCast(p));
    lb.gather(e, x, .{ .buf = mine.buffer, .off = mine.offset }, .{ .buf = out.buffer, .off = out.offset }, rows, width, out_stride);
}

fn loopFailed(p: *anyopaque) bool {
    const lb: *pair.Loopback = @ptrCast(@alignCast(p));
    return lb.failed();
}

fn loopArgmax(p: *anyopaque, e: *mtl.ComputeEncoder, x: u32, logits: q.forward.Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: q.forward.Ref) void {
    const lb: *pair.Loopback = @ptrCast(@alignCast(p));
    lb.argmaxBf16(e, x, .{ .buf = logits.buffer, .off = logits.offset }, stride, lo, n, rows, .{ .buf = picks.buffer, .off = picks.offset });
}

/// The wire, factored out: one thread per direction relays the producer's posted part into the consumer's slot.
fn pump(from: *pair.Loopback, to: *pair.Loopback, stop: *std.atomic.Value(bool)) void {
    var x: u32 = 1;
    while (!stop.load(.acquire)) {
        if (from.posted.value() < x) {
            std.atomic.spinLoopHint();
            continue;
        }
        const n: usize = from.bytesFor(x);
        to.arrive(x, from.win[from.sendOffset(x)..][0..n]);
        x +%= 1;
    }
}

fn runSide(s: *Side) void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    s.run();
}

/// KV rows one stream needs: every prompt id plus every decode step, padded so a round never runs out mid-way.
fn rowsFor(ids: usize, steps: usize) u32 {
    return @intCast(@min(ids + steps + 8, 32 * 1024));
}

/// Ids from a dump (`[248045, 846, …]`) or any run of decimals: brackets, commas and whitespace all separate.
fn readIds(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    var out: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, text, "[](), \t\r\n");
    while (it.next()) |token| try out.append(a, try std.fmt.parseInt(u32, token, 10));
    return out.toOwnedSlice(a);
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 1) {
        try std.Io.File.stdout().writeStreamingAll(io, "SKIP qwen27-tp-check: no --model DIR, so this host has no checkpoint to check\n");
        return;
    }
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return;
    }
    var dir: ?[]const u8 = null;
    var steps: usize = 8;
    var count: usize = 64;
    var chunk: usize = 1;
    var stage: u8 = 3;
    var mode: q.config.Mode = .full;
    var prompt_file: ?[]const u8 = null;
    var forced_file: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--dump")) {
            dump = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--decode-prompt")) {
            decode_prompt = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        const value = args[i];
        if (std.mem.eql(u8, arg, "--model")) dir = value else if (std.mem.eql(u8, arg, "--steps")) {
            steps = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--ids")) {
            count = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--chunk")) {
            chunk = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--stage")) {
            stage = try std.fmt.parseInt(u8, value, 10);
        } else if (std.mem.eql(u8, arg, "--mode")) {
            mode = if (std.mem.eql(u8, value, "mlp")) .mlp else if (std.mem.eql(u8, value, "full")) .full else return error.UnknownMode;
        } else if (std.mem.eql(u8, arg, "--prompt-file")) {
            prompt_file = value;
        } else if (std.mem.eql(u8, arg, "--forced-file")) {
            forced_file = value;
        } else if (std.mem.eql(u8, arg, "--taps")) {
            tap_ids = try readTaps(value);
        } else return error.UnknownOption;
    }
    if (dir == null or steps == 0 or steps > 256 or count == 0 or count > 512 or chunk == 0 or chunk > 512 or stage == 0 or stage > 3) return error.BadOptions;
    // a pair stages its widest round through the window `pair.stageBytes` gives it: 128 rows
    if (chunk > 128 and stage & 2 != 0) return error.PairChunkWiderThanTheSeamWindow;
    // synthetic ids: no tokenizer, deterministic, every id a valid embedding row
    const ids = if (prompt_file) |f| try readIds(a, io, f) else blk: {
        const out = try a.alloc(u32, count);
        for (out, 0..) |*id, j| id.* = @intCast(1000 + j);
        break :blk out;
    };
    if (ids.len == 0 or ids.len > 512) return error.BadOptions;
    count = ids.len;
    const force: ?[]const u32 = if (forced_file) |f| try readIds(a, io, f) else null;
    if (force) |f| if (f.len < steps) return error.ShortForcedFile;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const out = std.Io.File.stdout();

    var reference: ?Local = null;
    if (stage & 1 != 0) {
        const single = try localRun(a, io, dir.?, ids, steps, chunk, decode_prompt, force, false);
        const shard1 = try localRun(a, io, dir.?, ids, steps, chunk, decode_prompt, force, true);
        const same_tokens = std.mem.eql(u32, single.tokens, shard1.tokens);
        const same_state = std.mem.eql(u8, &single.state, &shard1.state);
        var first_logits: ?usize = null;
        for (single.logits, shard1.logits, 0..) |one, other, at| if (first_logits == null and !std.mem.eql(u8, &one, &other)) {
            first_logits = at;
        };
        const single_hex = std.fmt.bytesToHex(single.state, .lower);
        const shard1_hex = std.fmt.bytesToHex(shard1.state, .lower);
        const receipt = try std.json.Stringify.valueAlloc(a, .{
            .kind = "qwen27_tp_check",
            .stage = 1,
            .model = dir.?,
            .ids = count,
            .steps = steps,
            .chunk = chunk,
            .prompt_arithmetic = if (decode_prompt) "decode" else "prompt",
            .single_mac_tokens = single.tokens,
            .shard1_tokens_equal = same_tokens,
            .shard1_logits_byte_equal = first_logits == null,
            .shard1_first_logits_difference = first_logits,
            .shard1_prompt_state_equal = same_state,
            .single_mac_resident_bytes = single.resident,
            .shard1_resident_bytes = shard1.resident,
            .single_mac_prompt_state = @as([]const u8, &single_hex),
            .shard1_prompt_state = @as([]const u8, &shard1_hex),
        }, .{});
        try out.writeStreamingAll(io, receipt);
        try out.writeStreamingAll(io, "\n");
        if (!same_tokens or first_logits != null or !same_state) return error.ShardAtRanksOneDiffers;
        reference = single;
    }

    if (stage & 2 == 0) return;
    const ref = reference orelse return error.StageTwoNeedsTheSingleMacStream;

    // The window holds the widest prompt round: the MLP seam stages a bf16 row of this rank's half of `intermediate`.
    const bytes = pair.stageBytes(128, 17408 / 2);
    const zero = try a.create(Side);
    const one = try a.create(Side);
    zero.* = .{ .rank = 0, .ids = ids, .steps = steps, .chunk = chunk, .decode = decode_prompt, .force = force };
    one.* = .{ .rank = 1, .ids = ids, .steps = steps, .chunk = chunk, .decode = decode_prompt, .force = force };
    try zero.open(a, io, dir.?, mode, bytes);
    defer zero.close();
    const rank0_alone = zero.m.device.allocated();
    try one.open(a, io, dir.?, mode, bytes);
    defer one.close();
    const both = zero.m.device.allocated();
    const working = zero.m.device.maxWorkingSet();
    const global_vocab = zero.m.shard.global.vocab;

    // Dumping also keeps every step's logits row whole, so the pair's row can be hashed against the single Mac's row.
    const whole: ?[]u8 = if (dump) try a.alloc(u8, steps * @as(usize, global_vocab) * 2) else null;
    zero.whole = whole;
    one.whole = whole;

    var stop = std.atomic.Value(bool).init(false);
    const a2b = try std.Thread.spawn(.{}, pump, .{ &zero.lb, &one.lb, &stop });
    const b2a = try std.Thread.spawn(.{}, pump, .{ &one.lb, &zero.lb, &stop });
    const t0 = try std.Thread.spawn(.{}, runSide, .{zero});
    const t1 = try std.Thread.spawn(.{}, runSide, .{one});
    t0.join();
    t1.join();
    stop.store(true, .release);
    a2b.join();
    b2a.join();

    if (zero.err) |err| {
        try out.writeStreamingAll(io, try std.fmt.allocPrint(a, "{{\"kind\":\"qwen27_tp_check\",\"stage\":2,\"rank0_error\":\"{s}\"}}\n", .{@errorName(err)}));
        return err;
    }
    if (one.err) |err| {
        try out.writeStreamingAll(io, try std.fmt.allocPrint(a, "{{\"kind\":\"qwen27_tp_check\",\"stage\":2,\"rank1_error\":\"{s}\"}}\n", .{@errorName(err)}));
        return err;
    }

    if (dump) {
        // both ranks recorded a length per exchange: a disagreement means the layer loops entered the seam differently
        const x0 = zero.lb.x;
        const x1 = one.lb.x;
        var bad: usize = 0;
        var first_bad: ?usize = null;
        for (1..@min(x0, x1) + 1) |at| {
            const want = zero.lb.bytesFor(@intCast(at));
            const got = one.lb.bytesFor(@intCast(at));
            if (want == got) continue;
            bad += 1;
            if (first_bad == null) first_bad = at;
            if (bad <= 8) std.debug.print("[audit] x={d} rank0_bytes={d} rank1_bytes={d}\n", .{ at, want, got });
        }
        std.debug.print("[audit] x_rank0={d} x_rank1={d} mismatched_exchanges={d} first_bad={?d}\n", .{ x0, x1, bad, first_bad });
    }

    // A tapped layer whose hidden rows differ names where a seam stops being bit-exact; the logits only show the drift.
    var first_tap_diff: ?usize = null;
    if (tap_ids) |layers| for (layers, 0..) |layer, at| {
        const zero_hex = std.fmt.bytesToHex(zero.taps[at], .lower);
        const one_hex = std.fmt.bytesToHex(one.taps[at], .lower);
        const want_hex = std.fmt.bytesToHex(ref.taps[at], .lower);
        std.debug.print("[taps] layer={d} single={s} rank0={s} rank1={s}\n", .{ layer, want_hex[0..], zero_hex[0..], one_hex[0..] });
        if (first_tap_diff == null and !std.mem.eql(u8, &ref.taps[at], &zero.taps[at]) and !std.mem.eql(u8, &ref.taps[at], &one.taps[at])) first_tap_diff = layer;
    };

    // The token comparison only sees a tie when one flips; hashing every step's whole row shows the first bit of drift.
    var first_logit_step: ?usize = null;
    if (whole) |w| {
        const row_bytes = @as(usize, global_vocab) * 2;
        for (0..steps) |at| {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hash.update(w[at * row_bytes ..][0..row_bytes]);
            var got: [32]u8 = undefined;
            hash.final(&got);
            if (std.mem.eql(u8, &got, &ref.logits[at])) continue;
            first_logit_step = at;
            const merged = rep.mergeTop2(.{ 0, @intCast(zero.m.config.vocab) }, .{ zero.top2[at], one.top2[at] });
            const want = ref.top2[at];
            std.debug.print("[logits] first_differing_step={d} single={d}@{d},{d}@{d} pair={d}@{d},{d}@{d}\n", .{
                at, want.ids[0], want.logits[0], want.ids[1], want.logits[1], merged.ids[0], merged.logits[0], merged.ids[1], merged.logits[1],
            });
            break;
        }
        std.debug.print("[logits] steps_compared={d} first_differing_step={?d}\n", .{ steps, first_logit_step });
    }

    var first_difference: ?usize = null;
    for (ref.tokens, zero.tokens, 0..) |want, got, at| if (first_difference == null and want != got) {
        first_difference = at;
    };
    const ranks_agree = std.mem.eql(u32, zero.tokens, one.tokens);
    var bad_pick: ?usize = null;
    for (zero.tokens, one.tokens, 0..) |x, y, at| if (bad_pick == null and (x >= global_vocab or y >= global_vocab or x != y)) {
        bad_pick = at;
    };
    var merged: ?TopTwo = null;
    var single: ?TopTwo = null;
    if (first_difference) |at| {
        merged = rep.mergeTop2(.{ 0, @intCast(zero.m.config.vocab) }, .{ zero.top2[at], one.top2[at] });
        single = ref.top2[at];
    }
    const zero_state = std.fmt.bytesToHex(zero.state, .lower);
    const one_state = std.fmt.bytesToHex(one.state, .lower);
    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .kind = "qwen27_tp_check",
        .stage = 2,
        .model = dir.?,
        .mode = @tagName(mode),
        .ranks = 2,
        .ids = count,
        .steps = steps,
        .chunk = chunk,
        .prompt_arithmetic = if (decode_prompt) "decode" else "prompt",
        .rank0_tokens = zero.tokens,
        .rank1_tokens = one.tokens,
        .reference_tokens = ref.tokens,
        .ranks_token_equal = ranks_agree,
        .reference_token_equal = first_difference == null,
        .first_difference = first_difference,
        .single_top2 = single,
        .rank0_top2 = if (first_difference) |at| zero.top2[at] else null,
        .rank1_top2 = if (first_difference) |at| one.top2[at] else null,
        .pair_merged_top2 = merged,
        .pair_logits_compared = whole != null,
        .pair_logits_byte_equal = if (whole == null) null else first_logit_step == null,
        .pair_first_logits_difference = first_logit_step,
        .bad_pick_at = bad_pick,
        .seam_gave_up_rank0 = zero.lb.gaveUp(),
        .seam_gave_up_rank1 = one.lb.gaveUp(),
        .rank0_prompt_state = @as([]const u8, &zero_state),
        .rank1_prompt_state = @as([]const u8, &one_state),
        .global_vocab = global_vocab,
        .local_vocab = zero.m.config.vocab,
        .rank0_alone_resident_bytes = rank0_alone,
        .both_ranks_resident_bytes = both,
        .second_rank_increment_bytes = both - rank0_alone,
        .device_max_working_set = working,
    }, .{});
    try out.writeStreamingAll(io, receipt);
    try out.writeStreamingAll(io, "\n");
    if (!ranks_agree or bad_pick != null) return error.RanksDisagree;
    if (zero.lb.gaveUp() != 0 or one.lb.gaveUp() != 0) return error.SeamGaveUp;
    if (first_difference != null) return error.PairTokensDifferFromSingleMac;
    if (first_logit_step != null) return error.PairLogitsDifferFromSingleMac;
}
