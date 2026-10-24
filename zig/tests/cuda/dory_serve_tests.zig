//! Dory served on the real checkpoint: shared rounds == solo, graphs == eager, copy drafts == plain, kept == fresh.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const dory = @import("dory");
const check = @import("check.zig");
const Rig = @import("dory_tests.zig").Rig;
const Gpu = check.Gpu;

const fwd = dory.cuda_model;
const be = dory.cuda_backend;

pub fn hashOf(ids: []const u32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ids));
}

pub const Case = struct { prompt: []const u32, sampling: ?lanes.Sampling = null, drafts: bool = false };
pub const Run = struct { out: [][]u32, secs: f64, tokens: usize, rounds: u64, accepted: u64 };

pub fn freeRun(gpa: std.mem.Allocator, r: Run) void {
    for (r.out) |o| gpa.free(o);
    gpa.free(r.out);
}

/// Cases admitted together into one lane engine on the backend, run to the end (eos ignored: every reply `count` long).
pub fn lanesRun(gpu: Gpu, db: *be.Dory, cases: []const Case, count: u32) !Run {
    const gpa = gpu.gpa;
    var cfg = try lanes.Config.init(gpa, db.facts(), be.max_window, be.max_window - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, db.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    const props = try gpa.alloc(lanes.SuffixLookup, cases.len);
    defer gpa.free(props);
    for (cases, streams, props) |c, *s, *p| {
        p.* = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try lanes.Stream.init(gpa, .{ .id = "d", .prompt = c.prompt, .max_new = count, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer() });
    }
    defer for (streams, props) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    const t0 = check.now(gpu.io);
    while (engine.activeCount() > 0) try engine.step();
    var out: Run = .{ .out = try gpa.alloc([]u32, cases.len), .secs = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9, .tokens = 0, .rounds = 0, .accepted = 0 };
    for (out.out, streams) |*o, *s| {
        o.* = try gpa.dupe(u32, s.emitted());
        out.tokens += o.len;
        out.rounds += s.rounds;
        out.accepted += s.accepted;
    }
    return out;
}

/// TEXT's tokens after <s>, repeated until `need` of them.
fn tokens(gpu: Gpu, model: []const u8, path: []const u8, need: usize) ![]u32 {
    const gpa = gpu.gpa;
    var tok = try core.tokenizer.loadTokenizer(gpu.io, gpa, model);
    defer tok.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    if (ids.len == 0) return error.EmptyText;
    const out = try gpa.alloc(u32, need);
    out[0] = 1;
    for (out[1..], 0..) |*t, i| t.* = ids[i % ids.len];
    return out;
}

pub const Setup = struct { loops: usize, count: u32, ids: []u32, rig: Rig, db: be.Dory };

pub fn setup(x: *Setup, gpu: Gpu, rest: []const []const u8, need: usize, slots: usize) !void {
    if (rest.len < 4) return error.MissingArguments;
    x.loops = try std.fmt.parseInt(usize, rest[1], 10);
    x.count = try std.fmt.parseInt(u32, rest[3], 10);
    x.ids = try tokens(gpu, rest[0], rest[2], need);
    errdefer gpu.gpa.free(x.ids);
    try x.rig.initSlots(gpu, rest[0], x.loops, try be.capacity(need + x.count), slots);
    errdefer x.rig.deinit();
    x.db = try be.Dory.init(gpu.gpa, gpu.d, &x.rig.m, &x.rig.k, x.rig.s);
}

pub fn teardown(x: *Setup, gpa: std.mem.Allocator) void {
    x.db.deinit();
    x.rig.deinit();
    gpa.free(x.ids);
}

pub fn run(gpu: Gpu, cmd: []const u8, rest: []const []const u8) !void {
    if (std.mem.eql(u8, cmd, "dory-serve")) return serve(gpu, rest);
    if (std.mem.eql(u8, cmd, "dory-bench")) return bench(gpu, rest);
    return error.UnknownCommand;
}

/// Eight mixed prompts (short, past the window, past a chunk; greedy and sampled; a repeating one for copy drafts).
fn mixed(ids: []const u32, buf: *[8]Case, rep: []u32) []const Case {
    const sp: lanes.Sampling = .{ .seed = 11, .temperature = 0.8, .top_k = 40, .top_p = 0.95 };
    const sq: lanes.Sampling = .{ .seed = 5, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
    @memcpy(rep[0 .. rep.len / 2], ids[0 .. rep.len / 2]);
    @memcpy(rep[rep.len / 2 ..], ids[0 .. rep.len - rep.len / 2]);
    buf.* = .{
        .{ .prompt = ids[0..300] },
        .{ .prompt = ids[0..1500], .sampling = sp },
        .{ .prompt = ids[0..2600] },
        .{ .prompt = rep, .sampling = sq },
        .{ .prompt = ids[0..90], .sampling = sp },
        .{ .prompt = ids[0..3100] },
        .{ .prompt = ids[0..1100], .sampling = sq },
        .{ .prompt = ids[0..2049] },
    };
    return buf;
}

fn serve(gpu: Gpu, rest: []const []const u8) !void {
    const gpa = gpu.gpa;
    var x: Setup = undefined;
    try setup(&x, gpu, rest, 4000, 8);
    defer teardown(&x, gpa);
    const rep = try gpa.alloc(u32, 1800);
    defer gpa.free(rep);
    var buf: [8]Case = undefined;
    const cases = mixed(x.ids, &buf, rep);
    var bad: usize = 0;
    // each case alone, graphs on: the reference; then alone with graphs off, and alone with copy drafts
    var solo: [8][]u32 = undefined;
    for (cases, 0..) |c, i| {
        const r = try lanesRun(gpu, &x.db, &.{c}, x.count);
        solo[i] = r.out[0];
        gpa.free(r.out);
    }
    defer for (solo) |o| gpa.free(o);
    x.rig.m.graphed = false;
    for (cases, 0..) |c, i| {
        const r = try lanesRun(gpu, &x.db, &.{c}, x.count);
        defer freeRun(gpa, r);
        const eq = std.mem.eql(u32, r.out[0], solo[i]);
        if (!eq) bad += 1;
        std.debug.print("RESULT dory L={d} eager == graphs: case {d} ({d} prompt tokens) {x} vs {x} {s}\n", .{ x.loops, i, c.prompt.len, hashOf(r.out[0]), hashOf(solo[i]), if (eq) "same" else "DIFFER" });
    }
    x.rig.m.graphed = true;
    bad += try roundLogits(&x, cases);
    for (cases, 0..) |c, i| {
        var d = c;
        d.drafts = true;
        const r = try lanesRun(gpu, &x.db, &.{d}, x.count);
        defer freeRun(gpa, r);
        const eq = std.mem.eql(u32, r.out[0], solo[i]);
        if (!eq) bad += 1;
        std.debug.print("RESULT dory L={d} drafted == plain: case {d} {s} ({d} rounds for {d} tokens, {d} drafts landed)\n", .{ x.loops, i, if (eq) "same" else "DIFFER", r.rounds, r.tokens, r.accepted });
    }
    for ([_]usize{ 2, 4, 8 }) |k| for ([_]bool{ false, true }) |drafts| {
        var cs: [8]Case = undefined;
        for (cs[0..k], cases[0..k]) |*o, c| {
            o.* = c;
            o.drafts = drafts;
        }
        const r = try lanesRun(gpu, &x.db, cs[0..k], x.count);
        defer freeRun(gpa, r);
        var differ: usize = 0;
        for (r.out, solo[0..k]) |o, s| differ += @intFromBool(!std.mem.eql(u32, o, s));
        bad += differ;
        std.debug.print("RESULT dory L={d} concurrent == solo: {d} streams{s}, {d} of {d} replies differ, {d:.1} tok/s together\n", .{ x.loops, k, if (drafts) " drafted" else "", differ, k, @as(f64, @floatFromInt(r.tokens)) / r.secs });
    };
    try check.expect(bad == 0, "dory L={d} serve: {d} paths differ from the solo replies", .{ x.loops, bad });
    check.pass("dory L={d} serve: eager == graphs (tokens and a round's logits), drafted == plain, concurrent == solo at 2/4/8 ({d} tokens each)", .{ x.loops, x.count });
}

/// One decode round over every case's slot at its prompt end: the graph's fp32 logits bytes against eager's.
fn roundLogits(x: *Setup, cases: []const Case) !usize {
    const gpa = x.rig.gpu.gpa;
    var chains: [8]fwd.Chain = undefined;
    var rows: usize = 0;
    for (cases, 0..) |c, i| {
        const n = 1 + i % 3; // windows of 1 to 3 rows: the prompt's own last tokens, refed past its end
        var at: usize = 0;
        while (at < c.prompt.len) : (at += fwd.prompt_rows) {
            const end = @min(c.prompt.len, at + fwd.prompt_rows);
            gpa.free(try x.rig.slotRows(i, c.prompt[at..end], at, true, end - at - 1, 1));
        }
        chains[i] = .{ .slot = i, .p0 = c.prompt.len, .tokens = c.prompt[c.prompt.len - n ..] };
        rows += n;
    }
    var got: [2][]f32 = undefined;
    for ([_]bool{ true, false }, 0..) |g, j| {
        x.rig.m.graphed = g;
        try x.rig.m.forward(&x.rig.k, x.rig.s, chains[0..cases.len], false, 0, rows);
        try x.rig.s.synchronize();
        got[j] = try gpa.alloc(f32, rows * x.rig.w.config.vocab);
        try (cuda.DeviceBuffer{ .d = x.rig.gpu.d, .ptr = x.rig.m.s.logits, .len = got[j].len * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(got[j]));
    }
    x.rig.m.graphed = true;
    defer for (got) |g| gpa.free(g);
    const eq = std.mem.eql(u8, std.mem.sliceAsBytes(got[0]), std.mem.sliceAsBytes(got[1]));
    std.debug.print("RESULT dory L={d} graphs == eager: a {d}-row round over {d} slots, fp32 logits {s}\n", .{ x.loops, rows, cases.len, if (eq) "bit-equal" else "DIFFER" });
    return @intFromBool(!eq);
}

/// Greedy decode throughput at each stream count (copy drafts off, each stream its own 200-token prompt).
fn bench(gpu: Gpu, rest: []const []const u8) !void {
    const gpa = gpu.gpa;
    var x: Setup = undefined;
    try setup(&x, gpu, rest[0..4], 8 * 200 + 1, 8);
    defer teardown(&x, gpa);
    const counts = if (rest.len > 4) rest[4..] else &[_][]const u8{ "1", "2", "4", "8" };
    for (counts) |t| {
        const k = try std.fmt.parseInt(usize, t, 10);
        if (k == 0 or k > 8) return error.BadStreams;
        var cs: [8]Case = undefined;
        for (cs[0..k], 0..) |*c, i| c.* = .{ .prompt = x.ids[i * 200 .. (i + 1) * 200] };
        freeRun(gpa, try lanesRun(gpu, &x.db, cs[0..k], 8)); // graphs for these row counts captured
        const p = try lanesRun(gpu, &x.db, cs[0..k], 1); // the prompts and their first tokens alone
        defer freeRun(gpa, p);
        const r = try lanesRun(gpu, &x.db, cs[0..k], x.count);
        defer freeRun(gpa, r);
        const rate = @as(f64, @floatFromInt(r.tokens - p.tokens)) / (r.secs - p.secs);
        std.debug.print("RESULT dory bench L={d} streams {d}: {d:.1} tok/s decode together, {d:.1} a stream, {d:.2} ms a round ({d} tokens; prompts {d:.3} s)\n", .{ x.loops, k, rate, rate / @as(f64, @floatFromInt(k)), 1e3 * @as(f64, @floatFromInt(k)) / rate, r.tokens, p.secs });
    }
}
