//! Ripple drafted by DFlash2 on CUDA (ripple-drafted): drafted == plain at 1, 2 and 8 streams, and its admission.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");

const fwd = rc.model;
const wts = rc.weights;
const be = rc.backend;
const nat = rc.native;
const dfl = rc.dflash2;
const dw = rc.dflash2_weights;
const Stream = lanes.Stream;
const Backend = lanes.backend.Backend;
const Weight = cuda.qmmf.Weight;

const fixture = @import("ripple_forward_tests.zig"); // the release-width target on 10 layers

/// The streams' prompts: four past the tap ring's wrap, two past the sliding rings' (4,224 rows).
const prompts = [_]usize{ 4500, 37, 2100, 4300, 600, 2070, 1500, 4241 };
const slots = prompts.len;
const max_new = 40;

/// The release drafter's geometry at the fixture's vocabulary and an FFN of the target's width, its taps in 10 layers.
fn drafterConfig(vocab: u32) dfl.Config {
    var c: dfl.Config = .{ .layers = 5, .heads = 80, .kv_heads = 8, .hidden = 5120, .ffn = 12288, .vocab = vocab, .window = 4096, .block = 16, .mask = vocab - 384, .rank = 256, .top = 32, .group = 16, .conv = 4, .taps = .{ 1, 3, 5, 7, 8 }, .theta = 10000, .eps = 1e-6 };
    c.global[4] = true;
    c.rope[4] = false;
    return c;
}

fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// Device copies into the drafter's weights, as fixture.build makes the target's.
const Build = struct {
    w: *dw.Weights,
    d: *const cuda.Driver,
    r: std.Random,

    fn put(b: Build, bytes: []const u8) !u64 {
        var buf = try cuda.DeviceBuffer.fromHost(b.d, bytes);
        errdefer buf.free();
        try b.w.buffers.append(b.w.gpa, buf);
        b.w.bytes += bytes.len;
        return buf.ptr;
    }

    /// `n` bf16 values of magnitude in [2^e, 2^(e+1)), random signs and mantissas.
    fn values(b: Build, n: usize, e: i32) ![]u16 {
        const v = try b.w.gpa.alloc(u16, n);
        const exp: u16 = @intCast((e + 127) << 7);
        var bits: u64 = 0;
        for (v, 0..) |*x, i| {
            if (i % 4 == 0) bits = b.r.int(u64);
            x.* = exp | @as(u16, @truncate(bits & 0x807f));
            bits >>= 16;
        }
        return v;
    }

    fn small(b: Build, n: usize, mean: f32, sd: f32) !u64 {
        const v = try b.w.gpa.alloc(u16, n);
        defer b.w.gpa.free(v);
        for (v) |*x| x.* = bf16(mean + sd * b.r.floatNorm(f32));
        return b.put(std.mem.sliceAsBytes(v));
    }

    fn scaleOf(k: usize) i32 {
        return -@as(i32, std.math.log2_int(usize, k) / 2) - 1;
    }

    /// `src` [n, k] packed for the lane and prompt matmuls.
    fn pack(b: Build, src: []const u16, n: usize, k: usize) !Weight {
        const out = try b.w.gpa.alloc(u16, cuda.modelopt.padded(n) * k);
        defer b.w.gpa.free(out);
        cuda.qmmf.packBf16(out, src, n, k);
        return cuda.modelopt.bf16Weight(try b.put(std.mem.sliceAsBytes(out)), n, k);
    }

    fn dense(b: Build, n: usize, k: usize) !Weight {
        const src = try b.values(n * k, scaleOf(k));
        defer b.w.gpa.free(src);
        return b.pack(src, n, k);
    }

    fn conv(b: Build, c: dfl.Config) !dw.Conv {
        return .{ .base = try b.small(2 * @as(usize, c.conv) * c.hidden, 0, 0.2), .proj = try b.dense(c.dyn(), c.hidden) };
    }
};

/// Random drafter weights in B's layout: q|k|v and the absorb's k|v from the same k and v rows.
fn buildDrafter(gpu: check.Gpu, c: dfl.Config, seed: u64) !dw.Weights {
    const gpa = gpu.gpa;
    var w: dw.Weights = .{ .gpa = gpa, .config = c, .fc = undefined, .hproj = undefined, .hidden_norm = 0, .norm = 0, .layers = undefined, .pred = &.{}, .succ = &.{} };
    errdefer w.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const b: Build = .{ .w = &w, .d = gpu.d, .r = prng.random() };
    const h: usize = c.hidden;
    const qw = @as(usize, c.heads) * 128;
    const kvw = @as(usize, c.kv_heads) * 128;
    w.fc = try b.dense(h, 5 * h);
    w.hproj = try b.dense(c.rank, h);
    w.hidden_norm = try b.small(h, 1, 0.1);
    w.norm = try b.small(h, 1, 0.1);
    w.pred = try b.values(@as(usize, c.vocab) * c.rank, -4);
    w.succ = try b.values(@as(usize, c.vocab) * c.rank, -4);
    for (w.layers[0..c.layers]) |*l| {
        const qkv = try b.values((qw + 2 * kvw) * h, Build.scaleOf(h));
        defer gpa.free(qkv);
        l.* = .{
            .qkv = try b.pack(qkv, qw + 2 * kvw, h),
            .kv = try b.pack(qkv[qw * h ..], 2 * kvw, h),
            .o = try b.dense(h, qw),
            .gate_up = try b.dense(2 * @as(usize, c.ffn), h),
            .down = try b.dense(h, c.ffn),
            .attn_conv = try b.conv(c),
            .mlp_conv = try b.conv(c),
            .in_norm = try b.small(h, 1, 0.1),
            .post_norm = try b.small(h, 1, 0.1),
            .q_norm = try b.small(128, 1, 0.1),
            .k_norm = try b.small(128, 1, 0.1),
        };
    }
    return w;
}

fn sampling(j: usize) lanes.Sampling {
    return .{ .seed = 0x5eed + j, .temperature = 1.0, .top_p = 0.95, .top_k = 20 };
}

/// A run's streams' tokens and its drafts.
const Got = struct { ids: [slots][]const u32 = @splat(&.{}), drafted: u64 = 0, accepted: u64 = 0, rounds: u64 = 0 };

/// Streams `which` through one lanes.Engine as the lane host runs them: a prompt chunk between rounds, to max_new.
fn serve(gpu: check.Gpu, a: std.mem.Allocator, cfg: *const lanes.Config, backend: Backend, ids: *const [slots][]u32, which: []const usize, sampled: bool, drafts: bool) !Got {
    var wall: lanes.backend.WallClock = .{ .io = gpu.io };
    var le = lanes.Engine.init(gpu.gpa, cfg, backend, wall.clock());
    defer le.deinit();
    var ss: [slots]Stream = undefined;
    var made: usize = 0;
    defer for (ss[0..made]) |*s| s.deinit(gpu.gpa);
    for (which, 0..) |j, i| {
        ss[i] = try Stream.init(gpu.gpa, .{ .id = "drafted", .prompt = ids[j], .max_new = max_new, .drafts = drafts, .sampling = if (sampled) sampling(j) else null });
        made += 1;
    }
    var filling: usize = 0;
    var first = true;
    while (filling < made or le.activeCount() > 0) {
        if (filling < made) {
            first = if (try le.fillStream(&ss[filling], first)) blk: {
                filling += 1;
                break :blk true;
            } else false;
        }
        if (le.activeCount() > 0) try le.step();
    }
    var got: Got = .{};
    for (which, ss[0..made]) |j, *s| {
        got.ids[j] = try a.dupe(u32, s.emitted());
        got.drafted += s.drafted;
        got.accepted += s.accepted;
        got.rounds += s.rounds;
    }
    return got;
}

/// Each stream of `which` as its reference, token for token; the first difference printed.
fn same(comptime what: []const u8, args: anytype, got: *const Got, want: *const Got, which: []const usize) bool {
    var ok = true;
    for (which) |j| {
        const g = got.ids[j];
        const w = want.ids[j];
        if (g.len == max_new and std.mem.eql(u32, g, w)) continue;
        var i: usize = 0;
        while (i < @min(g.len, w.len) and g[i] == w[i]) i += 1;
        std.debug.print("  DIFF ", .{});
        std.debug.print(what, args);
        std.debug.print(": stream {d} (prompt {d}): token {d} of {d} is {d}, want {d}\n", .{ j, prompts[j], i, g.len, if (i < g.len) g[i] else 0, if (i < w.len) w[i] else 0 });
        ok = false;
    }
    return ok;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const c = fixture.config();
    try wts.gate(&c);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var w = try fixture.build(gpu, c, 0x7269_7070);
    defer w.deinit();
    var k = try fwd.Kernels.load(gpu.ctx);
    defer k.deinit();
    var m = try fwd.Model.init(gpu.gpa, gpu.d, s, &w, &k, c.context, slots, be.round_rows);
    defer m.deinit();
    var rb = try be.Ripple.init(gpu.gpa, gpu.d, &m);
    defer rb.deinit();
    var ids: [slots][]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x6472_6166);
    for (&ids, prompts) |*p, n| {
        p.* = try a.alloc(u32, n);
        for (p.*) |*t| t.* = prng.random().uintLessThan(u32, @intCast(c.vocab));
    }
    const all = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var fails: usize = 0;

    // the references: plain Ripple, no taps, every stream at once
    var plain_cfg = try lanes.Config.init(gpu.gpa, rb.facts(), be.max_window, be.max_window - 1);
    defer plain_cfg.deinit(gpu.gpa);
    const greedy = try serve(gpu, a, &plain_cfg, rb.backend(), &ids, &all, false, false);
    const sampled = try serve(gpu, a, &plain_cfg, rb.backend(), &ids, &all, true, false);

    // the drafter: its bytes past the weights (the scratch, 8 streams' rings, the tap rings setTaps makes) == runBytes
    const dc = drafterConfig(@intCast(c.vocab));
    try dfl.gate(&dc);
    const weights = try buildDrafter(gpu, dc, 0x6466_6c32);
    std.debug.print("  the drafter's weights: {d} bytes ({d} as B packs the release's layout)\n", .{ weights.bytes, dw.deviceBytes(dc) });
    const before = cuda.usage(false).device;
    const d = try dfl.Drafter.init(gpu.gpa, gpu.io, &m, weights, slots);
    defer d.deinit();
    var dd = try lanes.drafted.Drafted.init(gpu.gpa, rb.backend(), d.drafter());
    defer dd.deinit();
    const grew = cuda.usage(false).device - before;
    const want_run = dfl.runBytes(&dc, c, k.lane.clusters, slots);
    if (grew == want_run) {
        check.pass("ripple-drafted bytes: the drafter's scratch, rings and tap rings for {d} streams take {d} bytes, as admission counts them", .{ slots, grew });
    } else {
        std.debug.print("FAIL ripple-drafted bytes: the drafter took {d} bytes past its weights, admission counts {d}\n", .{ grew, want_run });
        fails += 1;
    }

    // admission: the drafter's bytes leave fewer streams, as they predict; one that leaves none is refused by name
    const plain: nat.Plan = .{ .w = &w, .k = &k, .positions = c.context };
    var drafted_plan = plain;
    drafted_plan.drafter = dc;
    const room = try drafted_plan.bytes(3) + 1;
    var predicted: usize = slots;
    while (predicted > 0 and try plain.bytes(predicted) + dfl.openBytes(&dc, c, k.lane.clusters, predicted) > room) predicted -= 1;
    const plain_n = try nat.admitted(room, slots, false, c.context, plain, null);
    const drafted_n = try nat.admitted(room, slots, false, c.context, plain, dc);
    const tight = try plain.bytes(1) + dfl.openBytes(&dc, c, k.lane.clusters, 1) - 1;
    const refused = if (nat.admitted(tight, slots, false, c.context, plain, dc)) |_| false else |e| e == error.DrafterPastMemory;
    if (drafted_n == predicted and predicted == 3 and plain_n > drafted_n and refused and nat.explain(null, error.DrafterPastMemory) != null) {
        check.pass("ripple-drafted admission: {d:.2} GiB of room holds {d} plain streams and {d} with the drafter ({d:.2} GiB at 3), as its bytes predict; {d:.2} GiB refused by name (DrafterPastMemory)", .{ gib(room), plain_n, drafted_n, gib(dfl.openBytes(&dc, c, k.lane.clusters, 3)), gib(tight) });
    } else {
        std.debug.print("FAIL ripple-drafted admission: {d} streams with the drafter, {d} predicted, {d} plain; the tight room refused by name: {}\n", .{ drafted_n, predicted, plain_n, refused });
        fails += 1;
    }

    // the target's windows timed (the server's calibration), then rounds that always offer drafts
    try d.timeWindows();
    var facts = dd.facts(rb.facts());
    facts.plain_guard = false; // a random drafter's drafts would lose to plain rounds; here every round drafts
    var cfg = try lanes.Config.init(gpu.gpa, facts, be.max_window, be.max_window - 1);
    defer cfg.deinit(gpu.gpa);

    // 1. the drafter loaded (taps on), streams without drafts: plain's tokens
    const unused = try serve(gpu, a, &cfg, dd.backend(), &ids, &all, false, false);
    if (same("loaded, unused", .{}, &unused, &greedy, &all) and unused.drafted == 0) {
        check.pass("ripple-drafted unused: {d} streams with the drafter loaded and no drafts == plain, token for token", .{slots});
    } else {
        std.debug.print("FAIL ripple-drafted unused: {d} drafts, or tokens differ\n", .{unused.drafted});
        fails += 1;
    }

    // 2. drafted == plain at 1, 2 and 8 streams, greedy and sampled
    const sets = [_][]const usize{ &.{0}, &.{ 1, 3 }, &all, &.{7}, &.{ 2, 0 }, &all };
    var drafted_all: u64 = 0;
    var accepted_all: u64 = 0;
    for (sets, 0..) |which, i| {
        const keyed = i >= 3;
        const got = try serve(gpu, a, &cfg, dd.backend(), &ids, which, keyed, true);
        const want = if (keyed) &sampled else &greedy;
        drafted_all += got.drafted;
        accepted_all += got.accepted;
        const kind = if (keyed) "sampled" else "greedy";
        if (same("{s} at {d} streams", .{ kind, which.len }, &got, want, which) and got.drafted > 0) {
            check.pass("ripple-drafted {s}, {d} stream{s}: == plain, token for token; {d} drafts offered, {d} accepted in {d} rounds", .{ kind, which.len, if (which.len == 1) "" else "s", got.drafted, got.accepted, got.rounds });
        } else {
            std.debug.print("FAIL ripple-drafted {s}, {d} streams: {d} drafts offered, or tokens differ\n", .{ kind, which.len, got.drafted });
            fails += 1;
        }
    }
    // a verify kept part of its window (drafts rejected past the pending row's)
    if (drafted_all > accepted_all) {
        check.pass("ripple-drafted accept count: {d} of {d} drafts accepted ({d:.1}%; a random drafter's, a report not a gate), every rejecting verify keeping part of its window", .{ accepted_all, drafted_all, 100 * @as(f64, @floatFromInt(accepted_all)) / @as(f64, @floatFromInt(@max(drafted_all, 1))) });
    } else {
        std.debug.print("FAIL ripple-drafted: no verify rejected a draft ({d} of {d} accepted)\n", .{ accepted_all, drafted_all });
        fails += 1;
    }
    try check.expect(fails == 0, "ripple-drafted: {d} checks differ", .{fails});
}
