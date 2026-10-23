//! `tf-dsv41 decode-tp PACK REF OUT`: one process per rank, against that world's reference tokens and digests.

const std = @import("std");
const pf = @import("forward_prefill.zig");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const tp = @import("tp");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const gate_lanes = @import("gate_lanes.zig");
const drive = @import("draft/drive.zig");
const tree = @import("draft/tree.zig");
const branches_oracle = @import("draft/branches_oracle.zig");
const dspark_emit = @import("dspark_emit.zig");
const dspark_gpu = @import("dspark_gpu.zig");

const rows_lanes = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
const named = @import("named.zig");
const load = @import("cuda_weights.zig");
const buffers = @import("buffers.zig");
const forms = @import("forms.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const gate_forward = @import("gate_forward.zig");
const pk = @import("serve_knobs.zig");
const Value = std.json.Value;

fn sha(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

pub fn main(gpa: std.mem.Allocator, io: std.Io, pack_dir: []const u8, ref: []const u8, out_dir: []const u8) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const tcfg = try tp.Config.fromEnv();
    const rank = tcfg.rank;
    const world = tcfg.world;
    try cwd.createDirPath(io, out_dir);
    const meta = try std.json.parseFromSliceLeaky(Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ ref, "ref.json" }), a, .limited(1 << 28)), .{});
    var lo: u32 = 0;
    var hi: u32 = 0;
    {
        var it = std.mem.tokenizeScalar(u8, meta.object.get("layers").?.string, '-');
        lo = try std.fmt.parseInt(u32, it.next().?, 10);
        hi = try std.fmt.parseInt(u32, it.next().?, 10);
    }
    const layers = try a.alloc(u32, hi - lo + 1);
    for (layers, 0..) |*l, i| l.* = lo + @as(u32, @intCast(i));
    // TF_DSV41_DECODE_TP_LANES (lanes checks, below): 1 drafts off, 2 the oracle's drafts, 3 DSpark's GPU pass
    const lanes_mode: u32 = if (std.c.getenv("TF_DSV41_DECODE_TP_LANES")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 0;
    // TF_DSV41_DECODE_TP_PREFILL=1: our prefill runs the reference's prompt from an empty slot; state equals its
    const own_prefill = if (std.c.getenv("TF_DSV41_DECODE_TP_PREFILL")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;

    var driver = try cuda.Driver.open();
    defer driver.close();
    try @import("dev_arena.zig").applySchedule(&driver, @intCast(tcfg.device)); // TF_CUDA_SCHED
    var ctx = try cuda.Context.init(&driver, @intCast(tcfg.device));
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var kernels = try dk.Kernels.load(&ctx);
    defer kernels.deinit();
    const session = try gpa.create(tp.Session);
    defer gpa.destroy(session);
    // TF_TP_KERNEL_IMAGE: tp_mailbox.fatbin; TF_COMM_BACKEND=roce then takes small exchanges over RoCE; unset: NCCL
    const image: []const u8 = if (std.c.getenv("TF_TP_KERNEL_IMAGE")) |p|
        try cwd.readFileAllocOptions(io, std.mem.span(p), a, .limited(1 << 26), .@"16", null)
    else
        &.{};
    try session.open(&driver, tcfg, image);
    defer session.close();
    // TF_DSV41_ARENA: every device buffer from here on (weights, forms, roles, scratch) carved from large chunks
    _ = try @import("dev_arena.zig").install(gpa, &ctx, rank);
    const comm = session.comm();

    const t0 = std.Io.Clock.awake.now(io);
    const cfg = try Config.read(gpa, io, pack_dir);
    var pack = try Pack.open(gpa, io, pack_dir);
    defer pack.deinit();
    const blocks = if (lanes_mode == 3) blk: {
        var ds: [8]u32 = undefined;
        break :blk try std.mem.concat(a, u32, &.{ layers, dspark_emit.blocks(&cfg, &ds) });
    } else layers;
    var pl = try plan.build(gpa, &cfg, &pack, .{ .rank = rank, .world = world, .blocks = blocks, .top = true, .dspark = lanes_mode == 3 });
    defer pl.deinit();
    var w: load.Weights = .{ .gpa = gpa };
    defer @import("close_or_exit.zig").finish(&w);
    for (pl.layers) |*l| {
        var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &pack };
        defer b.deinit();
        try b.layer(l, cfg.o_groups, rank, world);
        try w.upload(&driver, &b);
    }
    {
        var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &pack };
        defer b.deinit();
        try b.top(&pl);
        try b.dspark(&pl);
        try w.upload(&driver, &b);
    }
    var aot = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, try std.fs.path.join(a, &.{ ref, "aot" }));
    defer aot.deinit();
    const wd = try gate_forward.widths(a, &w, &pl);
    var runner: run.Runner = .{ .gpa = gpa, .d = &driver, .stream = stream, .kernels = &kernels, .triton = &aot, .weights = &w };
    defer runner.deinit();
    // TF_DSV41_PROFILE=<file>: every eager decode window's calls timed (profile.zig), written at the end
    runner.prof = if (try @import("profile.zig").Profile.fromEnv(gpa, &driver)) |p| p.timer() else null;
    defer if (runner.prof) |p| p.close(io);
    // TF_DSV41_DECODE_TP_TOPP (the reference's TF_DSV41_EXPERT_TOPP): unset = no top-p prune, else prune at that mass
    const topp: ?f64 = if (std.c.getenv("TF_DSV41_DECODE_TP_TOPP")) |v| try std.fmt.parseFloat(f64, std.mem.span(v)) else null;
    // R1 when the reference ran its knobs (ref.json's env)
    const r1 = blk: {
        const env = meta.object.get("env") orelse break :blk false;
        const v = env.object.get("TF_DSV41_MHC_DEFER") orelse break :blk false;
        break :blk v == .string and std.mem.eql(u8, v.string, "1");
    };
    var f = fwd.Forward.init(gpa, &cfg, &wd, .{ .world = world, .rank = rank, .expert_topp = topp, .r1 = r1 }, &runner, comm);
    defer @import("close_or_exit.zig").finish(&f);
    try @import("engine_hooks.zig").open(&f);
    f.layers = layers;
    // the served knobs the emitters do not fix (TF_DSV41_PF_DENSE, _PREFILL_CHUNK / _ROWS, ...)
    try @import("serve_knobs_apply.zig").apply(&f, a, io, rank);
    // the reference's slot limit (its rope.bin has limit + 2,048 rows): long prompts
    if (meta.object.get("limit")) |l| {
        f.opts.limit = l.integer;
        f.opts.rope_rows = l.integer + 2048;
    }
    var bp = blk: {
        // the plan at both trace settings (the traced program adds one role)
        f.opts.trace = true;
        // plan every row count the windows run (drafts, TF_DSV41_DECODE_TP_PROF_ROWS): decode-only roles that deep
        var rows: std.ArrayList(u32) = .empty;
        try rows.appendSlice(a, if (lanes_mode != 0) rows_lanes[0 .. cfg.dspark_block + 1] else &.{1});
        if (std.c.getenv("TF_DSV41_DECODE_TP_PROF_ROWS")) |spec| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(spec), ' ');
            while (it.next()) |word| {
                const n = try std.fmt.parseInt(u32, word, 10);
                if (std.mem.indexOfScalar(u32, rows.items, n) == null) try rows.append(a, n);
            }
        }
        var p = try f.plan(rows.items);
        if (lanes_mode == 3) try dspark_gpu.GpuPass.plan(f.arena.allocator(), &f, &p, cfg.dspark_block + 1);
        if (own_prefill) try f.planPrefill(&p, @intCast(f.opts.prefill_rows));
        f.programs.clearRetainingCapacity();
        f.opts.trace = false;
        try p.add(try f.program(1));
        break :blk p;
    };
    try runner.bind(&bp);
    const src: forms.Source = .{ .d = &driver, .weights = &w, .plan = &pl };
    for (bp.sizes.keys()) |role| {
        if (buffers.scopeOf(role) != .persistent) continue;
        var ra = std.heap.ArenaAllocator.init(gpa);
        defer ra.deinit();
        const bytes = (try forms.build(ra.allocator(), &src, role)) orelse continue;
        try cuda.DeviceBuffer.upload(.{ .d = &driver, .ptr = runner.addressOf(role).?, .len = bytes.len }, 0, bytes);
    }
    // TF_DSV41_L2PF as the served model installs it (l2pf.zig), so the check covers the prefetch side stream too
    const l2 = try @import("l2pf.zig").install(gpa, &f, &w);
    defer if (l2) |p| {
        runner.l2pf = null;
        p.deinit();
        gpa.destroy(p);
    };
    const rope = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ ref, "rope.bin" }), a, .limited(1 << 28));
    const half = (rope.len - 16) / 2;
    for ([_][]const u8{ "s.rope.main", "s.rope.comp" }, 0..) |role, i| if (runner.addressOf(role)) |p|
        try cuda.DeviceBuffer.upload(.{ .d = &driver, .ptr = p, .len = half }, 0, rope[16 + i * half ..][0..half]);
    const ehb = try cwd.readFileAllocOptions(io, try std.fs.path.join(a, &.{ ref, "engram-token-map.bin" }), a, .limited(1 << 26), .@"4", null);
    var lbuf: [eh.max_layers]u32 = undefined;
    const host = try eh.Host.load(ehb, cfg.engram_heads, cfg.engram_pad, cfg.engram_vocab, &lbuf);
    f.engram = &host;
    // this rank's slot after the reference's prompt
    const sdir = try std.fmt.allocPrint(a, "{s}/rank{d}/state", .{ ref, rank });
    const st = try std.json.parseFromSliceLeaky(Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ sdir, "state.json" }), a, .limited(1 << 24)), .{});
    var prefill_report: ?PrefillReport = null;
    if (own_prefill) {
        prefill_report = try ownPrefill(gpa, io, a, &f, &driver, &runner, meta, st, sdir, rank);
    } else {
        var sit = st.object.get("roles").?.object.iterator();
        while (sit.next()) |e| {
            const bytes = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ sdir, e.value_ptr.string }), a, .limited(1 << 32));
            if (bp.sizes.get(e.key_ptr.*) == null or bp.sizes.get(e.key_ptr.*).? < bytes.len) return error.StateRole;
            try cuda.DeviceBuffer.upload(.{ .d = &driver, .ptr = runner.addressOf(e.key_ptr.*).?, .len = bytes.len }, 0, bytes);
        }
        f.slot.pos = @intCast(st.object.get("pos").?.integer);
        const tail = st.object.get("tail").?.array.items;
        for (tail, 0..) |t, i| f.slot.tail[i] = @intCast(t.integer);
        f.slot.tail_len = tail.len;
    }
    try driver.synchronize();
    try comm.barrier();
    const load_s = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
    std.debug.print("rank {d}: {d} layers, {d:.2} GB, load {d:.1} s, position {d}\n", .{ rank, layers.len, @as(f64, @floatFromInt(w.bytes)) / 1e9, load_s, f.slot.pos });

    const want_tokens = meta.object.get("tokens").?.array.items;
    // TF_DSV41_DECODE_TP_LANES: the decode via the lanes engine on rank 0 (others follow), tokens only (gate_lanes)
    if (lanes_mode != 0) {
        const mode = lanes_mode;
        f.link = &session.plan;
        // every rank's DSpark pass (followers run it through f.drafter)
        const gp: ?*dspark_gpu.GpuPass = if (mode == 3) try dspark_gpu.GpuPass.init(gpa, &f, cfg.dspark_block + 1) else null;
        defer if (gp) |g| g.deinit();
        const tl0 = std.Io.Clock.awake.now(io);
        const want = try a.alloc(u32, want_tokens.len);
        for (want, want_tokens) |*d, v| d.* = @intCast(v.integer);
        var r: gate_lanes.Result = .{ .tokens = &.{}, .drafted = 0, .accepted = 0 };
        var equal: usize = 0;
        const trees = try tree.Settings.fromEnv();
        const tree_on = mode == 4 or (mode == 3 and trees.siblings > 0);
        var tr: gate_lanes.TreeResult = .{ .r = .{ .tokens = &.{}, .drafted = 0, .accepted = 0, .trees = 0, .sib_wins = 0, .rounds = 0 }, .chains = .{} };
        if (rank == 0) {
            const pids = meta.object.get("prompt_ids").?.array.items;
            const prompt = try a.alloc(u32, pids.len);
            for (prompt, pids) |*d, v| d.* = @intCast(v.integer);
            var nd: drive.NoDraft = .{};
            var oracle: drive.Oracle = .{ .serial = want, .prompt_len = prompt.len, .vocab = cfg.vocab };
            const pass = if (gp) |g| g.pass() else if (mode == 2) oracle.pass() else nd.pass();
            if (tree_on) {
                var to = try branches_oracle.TreeOracle.init(gpa, want, prompt.len, cfg.vocab);
                defer to.deinit(gpa);
                const set = if (mode == 4 and trees.siblings == 0) tree.Settings{ .siblings = 2 } else trees;
                tr = try gate_lanes.generateTrees(gpa, &f, if (gp) |g| g.pass() else to.pass(), prompt, want[0], @intCast(want.len), cfg.dspark_block + 1, set);
                r = .{ .tokens = tr.r.tokens, .drafted = tr.r.drafted, .accepted = tr.r.accepted, .off_anchor = to.off_anchor };
            } else {
                r = try gate_lanes.generate(gpa, &f, pass, mode >= 2, prompt, want[0], @intCast(want.len), cfg.dspark_block + 1);
                r.off_anchor = oracle.off_anchor;
            }
            try f.stop();
            for (r.tokens, want[0..@min(r.tokens.len, want.len)]) |g, wt| {
                if (g != wt) break;
                equal += 1;
            }
        } else try f.follow();
        defer if (rank == 0) gpa.free(r.tokens);
        const lanes_s = @as(f64, @floatFromInt(tl0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
        // the oracle must see accepts and rejects; DSpark's pass must at least draft
        const kept_some = switch (mode) {
            2 => r.accepted > 0 and r.accepted < r.drafted,
            3 => r.drafted > 0,
            // the tree oracle: siblings verified, some won, the serial anchors throughout
            4 => r.accepted > 0 and tr.r.trees > 0 and tr.r.sib_wins > 0 and tr.r.sib_wins < tr.r.trees and r.off_anchor == 0,
            else => true,
        };
        const ok = rank != 0 or (r.tokens.len == want.len and equal == want.len and kept_some);
        try cwd.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/decode-tp-lanes-rank{d}.json", .{ out_dir, rank }), .data = try std.fmt.allocPrint(a, "{{\"rank\": {d}, \"mode\": {d}, \"tokens\": {d}, \"want\": {d}, \"equal_prefix\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"off_anchor\": {d}, \"trees\": {d}, \"sib_wins\": {d}, \"rounds\": {d}, \"chains\": {d}, \"rerun_rows\": {d}, \"s\": {d:.1}, \"pass\": {}}}\n", .{ rank, mode, r.tokens.len, want.len, equal, r.drafted, r.accepted, r.off_anchor, tr.r.trees, tr.r.sib_wins, tr.r.rounds, tr.chains.chains, tr.chains.rerun_rows, lanes_s, ok }) });
        std.debug.print("{s} lanes mode {d} rank {d}: tokens {d}/{d} equal, drafted {d}, accepted {d}, trees {d}, sibling wins {d}, rounds {d}, {d:.1} s\n", .{ if (ok) "PASS" else "FAIL", mode, rank, equal, want.len, r.drafted, r.accepted, tr.r.trees, tr.r.sib_wins, tr.r.rounds, lanes_s });
        return if (ok) 0 else 1;
    }
    // greedy decoding against the reference's tokens
    const mine = meta.object.get("ranks").?.array.items[rank].object;
    const want_logits = mine.get("logits_sha256").?.array.items;
    const V: usize = (try @import("partition.zig").share(cfg.vocab, rank, world)).len;
    const host_logits = try a.alloc(u8, 4 * V);
    var trace: std.AutoArrayHashMapUnmanaged(u32, [64]u8) = .empty;
    defer trace.deinit(gpa);
    var tok: u32 = @intCast(want_tokens[0].integer);
    var log: std.Io.Writer.Allocating = .init(gpa);
    defer log.deinit();
    var tokens_equal: usize = 0;
    var logits_equal: usize = 0;
    var layers_equal: usize = 0;
    var first_layer: ?u32 = null;
    const steps = want_tokens.len - 1;
    const tw0 = std.Io.Clock.awake.now(io);
    for (0..steps) |step| {
        f.opts.trace = step == 0;
        f.trace = if (step == 0) &trace else null;
        const ids = [_]u32{tok};
        try f.window(&ids);
        try stream.synchronize();
        if (step == 0) {
            var it = mine.get("layers_first_window").?.object.iterator();
            while (it.next()) |e| {
                const L = try std.fmt.parseInt(u32, e.key_ptr.*, 10);
                const got = trace.get(L) orelse return error.TraceMissing;
                if (std.mem.eql(u8, &got, e.value_ptr.string)) layers_equal += 1 else if (first_layer == null) {
                    first_layer = L;
                }
            }
        }
        try cuda.DeviceBuffer.download(.{ .d = &driver, .ptr = runner.addressOf("w.logits").?, .len = host_logits.len }, 0, host_logits);
        const ld = sha(host_logits);
        const lok = std.mem.eql(u8, &ld, want_logits[step].string);
        logits_equal += @intFromBool(lok);
        var choice: [1]u32 = undefined;
        try f.greedy(&choice);
        tok = choice[0];
        const want: u32 = @intCast(want_tokens[step + 1].integer);
        const tok_ok = tok == want;
        tokens_equal += @intFromBool(tok_ok);
        try log.writer.print("{{\"step\": {d}, \"token\": {d}, \"want\": {d}, \"logits_equal\": {}}}\n", .{ step, tok, want, lok });
        try f.keep(0);
        if (!tok_ok) break; // the sequences have diverged: later steps compare nothing
    }
    const dec_s = @as(f64, @floatFromInt(tw0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
    f.trace = null;
    // the decode steps' tallies before anything else runs (a fault in the timed rows below still leaves them)
    std.debug.print("rank {d}: decode steps: tokens {d}/{d}, logits {d}/{d}\n", .{ rank, tokens_equal, steps, logits_equal, steps });
    // TF_DSV41_DECODE_TP_PROF_ROWS ("1 4 8"): after the check, each row count's windows timed (3 warm + 16, synced)
    if (std.c.getenv("TF_DSV41_DECODE_TP_PROF_ROWS")) |spec| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(spec), ' ');
        while (it.next()) |word| {
            const n = try std.fmt.parseInt(u32, word, 10);
            const ids = try a.alloc(u32, n);
            @memset(ids, tok);
            var best: f64 = std.math.floatMax(f64);
            var sum: f64 = 0;
            for (0..19) |rep_i| {
                const tr = std.Io.Clock.awake.now(io);
                try f.window(ids);
                try stream.synchronize();
                const ms = @as(f64, @floatFromInt(tr.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
                try f.keep(n - 1);
                if (rep_i < 3) continue;
                best = @min(best, ms);
                sum += ms;
            }
            try log.writer.print("{{\"prof_rows\": {d}, \"graphs\": {}, \"ms_mean\": {d:.3}, \"ms_min\": {d:.3}}}\n", .{ n, f.graphed != null and f.graphed.?.settings.on, sum / 16.0, best });
            std.debug.print("rank {d}: {d}-row windows {d:.3} ms mean, {d:.3} min\n", .{ rank, n, sum / 16.0, best });
        }
    }
    const nl = mine.get("layers_first_window").?.object.count();
    const pre_ok = if (prefill_report) |pr| pr.ok() else true;
    if (prefill_report) |pr| try log.writer.print("{{\"prefill\": {{\"rows\": {d}, \"seconds\": {d:.2}, \"roles_equal\": {d}, \"roles\": {d}, \"first_differing_role\": \"{s}\", \"pos_equal\": {}, \"tail_equal\": {}, \"first_token\": {d}, \"want\": {d}}}}}\n", .{ pr.rows, pr.seconds, pr.equal, pr.roles, pr.first_bad, pr.pos_ok, pr.tail_ok, pr.first, pr.want_first });
    const ok = tokens_equal == steps and logits_equal == steps and layers_equal == nl and pre_ok;
    try log.writer.print("{{\"rank\": {d}, \"steps\": {d}, \"tokens_equal\": {d}, \"logits_equal\": {d}, \"layers_equal\": {d}, \"layers\": {d}, \"first_differing_layer\": {?d}, \"load_s\": {d:.1}, \"decode_s\": {d:.1}, \"pass\": {}}}\n", .{ rank, steps, tokens_equal, logits_equal, layers_equal, nl, first_layer, load_s, dec_s, ok });
    try cwd.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/decode-tp-rank{d}.jsonl", .{ out_dir, rank }), .data = log.written() });
    if (prefill_report) |pr| std.debug.print("{s} Zig prefill rank {d}: {d} rows in {d:.2} s, state roles {d}/{d} equal (first differing: {s}), position {}, Engram tail {}, first token {d} vs {d}\n", .{ if (pr.ok()) "PASS" else "FAIL", rank, pr.rows, pr.seconds, pr.equal, pr.roles, pr.first_bad, pr.pos_ok, pr.tail_ok, pr.first, pr.want_first });
    std.debug.print("{s} decode TP rank {d}: tokens {d}/{d}, logits {d}/{d}, first window blocks {d}/{d} (first differing {?d}), decode {d:.1} s\n", .{ if (ok) "PASS" else "FAIL", rank, tokens_equal, steps, logits_equal, steps, layers_equal, nl, first_layer, dec_s });
    return if (ok) 0 else 1;
}

const PrefillReport = struct {
    rows: usize,
    seconds: f64,
    roles: usize,
    equal: usize,
    first_bad: []const u8,
    pos_ok: bool,
    tail_ok: bool,
    first: u32,
    want_first: u32,

    fn ok(p: PrefillReport) bool {
        return p.equal == p.roles and p.pos_ok and p.tail_ok and p.first == p.want_first;
    }
};

/// TF_DSV41_DECODE_TP_PREFILL: the reference's prompt through our prefill, then the slot and first token vs its dump.
fn ownPrefill(gpa: std.mem.Allocator, io: std.Io, a: std.mem.Allocator, f: *fwd.Forward, driver: *cuda.Driver, runner: *run.Runner, meta: Value, st: Value, sdir: []const u8, rank: u32) !PrefillReport {
    _ = rank;
    const cwd = std.Io.Dir.cwd();
    const pids = meta.object.get("prompt_ids").?.array.items;
    const prompt = try a.alloc(u32, pids.len);
    for (prompt, pids) |*d, v| d.* = @intCast(v.integer);
    const t0 = std.Io.Clock.awake.now(io);
    // the reference's prompt shape (ref.json "prompt_tail"; older references: prefill)
    const shape: pk.PromptTail = if (meta.object.get("prompt_tail")) |v| std.meta.stringToEnum(pk.PromptTail, v.string) orelse return error.BadPromptTail else .prefill;
    const sp = pk.promptSplit(prompt.len, shape);
    // the reference's prefill mode (ref.json "prefill"; older: full); replay needs TF_DSV41_PREFILL=replay here too
    const mode: pk.PrefillMode = if (meta.object.get("prefill")) |v| std.meta.stringToEnum(pk.PrefillMode, v.string) orelse return error.BadPrefillMode else .full;
    if (mode != pf.state(f).mode or (mode == .replay and shape != .verify)) {
        std.log.err("TF_DSV41_DECODE_TP_PREFILL: the reference prefilled in {t} mode (prompt tail {t}), these ranks run TF_DSV41_PREFILL={t}; replay needs both and the verify tail", .{ mode, shape, pf.state(f).mode });
        return error.PrefillMode;
    }
    try f.prefill(prompt[0..sp.prefill]);
    // CED: the decoder replay over the prompt's tail before the verify row (full mode: nothing)
    try f.finishPrompt();
    if (sp.window > 0) try f.window(prompt[sp.prefill..]);
    // the prompt's choice: the window's row, or the last chunk's last row
    const seg: usize = @intCast(f.opts.prefill_rows);
    const last: usize = if (sp.window > 0) sp.window else if (prompt.len % seg == 0) seg else prompt.len % seg;
    const picks = try a.alloc(u32, last);
    try f.greedy(picks);
    if (sp.window > 0) try f.keep(0);
    try driver.synchronize();
    const secs = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
    var rep: PrefillReport = .{ .rows = prompt.len, .seconds = secs, .roles = 0, .equal = 0, .first_bad = "", .pos_ok = false, .tail_ok = false, .first = picks[last - 1], .want_first = @intCast(meta.object.get("tokens").?.array.items[0].integer) };
    var it = st.object.get("roles").?.object.iterator();
    while (it.next()) |e| {
        const want = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ sdir, e.value_ptr.string }), a, .limited(1 << 32));
        const got = try gpa.alloc(u8, want.len);
        defer gpa.free(got);
        const ptr = runner.addressOf(e.key_ptr.*) orelse return error.StateRole;
        try cuda.DeviceBuffer.download(.{ .d = driver, .ptr = ptr, .len = got.len }, 0, got);
        rep.roles += 1;
        if (std.mem.eql(u8, got, want)) rep.equal += 1 else if (rep.first_bad.len == 0) rep.first_bad = e.key_ptr.*;
    }
    rep.pos_ok = f.slot.pos == @as(u64, @intCast(st.object.get("pos").?.integer));
    const tail = st.object.get("tail").?.array.items;
    rep.tail_ok = f.slot.tail_len == tail.len;
    for (tail, 0..) |t, i| if (i < f.slot.tail_len and f.slot.tail[i] != @as(u32, @intCast(t.integer))) {
        rep.tail_ok = false;
    };
    return rep;
}
