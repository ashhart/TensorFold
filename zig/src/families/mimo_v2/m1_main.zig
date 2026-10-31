//! tf-mimo: rank 1 behind the server (follow) and the decode forward against an answer key (m1).

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tp = @import("tp");
const mimo = @import("mimo_v2");

const prompts = [_][]const u8{ "en_story", "zh_explain", "code_py", "math", "tool_json", "edit", "list", "short" };

const Cmp = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    ora: *const core.safetensors.File,
    name: []const u8,
    step: usize,
    rank: usize,
    first_fail: ?[]const u8 = null,
    checked: usize = 0,
    host: []u8,
};

fn hook(ctx: *anyopaque, what: mimo.forward.Model.Tap, layer: usize, dev: u64, bytes: usize) anyerror!void {
    const c: *Cmp = @ptrCast(@alignCast(ctx));
    var nb: [128]u8 = undefined;
    const key = switch (what) {
        .attn => try std.fmt.bufPrint(&nb, "{s}.s{d}.attn.{d}", .{ c.name, c.step, layer }),
        .mlp => try std.fmt.bufPrint(&nb, "{s}.s{d}.res.{d}", .{ c.name, c.step, layer }),
        .att => try std.fmt.bufPrint(&nb, "{s}.s{d}.att.{d}", .{ c.name, c.step, layer }),
    };
    const want = c.ora.get(key) orelse return; // attention outputs exist only at two layers
    try c.s.synchronize();
    if (want.bytes.len != bytes) {
        std.debug.print("FAIL {s}: {d} bytes, oracle {d}\n", .{ key, bytes, want.bytes.len });
        return error.Shape;
    }
    try c.d.check(c.d.api.cuMemcpyDtoH_v2(@ptrCast(c.host.ptr), dev, bytes), "capture");
    c.checked += 1;
    if (std.mem.eql(u8, c.host[0..bytes], want.bytes)) return;
    if (c.first_fail == null) {
        c.first_fail = try c.gpa.dupe(u8, key);
        report(key, c.host[0..bytes], want.bytes, what == .att);
    }
}

fn report(key: []const u8, got: []const u8, want: []const u8, half: bool) void {
    var diff: usize = 0;
    var worst: f64 = 0;
    var first: ?usize = null;
    const n = if (half) got.len / 2 else got.len / 4;
    for (0..n) |i| {
        const g: f64, const w: f64 = if (half) .{ @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, got[2 * i ..][0..2], .little)))), @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, want[2 * i ..][0..2], .little)))) } else .{ @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little)))), @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, want[4 * i ..][0..4], .little)))) };
        if (g != w) {
            diff += 1;
            if (first == null) first = i;
            worst = @max(worst, @abs(g - w) / @max(@abs(w), 1e-6));
        }
    }
    std.debug.print("FIRST DIFF {s}: {d} of {d} values differ (first at {?d}), worst relative {e}\n", .{ key, diff, n, first, worst });
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const cmd: []const u8 = if (args.len >= 2) args[1] else "";
    const is_m1 = args.len >= 4 and std.mem.eql(u8, cmd, "m1");
    const is_follow = args.len >= 3 and std.mem.eql(u8, cmd, "follow");
    if (!is_m1 and !is_follow) {
        std.debug.print("usage: tf-mimo m1 MODEL ORACLE(rank<R>.safetensors) [CONTEXT [DUMP_OUT [DUMP_BASE]]]\n" ++
            "       tf-mimo follow MODEL   (TP rank 1 behind tensorfold-native serve on rank 0)\n", .{});
        return 2;
    }
    const gpa = init.gpa;
    const io = init.io;
    const context: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 4096;
    const cfg = try tp.Config.fromEnv();
    if (is_follow and (cfg.world < 2 or cfg.rank == 0)) return error.FollowNeedsRankOne; // world 1 opens no session
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(cfg.device));
    defer ctx.deinit();
    if (try ctx.capability() != 121) return error.MimoV2RequiresSm121; // as native.open: GB10 only
    var session: tp.Session = undefined;
    const world: usize = cfg.world;
    if (world > 1) try session.open(&driver, cfg, @import("tp_kernel").mailbox);
    defer if (world > 1) {
        session.announceStop(); // a clean stop: the peer exits 0 instead of failing fast (70) when this rank ends first
        session.close();
    };
    const rank: usize = cfg.rank;
    if (is_follow) return follow(gpa, io, &driver, &session, args[2]);
    var s = try cuda.Stream.init(&driver, false);
    defer s.deinit();

    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var w = try mimo.load.load(gpa, io, &driver, args[2], rank, world, rank == 0);
    defer w.deinit();
    var k3 = try mimo.exl3.Kernels.load(&driver);
    defer k3.deinit();
    var ops = try mimo.ops.Ops.load(&driver);
    defer ops.unload();
    const comm = if (world > 1) session.comm() else null;
    var model = try mimo.forward.Model.init(gpa, &driver, &w, &ops, &k3, comm, context, 1, 1, s);
    defer model.deinit();
    std.debug.print("[m1] rank {d}: loaded in {d:.0}s ({d:.1} GiB weights)\n", .{ rank, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9, @as(f64, @floatFromInt(w.arena.total)) / (1 << 30) });

    var ora = try core.safetensors.File.open(gpa, io, args[3]);
    defer ora.close(io);
    const V = w.full.vocab;
    const logits = try gpa.alloc(f32, V);
    defer gpa.free(logits);
    const host = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(host);
    var cmp: Cmp = .{ .gpa = gpa, .d = &driver, .s = s, .ora = &ora, .name = "", .step = 0, .rank = rank, .host = host };
    var total_steps: usize = 0;
    var argmax_ok: usize = 0;
    var logits_ok: usize = 0;
    var logits_n: usize = 0;
    var kl_sum: f64 = 0;
    var kl_max: f64 = 0;
    // optional: every step's fp32 logits written to DUMP_OUT, and compared with DUMP_BASE's (another build's dump)
    const cwd = std.Io.Dir.cwd();
    const dump = if (args.len > 5) try cwd.createFile(io, args[5], .{}) else null;
    defer if (dump) |f| f.close(io);
    const base = if (args.len > 6) try cwd.openFile(io, args[6], .{}) else null;
    defer if (base) |f| f.close(io);
    const base_row = try gpa.alloc(f32, V);
    defer gpa.free(base_row);
    var at: u64 = 0;
    var base_kl: f64 = 0;
    var base_max: f64 = 0;
    var base_am: usize = 0;
    var all_ok = true;
    var nb: [128]u8 = undefined;
    // the prompt path over each prompt's whole sequence (one chunk, then two): its last row against the decode step's
    const pr = try mimo.prompt.Prompt.init(&model, gpa, 512, s);
    defer pr.deinit();
    const plog = try gpa.alloc(f32, V);
    defer gpa.free(plog);
    const whole = try gpa.alloc(f32, V);
    defer gpa.free(whole);
    var prompt_eq: usize = 0;
    var prompt_n: usize = 0;
    var chunk_eq: usize = 0;
    var prompt_kl: f64 = 0;
    for (prompts) |name| {
        const ids_t = ora.get(try std.fmt.bufPrint(&nb, "{s}.ids", .{name})) orelse return error.NoIds;
        const greedy_t = ora.get(try std.fmt.bufPrint(&nb, "{s}.greedy", .{name})) orelse return error.NoGreedy;
        const am_t = ora.get(try std.fmt.bufPrint(&nb, "{s}.argmax", .{name})) orelse return error.NoArgmax;
        const lg_t = ora.get(try std.fmt.bufPrint(&nb, "{s}.logits", .{name})) orelse return error.NoLogits;
        const ids = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, ids_t.bytes));
        const greedy = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, greedy_t.bytes));
        const am = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, am_t.bytes));
        // the sequence the capture fed: the prompt, then each greedy token that was not EOS
        var seq: std.ArrayList(i64) = .empty;
        defer seq.deinit(gpa);
        for (ids) |t| try seq.append(gpa, t);
        for (greedy) |t| if (!w.full.isEos(@intCast(t))) try seq.append(gpa, t);
        var ok_here: usize = 0;
        var first_bad: ?usize = null;
        cmp.name = name;
        const nsteps = am.len;
        for (0..nsteps) |step| {
            cmp.step = step;
            const deep = step < 4;
            try model.forward(s, &.{.{ .slot = 0, .pos = @intCast(step), .token = seq.items[step] }}, logits, if (deep) .{ .hook = &hook, .ctx = &cmp } else null);
            const got = mimo.forward.argmax(logits);
            if (dump) |f| try f.writePositionalAll(io, std.mem.sliceAsBytes(logits), at);
            if (base) |f| {
                if (try f.readPositionalAll(io, std.mem.sliceAsBytes(base_row), at) != V * 4) return error.ShortBase;
                const kl = klDiv(base_row, logits);
                base_kl += kl;
                base_max = @max(base_max, kl);
                if (mimo.forward.argmax(base_row) == got) base_am += 1;
            }
            at += V * 4;
            if (got == am[step]) ok_here += 1 else if (first_bad == null) first_bad = step;
            if (step < lg_t.shape[0]) {
                logits_n += 1;
                const want = lg_t.bytes[step * V * 4 ..][0 .. V * 4];
                const kl = klDiv(std.mem.bytesAsSlice(f32, @as([]align(1) const u8, want)), logits);
                kl_sum += kl;
                kl_max = @max(kl_max, kl);
                if (std.mem.eql(u8, std.mem.sliceAsBytes(logits), want)) logits_ok += 1 else if (cmp.first_fail == null) {
                    cmp.first_fail = try std.fmt.allocPrint(gpa, "{s}.logits[{d}]", .{ name, step });
                    report(cmp.first_fail.?, std.mem.sliceAsBytes(logits), want, false);
                }
            }
        }
        for ([_]usize{ nsteps, nsteps / 2 }) |cut| {
            try pr.chunk(s, 0, 0, seq.items[0..cut], null);
            if (cut < nsteps) try pr.chunk(s, 0, cut, seq.items[cut..nsteps], null);
            try model.logitsTo(s, 1, plog);
            prompt_n += 1;
            prompt_kl += klDiv(logits, plog);
            if (std.mem.eql(f32, plog, logits)) prompt_eq += 1;
            if (cut == nsteps) @memcpy(whole, plog) else if (std.mem.eql(f32, plog, whole)) chunk_eq += 1 else std.debug.print("[m1] rank {d} {s}: two chunks differ from one\n", .{ rank, name });
        }
        total_steps += nsteps;
        argmax_ok += ok_here;
        if (first_bad != null) all_ok = false;
        std.debug.print("[m1] rank {d} {s}: {d} steps, argmax equal at {d}{s}\n", .{ rank, name, nsteps, ok_here, if (first_bad) |b| try std.fmt.bufPrint(&nb, " (first difference at step {d})", .{b}) else "" });
    }
    const exact = all_ok and cmp.first_fail == null and logits_ok == logits_n;
    if (base != null) std.debug.print("[m1] rank {d}: against the base dump over {d} steps: argmax equal {d}, KL mean {e:.3}, max {e:.3}\n", .{ rank, total_steps, base_am, base_kl / @as(f64, @floatFromInt(@max(total_steps, 1))), base_max });
    std.debug.print("[m1] rank {d}: prompt path on the last row: two chunks == one {d}/{d}; == decode {d}/{d}, KL mean {e:.3}\n", .{ rank, chunk_eq, prompt_n / 2, prompt_eq, prompt_n, prompt_kl / @as(f64, @floatFromInt(@max(prompt_n, 1))) });
    std.debug.print("[m1] rank {d}: KL(answer key || this) over {d} logit rows: mean {e:.3}, max {e:.3}\n", .{ rank, logits_n, kl_sum / @as(f64, @floatFromInt(@max(logits_n, 1))), kl_max });
    std.debug.print("{s} m1 rank {d}: argmax {d}/{d} steps; logits byte-equal {d}/{d}; {d} layer captures checked{s}{s}\n", .{ if (exact) "PASS" else "FAIL", rank, argmax_ok, total_steps, logits_ok, logits_n, cmp.checked, if (cmp.first_fail != null) "; first difference " else "", cmp.first_fail orelse "" });
    return if (exact) 0 else 1;
}

/// KL(p || q) in nats of the softmaxes of two fp32 logit rows (p: the answer key's).
fn klDiv(want: []align(1) const f32, got: []const f32) f64 {
    var mp: f64 = -std.math.inf(f64);
    var mq: f64 = -std.math.inf(f64);
    for (want, got) |a, b| {
        mp = @max(mp, @as(f64, a));
        mq = @max(mq, @as(f64, b));
    }
    var zp: f64 = 0;
    var zq: f64 = 0;
    for (want, got) |a, b| {
        zp += @exp(@as(f64, a) - mp);
        zq += @exp(@as(f64, b) - mq);
    }
    var kl: f64 = 0;
    for (want, got) |a, b| {
        const lp = @as(f64, a) - mp - @log(zp);
        const lq = @as(f64, b) - mq - @log(zq);
        kl += @exp(lp) * (lp - lq);
    }
    return kl;
}

/// Rank 1 behind the native server: the model rank 0 announces (hello), then every step rank 0 sends, until it stops.
fn follow(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, session: *tp.Session, model_dir: []const u8) !u8 {
    var buf: std.ArrayList(i64) = .empty;
    defer buf.deinit(gpa);
    const su = try mimo.engine.setupOf(try session.plan.recvGrow(gpa, &buf));
    std.debug.print("[follow] rank {d}: context {d}, {d} slots, drafter {s}\n", .{ session.cfg.rank, su.context, su.slots, if (su.drafts) "on" else "off" });
    const e = try mimo.engine.Engine.init(gpa, io, d, model_dir, su, session);
    defer e.deinit();
    std.debug.print("[follow] loaded; following\n", .{});
    var steps: usize = 0;
    while (true) {
        const words = session.plan.recvGrow(gpa, &buf) catch |err| switch (err) {
            error.PeerClosed => break,
            else => return err,
        };
        if (!try e.follow(words)) break;
        steps += 1;
    }
    std.debug.print("[follow] stopped after {d} steps\n", .{steps});
    return 0;
}
