//! tf-mimo: rank 1 behind the server (follow), the decode forward against the Python engine's bytes (m1), kv8check.

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
    const is_m1 = args.len >= 5 and std.mem.eql(u8, cmd, "m1");
    const is_follow = args.len >= 4 and std.mem.eql(u8, cmd, "follow");
    const is_kv8 = args.len >= 4 and std.mem.eql(u8, cmd, "kv8check");
    if (!is_m1 and !is_follow and !is_kv8) {
        std.debug.print("usage: tf-mimo m1 MODEL ORACLE(rank<R>.safetensors) AOT_DIR [CONTEXT]\n" ++
            "       tf-mimo follow MODEL AOT_DIR   (TP rank 1 behind tensorfold-native serve on rank 0)\n" ++
            "       tf-mimo kv8check MODEL AOT_DIR   (the 8-bit cache's decode attention against Triton's)\n", .{});
        return 2;
    }
    const gpa = init.gpa;
    const io = init.io;
    const context: usize = if (is_kv8) 8192 else if (is_m1 and args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else 4096;
    const cfg = try tp.Config.fromEnv();
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(cfg.device));
    defer ctx.deinit();
    var session: tp.Session = undefined;
    const world: usize = cfg.world;
    if (world > 1) try session.open(&driver, cfg, @import("tp_kernel").mailbox);
    defer if (world > 1) {
        session.announceStop(); // a clean stop: the peer exits 0 instead of failing fast (70) when this rank ends first
        session.close();
    };
    const rank: usize = cfg.rank;
    if (is_follow) return follow(gpa, io, &driver, &session, args[2], args[3]);
    var s = try cuda.Stream.init(&driver, false);
    defer s.deinit();

    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var w = try mimo.load.load(gpa, io, &driver, args[2], rank, world, rank == 0);
    defer w.deinit();
    var k3 = try cuda.exl3.Kernels.load(&driver);
    defer k3.deinit();
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, args[if (is_kv8) 3 else 4]);
    defer set.deinit();
    const comm = if (world > 1) session.comm() else null;
    var model = try mimo.forward.Model.init(gpa, &driver, &w, &set, &k3, comm, context, 1, 1, false, s);
    defer model.deinit();
    std.debug.print("[m1] rank {d}: loaded in {d:.0}s ({d:.1} GiB weights)\n", .{ rank, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9, @as(f64, @floatFromInt(w.arena.total)) / (1 << 30) });
    if (is_kv8) return if (try mimo.kv8.check(&model, io, s, rank)) 0 else 1;

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
    var all_ok = true;
    var nb: [128]u8 = undefined;
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
            if (got == am[step]) ok_here += 1 else if (first_bad == null) first_bad = step;
            if (step < lg_t.shape[0]) {
                logits_n += 1;
                const want = lg_t.bytes[step * V * 4 ..][0 .. V * 4];
                if (std.mem.eql(u8, std.mem.sliceAsBytes(logits), want)) logits_ok += 1 else if (cmp.first_fail == null) {
                    cmp.first_fail = try std.fmt.allocPrint(gpa, "{s}.logits[{d}]", .{ name, step });
                    report(cmp.first_fail.?, std.mem.sliceAsBytes(logits), want, false);
                }
            }
        }
        total_steps += nsteps;
        argmax_ok += ok_here;
        if (first_bad != null) all_ok = false;
        std.debug.print("[m1] rank {d} {s}: {d} steps, argmax equal at {d}{s}\n", .{ rank, name, nsteps, ok_here, if (first_bad) |b| try std.fmt.bufPrint(&nb, " (first difference at step {d})", .{b}) else "" });
    }
    const exact = all_ok and cmp.first_fail == null and logits_ok == logits_n;
    std.debug.print("{s} m1 rank {d}: argmax {d}/{d} steps; logits byte-equal {d}/{d}; {d} layer captures checked{s}{s}\n", .{ if (exact) "PASS" else "FAIL", rank, argmax_ok, total_steps, logits_ok, logits_n, cmp.checked, if (cmp.first_fail != null) "; first difference " else "", cmp.first_fail orelse "" });
    return if (exact) 0 else 1;
}

/// Rank 1 behind the native server: the model rank 0 announces (hello), then every step rank 0 sends, until it stops.
fn follow(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, session: *tp.Session, model_dir: []const u8, aot_dir: []const u8) !u8 {
    if (session.cfg.world < 2 or session.cfg.rank == 0) return error.FollowNeedsRankOne;
    var buf: std.ArrayList(i64) = .empty;
    defer buf.deinit(gpa);
    const su = try mimo.engine.setupOf(try session.plan.recvGrow(gpa, &buf));
    std.debug.print("[follow] rank {d}: context {d}, {d} slots, drafter {s}, {s} cache\n", .{ session.cfg.rank, su.context, su.slots, if (su.drafts) "on" else "off", if (su.kv8) "8-bit" else "16-bit" });
    const e = try mimo.engine.Engine.init(gpa, io, d, model_dir, aot_dir, su, session);
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
