//! The M5 tensor-unit prompt's quality gate on the release (TF_DSV41_REAL, TF_DSV41_GATE_IDS; units on, the default).
const std = @import("std");
const pkg = @import("dsv41_metal");
const pr = pkg.prompt;

fn env(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

/// A row's negative log-likelihood of token `t` (float64) and whether its first maximum is `t`.
fn rowNll(x: []const f32, t: u32) struct { nll: f64, top: bool } {
    var mx: f64 = -std.math.inf(f64);
    var arg: usize = 0;
    for (x, 0..) |v, i| if (v > mx) {
        mx = v;
        arg = i;
    };
    var sum: f64 = 0;
    for (x) |v| sum += @exp(@as(f64, v) - mx);
    return .{ .nll = mx + @log(sum) - @as(f64, x[t]), .top = arg == t };
}

/// Per-position paired sums of arm B against arm A: NLL of the true next token and top-1 hits.
const Pairs = struct { n: usize = 0, nll_a: f64 = 0, nll_b: f64 = 0, top_a: usize = 0, top_b: usize = 0, dn: f64 = 0, dn2: f64 = 0, dt: f64 = 0, dt2: f64 = 0 };

fn add(q: *Pairs, nll_a: f64, top_a: bool, nll_b: f64, top_b: bool) void {
    const dn = nll_b - nll_a;
    const dt = @as(f64, @floatFromInt(@intFromBool(top_b))) - @as(f64, @floatFromInt(@intFromBool(top_a)));
    q.n += 1;
    q.nll_a += nll_a;
    q.nll_b += nll_b;
    q.top_a += @intFromBool(top_a);
    q.top_b += @intFromBool(top_b);
    q.dn += dn;
    q.dn2 += dn * dn;
    q.dt += dt;
    q.dt2 += dt * dt;
}

/// Quality gate 2 (fixed before any run): B's NLL no worse than A's by 2 SE of the paired difference, top-1 likewise.
const Verdict = struct { dn: f64, se_n: f64, dt: f64, se_t: f64, ok_n: bool, ok_t: bool };

fn verdict(q: Pairs) Verdict {
    const nf: f64 = @floatFromInt(q.n);
    const dn = q.dn / nf;
    const se_n = @sqrt(@max(0, q.dn2 / nf - dn * dn) / (nf - 1));
    const dt = q.dt / nf;
    const se_t = @sqrt(@max(0, q.dt2 / nf - dt * dt) / (nf - 1));
    return .{ .dn = dn, .se_n = se_n, .dt = dt, .se_t = se_t, .ok_n = dn <= 2 * se_n, .ok_t = dt >= -2 * se_t };
}

test "gate 2's verdict: equal arms pass, a planted-worse arm fails on NLL and on top-1" {
    var prng = std.Random.DefaultPrng.init(2);
    const r = prng.random();
    var same: Pairs = .{};
    var worse: Pairs = .{};
    var lower: Pairs = .{};
    for (0..4000) |_| {
        const base = 1.0 + r.float(f64);
        const noise = (r.float(f64) - 0.5) * 0.01;
        const top = r.float(f64) < 0.7;
        add(&same, base, top, base + noise, top);
        add(&worse, base, top, base + 0.05 + noise, top);
        add(&lower, base, top, base + noise, top and r.float(f64) < 0.95);
    }
    const a = verdict(same);
    const b = verdict(worse);
    const c = verdict(lower);
    try std.testing.expect(a.ok_n and a.ok_t);
    try std.testing.expect(!b.ok_n);
    try std.testing.expect(!c.ok_t);
}

fn has(list: []const u8, name: []const u8) bool {
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |w| if (std.mem.eql(u8, w, name)) return true;
    return false;
}

/// KL(p || q) of two logit rows in float64, and whether their first maxima agree.
fn rowKl(x: []const f32, y: []const f32) struct { kl: f64, same: bool } {
    var mx: f64 = -std.math.inf(f64);
    var my: f64 = -std.math.inf(f64);
    var ax: usize = 0;
    var ay: usize = 0;
    for (x, y, 0..) |s, t, i| {
        if (s > mx) {
            mx = s;
            ax = i;
        }
        if (t > my) {
            my = t;
            ay = i;
        }
    }
    var zx: f64 = 0;
    var zy: f64 = 0;
    for (x, y) |s, t| {
        zx += @exp(@as(f64, s) - mx);
        zy += @exp(@as(f64, t) - my);
    }
    var kl: f64 = 0;
    for (x, y) |s, t| {
        const lp = @as(f64, s) - mx - @log(zx);
        const lq = @as(f64, t) - my - @log(zy);
        kl += @exp(lp) * (lp - lq);
    }
    return .{ .kl = kl, .same = ax == ay };
}

test "dsv41 metal gate: each passage's last 128 rows, decode-arithmetic prompt rows against tensor-unit ones" {
    const dir = env("TF_DSV41_REAL") orelse return error.SkipZigTest;
    const ids_path = env("TF_DSV41_GATE_IDS") orelse return error.SkipZigTest;
    const a = std.heap.page_allocator;
    const io = std.testing.io;
    const limit: u32 = 8192;
    const m = try pkg.boot.Boot.open(a, io, dir, .{ .slots = 1, .context = limit, .drafts = false });
    const te = m.t.env;
    const c = te.c;
    const p = @constCast(te.p);
    if (p.mma.nax == null) return error.SkipZigTest;
    const fp8 = p.mma.nax;
    const penv: pr.Env = .{ .c = c, .p = te.p, .tables = te.tables, .model = te.model, .lookup = te.lookup, .queue = te.queue };
    const o: pr.Opts = .{ .chunk = 4096, .head_rows = 128, .taps_rows = 128 };
    var prompt = try pr.Prompt.init(te.device, c, o, limit);
    // floor: decode arithmetic with the FP8 prompt rows' tile roots added in a chain, a pure fp32 reordering
    const only = env("TF_DSV41_GATE_ONLY") orelse "";
    // TF_DSV41_GATE_CTL=worse: arm B's logits at half scale, a planted-worse control that must fail the gate
    const worse = std.mem.eql(u8, env("TF_DSV41_GATE_CTL") orelse "", "worse");
    const plain = p.mma;
    const chain = if (std.mem.eql(u8, only, "floor")) blk: {
        const src = try std.Io.Dir.cwd().readFileAlloc(io, "zig/kernels/metal/deepseek_v41/fp8_mma.metal", a, .limited(1 << 22));
        const text = try std.fmt.allocPrint(a, "#define TF_CTL 2\n{s}", .{src});
        break :blk try pkg.pipelines.fp8_mma.Pipelines.loadSource(te.device, text);
    } else plain;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, a, .limited(1 << 24));
    const parsed = try std.json.parseFromSlice([][]u32, a, bytes, .{});
    const V: usize = c.vocab;
    var kl_sum: f64 = 0;
    var kl_max: f64 = 0;
    var rows: usize = 0;
    var agree: usize = 0;
    var q: Pairs = .{};
    for (parsed.value, 0..) |ids, pi| {
        var out: [2][]f32 = undefined;
        for (0..2) |mode| {
            const all = only.len == 0;
            const f = mode == 1 and std.mem.eql(u8, only, "floor");
            p.mma = if (f) chain else plain;
            p.mma.nax = if (mode == 1 and (all or has(only, "fp8"))) fp8 else null;
            var slot = try pkg.kv.Slot.init(a, te.device, c, limit);
            defer slot.deinit();
            var tail = try pr.Tail.init(te.device, c, try pr.tailRows(c, o));
            defer tail.deinit();
            try prompt.run(penv, &slot, &tail, ids);
            out[mode] = try a.dupe(f32, prompt.logits(c)[0 .. prompt.head * V]);
            if (mode == 1 and worse) for (out[1]) |*x| {
                x.* *= 0.5;
            };
        }
        // gate 2: each scored row's true next token (the last row has none in the passage)
        const n = ids.len;
        for (0..prompt.head - 1) |r| {
            const t = ids[n - prompt.head + r + 1];
            const ea = rowNll(out[0][r * V ..][0..V], t);
            const eb = rowNll(out[1][r * V ..][0..V], t);
            add(&q, ea.nll, ea.top, eb.nll, eb.top);
        }
        var pk: f64 = 0;
        var pa: usize = 0;
        for (0..prompt.head) |r| {
            const d = rowKl(out[0][r * V ..][0..V], out[1][r * V ..][0..V]);
            pk += d.kl;
            kl_max = @max(kl_max, d.kl);
            pa += @intFromBool(d.same);
        }
        std.debug.print("gate passage {d}: {d} tokens, {d} rows, mean KL {e:.3}, top-1 {d}/{d}\n", .{ pi, ids.len, prompt.head, pk / @as(f64, @floatFromInt(prompt.head)), pa, prompt.head });
        kl_sum += pk;
        agree += pa;
        rows += prompt.head;
    }
    const mean = kl_sum / @as(f64, @floatFromInt(rows));
    const share = @as(f64, @floatFromInt(agree)) / @as(f64, @floatFromInt(rows));
    const pass = mean <= 1e-4 and share >= 0.99;
    std.debug.print("gate {s}: {d} rows, mean KL {e:.3} nats (max {e:.3}), top-1 agreement {d:.2}%\n", .{ if (pass) "PASS" else "FAIL", rows, mean, kl_max, share * 100 });
    const v = verdict(q);
    const nf: f64 = @floatFromInt(q.n);
    const ok = v.ok_n and v.ok_t;
    std.debug.print("gate2 {s}{s}: {d} positions; NLL A {d:.5} B {d:.5}, B-A {d:.6} (SE {d:.6}, {s}); top-1 A {d:.4} B {d:.4}, B-A {d:.5} (SE {d:.5}, {s})\n", .{ if (ok) "PASS" else "FAIL", if (worse) " (planted-worse control)" else "", q.n, q.nll_a / nf, q.nll_b / nf, v.dn, v.se_n, if (v.ok_n) "ok" else "worse", @as(f64, @floatFromInt(q.top_a)) / nf, @as(f64, @floatFromInt(q.top_b)) / nf, v.dt, v.se_t, if (v.ok_t) "ok" else "worse" });
    // the gate fails the step; under the planted control it must fail, and the step fails if it does not
    try std.testing.expect(ok != worse);
}
