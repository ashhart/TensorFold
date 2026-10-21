//! flashnext-exl3-engine: the Qwen3.8-Flash-Next EXL3 pack through the Flash Next family's own Engine
//! (families/flashnext/cuda_prompt.zig), with the offload pool attached exactly as the server attaches it
//! (cuda_exl3_pass.attach, called from cuda_lanes_load.open), checked against the independent reference golden
//! (zig/tests/fixtures/exl3_ref_golden.json: the Python engine's bf16 logits and greedy ids).
//!
//!   flashnext-exl3-engine DIR tf GOLDEN [LINES]     the golden's 16-id prompt teacher-forced in one pass: every
//!                                                    position's top-1 and top-5 against the golden's, the logit
//!                                                    difference on the golden's top-5 ids, and the EXL3 lm_head on the
//!                                                    pass's own head rows against an exact float64 product
//!   flashnext-exl3-engine DIR tfc GOLDEN [LINES]    each golden prompt and the golden's 24 greedy ids teacher-forced:
//!                                                    our top-1 at every continuation step, with both margins
//!   flashnext-exl3-engine DIR gen GOLDEN [LINES]    greedy 24 tokens for each golden prompt (prefill, then
//!                                                    Engine.round a token): ids equal to the golden's
//!   flashnext-exl3-engine DIR bench [LINES]         prefill rows/s at 16, 256 and 2048 tokens and decode ms/token,
//!                                                    with the pool's hit rate (a measurement on an idle card)
//!
//! LINES (or TF_FLASHNEXT_EXL3_LINES) fixes the pool's lines a layer; without it the pool takes free memory.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");

const Gpu = check.Gpu;
const st = core.safetensors;
const fmt = core.exl3_format;
const prompt = flash.prompt;
const xpass = flash.exl3.pass;
const vocab = prompt.vocabulary;
const dims: usize = 2560;
const cap_default: usize = 32768;
/// Rows of logits a teacher-forced pass returns at most.
const out_rows: usize = 64;

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn ms(t0: i96, t1: i96) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

const Eng = struct {
    gpu: Gpu,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    out: cuda.DeviceBuffer,
    cap: usize,
    attached: xpass.Attached,
    load_s: f64,

    fn init(gpu: Gpu, dir: []const u8, lines: ?usize, cap: usize) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.cap = cap;
        g.stream = try cuda.Stream.init(gpu.d, true);
        g.glue = try flash.Glue.load(gpu.d);
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, 1, 0);
        const t1 = check.now(gpu.io);
        // as cuda_lanes_load.open: the pass's scratch for a full chunk and the block scores, then the stream's caches
        _ = try g.e.reserve(@min(cap, g.e.chunk));
        _ = try g.e.reserveScores(cap, @min(cap, g.e.chunk));
        g.seq = try g.e.newSeq(cap);
        g.out = try cuda.DeviceBuffer.alloc(gpu.d, out_rows * vocab * 2);
        var b = try xpass.budgetFromEnv(0);
        if (lines) |n| b.lines = n;
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, b);
        const t2 = check.now(gpu.io);
        g.load_s = ms(t0, t2) / 1e3;
        var free: usize = 0;
        var total: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        std.debug.print("engine: loaded in {d:.1} s (Engine.initRank {d:.1} s incl. the pinned bank's copies {d:.1} s; pools {d:.1} s); {d} lines a layer x {d} layers ({d:.2} GiB), {d:.2} GiB free after, chunk {d}, context {d}\n", .{ g.load_s, ms(t0, t1) / 1e3, @as(f64, @floatFromInt(g.attached.bank_ns)) / 1e9, ms(t1, t2) / 1e3, g.attached.lines, g.attached.layers, @as(f64, @floatFromInt(g.attached.device_bytes)) / (1 << 30), @as(f64, @floatFromInt(free)) / (1 << 30), g.e.chunk, cap });
        return g;
    }

    fn deinit(g: *Eng) void {
        g.out.free();
        g.seq.deinit(g.gpu.gpa);
        g.e.deinit();
        g.glue.deinit();
        g.stream.deinit();
        g.gpu.gpa.destroy(g);
    }

    fn tri(g: *Eng) flash.Tri {
        return .{ .own = &g.glue, .s = g.stream };
    }

    /// The sequence as Engine.newSeq makes it, without a new allocation (the pools took the free memory).
    fn reset(g: *Eng) !void {
        const s = &g.seq;
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| for (bs) |b| try b.fill8(0, g.stream.handle);
        try s.tail.fill8(0, g.stream.handle);
        try g.stream.synchronize();
        s.fed = 0;
        s.history = .{ g.e.eos, g.e.eos };
    }

    fn stats(g: *Eng) flash.exl3_pool.Stats {
        return xpass.poolStats(g.e.layers[0..g.e.n_layers], null);
    }
};

const Golden = struct {
    parsed: std.json.Parsed(std.json.Value),

    fn open(gpu: Gpu, path: []const u8) !Golden {
        const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 26));
        defer gpu.gpa.free(text);
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, gpu.gpa, text, .{}) };
    }

    fn prompts(g: *const Golden) []std.json.Value {
        return g.parsed.value.object.get("prompts").?.array.items;
    }
};

fn ints(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const items = v.array.items;
    const out = try a.alloc(u32, items.len);
    for (items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
}

fn top5(row: []const u16) [5]u32 {
    var ids: [5]u32 = @splat(0);
    var vals: [5]f32 = @splat(-std.math.inf(f32));
    for (row, 0..) |b, i| {
        const v = promote(b);
        if (!(v > vals[4])) continue;
        var at: usize = 4;
        while (at > 0 and v > vals[at - 1]) : (at -= 1) {
            vals[at] = vals[at - 1];
            ids[at] = ids[at - 1];
        }
        vals[at] = v;
        ids[at] = @intCast(i);
    }
    return ids;
}

// -- the EXL3 head in float64 ------------------------------------------------------------------------------------

const Raw = struct {
    files: [7]st.File = undefined,
    n: usize = 0,

    fn open(gpu: Gpu, dir: []const u8) !Raw {
        var r: Raw = .{};
        var buf: [512]u8 = undefined;
        for (0..7) |i| {
            r.files[i] = try st.File.open(gpu.gpa, gpu.io, try std.fmt.bufPrint(&buf, "{s}/model-{d:0>5}-of-00007.safetensors", .{ dir, i + 1 }));
            r.n += 1;
        }
        return r;
    }

    fn close(r: *Raw, io: std.Io) void {
        for (r.files[0..r.n]) |*f| f.close(io);
    }

    fn get(r: *const Raw, name: []const u8) !st.Tensor {
        for (r.files[0..r.n]) |*f| if (f.get(name)) |t| return t;
        return error.MissingTensor;
    }
};

const HeadWork = struct {
    head: flash.exl3_mtp.Head,
    x: []const f64, // [rows, K]
    rows: usize,
    out: []f64, // [rows, N]
    next: std.atomic.Value(usize) = .init(0),
    failed: ?anyerror = null,
};

fn headWorker(w: *HeadWork, gpa: std.mem.Allocator) void {
    headWork(w, gpa) catch |err| {
        w.failed = err;
    };
}

/// logits[r][v] = sum_k x[r][k] * W[k][v], W decoded in float64 a 128-column block at a time (exl3_format.dequantize).
fn headWork(w: *HeadWork, gpa: std.mem.Allocator) !void {
    const h = w.head;
    const bn: usize = fmt.hadamard_dim;
    const tiles_k = h.k / 16;
    const tiles_n = h.n / 16;
    const per = bn / 16;
    const words = h.words();
    const sub = try gpa.alloc(i16, tiles_k * per * words);
    defer gpa.free(sub);
    const vals = try gpa.alloc(f64, h.k * bn);
    defer gpa.free(vals);
    const scratch = try gpa.alloc(f64, 2 * h.k * bn);
    defer gpa.free(scratch);
    const suh = try gpa.alloc(f16, h.k);
    defer gpa.free(suh);
    for (suh, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u16, h.suh[2 * i ..][0..2], .little));
    var svh: [fmt.hadamard_dim]f16 = undefined;
    while (true) {
        const nb = w.next.fetchAdd(1, .monotonic);
        if (nb >= h.n / bn) return;
        for (0..tiles_k) |tk| for (0..per) |j| {
            const src = ((tk * tiles_n) + nb * per + j) * words * 2;
            const dst = sub[(tk * per + j) * words ..][0..words];
            for (dst, 0..) |*d, i| d.* = std.mem.readInt(i16, h.trellis[src + 2 * i ..][0..2], .little);
        };
        for (&svh, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u16, h.svh[2 * (nb * bn + i) ..][0..2], .little));
        fmt.dequantize(.{ .halves = h.halves }, .mul1, sub, tiles_k, per, suh, &svh, vals, scratch);
        for (0..w.rows) |r| {
            const xr = w.x[r * h.k ..][0..h.k];
            var acc: [fmt.hadamard_dim]f64 = @splat(0);
            for (0..h.k) |k| {
                const xv = xr[k];
                const row = vals[k * bn ..][0..bn];
                for (&acc, row) |*a, v| a.* += xv * v;
            }
            @memcpy(w.out[r * h.n + nb * bn ..][0..bn], &acc);
        }
    }
}

fn headF64(gpa: std.mem.Allocator, raw: *const Raw, x: []const f64, rows: usize) ![]f64 {
    const head = try flash.exl3_mtp.Head.of(try raw.get("lm_head.trellis"), try raw.get("lm_head.suh"), try raw.get("lm_head.svh"), raw.get("lm_head.mul1") catch null);
    const out = try gpa.alloc(f64, rows * head.n);
    errdefer gpa.free(out);
    var w: HeadWork = .{ .head = head, .x = x, .rows = rows, .out = out };
    var pool: [64]std.Thread = undefined;
    const n = @min(pool.len, @max(1, std.Thread.getCpuCount() catch 1));
    var spawned: usize = 0;
    for (0..n) |_| {
        pool[spawned] = try std.Thread.spawn(.{}, headWorker, .{ &w, gpa });
        spawned += 1;
    }
    for (pool[0..spawned]) |t| t.join();
    if (w.failed) |err| return err;
    return out;
}

// -- modes ---------------------------------------------------------------------------------------------------------

fn tfMode(gpu: Gpu, dir: []const u8, golden_path: []const u8, lines: ?usize) !void {
    const a = gpu.gpa;
    var golden = try Golden.open(gpu, golden_path);
    defer golden.parsed.deinit();
    const p = golden.prompts()[0];
    const ids = try ints(a, p.object.get("ids").?);
    defer a.free(ids);
    const tf = p.object.get("teacher_forced").?.array.items;
    const g = try Eng.init(gpu, dir, lines, cap_default);
    defer g.deinit();
    try g.reset();
    const n = ids.len;
    const ops = g.e.layers[0].exl3.?.ops;
    const before = ops.fingerprint();
    const t0 = check.now(gpu.io);
    try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, .{ .logits = .all, .logits_out = g.out.ptr });
    try g.stream.synchronize();
    const t1 = check.now(gpu.io);
    const logits = try a.alloc(u16, n * vocab);
    defer a.free(logits);
    try g.out.download(0, std.mem.sliceAsBytes(logits));
    // the head rows the pass fed lm_head (the final mixer's read-out, bf16 [n, dims])
    const hrows = try a.alloc(u16, n * dims);
    defer a.free(hrows);
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(hrows.ptr, g.e.batch_s.?.mixed, n * dims * 2), "cuMemcpyDtoH");
    std.debug.print("teacher-forced pass: {d} rows in {d:.1} ms (first pass: includes every miss's copy)\n", .{ n, ms(t0, t1) });
    // one decode round too, then every EXL3 scratch buffer must be where Ops.reserve put it
    {
        var tok = prompt.argmax(g.seq.logits);
        const q = [_]*prompt.Seq{&g.seq};
        try g.e.round(flash.Tri, g.tri(), &q, (&tok)[0..1], g.out.ptr);
        tok = prompt.argmax(g.seq.logits);
    }
    const after = ops.fingerprint();
    try check.expect(std.mem.eql(u8, std.mem.sliceAsBytes(&before), std.mem.sliceAsBytes(&after)), "an EXL3 scratch buffer grew inside a pass", .{});
    check.pass("EXL3 scratch reserved up front: the {d} scratch buffers kept their addresses and sizes through a {d}-row prompt pass and a decode round", .{ blk: {
        var k: usize = 0;
        for (after) |e| k += @intFromBool(e[1] != 0);
        break :blk k;
    }, n });

    std.debug.print(" pos  input  ours top1 (logit)  golden top1 (logit)  match  top5 overlap  max|dlogit| on golden top5\n", .{});
    var top1_ok: usize = 0;
    var top5_same: usize = 0;
    var worst: f64 = 0;
    var sum_abs: f64 = 0;
    var count: usize = 0;
    for (tf, 0..) |row, pos| {
        const r = logits[pos * vocab ..][0..vocab];
        const ours = top5(r);
        const g5 = try ints(a, row.object.get("top5_ids").?);
        defer a.free(g5);
        const gl = row.object.get("top5_logits").?.array.items;
        var overlap: usize = 0;
        for (ours) |o| for (g5) |x| {
            overlap += @intFromBool(o == x);
        };
        var dmax: f64 = 0;
        for (g5, gl) |id, lv| {
            const d = @abs(@as(f64, promote(r[id])) - num(lv));
            dmax = @max(dmax, d);
            sum_abs += d;
            count += 1;
        }
        worst = @max(worst, dmax);
        const ok = ours[0] == g5[0];
        top1_ok += @intFromBool(ok);
        top5_same += @intFromBool(std.mem.eql(u32, &ours, g5));
        std.debug.print("{d:4} {d:7} {d:9} ({d:7.3}) {d:9} ({d:7.3})  {s:5}  {d}/5 {s}  {d:.4}\n", .{ pos, ids[pos], ours[0], promote(r[ours[0]]), g5[0], num(gl[0]), if (ok) "yes" else "NO", overlap, if (std.mem.eql(u32, &ours, g5)) "(same order)" else "            ", dmax });
    }
    std.debug.print("TF top1 {d}/{d}, top5 identical in order {d}/{d}, |our logit - golden bf16 logit| on the golden's top-5 ids: max {d:.4}, mean {d:.4}\n", .{ top1_ok, tf.len, top5_same, tf.len, worst, sum_abs / @as(f64, @floatFromInt(count)) });

    // the EXL3 lm_head itself, on the same head rows, against an exact float64 product over all 248,320 columns
    var raw = try Raw.open(gpu, dir);
    defer raw.close(gpu.io);
    const xf = try a.alloc(f64, n * dims);
    defer a.free(xf);
    for (xf, hrows) |*o, b| o.* = promote(b);
    const t2 = check.now(gpu.io);
    const exact = try headF64(a, &raw, xf, n);
    defer a.free(exact);
    const t3 = check.now(gpu.io);
    var head_abs: f64 = 0;
    var head_rel: f64 = 0;
    var head_top1: usize = 0;
    for (0..n) |r| {
        var peak: f64 = 0;
        var best: usize = 0;
        var dmax: f64 = 0;
        for (0..vocab) |v| {
            const want = exact[r * vocab + v];
            peak = @max(peak, @abs(want));
            if (want > exact[r * vocab + best]) best = v;
            dmax = @max(dmax, @abs(@as(f64, promote(logits[r * vocab + v])) - want));
        }
        head_abs = @max(head_abs, dmax);
        head_rel = @max(head_rel, dmax / peak);
        const ours = top5(logits[r * vocab ..][0..vocab])[0];
        // a bf16 tie at the top is no disagreement
        head_top1 += @intFromBool(ours == best or promote(logits[r * vocab + ours]) == promote(logits[r * vocab + best]));
    }
    std.debug.print("EXL3 lm_head (Dense, bf16 out) vs exact float64 on the pass's {d} head rows, {d} columns ({d:.1} s on the host): max |d| {d:.4} (rel to the row peak {e:.2}), top-1 equal on {d}/{d}\n", .{ n, vocab, ms(t2, t3) / 1e3, head_abs, head_rel, head_top1, n });
    // bf16 output rounding is 2^-9 of a value (0.06 at a logit of 30); the fp16 rotated input adds ~2.4e-4 of the peak
    try check.expect(head_rel < 4e-3 and head_top1 == n, "lm_head within bf16 rounding of float64", .{});
    check.pass("lm_head: EXL3 hook through exl3.Dense, {d} rows, max |d| {d:.4} (rel {e:.2}) vs float64, top-1 equal on every row", .{ n, head_abs, head_rel });
    try check.expect(top1_ok == tf.len, "teacher-forced top-1 equals the golden on every position ({d}/{d})", .{ top1_ok, tf.len });
    check.pass("teacher-forced top-1 {d}/{d} vs the golden; max |dlogit| {d:.4} on the golden's top-5", .{ top1_ok, tf.len, worst });
    printPool(g, "after tf");
}

fn printPool(g: *Eng, what: []const u8) void {
    const s = g.stats();
    const routed = s.hit_slots + s.miss_slots;
    std.debug.print("pool {s}: {d} rounds, {d} passes, hit rate {d:.1}% ({d}/{d} routed slots), {d} copies, {d} evictions; host cut {d:.1} ms over {d} blocks, plan {d:.1} ms\n", .{ what, s.rounds, s.passes, 100 * @as(f64, @floatFromInt(s.hit_slots)) / @as(f64, @floatFromInt(@max(routed, 1))), s.hit_slots, routed, s.misses, s.evictions, @as(f64, @floatFromInt(xpass.pool_time.cut_ns)) / 1e6, xpass.pool_time.blocks, @as(f64, @floatFromInt(xpass.pool_time.plan_ns)) / 1e6 });
}

/// Greedy `steps` tokens after `ids`: the prefill's last row, then a decode round a token.
fn greedyRun(g: *Eng, ids: []const u32, out: []u32, step_ms: ?*f64) !void {
    try g.reset();
    try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, .{ .logits = .last });
    out[0] = prompt.argmax(g.seq.logits);
    const t0 = check.now(g.gpu.io);
    for (1..out.len) |i| {
        const q = [_]*prompt.Seq{&g.seq};
        try g.e.round(flash.Tri, g.tri(), &q, out[i - 1 ..][0..1], g.out.ptr);
        out[i] = prompt.argmax(g.seq.logits);
    }
    if (step_ms) |p| p.* = ms(t0, check.now(g.gpu.io)) / @as(f64, @floatFromInt(@max(out.len - 1, 1)));
}

/// Each golden prompt followed by the golden's own 24 greedy ids, teacher-forced in one pass: our top-1 at every
/// continuation position against the golden's greedy id, with both margins (top-1 minus top-2).
fn tfcMode(gpu: Gpu, dir: []const u8, golden_path: []const u8, lines: ?usize) !void {
    const a = gpu.gpa;
    var golden = try Golden.open(gpu, golden_path);
    defer golden.parsed.deinit();
    const g = try Eng.init(gpu, dir, lines, cap_default);
    defer g.deinit();
    var all_ok = true;
    for (golden.prompts()) |p| {
        const ids = try ints(a, p.object.get("ids").?);
        defer a.free(ids);
        const cont = try ints(a, p.object.get("greedy_ids").?);
        defer a.free(cont);
        const steps = p.object.get("greedy_steps").?.array.items;
        const all = try a.alloc(u32, ids.len + cont.len - 1);
        defer a.free(all);
        @memcpy(all[0..ids.len], ids);
        @memcpy(all[ids.len..], cont[0 .. cont.len - 1]);
        if (all.len > out_rows) return error.UnexpectedTensor;
        try g.reset();
        try g.e.forward(flash.Tri, g.tri(), &g.seq, all, .{ .logits = .all, .logits_out = g.out.ptr });
        try g.stream.synchronize();
        const logits = try a.alloc(u16, all.len * vocab);
        defer a.free(logits);
        try g.out.download(0, std.mem.sliceAsBytes(logits[0 .. all.len * vocab]));
        std.debug.print("TFC {s}\n step  golden id  ours top1  our margin  golden margin  max|dlogit| golden top5\n", .{p.object.get("name").?.string});
        var ok_n: usize = 0;
        for (cont, 0..) |want, k| {
            const r = logits[(ids.len - 1 + k) * vocab ..][0..vocab];
            const t = top5(r);
            const margin = promote(r[t[0]]) - promote(r[t[1]]);
            const g5 = try ints(a, steps[k].object.get("top5_ids").?);
            defer a.free(g5);
            var dmax: f64 = 0;
            for (g5, steps[k].object.get("top5_logits").?.array.items) |id, lv| dmax = @max(dmax, @abs(@as(f64, promote(r[id])) - num(lv)));
            const ok = t[0] == want;
            ok_n += @intFromBool(ok);
            std.debug.print(" {d:4} {d:10} {d:9} {s} {d:10.4} {d:14.4} {d:10.4}\n", .{ k, want, t[0], if (ok) " " else "*", margin, num(steps[k].object.get("margin12").?), dmax });
        }
        std.debug.print("TFC {s}: top-1 equals the golden's greedy id on {d}/{d} continuation positions\n", .{ p.object.get("name").?.string, ok_n, cont.len });
        if (ok_n != cont.len) all_ok = false;
    }
    printPool(g, "after tfc");
    try check.expect(all_ok, "teacher-forced continuation top-1 equals the golden everywhere", .{});
    check.pass("teacher-forced continuation: top-1 equals the golden's greedy id at every step of all {d} prompts", .{golden.prompts().len});
}

fn genMode(gpu: Gpu, dir: []const u8, golden_path: []const u8, lines: ?usize) !void {
    const a = gpu.gpa;
    var golden = try Golden.open(gpu, golden_path);
    defer golden.parsed.deinit();
    const g = try Eng.init(gpu, dir, lines, cap_default);
    defer g.deinit();
    var all_ok = true;
    for (golden.prompts()) |p| {
        const ids = try ints(a, p.object.get("ids").?);
        defer a.free(ids);
        const want = try ints(a, p.object.get("greedy_ids").?);
        defer a.free(want);
        const got = try a.alloc(u32, want.len);
        defer a.free(got);
        var step: f64 = 0;
        try greedyRun(g, ids, got, &step);
        var first_diff: ?usize = null;
        for (got, want, 0..) |x, y, i| if (x != y and first_diff == null) {
            first_diff = i;
        };
        std.debug.print("GEN {s}: ", .{p.object.get("name").?.string});
        for (got) |x| std.debug.print("{d},", .{x});
        std.debug.print("  {s} ({d:.1} ms/step decode)\n", .{ if (first_diff != null) "DIFFERS" else "equal to the golden", step });
        if (first_diff) |d| {
            std.debug.print("  first difference at step {d}: ours {d}, golden {d}; golden margin there {d:.4}\n", .{ d, got[d], want[d], num(p.object.get("greedy_steps").?.array.items[d].object.get("margin12").?) });
            all_ok = false;
        }
    }
    printPool(g, "after gen");
    try check.expect(all_ok, "greedy 24 equals the golden on every prompt", .{});
    check.pass("greedy 24 tokens equal the golden on all {d} prompts", .{golden.prompts().len});
}

fn benchMode(gpu: Gpu, dir: []const u8, lines: ?usize) !void {
    const a = gpu.gpa;
    const g = try Eng.init(gpu, dir, lines, cap_default);
    defer g.deinit();
    const max_n: usize = 2048;
    const ids = try a.alloc(u32, max_n);
    defer a.free(ids);
    // text-like ids: a fixed pseudo-random walk over the common part of the vocabulary
    var prng = std.Random.DefaultPrng.init(0x5eed);
    for (ids) |*x| x.* = 200 + prng.random().uintLessThan(u32, 60000);
    // warm-up: one pass to load kernels, settle the pool
    try g.reset();
    try g.e.forward(flash.Tri, g.tri(), &g.seq, ids[0..64], .{ .logits = .last });
    for ([_]usize{ 16, 256, 2048 }) |n| {
        var best: f64 = std.math.inf(f64);
        for (0..2) |_| {
            try g.reset();
            const before = g.stats();
            const t0 = check.now(gpu.io);
            try g.e.forward(flash.Tri, g.tri(), &g.seq, ids[0..n], .{ .logits = .last });
            const t = ms(t0, check.now(gpu.io));
            const after = g.stats();
            const hits = after.hit_slots - before.hit_slots;
            const all = hits + after.miss_slots - before.miss_slots;
            std.debug.print("  prefill {d:5} tokens: {d:8.1} ms = {d:7.1} rows/s  (hit {d:.1}%, {d} copies)\n", .{ n, t, @as(f64, @floatFromInt(n)) / (t / 1e3), 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(all, 1))), after.misses - before.misses });
            best = @min(best, t);
        }
        std.debug.print("BENCH prefill {d} tokens: best {d:.1} ms, {d:.1} rows/s\n", .{ n, best, @as(f64, @floatFromInt(n)) / (best / 1e3) });
        // decode after this prompt: 32 tokens, greedy
        var tok = prompt.argmax(g.seq.logits);
        const before = g.stats();
        const t0 = check.now(gpu.io);
        const steps: usize = 32;
        for (0..steps) |_| {
            const q = [_]*prompt.Seq{&g.seq};
            try g.e.round(flash.Tri, g.tri(), &q, (&tok)[0..1], g.out.ptr);
            tok = prompt.argmax(g.seq.logits);
        }
        const t = ms(t0, check.now(gpu.io)) / steps;
        const after = g.stats();
        const hits = after.hit_slots - before.hit_slots;
        const all = hits + after.miss_slots - before.miss_slots;
        std.debug.print("BENCH decode after {d} tokens: {d:.2} ms/token over {d} steps (hit {d:.1}%, {d:.1} copies a token)\n", .{ n, t, steps, 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(all, 1))), @as(f64, @floatFromInt(after.misses - before.misses)) / steps });
    }
    printPool(g, "after bench");
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const mode = rest[1];
    const lineArg = struct {
        fn f(r: []const [:0]const u8, i: usize) !?usize {
            return if (r.len > i) try std.fmt.parseInt(usize, r[i], 10) else null;
        }
    }.f;
    if (std.mem.eql(u8, mode, "tf")) return tfMode(gpu, dir, if (rest.len > 2) rest[2] else return error.MissingArgument, try lineArg(rest, 3));
    if (std.mem.eql(u8, mode, "tfc")) return tfcMode(gpu, dir, if (rest.len > 2) rest[2] else return error.MissingArgument, try lineArg(rest, 3));
    if (std.mem.eql(u8, mode, "gen")) return genMode(gpu, dir, if (rest.len > 2) rest[2] else return error.MissingArgument, try lineArg(rest, 3));
    if (std.mem.eql(u8, mode, "bench")) return benchMode(gpu, dir, try lineArg(rest, 2));
    return error.UnknownMode;
}
