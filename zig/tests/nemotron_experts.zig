//! Nemotron's routed experts at window widths: member-row pass shapes bit-checked against the engine's kernels, then timed.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const E = 128; // experts a layer
const TOPK = 6;
const GS = 64;
const LAYERS = 4; // weight sets streamed per timing (each ~718 MB: nothing stays cached)
const widths = [_]usize{ 1, 2, 3, 4, 8, 16, 32, 64 };

/// A pass shape: member rows a pass (mix: MB, pairs, singles; words: tf_experts_w), outputs a simdgroup, simdgroups.
const Variant = struct { name: []const u8, mb: usize, rps: usize, sg: usize, mix: bool, words: bool = false };
const variants = [_]Variant{
    .{ .name = "rows2r2", .mb = 2, .rps = 2, .sg = 2, .mix = false },
    .{ .name = "rows2r2s4", .mb = 2, .rps = 2, .sg = 4, .mix = false },
    .{ .name = "rows2r1", .mb = 2, .rps = 1, .sg = 2, .mix = false },
    .{ .name = "rows2r1s4", .mb = 2, .rps = 1, .sg = 4, .mix = false },
    .{ .name = "rows2s4", .mb = 2, .rps = 4, .sg = 4, .mix = false },
    .{ .name = "rows2s1", .mb = 2, .rps = 4, .sg = 1, .mix = false },
    .{ .name = "rows2r8", .mb = 2, .rps = 8, .sg = 2, .mix = false },
    .{ .name = "rows4", .mb = 4, .rps = 4, .sg = 2, .mix = false },
    .{ .name = "rows3", .mb = 3, .rps = 4, .sg = 2, .mix = false },
    .{ .name = "mix4r2", .mb = 4, .rps = 2, .sg = 2, .mix = true },
    .{ .name = "w2s4", .mb = 2, .rps = 4, .sg = 4, .mix = false, .words = true },
    .{ .name = "w3s4", .mb = 3, .rps = 4, .sg = 4, .mix = false, .words = true }, // the engine's tf_xup_w3 / tf_xdown_w3
    .{ .name = "w4s4", .mb = 4, .rps = 4, .sg = 4, .mix = false, .words = true },
    .{ .name = "w4s2", .mb = 4, .rps = 4, .sg = 2, .mix = false, .words = true },
    .{ .name = "w6s4", .mb = 6, .rps = 4, .sg = 4, .mix = false, .words = true },
    .{ .name = "w4r8", .mb = 4, .rps = 8, .sg = 4, .mix = false, .words = true },
};

/// tf_experts_rows with the big passes first, then pairs, then singles: every row's sums unchanged in any pass.
const mix_source =
    \\template <int K, int N, int GS, int RPS, int TOPK, int MB, bool UP>
    \\inline void tf_pass(const device bfloat16_t* X, const device int32_t* MEMBERS, int m, const device uint8_t* wr,
    \\                    const device bfloat16_t* S, const device bfloat16_t* B, size_t at, int row0, uint lane, device bfloat16_t* OUT) {
    \\  int p[MB], xoff[MB];
    \\  for (int b = 0; b < MB; b++) { p[b] = MEMBERS[m + b]; xoff[b] = (UP ? p[b] / TOPK : p[b]) * K; }
    \\  float acc[MB][RPS];
    \\  tf_rowdot_rows<K, GS, RPS, MB>(wr, S + at * (K / GS), B + at * (K / GS), xoff, X, lane, acc);
    \\  if (lane == 0)
    \\    for (int b = 0; b < MB; b++)
    \\      for (int j = 0; j < RPS; j++) {
    \\        if (UP) {
    \\          const float h = metal::max(float(bfloat(acc[b][j])), 0.0f);
    \\          OUT[size_t(p[b]) * N + row0 + j] = bfloat(h * h);
    \\        } else OUT[size_t(p[b]) * N + row0 + j] = bfloat(acc[b][j]);
    \\      }
    \\}
    \\template <int K, int N, int GS, int RPS, int SG, int TOPK, int MB, bool UP>
    \\[[kernel]] void tf_experts_mix(
    \\  const device bfloat16_t* X [[buffer(0)]], const device uint32_t* UIDS [[buffer(1)]], const device int32_t* START [[buffer(2)]],
    \\  const device int32_t* COUNT [[buffer(3)]], const device int32_t* MEMBERS [[buffer(4)]], const constant int32_t* UCOUNT [[buffer(5)]],
    \\  const device uint32_t* W [[buffer(6)]], const device bfloat16_t* S [[buffer(7)]], const device bfloat16_t* B [[buffer(8)]],
    \\  device bfloat16_t* OUT [[buffer(9)]], uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    \\  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const int u = int(threadgroup_position_in_grid.z);
    \\  if (u >= UCOUNT[0]) return;
    \\  const size_t e = size_t(UIDS[u]);
    \\  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
    \\  const size_t at = e * N + size_t(row0);
    \\  const int first = START[u], last = START[u] + COUNT[u];
    \\  const device uint8_t* wr = (const device uint8_t*)W + at * (K / 2);
    \\  int m = first;
    \\  #pragma clang loop unroll(disable)
    \\  for (; m + MB <= last; m += MB) tf_pass<K, N, GS, RPS, TOPK, MB, UP>(X, MEMBERS, m, wr, S, B, at, row0, lane, OUT);
    \\  if (MB > 2 && m + 2 <= last) { tf_pass<K, N, GS, RPS, TOPK, 2, UP>(X, MEMBERS, m, wr, S, B, at, row0, lane, OUT); m += 2; }
    \\  if (m < last) {
    \\    const int p = MEMBERS[m];
    \\    float acc[RPS];
    \\    tf_rowdot<K, GS, RPS>(wr, S + at * (K / GS), B + at * (K / GS), X + size_t(UP ? p / TOPK : p) * K, lane, acc);
    \\    if (lane == 0)
    \\      for (int j = 0; j < RPS; j++) {
    \\        if (UP) {
    \\          const float h = metal::max(float(bfloat(acc[j])), 0.0f);
    \\          OUT[size_t(p) * N + row0 + j] = bfloat(h * h);
    \\        } else OUT[size_t(p) * N + row0 + j] = bfloat(acc[j]);
    \\      }
    \\  }
    \\}
    \\
;

/// Shapes the engine's source already instantiates (tf_xup_rows2/4, tf_xdown_rows2/4, tf_xup_w3, tf_xdown_w3).
fn builtIn(v: Variant) bool {
    if (v.words) return v.mb == 3 and v.rps == 4 and v.sg == 4;
    return !v.mix and v.rps == 4 and v.sg == 2 and (v.mb == 2 or v.mb == 4);
}

fn bf16(f: f32) u16 {
    return @truncate(@as(u32, @bitCast(f)) >> 16);
}

/// One projection's weights for LAYERS layers: W [E, N, K/2] 4-bit, S and B [E, N, K/GS] bf16.
const Proj = struct {
    n: usize,
    k: usize,
    w: [LAYERS]mtl.Buffer,
    s: [LAYERS]mtl.Buffer,
    b: [LAYERS]mtl.Buffer,

    fn init(d: mtl.Device, n: usize, k: usize, rng: std.Random) !Proj {
        var p: Proj = .{ .n = n, .k = k, .w = undefined, .s = undefined, .b = undefined };
        for (0..LAYERS) |l| {
            p.w[l] = try d.buffer(E * n * k / 2, opts);
            rng.bytes(p.w[l].slice(u8, E * n * k / 2));
            const g = E * n * (k / GS);
            p.s[l] = try d.buffer(g * 2, opts);
            p.b[l] = try d.buffer(g * 2, opts);
            for (p.s[l].slice(u16, g), p.b[l].slice(u16, g)) |*sv, *bv| {
                sv.* = bf16(0.002 + 0.018 * rng.float(f32));
                bv.* = bf16(-0.15 * rng.float(f32));
            }
        }
        return p;
    }
};

/// A window's routing for each layer, laid out as route_group writes it (experts ascending, members by pair).
const Route = struct {
    uids: [LAYERS]mtl.Buffer,
    start: [LAYERS]mtl.Buffer,
    count: [LAYERS]mtl.Buffer,
    members: [LAYERS]mtl.Buffer,
    ucount: [LAYERS]i32,
    distinct: usize = 0,
    ids0: [64 * TOPK]u32 = undefined, // layer 0's experts by pair (row * 6 + k)

    /// Each row's 6 experts drawn without replacement from a Zipf-like popularity (s = 1), a fresh expert order a layer.
    fn init(d: mtl.Device, rows: usize, rng: std.Random) !Route {
        var r: Route = undefined;
        r.distinct = 0;
        var weight: [E]f64 = undefined;
        for (&weight, 0..) |*w, i| w.* = 1.0 / @as(f64, @floatFromInt(i + 1));
        for (0..LAYERS) |l| {
            var perm: [E]u32 = undefined;
            for (&perm, 0..) |*p, i| p.* = @intCast(i);
            rng.shuffle(u32, &perm);
            const ids = try std.heap.page_allocator.alloc(u32, rows * TOPK);
            defer std.heap.page_allocator.free(ids);
            for (0..rows) |row| {
                var taken: [E]bool = @splat(false);
                for (0..TOPK) |k| {
                    var total: f64 = 0;
                    for (0..E) |i| total += if (taken[i]) 0 else weight[i];
                    var x = rng.float(f64) * total;
                    var pick: usize = 0;
                    for (0..E) |i| {
                        if (taken[i]) continue;
                        pick = i;
                        x -= weight[i];
                        if (x <= 0) break;
                    }
                    taken[pick] = true;
                    ids[row * TOPK + k] = perm[pick];
                }
            }
            if (l == 0) @memcpy(r.ids0[0 .. rows * TOPK], ids);
            r.uids[l] = try d.buffer(E * 4, opts);
            r.start[l] = try d.buffer(E * 4, opts);
            r.count[l] = try d.buffer(E * 4, opts);
            r.members[l] = try d.buffer(@max(rows * TOPK, 8) * 4, opts);
            var u: usize = 0;
            var m: usize = 0;
            for (0..E) |e| {
                var c: usize = 0;
                for (ids) |id| c += @intFromBool(id == e);
                if (c == 0) continue;
                r.uids[l].slice(u32, E)[u] = @intCast(e);
                r.start[l].slice(i32, E)[u] = @intCast(m);
                r.count[l].slice(i32, E)[u] = @intCast(c);
                for (ids, 0..) |id, p| if (id == e) {
                    r.members[l].slice(i32, rows * TOPK)[m] = @intCast(p);
                    m += 1;
                };
                u += 1;
            }
            r.ucount[l] = @intCast(u);
            r.distinct += u;
        }
        return r;
    }
};

const Kernel = struct { up: mtl.Pipeline, down: mtl.Pipeline, rps: usize, sg: usize };

fn encode(enc: mtl.ComputeEncoder, k: Kernel, up: bool, p: *const Proj, rt: *const Route, l: usize, x: mtl.Buffer, out: mtl.Buffer, rows: usize) void {
    enc.setPipeline(if (up) k.up else k.down);
    enc.setBuffer(x, 0, 0);
    enc.setBuffer(rt.uids[l], 0, 1);
    enc.setBuffer(rt.start[l], 0, 2);
    enc.setBuffer(rt.count[l], 0, 3);
    enc.setBuffer(rt.members[l], 0, 4);
    enc.setBytes(std.mem.asBytes(&rt.ucount[l]), 5);
    enc.setBuffer(p.w[l], 0, 6);
    enc.setBuffer(p.s[l], 0, 7);
    enc.setBuffer(p.b[l], 0, 8);
    enc.setBuffer(out, 0, 9);
    const groups = @min(rows * TOPK, E); // as the engine dispatches: threadgroups past UCOUNT return at once
    enc.dispatchThreads(mtl.Size.of(32 * k.sg, p.n / (k.rps * k.sg), groups), mtl.Size.of(32 * k.sg, 1, 1));
}

fn runOnce(q: mtl.Queue, k: Kernel, up: bool, p: *const Proj, rt: *const Route, x: mtl.Buffer, out: mtl.Buffer, rows: usize, dst: []u16) void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    @memset(out.slice(u8, rows * TOPK * p.n * 2), 0xff);
    const cb = q.commandBuffer();
    const enc = cb.compute(.serial);
    encode(enc, k, up, p, rt, 0, x, out, rows);
    enc.end();
    cb.commit();
    cb.wait();
    @memcpy(dst, out.slice(u16, rows * TOPK * p.n));
}

/// Median GPU us a layer over LAYERS layers a command buffer; timed twice, the second kept (first runs pay first use).
fn time(q: mtl.Queue, k: Kernel, up: bool, p: *const Proj, rt: *const Route, x: mtl.Buffer, out: mtl.Buffer, rows: usize, reps: usize) f64 {
    var t: [32]f64 = undefined;
    for (0..2) |_| {
        for (0..reps + 2) |i| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = q.commandBuffer();
            const enc = cb.compute(.serial);
            for (0..LAYERS) |l| encode(enc, k, up, p, rt, l, x, out, rows);
            enc.end();
            cb.commit();
            cb.wait();
            if (i >= 2) t[i - 2] = cb.gpuSeconds() * 1e6 / LAYERS;
        }
    }
    std.mem.sort(f64, t[0..reps], {}, std.sort.asc(f64));
    return t[reps / 2];
}

/// The engine's one-row tables for row r of layer 0: its 6 experts in route order, pair k of the row as member k.
fn oneRow(d: mtl.Device, rt: *const Route, r: usize) !Route {
    var o: Route = undefined;
    o.distinct = TOPK;
    o.uids[0] = try d.buffer(E * 4, opts);
    o.start[0] = try d.buffer(E * 4, opts);
    o.count[0] = try d.buffer(E * 4, opts);
    o.members[0] = try d.buffer(8 * 4, opts);
    for (0..TOPK) |k| {
        o.uids[0].slice(u32, E)[k] = rt.ids0[r * TOPK + k];
        o.start[0].slice(i32, E)[k] = @intCast(k);
        o.count[0].slice(i32, E)[k] = 1;
        o.members[0].slice(i32, 8)[k] = @intCast(k);
    }
    o.ucount[0] = TOPK;
    return o;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const args = try init.minimal.args.toSlice(a);
    const reps: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 7;
    if (reps < 1 or reps > 30) return error.BadReps;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const q = try device.queue();
    std.debug.print("device {s}; compiling\n", .{device.name()});
    var engine = try tf.nemotron.kernels.load(gpa, device);
    defer engine.deinit();
    var src: std.ArrayList(u8) = .empty;
    try src.appendSlice(a, sources.nemotron_experts);
    try src.appendSlice(a, mix_source);
    for (variants) |v| {
        if (builtIn(v)) continue;
        const kname = if (v.mix) "tf_experts_mix" else if (v.words) "tf_experts_w" else "tf_experts_rows";
        try src.print(a, "template [[host_name(\"p_up_{s}\")]] [[kernel]] decltype({s}<2688, 1856, 64, {d}, {d}, 6, {d}, true>) {s}<2688, 1856, 64, {d}, {d}, 6, {d}, true>;\n", .{ v.name, kname, v.rps, v.sg, v.mb, kname, v.rps, v.sg, v.mb });
        try src.print(a, "template [[host_name(\"p_down_{s}\")]] [[kernel]] decltype({s}<1856, 2688, 64, {d}, {d}, 6, {d}, false>) {s}<1856, 2688, 64, {d}, {d}, 6, {d}, false>;\n", .{ v.name, kname, v.rps, v.sg, v.mb, kname, v.rps, v.sg, v.mb });
    }
    const lib = try mtl.Library.fromSource(device, src.items, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var kernels: [variants.len]Kernel = undefined;
    for (variants, &kernels) |v, *k| {
        const up_name = if (!builtIn(v)) try std.fmt.allocPrint(a, "p_up_{s}", .{v.name}) else if (v.words) "tf_xup_w3" else try std.fmt.allocPrint(a, "tf_xup_rows{d}", .{v.mb});
        const down_name = if (!builtIn(v)) try std.fmt.allocPrint(a, "p_down_{s}", .{v.name}) else if (v.words) "tf_xdown_w3" else try std.fmt.allocPrint(a, "tf_xdown_rows{d}", .{v.mb});
        k.* = .{ .up = try mtl.Pipeline.init(device, lib, up_name, false), .down = try mtl.Pipeline.init(device, lib, down_name, false), .rps = v.rps, .sg = v.sg };
    }
    const window: Kernel = .{ .up = engine.get("tf_xup_rows2"), .down = engine.get("tf_xdown_rows2"), .rps = 4, .sg = 2 };
    // the one-row forward's kernels: the generated ones at (4, 2), then the engine's geometries
    const one = [_]struct { name: []const u8, k: Kernel }{
        .{ .name = "one 4x2", .k = .{ .up = engine.get("expert_up"), .down = engine.get("expert_down"), .rps = 4, .sg = 2 } },
        .{ .name = "one 2x2", .k = .{ .up = engine.get("tf_xup_2_2"), .down = engine.get("tf_xdown_2_2"), .rps = 2, .sg = 2 } },
        .{ .name = "one 8x2", .k = .{ .up = engine.get("tf_xup_8_2"), .down = engine.get("tf_xdown_8_2"), .rps = 8, .sg = 2 } },
        .{ .name = "one 4x4", .k = .{ .up = engine.get("tf_xup_4_4"), .down = engine.get("tf_xdown_4_4"), .rps = 4, .sg = 4 } },
        .{ .name = "one 8x4", .k = .{ .up = engine.get("tf_xup_8_4"), .down = engine.get("tf_xdown_8_4"), .rps = 8, .sg = 4 } },
        .{ .name = "one 4x1", .k = .{ .up = engine.get("tf_xup_4_1"), .down = engine.get("tf_xdown_4_1"), .rps = 4, .sg = 1 } },
        .{ .name = "one 2x4", .k = .{ .up = engine.get("tf_xup_2_4"), .down = engine.get("tf_xdown_2_4"), .rps = 2, .sg = 4 } },
    };
    var prng: std.Random.DefaultPrng = .init(20261004);
    const rng = prng.random();
    std.debug.print("weights: {d} layers x 128 experts, synthetic 4-bit\n", .{LAYERS});
    const up = try Proj.init(device, 1856, 2688, rng);
    const down = try Proj.init(device, 2688, 1856, rng);
    const wmax = widths[widths.len - 1];
    const x = try device.buffer(wmax * 2688 * 2, opts);
    for (x.slice(u16, wmax * 2688)) |*v| v.* = bf16(rng.floatNorm(f32));
    const act = try device.buffer(wmax * TOPK * 1856 * 2, opts);
    for (act.slice(u16, wmax * TOPK * 1856)) |*v| v.* = bf16(@abs(rng.floatNorm(f32)) * 0.5);
    const x1 = try device.buffer(2688 * 2, opts);
    const act1 = try device.buffer(TOPK * 1856 * 2, opts);
    const out = try device.buffer(wmax * TOPK * 2688 * 2, opts);
    const ref = try a.alloc(u16, wmax * TOPK * 2688);
    const got = try a.alloc(u16, wmax * TOPK * 2688);
    var all_exact = true;
    for (widths) |w| {
        const rt = try Route.init(device, w, rng);
        std.debug.print("\n{d} rows: {d:.1} distinct experts a layer ({d:.2} members an expert); us a layer, up + down:\n", .{ w, @as(f64, @floatFromInt(rt.distinct)) / LAYERS, @as(f64, @floatFromInt(w * TOPK * LAYERS)) / @as(f64, @floatFromInt(rt.distinct)) });
        const ref_k = if (w == 1) one[0].k else window;
        const n_cands = if (w == 1) one.len else variants.len + 1;
        for (0..n_cands) |vi| {
            const k = if (w == 1) one[vi].k else if (vi == 0) window else kernels[vi - 1];
            const name = if (w == 1) one[vi].name else if (vi == 0) "engine" else variants[vi - 1].name;
            var same = true;
            for ([_]bool{ true, false }) |is_up| {
                const p = if (is_up) &up else &down;
                const xin = if (is_up) x else act;
                const n = w * TOPK * p.n;
                runOnce(q, ref_k, is_up, p, &rt, xin, out, w, ref[0..n]);
                runOnce(q, k, is_up, p, &rt, xin, out, w, got[0..n]);
                same = same and std.mem.eql(u16, ref[0..n], got[0..n]);
            }
            all_exact = all_exact and same;
            const tu = time(q, k, true, &up, &rt, x, out, w, reps);
            const td = time(q, k, false, &down, &rt, act, out, w, reps);
            const gbs = @as(f64, @floatFromInt(rt.distinct)) / LAYERS * 2 * 2.8e6 / ((tu + td) * 1e3);
            std.debug.print("  {s:<10} bits {s} | up {d:8.1} down {d:8.1} sum {d:8.1} | {d:4.0} GB/s | 23 layers {d:6.2} ms\n", .{ name, if (same) "==" else "DIFFER", tu, td, tu + td, gbs, (tu + td) * 23 / 1e3 });
        }
        if (w != 8) continue;
        // each row of the window alone through the one-row forward's kernel: the same bits for its 6 pairs
        var rows_ok = true;
        for ([_]bool{ true, false }) |is_up| {
            const p = if (is_up) &up else &down;
            runOnce(q, window, is_up, p, &rt, if (is_up) x else act, out, w, ref[0 .. w * TOPK * p.n]);
            for (0..w) |r| {
                const o = try oneRow(device, &rt, r);
                if (is_up) @memcpy(x1.slice(u16, 2688), x.slice(u16, wmax * 2688)[r * 2688 .. (r + 1) * 2688]) else @memcpy(act1.slice(u16, TOPK * 1856), act.slice(u16, wmax * TOPK * 1856)[r * TOPK * 1856 .. (r + 1) * TOPK * 1856]);
                runOnce(q, one[0].k, is_up, p, &o, if (is_up) x1 else act1, out, 1, got[0 .. TOPK * p.n]);
                if (!std.mem.eql(u16, got[0 .. TOPK * p.n], ref[r * TOPK * p.n .. (r + 1) * TOPK * p.n])) rows_ok = false;
            }
        }
        all_exact = all_exact and rows_ok;
        std.debug.print("  each row alone through the one-row kernel == its rows in the 8-row window: {s}\n", .{if (rows_ok) "exact" else "VARIANT"});
    }
    std.debug.print("{s}\n", .{if (all_exact) "every kernel: bits == the engine's, rows exact" else "SOMETHING DIFFERS (see above)"});
    if (!all_exact) std.process.exit(1);
}
