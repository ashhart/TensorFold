//! tf-cuda-test qwen35-tp-cut PACK [STEP]: a rank's cut of the 27B's EXL3 weights at the whole plan, one GPU's bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen35 = @import("qwen35");
const check = @import("check.zig");

const exl3 = cuda.exl3;
const cs = qwen35.shard;
const plans = qwen35.shard_plan;
const admit = qwen35.exl3_admit;
const Part = plans.Part;

const world = 2;
const most = 1536; // the engine's default prompt chunk
/// Rows: a decode row, 2, 16, then 17 to 129 across linear_wc, the mid-M groups (36 is one) and 128-row blocks.
const counts = [_]usize{ 1, 2, 16, 17, 36, 48, 49, 128, 129, most };
/// The head reads the last rows of a chunk, so it stops at the first count past a 128-row block.
const head_most = 129;
const root = "model.language_model.layers.{d}.";

/// A weight of a layer by its tensor name, and the set of weights one input feeds (the engine groups those it can).
const Member = struct { part: Part, name: []const u8 };
const m_q: Member = .{ .part = .q_proj, .name = "self_attn.q_proj" };
const m_k: Member = .{ .part = .k_proj, .name = "self_attn.k_proj" };
const m_v: Member = .{ .part = .v_proj, .name = "self_attn.v_proj" };
const m_o: Member = .{ .part = .o_proj, .name = "self_attn.o_proj" };
const m_qkv: Member = .{ .part = .in_proj_qkv, .name = "linear_attn.in_proj_qkv" };
const m_z: Member = .{ .part = .in_proj_z, .name = "linear_attn.in_proj_z" };
const m_out: Member = .{ .part = .out_proj, .name = "linear_attn.out_proj" };
const m_gate: Member = .{ .part = .gate_proj, .name = "mlp.gate_proj" };
const m_up: Member = .{ .part = .up_proj, .name = "mlp.up_proj" };
const m_down: Member = .{ .part = .down_proj, .name = "mlp.down_proj" };
const m_head: Member = .{ .part = .lm_head, .name = "lm_head" };
const attention = [_][]const Member{ &.{ m_q, m_k, m_v }, &.{m_o} };
const delta = [_][]const Member{ &.{m_qkv}, &.{m_z}, &.{m_out} };
const mlp = [_][]const Member{ &.{ m_gate, m_up }, &.{m_down} };

/// What a run counted, per part: weights, the whole plan and a half's own, and how often the own-plan columns matched.
const Tally = struct {
    weights: std.EnumArray(Part, usize) = .initFill(0),
    own_same: std.EnumArray(Part, usize) = .initFill(0),
    own_diff: std.EnumArray(Part, usize) = .initFill(0),
    whole: std.EnumArray(Part, [2]u32) = .initFill(.{ 0, 0 }),
    own: std.EnumArray(Part, [2]u32) = .initFill(.{ 0, 0 }),
    checks: usize = 0,
    groups: usize = 0,
    /// By row count: pinned views equal, own-plan columns equal and different, and the whole run's kernels.
    pinned_equal: [counts.len]usize = @splat(0),
    rows_same: [counts.len]usize = @splat(0),
    rows_diff: [counts.len]usize = @splat(0),
    kernel: [2][counts.len][3]usize = @splat(@splat(@splat(0))), // by sweep: linear, linear_wc and mid-M launches
};

const Ctx = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    dense: *exl3.Dense,
    ck: *core.Checkpoint,
    rng: std.Random,
    t: *Tally,
};

/// A host buffer on the device, freed with `bufs`.
fn upload(c: *Ctx, bufs: *std.ArrayList(cuda.DeviceBuffer), bytes: []const u8) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.fromHost(c.gpu.d, bytes);
    try bufs.append(c.gpu.gpa, b);
    return b;
}

/// The whole weight `base` of the pack on the device, as the family's loader places it.
fn load(c: *Ctx, bufs: *std.ArrayList(cuda.DeviceBuffer), base: []const u8, d: plans.Dims) !exl3.Weight {
    const g = admit.denseGroup(c.ck, base, d.n, d.k) catch |e| {
        std.debug.print("FAIL {s}: not the 27B's {d} x {d} EXL3 weight ({s})\n", .{ base, d.n, d.k, @errorName(e) });
        return error.TestFailed;
    };
    const st = exl3.strides(d.k, d.n, g.halves, false);
    return .{
        .trellis = (try upload(c, bufs, g.trellis.bytes)).ptr,
        .suh = (try upload(c, bufs, g.suh.bytes)).ptr,
        .svh = (try upload(c, bufs, g.svh.bytes)).ptr,
        .k = @intCast(d.k),
        .n = @intCast(d.n),
        .halves = g.halves,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
}

/// One call as the engine makes it: decode rows to 16, the prompt entry above.
fn launch(c: *Ctx, w: exl3.Weight, x: u64, m: usize, y: u64, dt: exl3.DType) !void {
    if (m <= 16) return c.dense.decode(c.s, w, x, .bf16, m, y, dt);
    return c.dense.prompt(c.s, w, x, .bf16, m, y, dt);
}

/// `len` bytes of the buffer, after the stream is done and before the next call fills it again.
fn back(c: *Ctx, a: std.mem.Allocator, b: cuda.DeviceBuffer, len: usize) ![]u8 {
    const host = try a.alloc(u8, len);
    try c.s.synchronize();
    try b.download(0, host);
    return host;
}

/// Columns [lo, lo + len) of each of `m` rows of `whole` (n columns of `es` bytes): a rank's view of the rows.
fn kept(a: std.mem.Allocator, whole: []const u8, m: usize, n: usize, es: usize, span: cs.Span) ![]u8 {
    const w = span.len() * es;
    const rows = try a.alloc(u8, m * w);
    for (0..m) |row| @memcpy(rows[row * w ..][0..w], whole[(row * n + span.lo) * es ..][0..w]);
    return rows;
}

/// Unit-normal bf16 rows; the first row's first three columns NaN, +inf and -inf when `nonfinite`.
fn inputs(a: std.mem.Allocator, rng: std.Random, len: usize, nonfinite: bool) ![]u16 {
    const x = try a.alloc(u16, len);
    for (x) |*h| h.* = core.bf16_round.bits(rng.floatNorm(f32));
    if (nonfinite) x[0..3].* = .{ 0x7fc0, 0x7f80, 0xff80 };
    return x;
}

/// A weight's views of one rank: each span's columns at the whole plan, and the same columns planned on their own.
const Views = struct { pinned: [3]exl3.Weight, own: [3]exl3.Weight, spans: [3]cs.Span, len: usize };

/// Every comparison of one input set: the whole weights against each rank's views, at every row count and both outputs.
fn setChecks(c: *Ctx, members: []const Member, prefix: []const u8) !void {
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(c.gpu.gpa);
    }
    const shape = cs.qwen27;
    var whole: [3]exl3.Weight = undefined;
    var plan_differs: [3]bool = @splat(false);
    var views: [3][world]Views = undefined;
    var widest: usize = 0;
    for (members, 0..) |m, i| {
        var name_buf: [160]u8 = undefined;
        const d = plans.dims(shape, m.part);
        whole[i] = try load(c, &bufs, try std.fmt.bufPrint(&name_buf, "{s}{s}", .{ prefix, m.name }), d);
        widest = @max(widest, d.n);
        for (0..world) |rank| {
            const cut = plans.cutOf(m.part, shape, try cs.Share.of(shape, rank, world));
            var vw: Views = .{ .pinned = undefined, .own = undefined, .spans = cut.spans, .len = cut.len };
            for (cut.items(), 0..) |span, j| {
                vw.pinned[j] = try plans.view(whole[i], span);
                vw.own[j] = try whole[i].columns(@intCast(span.lo), @intCast(span.len()));
                try check.expect(vw.pinned[j].sk == cut.sk and vw.pinned[j].wk == cut.wk, "{s}: the view keeps the whole's {d} splits of {d} warps", .{ m.name, cut.sk, cut.wk });
                const own = exl3.planOf(vw.own[j]);
                if (own.sk != cut.sk or own.wk != cut.wk) plan_differs[i] = true;
                c.t.whole.set(m.part, .{ cut.sk, cut.wk });
                c.t.own.set(m.part, .{ own.sk, own.wk });
            }
            views[i][rank] = vw;
        }
        c.t.weights.getPtr(m.part).* += 1;
    }
    defer c.dense.lin.use_wc = true;
    const rows_most: usize = if (members[0].part == .lm_head) head_most else most;
    const big = try cuda.DeviceBuffer.alloc(c.gpu.d, rows_most * widest * 4);
    try bufs.append(c.gpu.gpa, big);
    const ys = try cuda.DeviceBuffer.alloc(c.gpu.d, rows_most * widest * 4);
    try bufs.append(c.gpu.gpa, ys);
    var group_ys: [3]cuda.DeviceBuffer = undefined;
    for (0..members.len) |i| {
        group_ys[i] = try cuda.DeviceBuffer.alloc(c.gpu.d, 16 * plans.dims(shape, members[i].part).n * 4);
        try bufs.append(c.gpu.gpa, group_ys[i]);
    }
    const kk = plans.dims(shape, members[0].part).k;
    var witnessed: [3]bool = @splat(false); // an own plan that differs from the whole's must give other bytes in some run
    for ([_]bool{ false, true }) |nonfinite| {
        var arena = std.heap.ArenaAllocator.init(c.gpu.gpa);
        defer arena.deinit();
        const x = try upload(c, &bufs, std.mem.sliceAsBytes(try inputs(arena.allocator(), c.rng, rows_most * kk, nonfinite)));
        // two sweeps: linear_wc allowed as the engine has it, then off so the mid-M kernels take the rows it would
        for (0..2 * counts.len) |pass| {
            const ci = pass % counts.len;
            const m = counts[ci];
            c.dense.lin.use_wc = pass < counts.len;
            if (m > rows_most or (pass >= counts.len and (m <= 16 or m > 128))) continue;
            for ([_]exl3.DType{ .bf16, .f32 }) |dt| {
                var iter = std.heap.ArenaAllocator.init(c.gpu.gpa);
                defer iter.deinit();
                const a = iter.allocator();
                const es: usize = if (dt == .f32) 4 else 2;
                var want: [3][]u8 = undefined;
                for (members, 0..) |mem, i| {
                    const n = plans.dims(shape, mem.part).n;
                    try big.fill8(0xff, c.s.handle); // a launch that writes nothing fails its comparison
                    if (dt == .bf16 and !nonfinite) {
                        var at: usize = 0; // Dense takes 128 rows a launch
                        while (at < m) : (at += 128) {
                            const how = exl3.pick(whole[i], @min(m - at, 128), c.dense.lin.midm, c.dense.lin.use_wc, c.dense.lin.min_rows).?;
                            const slot: usize = switch (how) {
                                .linear => 0,
                                .wc => 1,
                                .mpg => 2,
                            };
                            c.t.kernel[pass / counts.len][ci][slot] += 1;
                        }
                    }
                    try launch(c, whole[i], x.ptr, m, big.ptr, dt);
                    want[i] = try back(c, a, big, m * n * es);
                    for (0..world) |rank| {
                        const vw = views[i][rank];
                        for (vw.spans[0..vw.len], 0..) |span, j| {
                            const exp = try kept(a, want[i], m, n, es, span);
                            try ys.fill8(0xff, c.s.handle);
                            try launch(c, vw.pinned[j], x.ptr, m, ys.ptr, dt);
                            const what = try std.fmt.allocPrint(a, "{s} rank {d} span {d}, {d} rows, {s}, nonfinite={any}", .{ mem.name, rank, j, m, @tagName(dt), nonfinite });
                            try check.sameBytes(what, try back(c, a, ys, exp.len), exp);
                            c.t.checks += 1;
                            c.t.pinned_equal[ci] += 1;
                            try ys.fill8(0xff, c.s.handle);
                            try launch(c, vw.own[j], x.ptr, m, ys.ptr, dt);
                            const alike = std.mem.eql(u8, try back(c, a, ys, exp.len), exp);
                            const tally = if (alike) &c.t.own_same else &c.t.own_diff;
                            tally.getPtr(mem.part).* += 1;
                            (if (alike) &c.t.rows_same else &c.t.rows_diff)[ci] += 1;
                            if (!alike and dt == .f32 and !nonfinite) witnessed[i] = true;
                        }
                    }
                }
                if (members.len < 2 or m > 16) continue;
                for (0..world) |rank| {
                    var vs: [3]exl3.Weight = undefined;
                    var outs: [3]u64 = undefined;
                    for (members, 0..) |_, i| {
                        vs[i] = views[i][rank].pinned[0];
                        outs[i] = group_ys[i].ptr;
                        try group_ys[i].fill8(0xff, c.s.handle);
                    }
                    try c.dense.decodeGroup(c.s, vs[0..members.len], x.ptr, .bf16, m, outs[0..members.len], dt);
                    if (exl3.groupable(vs[0..members.len], m)) c.t.groups += 1;
                    for (members, 0..) |mem, i| {
                        const n = plans.dims(shape, mem.part).n;
                        const exp = try kept(a, want[i], m, n, es, views[i][rank].spans[0]);
                        const what = try std.fmt.allocPrint(a, "{s} rank {d} in a group of {d}, {d} rows, {s}, nonfinite={any}", .{ mem.name, rank, members.len, m, @tagName(dt), nonfinite });
                        try check.sameBytes(what, try back(c, a, group_ys[i], exp.len), exp);
                        c.t.checks += 1;
                    }
                }
            }
        }
    }
    for (members, 0..) |m, i| {
        try check.expect(!plan_differs[i] or witnessed[i], "{s}: the half's own plan gives the whole's bytes, so the pin cannot be shown to matter", .{m.name});
    }
}

pub fn run(gpu: check.Gpu, dir: []const u8, step_arg: []const u8) !void {
    const step = try std.fmt.parseInt(usize, step_arg, 10);
    if (step == 0) return error.BadArguments;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var dense = try exl3.Dense.load(gpu.d);
    defer dense.unload();
    var ck = try core.Checkpoint.openModel(gpu.gpa, gpu.io, dir);
    defer ck.close();
    var prng = std.Random.DefaultPrng.init(0x7f27_c07);
    var t: Tally = .{};
    var c: Ctx = .{ .gpu = gpu, .s = s, .dense = &dense, .ck = &ck, .rng = prng.random(), .t = &t };
    try setChecks(&c, &.{m_head}, "");
    var layer: usize = 0;
    var kept_layers: usize = 0;
    while (true) : (layer += 1) {
        var buf: [96]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&buf, root, .{layer});
        var probe: [128]u8 = undefined;
        if (!ck.has(try std.fmt.bufPrint(&probe, "{s}mlp.gate_proj.trellis", .{prefix}))) break;
        if (layer % step != 0) continue;
        kept_layers += 1;
        const attn = ck.has(try std.fmt.bufPrint(&probe, "{s}self_attn.q_proj.trellis", .{prefix}));
        const sets: []const []const Member = if (attn) &attention else &delta;
        for (sets) |set| try setChecks(&c, set, prefix);
        for (&mlp) |set| try setChecks(&c, set, prefix);
    }
    try check.expect(kept_layers > 0, "the pack has no layer to check", .{});
    for (std.enums.values(Part)) |part| {
        const n = t.weights.get(part);
        if (n == 0) continue;
        const whole = t.whole.get(part);
        const own = t.own.get(part);
        std.debug.print("RESULT qwen35-tp-cut {s}: {d} weights, whole plan {d} splits x {d} warps, a half's own {d} x {d}; own-plan columns equal the whole run's in {d} and differ in {d} comparisons\n", .{ @tagName(part), n, whole[0], whole[1], own[0], own[1], t.own_same.get(part), t.own_diff.get(part) });
    }
    for (counts, 0..) |m, ci| {
        const on = t.kernel[0][ci];
        const off = t.kernel[1][ci];
        std.debug.print("RESULT qwen35-tp-cut {d} rows: whole run's launches (bf16, finite) linear {d}, linear_wc {d}, mid-M {d}; linear_wc off: linear {d}, mid-M {d}; pinned views (linear) equal in {d}; own-plan columns equal in {d}, differ in {d}\n", .{ m, on[0], on[1], on[2], off[0], off[2], t.pinned_equal[ci], t.rows_same[ci], t.rows_diff[ci] });
    }
    check.pass("qwen35-tp-cut: {d} layers, {d} byte comparisons (bf16 and fp32 outputs, finite and non-finite rows), {d} projection groups; " ++
        "each rank's columns at the whole plan equal one run's bytes at 1 to {d} rows (the head to {d})", .{ kept_layers, t.checks, t.groups, most, head_most });
}
