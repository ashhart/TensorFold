//! GLM-5.3-Flash's products at widths its generated kernels do not read (glm_rows.metal): a launch a stack member.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const affine = @import("affine.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const kernels = @import("kernels.zig");
pub const gather = @import("gather.zig");
const Affine = affine.Affine;
const Ref = affine.Ref;

const bit_widths = affine.row_bits;
const nw = bit_widths.len;

/// Whether `q` takes the generated kernels: one width, 4 bits in groups of 64 (the uniform checkpoints' launches).
pub fn native(q: Affine) bool {
    return q.more == 0 and q.bits == 4 and q.group == 64;
}

pub const RowDims = extern struct { k: i32, n: i32, group: i32, rows: i32, heads: i32 = 1, ldx: i32, hsx: i32 = 0, ldo: i32, hso: i32 = 0, hsw: i32 = 0 };
pub const PickDims = extern struct { k: i32, n: i32, group: i32, slots: i32, repeat: i32, experts: i32, ldo: i32, lim: f32 };

pub const Pipes = struct {
    rows: [nw]mtl.Pipeline, // dense, strided, by head
    ids: [nw]mtl.Pipeline, // a slot through its expert
    glu_ids: [nw]mtl.Pipeline, // gate and up through it, SwiGLU'd
    sorted: [nw]mtl.Pipeline, // an expert's sorted pairs (prompt chunks)
    glu_sorted: [nw]mtl.Pipeline,
    moe: [nw]mtl.Pipeline, // routed experts by pick at each width and the 8-bit shared expert, one launch
    moe_glu: [nw]mtl.Pipeline, // the same through gate and up, SwiGLU'd
    gather: [2]mtl.Pipeline, // four gathered rows sharing the sorted row arithmetic
    bf16: mtl.Pipeline, // an unquantized matrix (the MTP head's eh_proj, scalar path; decode-side path keeps this)
    bf16_nax: ?mtl.Pipeline, // bf16 GEMM on tensor units: prefill-side eh_proj replacement for `bf16`
    kda_pre: mtl.Pipeline, // kda_rows reading f_b's and g_b's products made apart (any width but 4/64)
    kda_fg: [nw]mtl.Pipeline, // kda_rows making f_b's and g_b's dots of one width (g64) as glm_rows does: .pre's bits
    embed: [2]mtl.Pipeline, // the embedding at 6 and 8 bits, groups of 64
    mm8: affine_mm.Pipes, // prompt chunks' products of one width at 8 bits in groups of 64 (core/affine_mm.zig)
    mm2: affine_mm.Pipes, // and at 2 bits in groups of 64: oQ2's routed experts, gathered
    mm5: affine_mm.Pipes, // and at 5 and 6 bits (oQ4's), split stacks' members too
    mm6: affine_mm.Pipes,
    zeros: mtl.Buffer, // expert ids all 0: a dense GLU through the by-ids kernel (the shared expert)
    hash: u64, // every source compiled here

    pub fn deinit(p: *Pipes) void {
        inline for (.{
            &p.gather,
            &p.rows,
            &p.ids,
            &p.glu_ids,
            &p.sorted,
            &p.glu_sorted,
            &p.moe,
            &p.moe_glu,
            &p.embed,
            &p.kda_fg,
        }) |set| for (set) |x| x.deinit();
        p.bf16.deinit();
        if (p.bf16_nax) |x| x.deinit();
        p.kda_pre.deinit();
        p.zeros.deinit();
        inline for (.{ p.mm8, p.mm2, p.mm5, p.mm6 }) |set| for (set) |x| x.deinit();
    }
};

/// kda_rows's f_b dot read from a product made apart (a row's sum in its first lane of four, zeros in the rest: exact).
const pre_f = .{ "result = quad_dot<FB, PER>(x, wb, s, bb);", "result = qlid == 0u ? float(((device const bfloat*)FBW)[(size_t)r * W + row]) : 0.0f;" };
const pre_g = .{ "result = quad_dot<GB, PER>(x, wb, s, bb);", "result = qlid == 0u ? float(((device const bfloat*)GBW)[(size_t)r * W + row]) : 0.0f;" };

/// kda_rows's f_b and g_b dots by quads of lanes, cut (from its block's start to the barrier after it) for `fg_block`.
const fg_cut = .{
    "    {\n      constexpr int PER = D / 4;",
    "\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint idx = tid; idx < 3u * (uint)D;",
};
/// The head's f_b and g_b rows as glm_rows sums them (core tf_rp_rows, 4 rows a simdgroup, then bf16): .pre's values.
const fg_block =
    \\    for (uint t0 = simdgroup_index_in_threadgroup * 4u; t0 < 2u * (uint)D; t0 += (uint)TY * 4u) {
    \\      const bool f = t0 < (uint)D;
    \\      const uint d0 = f ? t0 : t0 - (uint)D;
    \\      const size_t row = (size_t)h * D + d0;
    \\      float acc[1][4];
    \\      if (f) tf_rp_rows<4, 1, FB, false>((const device uint8_t*)FBW + row * (D * FB / 8), FBS + row * (D / 64),
    \\                                         FBB + row * (D / 64), prow + FA, D, 64, lane, acc);
    \\      else tf_rp_rows<4, 1, GB, false>((const device uint8_t*)GBW + row * (D * GB / 8), GBS + row * (D / 64),
    \\                                       GBB + row * (D / 64), prow + GA, D, 64, lane, acc);
    \\      if (lane == 0u)
    \\        for (int j = 0; j < 4; j++) (f ? sa : sgate)[d0 + j] = float(bfloat(acc[0][j]));
    \\    }
;
/// core/row_projection.metal's row sums only (its kernels and instances cut).
const rp_sums = blk: {
    @setEvalBranchQuota(100_000);
    const r = sources.core_row_projection;
    break :blk r[0..std.mem.indexOf(u8, r, "// A value out: bf16").?];
};

/// glm_glue.metal's bfloat overloads and sigmoid_fast, the text glm_swiglu compiles: glm_rows's SwiGLU takes its bits.
pub const glue_math = blk: {
    @setEvalBranchQuota(100_000);
    const g = sources.glm_glue;
    const a = std.mem.indexOf(u8, g, "// MLX's bfloat math overloads").?;
    const b = a + std.mem.indexOf(u8, g[a..], "// The dense MLP's SwiGLU").?;
    break :blk "typedef bfloat bfloat16_t;\n" ++ g[a..b];
};

pub const gather_source = sources.core_row_projection ++ glue_math ++ sources.glm_rows ++ sources.glm_gather;

/// The pipelines built so far: a failed compile releases them all.
const Made = struct {
    list: [nw * 8 + 7 + 4 * affine_mm.names.len]mtl.Pipeline = undefined,
    n: usize = 0,
    released: usize = 0,

    fn add(m: *Made, device: mtl.Device, l: mtl.Library, name: []const u8) !mtl.Pipeline {
        m.list[m.n] = try mtl.Pipeline.init(device, l, name, false);
        m.n += 1;
        return m.list[m.n - 1];
    }

    fn drop(m: *Made) void {
        for (m.list[0..m.n]) |p| p.deinit();
        m.released = m.n;
        m.n = 0;
    }
};

fn lib(m: *Made, device: mtl.Device, src: []const u8, names: []const [:0]const u8, out: []mtl.Pipeline) !void {
    const l = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer l.deinit();
    for (names, out) |n, *p| p.* = try m.add(device, l, n);
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Pipes {
    var made: Made = .{};
    return build(gpa, device, comptime kernels.generatedSource("kda_rows"), &made);
}

fn build(gpa: std.mem.Allocator, device: mtl.Device, kda: []const u8, made: *Made) !Pipes {
    errdefer made.drop();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var p: Pipes = undefined;
    var h = std.hash.Wyhash.init(0x77);
    const rows_src = try std.mem.concat(gpa, u8, &.{ sources.core_row_projection, glue_math, sources.glm_rows, sources.glm_gather });
    defer gpa.free(rows_src);
    h.update(rows_src);
    {
        const l = try mtl.Library.fromSource(device, rows_src, mtl.CompileOptions.mlx());
        defer l.deinit();
        inline for (bit_widths, 0..) |b, i| {
            const q = std.fmt.comptimePrint("_q{d}", .{b});
            p.rows[i] = try made.add(device, l, "glm_rows" ++ q);
            p.ids[i] = try made.add(device, l, "glm_rows_ids" ++ q);
            p.glu_ids[i] = try made.add(device, l, "glm_rows_glu_ids" ++ q);
            p.sorted[i] = try made.add(device, l, "glm_rows_sorted" ++ q);
            p.glu_sorted[i] = try made.add(device, l, "glm_rows_glu_sorted" ++ q);
            p.moe[i] = try made.add(device, l, "glm_moe_rows" ++ q ++ "_s8");
            p.moe_glu[i] = try made.add(device, l, "glm_moe_rows_glu" ++ q ++ "_s8");
        }
        p.bf16 = try made.add(device, l, "glm_rows_bf16");
        for (gather.names, &p.gather) |name, *pipe| pipe.* = try made.add(device, l, name);
        p.bf16_nax = made.add(device, l, "glm_rows_bf16_nax") catch blk: {
            std.log.warn("glm: the eh_proj tile kernel cannot build here; eh_proj serves on the scalar path", .{});
            break :blk null;
        };
    }
    const pre_a = try std.mem.replaceOwned(u8, gpa, kda, pre_f[0], pre_f[1]); // its own instance, under its own name
    defer gpa.free(pre_a);
    const pre_src = try std.mem.replaceOwned(u8, gpa, pre_a, pre_g[0], pre_g[1]);
    defer gpa.free(pre_src);
    if (std.mem.indexOf(u8, pre_src, pre_f[0]) != null or std.mem.indexOf(u8, pre_src, pre_g[0]) != null or std.mem.indexOf(u8, pre_src, pre_f[1]) == null) return error.KernelCompile;
    h.update(pre_src);
    try lib(made, device, pre_src, &.{comptime kernels.generatedFunction("kda_rows")}, @as(*[1]mtl.Pipeline, &p.kda_pre));
    const a = std.mem.indexOf(u8, kda, fg_cut[0]) orelse return error.KernelCompile;
    const z = a + (std.mem.indexOf(u8, kda[a..], fg_cut[1]) orelse return error.KernelCompile);
    const own = z + (std.mem.lastIndexOf(u8, kda[z..], "\ntemplate [[host_name(") orelse return error.KernelCompile);
    const fg_src = try std.mem.concat(
        gpa,
        u8,
        &.{ rp_sums, kda[0..a], fg_block, kda[z..own], fg_instances },
    ); // not its own
    defer gpa.free(fg_src);
    h.update(fg_src);
    try lib(made, device, fg_src, &fg_names, &p.kda_fg);
    h.update(sources.ops_embed_norm);
    try lib(made, device, sources.ops_embed_norm, &.{ "tf_embed_b6_g64", "tf_embed_b8_g64" }, &p.embed);
    for ([_]u8{ 8, 2, 5, 6 }, [_]*affine_mm.Pipes{ &p.mm8, &p.mm2, &p.mm5, &p.mm6 }) |bits, out| {
        const src = try affine_mm.source(device, gpa, .{ .bits = bits, .group = 64 });
        defer gpa.free(src);
        h.update(src);
        try lib(made, device, src, &affine_mm.names, out);
    }
    p.zeros = try device.buffer(64 * 4, mtl.ResourceOptions.shared);
    @memset(p.zeros.slice(i32, 64), 0);
    p.hash = h.final();
    return p;
}

/// kda_rows with f_b and g_b at one width a pipeline (`fg_block`), under the names glm_kda_fg_q<bits>.
const fg_names = blk: {
    var n: [nw][:0]const u8 = undefined;
    for (bit_widths, 0..) |b, i| n[i] = std.fmt.comptimePrint("glm_kda_fg_q{d}", .{b});
    break :blk n;
};
const fg_instances = blk: {
    var s: []const u8 = "";
    for (bit_widths, fg_names) |b, name| s = s ++ kernels.instance(
        "kda_rows",
        std.fmt.comptimePrint("64, 128, 4, 32, {d}, {d}", .{ b, b }),
        name,
    );
    break :blk s;
};

fn bind(e: mtl.ComputeEncoder, first: usize, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, first + i);
}

fn wBytes(n: u32, k: u32, q: Affine) usize {
    return affine.wBytes(n, k, q.width());
}

fn sBytes(n: u32, k: u32, q: Affine) usize {
    return affine.sBytes(n, k, q.width());
}

fn at(bits: u8) usize {
    return std.mem.indexOfScalar(u8, &bit_widths, bits).?;
}

/// Rows of x (`ldx` apart) through a matrix into out (`ldo` apart).
pub const Io = struct { in: Ref, ldx: u32, out: Ref, ldo: u32, rows: u32 };

fn launch(p: *const Pipes, e: mtl.ComputeEncoder, w: [3]Ref, bits: u8, d: RowDims, in: Ref, out: Ref) void {
    e.setPipeline(p.rows[at(bits)]);
    bind(e, 0, .{ in, w[0], w[1], w[2] });
    e.setValue(d, 4);
    bind(e, 5, .{out});
    e.dispatchGroups(mtl.Size.of(@intCast(@divTrunc(d.n + 7, 8)), @intCast(d.heads), 1), mtl.Size.of(64, 1, 1));
}

/// x through every member of `q` at its own width, each into its own output rows.
pub fn dense(p: *const Pipes, e: mtl.ComputeEncoder, q: Affine, io: Io) void {
    for (0..@as(usize, q.more) + 1) |i| {
        const r = q.run(i);
        launch(p, e, .{ r.w, r.s, r.b }, r.bits, .{ .k = @intCast(q.k), .n = @intCast(r.n), .group = r.group, .rows = @intCast(io.rows), .ldx = @intCast(io.ldx), .ldo = @intCast(io.ldo) }, io.in, io.out.at(@as(usize, r.row) * 2));
    }
}

/// `heads` matrices stacked [heads, n / heads, k] (embed_q, unembed_out): head h reads x + h hsx, writes out + h hso.
pub fn perHead(p: *const Pipes, e: mtl.ComputeEncoder, q: Affine, heads: u32, io: Io, hsx: u32, hso: u32) void {
    const r = q.n / heads;
    launch(p, e, .{ q.w, q.s, q.b }, q.bits, .{ .k = @intCast(q.k), .n = @intCast(r), .group = q.group, .rows = @intCast(io.rows), .heads = @intCast(heads), .ldx = @intCast(io.ldx), .hsx = @intCast(hsx), .ldo = @intCast(io.ldo), .hso = @intCast(hso), .hsw = @intCast(r) }, io.in, io.out);
}

/// A chunk's x slices (`kx` apart) through q's heads into out [rows, heads, n / heads]: one affine_mm batch, else rows.
pub fn headsBatch(p: *const Pipes, e: mtl.ComputeEncoder, q: Affine, h: u32, io: Io, kx: u32, sums: Ref) void {
    const ny = q.n / h;
    const pipes = mm(p, q) orelse return perHead(p, e, q, h, io, kx, ny);
    var head = q;
    head.n = ny;
    affine_mm.rowSumsAt(e, pipes, 64, io.in, sums, io.rows, h * kx, io.ldx); // a head's sums one row apart
    affine_mm.denseBatch(e, pipes, io.in, sums, head, io.out, io.rows, h, .{
        .x_row = @intCast(io.ldx),
        .y_row = @intCast(io.ldo),
        .sums_row = @intCast(h),
        .x_batch = @intCast(kx),
        .y_batch = @intCast(ny),
        .sums_batch = 1,
        .w_batch = @intCast(ny),
    });
}

/// x [rows, k] through bf16 W [n, k]: scalar path, kept for tests / non-prefill callers (decode-side eh uses mtp.zig).
pub fn bf16(p: *const Pipes, e: mtl.ComputeEncoder, w: Ref, n: u32, k: u32, io: Io) void {
    e.setPipeline(p.bf16);
    bind(e, 0, .{ io.in, w });
    e.setValue(RowDims{ .k = @intCast(k), .n = @intCast(n), .group = 0, .rows = @intCast(io.rows), .ldx = @intCast(io.ldx), .ldo = @intCast(io.ldo) }, 4);
    bind(e, 5, .{io.out});
    e.dispatchGroups(mtl.Size.of(@intCast(@divTrunc(n + 7, 8)), 1, 1), mtl.Size.of(64, 1, 1));
}

/// x [rows, k] through bf16 W [n, k]: prefill-side eh_proj on tensor units (8x8 simdgroup MMA in fp32).
/// Caller must ensure rows and n are multiples of 8 and 64 respectively (eh_proj: rows = chunk rows, n = D=4096).
pub fn bf16Nax(p: *const Pipes, e: mtl.ComputeEncoder, w: Ref, n: u32, k: u32, io: Io) void {
    const pipe = p.bf16_nax orelse { return bf16(p, e, w, n, k, io); };
    e.setPipeline(pipe);
    bind(e, 0, .{ io.in, w });
    e.setValue(RowDims{ .k = @intCast(k), .n = @intCast(n), .group = 0, .rows = @intCast(io.rows), .ldx = @intCast(io.ldx), .ldo = @intCast(io.ldo) }, 4);
    bind(e, 5, .{io.out});
    // kernel template: <8, 64> -> BM=8, BN=64. Grid: (n / 64, rows / 8).
    e.dispatchGroups(
        mtl.Size.of(@intCast(@divTrunc(n + 63, 64)), @intCast(@divTrunc(io.rows + 7, 8)), 1),
        mtl.Size.of(8 * 32, 1, 1), // 8 simdgroups * 32 lanes
    );
}

/// The routed experts `w` (stacked [experts, n, k]) by pick: with `up`, gate and up SwiGLU'd.
pub const Picks = struct { w: Affine, up: ?Affine = null, experts: u32, repeat: u32, lim: f32 = 0 };

fn pickDims(m: Picks, slots: u32, ldo: u32) PickDims {
    return .{ .k = @intCast(m.w.k), .n = @intCast(m.w.n), .group = m.w.group, .slots = @intCast(slots), .repeat = @intCast(m.repeat), .experts = @intCast(m.experts), .ldo = @intCast(ldo), .lim = m.lim };
}

/// Slot y [0, slots) through expert ids[y] (i32) on x row y / repeat, into out row y.
pub fn byIds(p: *const Pipes, e: mtl.ComputeEncoder, m: Picks, x: Ref, ids: Ref, out: Ref, slots: u32) void {
    const u = m.up orelse m.w;
    e.setPipeline(if (m.up != null) p.glu_ids[at(m.w.bits)] else p.ids[at(m.w.bits)]);
    bind(e, 0, .{ x, m.w.w, m.w.s, m.w.b });
    e.setValue(pickDims(m, slots, m.w.n), 4);
    bind(e, 5, .{ out, ids, u.w, u.s, u.b });
    e.dispatchGroups(mtl.Size.of((m.w.n + 7) / 8, slots, 1), mtl.Size.of(64, 1, 1));
}

/// x through [gate; up] (one width, `n` rows each) SwiGLU'd into out: the by-ids GLU, every id 0 (glm_swiglu's bits).
pub fn glu(p: *const Pipes, e: mtl.ComputeEncoder, gu: Affine, n: u32, x: Ref, out: Ref, rows: u32, lim: f32) void {
    std.debug.assert(gu.more == 0 and gu.n == 2 * n and rows <= 64);
    var gate = gu;
    gate.n = n;
    var up = gate;
    up.w = gu.w.at(wBytes(n, gu.k, gu));
    up.s = gu.s.at(sBytes(n, gu.k, gu));
    up.b = gu.b.at(sBytes(n, gu.k, gu));
    byIds(p, e, .{ .w = gate, .up = up, .experts = 1, .repeat = 1, .lim = lim }, x, .{ .buf = p.zeros }, out, rows);
}

/// A MoE block's routed experts and shared expert (`sh`: [gate; up] one run, then down), all g64, the shared at 8 bits.
pub const Moe = struct { gate: Affine, up: Affine, down: Affine, sh_gate_up: Affine, sh_down: Affine };

/// Whether `m` takes `moe`: every matrix one run, g64, the routed of one width (not 4/64), the shared at 8 bits.
pub fn moeOne(m: Moe) bool {
    if (native(m.gate) and native(m.up) and native(m.down)) return false; // routed 4/64 keeps the generated gathers
    for ([_]Affine{ m.gate, m.up, m.down, m.sh_gate_up, m.sh_down }) |q| if (q.more != 0 or q.group != 64) return false;
    return m.up.bits == m.gate.bits and m.down.bits == m.gate.bits and m.sh_gate_up.bits == 8 and m.sh_down.bits == 8;
}

/// byIds's GLU and down for the slots, glu and dense for the shared expert's rows: a launch each step, the same bits.
pub fn moe(
    p: *const Pipes,
    e: mtl.ComputeEncoder,
    m: Moe,
    experts: u32,
    topk: u32,
    lim: f32,
    io: MoeIo,
    rows: u32,
) void {
    const slots = rows * topk;
    const g = m.sh_gate_up;
    const steps = [2]struct {
        pipe: mtl.Pipeline,
        w: Affine,
        up: ?Affine,
        sh: Affine,
        x: Ref,
        x2: Ref,
        out: Ref,
        out2: Ref,
        repeat: u32,
    }{
        .{
            .pipe = p.moe_glu[at(m.gate.bits)],
            .w = m.gate,
            .up = m.up,
            .sh = g,
            .x = io.x,
            .x2 = io.x,
            .out = io.act,
            .out2 = io.acts,
            .repeat = topk,
        },
        .{
            .pipe = p.moe[at(m.down.bits)],
            .w = m.down,
            .up = null,
            .sh = m.sh_down,
            .x = io.act,
            .x2 = io.acts,
            .out = io.ye,
            .out2 = io.ys,
            .repeat = 1,
        },
    };
    for (steps) |s| {
        const u = s.up orelse s.w;
        e.setPipeline(s.pipe);
        bind(e, 0, .{ s.x, s.w.w, s.w.s, s.w.b });
        e.setValue(PickDims{
            .k = @intCast(s.w.k),
            .n = @intCast(s.w.n),
            .group = 64,
            .slots = @intCast(slots),
            .repeat = @intCast(s.repeat),
            .experts = @intCast(experts),
            .ldo = @intCast(s.w.n),
            .lim = lim,
        }, 4);
        bind(e, 5, .{ s.out, io.ids, u.w, u.s, u.b, s.x2, s.sh.w, s.sh.s, s.sh.b, s.out2 });
        e.dispatchGroups(mtl.Size.of((s.w.n + 7) / 8, slots + rows, 1), mtl.Size.of(64, 1, 1));
    }
}

/// `moe`'s buffers: x rows, the picks' expert ids, act and acts (gate|up out), ye and ys (down out).
pub const MoeIo = struct { x: Ref, ids: Ref, act: Ref, acts: Ref, ye: Ref, ys: Ref };

/// `pairs` pairs by expert (offsets, order): with `up` x row order / repeat to row i, else x row i to row order[i].
pub fn bySorted(
    p: *const Pipes,
    e: mtl.ComputeEncoder,
    m: Picks,
    x: Ref,
    offsets: Ref,
    order: Ref,
    out: Ref,
    ldo: u32,
    pairs: u32,
) void {
    const u = m.up orelse m.w;
    e.setPipeline(if (m.up != null) p.glu_sorted[at(m.w.bits)] else p.sorted[at(m.w.bits)]);
    bind(e, 0, .{ x, m.w.w, m.w.s, m.w.b });
    e.setValue(pickDims(m, pairs, ldo), 4);
    bind(e, 5, .{ out, offsets, order, u.w, u.s, u.b });
    e.dispatchGroups(mtl.Size.of((m.w.n + 7) / 8, m.experts, 1), mtl.Size.of(64, 1, 1));
}

/// A prompt chunk's affine_mm pipelines for `q` (one width, a format wide.zig compiles), else null: the row kernels.
pub fn mm(p: *const Pipes, q: Affine) ?affine_mm.Pipes {
    if (q.more != 0 or q.group != 64) return null;
    return switch (q.bits) {
        8 => p.mm8,
        2 => p.mm2,
        5 => p.mm5,
        6 => p.mm6,
        else => null,
    };
}

/// A chunk's product into io.out: each member on affine_mm at its width if it fills 64-row tiles, else the row kernels.
pub fn denseMm(p: *const Pipes, e: mtl.ComputeEncoder, q: Affine, io: Io, sums: Ref) void {
    var summed = false;
    for (0..@as(usize, q.more) + 1) |i| {
        const r = q.run(i);
        const one: Affine = .{ .w = r.w, .s = r.s, .b = r.b, .n = r.n, .k = q.k, .bits = r.bits, .group = r.group };
        var part = io;
        part.out = io.out.at(@as(usize, r.row) * 2);
        const pipes = (if (q.more == 0 or r.n % 64 == 0) mm(p, one) else null) orelse {
            dense(p, e, one, part);
            continue;
        };
        if (!summed) affine_mm.rowSumsAt(e, pipes, 64, io.in, sums, io.rows, q.k, io.ldx); // every member's: one k, g64
        summed = true;
        affine_mm.denseBatch(
            e,
            pipes,
            io.in,
            sums,
            one,
            part.out,
            io.rows,
            1,
            .{ .x_row = @intCast(io.ldx), .y_row = @intCast(io.ldo) },
        );
    }
}

/// Existing tensor gathers keep their dispatch; other supported formats gather with the sorted row arithmetic.
pub const Routed = enum { mm4, mm2, gather, rows };

pub fn routed(gate: Affine, up: Affine, down: Affine) Routed {
    if (native(gate) and native(up) and native(down)) return .mm4;
    const two = for ([_]Affine{ gate, up, down }) |q| {
        if (q.more != 0 or q.bits != 2 or q.group != 64) break false;
    } else true;
    if (two) return .mm2;
    for ([_]Affine{ gate, up, down }) |q| if (!gather.supported(q)) return .rows;
    return .gather;
}

/// f_b/g_b dots: generated (both 4/64), `.fused` (one width, g64, glm_rows's sums inside), else `.pre`: same bits.
pub const KdaGate = union(enum) { generated, fused: mtl.Pipeline, pre: mtl.Pipeline };

pub fn kdaGate(p: ?*const Pipes, f: Affine, g: Affine) KdaGate {
    if (native(f) and native(g)) return .generated;
    if (f.more == 0 and g.more == 0 and f.group == 64 and g.group == 64 and f.bits == g.bits)
        return .{ .fused = p.?.kda_fg[at(f.bits)] };
    return .{ .pre = p.?.kda_pre };
}

test "4/64 of one width alone takes the generated kernels" {
    const r: Ref = .{ .buf = undefined };
    const q: Affine = .{ .w = r, .s = r, .b = r, .n = 64, .k = 128, .bits = 4, .group = 64 };
    try std.testing.expect(native(q));
    var m = q;
    m.bits = 8;
    try std.testing.expect(!native(m));
    m = q;
    m.more = 1;
    try std.testing.expect(!native(m));
    try std.testing.expect(kdaGate(null, q, q) == .generated);
}

test "a failed compile releases every pipeline built before it; a load counts each one it builds" {
    const gpa = std.testing.allocator;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var made: Made = .{};
    try std.testing.expectError(error.KernelCompile, build(gpa, device, "", &made)); // after the row kernels
    try std.testing.expectEqual(@as(usize, nw * 7 + 4), made.released);
    try std.testing.expectEqual(@as(usize, 0), made.n);
    var all: Made = .{};
    var p = try build(gpa, device, comptime kernels.generatedSource("kda_rows"), &all);
    p.deinit();
    try std.testing.expectEqual(all.list.len, all.n);
}

test "glm_rows's SwiGLU compiles glm_glue's sigmoid_fast and bfloat overloads, no copy of its own" {
    try std.testing.expect(std.mem.indexOf(u8, glue_math, "inline U sigmoid_fast(U x)") != null);
    try std.testing.expect(std.mem.indexOf(u8, glue_math, "METAL_FUNC bfloat16_t exp(bfloat16_t x)") != null);
    try std.testing.expect(std.mem.indexOf(u8, glue_math, "kernel void") == null); // the helpers only
    try std.testing.expect(std.mem.indexOf(u8, sources.glm_rows, "exp(") == null);
    try std.testing.expect(std.mem.indexOf(u8, sources.glm_rows, "sigmoid_fast(gt)") != null);
}

test "prompt chunks retain the existing uniform tensor gathers and gather other supported formats" {
    const r: Ref = .{ .buf = undefined };
    const q: Affine = .{ .w = r, .s = r, .b = r, .n = 64, .k = 128, .bits = 2, .group = 64 };
    var four = q;
    four.bits = 4;
    try std.testing.expectEqual(Routed.mm4, routed(four, four, four));
    try std.testing.expectEqual(Routed.mm2, routed(q, q, q));
    var three = q;
    three.bits = 3;
    var g32 = q;
    g32.group = 32;
    try std.testing.expectEqual(Routed.gather, routed(q, q, three));
    try std.testing.expectEqual(Routed.gather, routed(q, q, four));
    try std.testing.expectEqual(Routed.gather, routed(g32, g32, g32));
}

test "f_b and g_b of one width (g64) take glm_rows's dots inside kda_rows, two widths the products made apart" {
    const r: Ref = .{ .buf = undefined };
    const q: Affine = .{ .w = r, .s = r, .b = r, .n = 64, .k = 128, .bits = 8, .group = 64 };
    var four = q;
    four.bits = 4;
    var g32 = q;
    g32.group = 32;
    const p: Pipes = undefined;
    for ([_][2]Affine{
        .{ four, q },
        .{ q, four },
        .{ g32, g32 },
    }) |fg| try std.testing.expect(kdaGate(&p, fg[0], fg[1]) == .pre);
    var five = q;
    five.bits = 5;
    for ([_]Affine{ q, five }) |x| try std.testing.expect(kdaGate(&p, x, x) == .fused);
}
