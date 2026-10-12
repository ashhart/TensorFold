//! GLM-5.3-Flash's products at widths its generated kernels do not read (glm_rows.metal): a launch a stack member.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const affine = @import("affine.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const kernels = @import("kernels.zig");
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
    bf16: mtl.Pipeline, // an unquantized matrix (the MTP head's eh_proj)
    kda_pre: mtl.Pipeline, // kda_rows reading f_b's and g_b's products made apart (any width but 4/64)
    embed: [2]mtl.Pipeline, // the embedding at 6 and 8 bits, groups of 64
    mm8: affine_mm.Pipes, // prompt chunks' products of one width at 8 bits in groups of 64 (core/affine_mm.zig)
    mm2: affine_mm.Pipes, // and at 2 bits in groups of 64: oQ2's routed experts, gathered
    hash: u64, // every source compiled here

    pub fn deinit(p: *Pipes) void {
        inline for (.{ &p.rows, &p.ids, &p.glu_ids, &p.sorted, &p.glu_sorted, &p.embed }) |set| for (set) |x| x.deinit();
        p.bf16.deinit();
        p.kda_pre.deinit();
        for (p.mm8) |x| x.deinit();
        for (p.mm2) |x| x.deinit();
    }
};

/// kda_rows's f_b dot read from a product made apart (a row's sum in its first lane of four, zeros in the rest: exact).
const pre_f = .{ "result = quad_dot<FB, PER>(x, wb, s, bb);", "result = qlid == 0u ? float(((device const bfloat*)FBW)[(size_t)r * W + row]) : 0.0f;" };
const pre_g = .{ "result = quad_dot<GB, PER>(x, wb, s, bb);", "result = qlid == 0u ? float(((device const bfloat*)GBW)[(size_t)r * W + row]) : 0.0f;" };

/// glm_glue.metal's bfloat overloads and sigmoid_fast, the text glm_swiglu compiles: glm_rows's SwiGLU takes its bits.
pub const glue_math = blk: {
    @setEvalBranchQuota(100_000);
    const g = sources.glm_glue;
    const a = std.mem.indexOf(u8, g, "// MLX's bfloat math overloads").?;
    const b = a + std.mem.indexOf(u8, g[a..], "// The dense MLP's SwiGLU").?;
    break :blk "typedef bfloat bfloat16_t;\n" ++ g[a..b];
};

/// The pipelines built so far: a failed compile releases them all.
const Made = struct {
    list: [nw * 5 + 4 + 2 * affine_mm.names.len]mtl.Pipeline = undefined,
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
    const rows_src = try std.mem.concat(gpa, u8, &.{ sources.core_row_projection, glue_math, sources.glm_rows });
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
        }
        p.bf16 = try made.add(device, l, "glm_rows_bf16");
    }
    const pre_a = try std.mem.replaceOwned(u8, gpa, kda, pre_f[0], pre_f[1]); // its own instance, under its own name
    defer gpa.free(pre_a);
    const pre_src = try std.mem.replaceOwned(u8, gpa, pre_a, pre_g[0], pre_g[1]);
    defer gpa.free(pre_src);
    if (std.mem.indexOf(u8, pre_src, pre_f[0]) != null or std.mem.indexOf(u8, pre_src, pre_g[0]) != null or std.mem.indexOf(u8, pre_src, pre_f[1]) == null) return error.KernelCompile;
    h.update(pre_src);
    try lib(made, device, pre_src, &.{comptime kernels.generatedFunction("kda_rows")}, @as(*[1]mtl.Pipeline, &p.kda_pre));
    h.update(sources.ops_embed_norm);
    try lib(made, device, sources.ops_embed_norm, &.{ "tf_embed_b6_g64", "tf_embed_b8_g64" }, &p.embed);
    const mm8 = try affine_mm.source(device, gpa, .{ .bits = 8, .group = 64 });
    defer gpa.free(mm8);
    h.update(mm8);
    try lib(made, device, mm8, &affine_mm.names, &p.mm8);
    const mm2 = try affine_mm.source(device, gpa, .{ .bits = 2, .group = 64 });
    defer gpa.free(mm2);
    h.update(mm2);
    try lib(made, device, mm2, &affine_mm.names, &p.mm2);
    p.hash = h.final();
    return p;
}

fn bind(e: mtl.ComputeEncoder, first: usize, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, first + i);
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

/// x [rows, k] through bf16 W [n, k].
pub fn bf16(p: *const Pipes, e: mtl.ComputeEncoder, w: Ref, n: u32, k: u32, io: Io) void {
    e.setPipeline(p.bf16);
    bind(e, 0, .{ io.in, w });
    e.setValue(RowDims{ .k = @intCast(k), .n = @intCast(n), .group = 0, .rows = @intCast(io.rows), .ldx = @intCast(io.ldx), .ldo = @intCast(io.ldo) }, 4);
    bind(e, 5, .{io.out});
    e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(64, 1, 1));
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

/// Each expert's sorted pairs (offsets, order): with `up` x row order / repeat to row i, else x row i to row order[i].
pub fn bySorted(p: *const Pipes, e: mtl.ComputeEncoder, m: Picks, x: Ref, offsets: Ref, order: Ref, out: Ref, ldo: u32) void {
    const u = m.up orelse m.w;
    e.setPipeline(if (m.up != null) p.glu_sorted[at(m.w.bits)] else p.sorted[at(m.w.bits)]);
    bind(e, 0, .{ x, m.w.w, m.w.s, m.w.b });
    e.setValue(pickDims(m, 0, ldo), 4);
    bind(e, 5, .{ out, offsets, order, u.w, u.s, u.b });
    e.dispatchGroups(mtl.Size.of((m.w.n + 7) / 8, m.experts, 1), mtl.Size.of(64, 1, 1));
}

/// A prompt chunk's routed experts: affine_mm gathers at 4/64 (`mm4`) or 2/64 (`mm2`, oQ2's), else sorted row products.
pub const Routed = enum { mm4, mm2, rows };

pub fn routed(gate: Affine, up: Affine, down: Affine) Routed {
    if (native(gate) and native(up) and native(down)) return .mm4;
    for ([_]Affine{ gate, up, down }) |q| if (q.more != 0 or q.bits != 2 or q.group != 64) return .rows;
    return .mm2;
}

/// kda_rows's f_b/g_b dots: generated (both 4/64), else (`.pre`) glm_rows products made apart, the fp64-checked ones.
pub const KdaGate = union(enum) { generated, pre: mtl.Pipeline };

pub fn kdaGate(p: ?*const Pipes, f: Affine, g: Affine) KdaGate {
    if (native(f) and native(g)) return .generated;
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
    const device = try mtl.Device.init();
    defer device.deinit();
    var made: Made = .{};
    try std.testing.expectError(error.KernelCompile, build(gpa, device, "", &made)); // after the row kernels
    try std.testing.expectEqual(@as(usize, nw * 5 + 1), made.released);
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

test "prompt chunks gather routed experts on affine_mm at 4/64 and 2/64 only; other widths take the sorted row products" {
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
    try std.testing.expectEqual(Routed.rows, routed(q, q, three));
    try std.testing.expectEqual(Routed.rows, routed(q, q, four));
    try std.testing.expectEqual(Routed.rows, routed(g32, g32, g32));
}

test "f_b and g_b at any other width (8/64 too) take the products made apart, not the template's own dots" {
    const r: Ref = .{ .buf = undefined };
    const q: Affine = .{ .w = r, .s = r, .b = r, .n = 64, .k = 128, .bits = 8, .group = 64 };
    var four = q;
    four.bits = 4;
    const p: Pipes = undefined;
    for ([_][2]Affine{ .{ q, q }, .{ four, q }, .{ q, four } }) |fg| try std.testing.expect(kdaGate(&p, fg[0], fg[1]) == .pre);
}
