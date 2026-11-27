//! An EXL3 pack's layer on this GPU: every trellis stage bit-equal to the serial chain on the GPU's own inputs.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");
const ref = @import("ref.zig");

const testing = std.testing;
const xr = pkg.experts_exl3.ref;
const Matrix = pkg.weights.X3Matrix;
const Ref = pkg.ctx.Ref;

/// The FP8 fixture's geometry as an EXL3 pack: layer 1 a ratio-2 KV and index source.
fn config() pkg.host.config.Config {
    var c = fixture.config();
    c.format = .exl3;
    return c;
}

fn host(comptime T: type, buf: mtl.Buffer, off: usize, n: usize) []const T {
    return @alignCast(std.mem.bytesAsSlice(T, (buf.contents() + off)[0 .. n * @sizeOf(T)]));
}

/// A group as the serial chain reads it: its trellis tiles and fp16 scales from the Metal views.
fn serialOf(m: Matrix) xr.Matrix {
    const words = m.k / 16 * (m.n / 16) * m.width.tileWords();
    return .{ .width = m.width, .book = m.codebook, .trellis = host(i16, m.trellis.buffer, m.trellis.offset, words), .su = host(f16, m.suh.buffer, m.suh.offset, m.k), .sv = host(f16, m.svh.buffer, m.svh.offset, m.n) };
}

/// A routed expert's group from its descriptor: the arena's GPU address mapped back to its bytes.
fn expertOf(arena: mtl.Buffer, d: pkg.weights_exl3.Descriptor, k: usize, n: usize) xr.Matrix {
    const base = arena.gpuAddress();
    const width: @TypeOf(@as(Matrix, undefined).width) = .{ .halves = @intCast(d.halves) };
    const words = k / 16 * (n / 16) * width.tileWords();
    return .{ .width = width, .book = @fromBackingInt(@intCast(d.codebook)), .trellis = host(i16, arena, d.trellis - base, words), .su = host(f16, arena, d.suh - base, k), .sv = host(f16, arena, d.svh - base, n) };
}

/// Metal's flush of subnormal fp32 operands and results, as experts_exl3_ref's arithmetic.
fn ftz(v: f32) f32 {
    return if (v != 0 and @abs(v) < std.math.floatMin(f32)) std.math.copysign(@as(f32, 0), v) else v;
}

/// Rows of x [rows][k] bf16 through `m`: rotate_in, the one-row dot, rotate_out to bf16 bits or fp32 bits.
fn serial(a: std.mem.Allocator, m: xr.Matrix, k: usize, n: usize, x: []const u16, rows: usize, f32_out: bool) ![]u32 {
    const out = try a.alloc(u32, rows * n);
    const rot = try a.alloc(f16, k);
    defer a.free(rot);
    const sums = try a.alloc(f32, n);
    defer a.free(sums);
    const bf = try a.alloc(u16, n);
    defer a.free(bf);
    for (0..rows) |r| {
        xr.rotIn(x[r * k ..][0..k], m.su, rot);
        xr.dot(m, rot, sums);
        if (f32_out) {
            for (0..n / 128) |b| {
                var v: [128]f32 = sums[b * 128 ..][0..128].*;
                xr.hadamard(&v);
                for (v, 0..) |y, i| out[r * n + b * 128 + i] = @bitCast(ftz(ftz(ftz(y) * 0.08838834764831845) * ftz(m.sv[b * 128 + i])));
            }
        } else {
            xr.rotOut(sums, m.sv, bf);
            for (bf, 0..) |v, i| out[r * n + i] = v;
        }
    }
    return out;
}

/// The device's silu at every bf16 gate (the one value the serial chain takes from the GPU).
fn siluTable(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !*xr.Silu {
    const lib = try mtl.Library.fromSource(device, pkg.experts_exl3.text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const p = try mtl.Pipeline.init(device, lib, "tf_dsv41_x3_silu", false);
    defer p.deinit();
    const buf = try device.buffer(65536 * 4, mtl.ResourceOptions.shared);
    defer buf.deinit();
    const pass = fixture.Pass.begin(queue);
    pass.e.setPipeline(p);
    pass.e.setBuffer(buf, 0, 0);
    pass.e.dispatchGroups(.{ .width = 256 }, .{ .width = 256 });
    try pass.finish();
    const t = try a.create(xr.Silu);
    @memcpy(t, buf.slice(f32, 65536));
    return t;
}

const Check = struct {
    a: std.mem.Allocator,
    r: *fixture.Rig,
    rows: usize,
    bad: usize = 0,

    fn bits(_: *Check, comptime T: type, at: Ref, n: usize) []const T {
        return host(T, at.buf, at.off, n);
    }

    /// `name`: y [rows][n] of the GPU against `m` on the GPU's own x [rows][k].
    fn stage(ck: *Check, name: []const u8, m: Matrix, x: Ref, y: Ref, f32_out: bool) !void {
        const want = try serial(ck.a, serialOf(m), m.k, m.n, ck.bits(u16, x, ck.rows * m.k), ck.rows, f32_out);
        defer ck.a.free(want);
        for (want, 0..) |w, i| {
            const g: u32 = if (f32_out) ck.bits(u32, y, want.len)[i] else ck.bits(u16, y, want.len)[i];
            if (g != w) {
                std.debug.print("FAIL dsv41 metal exl3 {s}: row {d} column {d}: gpu {x}, serial {x}\n", .{ name, i / m.n, i % m.n, g, w });
                ck.bad += 1;
                return;
            }
        }
        std.debug.print("PASS dsv41 metal exl3 {s}: {d} rows x {d} bit-equal to the serial chain\n", .{ name, ck.rows, m.n });
    }
};

fn plus(r: Ref, bytes: usize) Ref {
    return .{ .buf = r.buf, .off = r.off + bytes };
}

test "dsv41 metal exl3: a layer's trellis stages bit-equal to the serial chain on the GPU's own inputs" {
    const a = testing.allocator;
    const r = try fixture.Rig.initWith(a, testing.io, config(), 2048, 43);
    defer r.deinit();
    const c = &r.c;
    const s = &r.s;
    const p0 = 1101;
    const rows = 3;
    fixture.fillPast(c, &r.slot, p0, 7);
    fixture.fillInput(r, rows, 11);
    const w = &r.layers[1];
    const m = &w.x3.?.m;
    const ctx = r.ctx(&r.slot, p0, rows);
    try s.writeRows(c, p0, rows);
    const pass = fixture.Pass.begin(r.queue);
    const st = r.streams();
    const pr = r.pres();
    try pkg.layer.encode(&ctx, pass.e, w, .{ .x = st[0], .pre = pr[0], .x_out = st[1], .pre_out = pr[1] });
    try pass.finish();
    var ck: Check = .{ .a = a, .r = r, .rows = rows };
    try ck.stage("wq_a", m.wq_a, s.hq, s.qa, false);
    try ck.stage("wkv", m.wkv, s.hq, s.kv, false);
    try ck.stage("wq_b", m.wq_b, s.qrq, s.q, false);
    try ck.stage("indexer wq_b", m.ix_wq_b.?, s.qrq, s.ixq, false);
    try ck.stage("compressor wkv (fp32)", m.comp_wkv.?, s.hn, s.cf, true);
    try ck.stage("compressor wgate (fp32)", m.comp_wgate.?, s.hn, s.cg, true);
    try ck.stage("indexer wk", m.ix_wk.?, s.latn, s.ik, false);
    const kg = c.heads / c.o_groups * c.head_dim;
    for (0..c.o_groups) |g| try ck.stage("wo_a group", m.wo_a[g].?, plus(s.oq, g * rows * kg * 2), plus(s.oa, g * rows * c.o_lora * 2), false);
    try ck.stage("wo_b", m.wo_b, s.ob, s.aout, false);
    try ck.stage("shared w2", m.shared[1], s.sactq, s.sout, false);
    const silu = try siluTable(a, r.device, r.queue);
    defer a.destroy(silu);
    // the shared gate/up: both rotations, the clamped SwiGLU at weight 1, then its act_quant
    const H = c.hidden;
    const W = c.expert_width;
    const hq2 = ck.bits(u16, s.hq2, rows * H);
    const g1 = try serial(a, serialOf(m.shared[0]), H, W, hq2, rows, false);
    defer a.free(g1);
    const u_3 = try serial(a, serialOf(m.shared[2]), H, W, hq2, rows, false);
    defer a.free(u_3);
    const act = try a.alloc(u16, rows * W);
    defer a.free(act);
    for (act, g1, u_3) |*y, g, u| y.* = xr.act(@intCast(g), @intCast(u), 1, c.swiglu_limit, silu);
    const actq = try a.alloc(u16, rows * W);
    defer a.free(actq);
    for (0..rows * W / 32) |b| ref.actq32(act[b * 32 ..][0..32], actq[b * 32 ..][0..32]);
    const shared_ok = std.mem.eql(u16, act, ck.bits(u16, s.sact, rows * W)) and std.mem.eql(u16, actq, ck.bits(u16, s.sactq, rows * W));
    std.debug.print("{s} dsv41 metal exl3 shared gate/up: SwiGLU and its act_quant against the serial chain\n", .{if (shared_ok) "PASS" else "FAIL"});
    ck.bad += @intFromBool(!shared_ok);
    // each routed pair's output against upPair and downPair at its route weight, found where the plan put it
    const K = c.topk;
    const t = w.x3.?.experts;
    const descs = host(pkg.weights_exl3.Descriptor, w.table, 0, 3 * t.count);
    const picks = ck.bits(i32, s.picks, rows * K);
    const wts = ck.bits(f32, s.wts, rows * K);
    const where = ck.bits(u32, s.plan[3], rows * K);
    const eout = ck.bits(u16, s.eout, rows * K * H);
    var work = try xr.Work.init(a, H, W);
    defer work.deinit(a);
    const sums = try a.alloc(f32, 2 * @max(H, W));
    defer a.free(sums);
    const pact = try a.alloc(u16, W);
    defer a.free(pact);
    const pactq = try a.alloc(u16, W);
    defer a.free(pactq);
    const pout = try a.alloc(u16, H);
    defer a.free(pout);
    var routed_bad: usize = 0;
    for (0..rows * K) |q| {
        const e: usize = @intCast(picks[q]);
        const gate = expertOf(w.arena, descs[e], H, W);
        const up = expertOf(w.arena, descs[t.count + e], H, W);
        const down = expertOf(w.arena, descs[2 * t.count + e], W, H);
        xr.upPair(gate, up, hq2[q / K * H ..][0..H], wts[q], c.swiglu_limit, silu, .{ sums[0..W], sums[W .. 2 * W] }, pact, &work);
        for (0..W / 32) |b| ref.actq32(pact[32 * b ..][0..32], pactq[32 * b ..][0..32]);
        xr.downPair(down, pactq, sums[0..H], pout, &work);
        routed_bad += @intFromBool(!std.mem.eql(u16, pout, eout[where[q] * H ..][0..H]));
    }
    std.debug.print("{s} dsv41 metal exl3 routed experts: {d} of {d} pairs' outputs differ from upPair, act_quant and downPair\n", .{ if (routed_bad == 0) "PASS" else "FAIL", routed_bad, rows * K });
    ck.bad += routed_bad;
    try testing.expectEqual(@as(usize, 0), ck.bad);
}

test "dsv41 metal exl3: the forward fixture loads Engram's wkv and the head as trellis groups" {
    const a = testing.allocator;
    var c = fixture.forwardConfig();
    c.format = .exl3;
    c.dspark_targets = pkg.host.config.Ids.of(&.{});
    const r = try fixture.Rig.initWith(a, testing.io, c, 2048, 47);
    defer r.deinit();
    try testing.expect(r.top.?.head_x3 != null);
    try testing.expect(r.layers[1].engram != null and r.layers[1].x3.?.m.engram_wkv != null);
}
