//! The recorded q4_attn_parts with two query heads of one KV head a threadgroup: each key's K and V loaded once.
const std = @import("std");
const mtl = @import("metal");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;
const Variant = replay.Variant;
const patch = @import("tree.zig").patch;

/// Query heads a threadgroup (the recorded grid's width divided by it).
pub const heads = 2;

/// Each head keeps its own scores, online softmax and combine in the recorded order; only K and V loads are shared.
const edits = [_][2][]const u8{
    .{ "  constexpr int PER = D / 32;\n", "  constexpr int PER = D / 32, HPT = 2;\n" },
    .{ "  const int h = int(threadgroup_position_in_grid.x);\n", "  const int h0 = int(threadgroup_position_in_grid.x) * HPT;\n" },
    .{ "  const int kvh = h / (H / KVH);\n", "  const int kvh = h0 / (H / KVH);\n" },
    .{
        \\  const device bfloat* qp = Q + (size_t(r) * H + h) * D + lane * PER;
        \\  float q[PER], o[PER];
        \\  for (int i = 0; i < PER; i++) { q[i] = SCALE[0] * float(qp[i]); o[i] = 0.0f; }
        \\  float m = -INFINITY, l = 0.0f;
        \\  for (int j = lo + int(g); j < hi; j += 8) {
        \\    const size_t key = size_t(sparse ? ids[j] : j) * D;
        \\    float sc = 0.0f;
        \\    for (int i = 0; i < PER; i++) sc = fma(q[i], float(kb[key + i]), sc);
        \\    sc = simd_sum(sc);
        \\    const float mn = metal::max(m, sc);
        \\    const float f = metal::exp(m - mn), e = metal::exp(sc - mn);
        \\    l = fma(l, f, e);
        \\    for (int i = 0; i < PER; i++) o[i] = fma(e, float(vb[key + i]), o[i] * f);
        \\    m = mn;
        \\  }
        \\
        ,
        \\  float q[HPT][PER], o[HPT][PER], m[HPT], l[HPT];
        \\  for (int hh = 0; hh < HPT; hh++) {
        \\    const device bfloat* qp = Q + (size_t(r) * H + h0 + hh) * D + lane * PER;
        \\    for (int i = 0; i < PER; i++) { q[hh][i] = SCALE[0] * float(qp[i]); o[hh][i] = 0.0f; }
        \\    m[hh] = -INFINITY; l[hh] = 0.0f;
        \\  }
        \\  for (int j = lo + int(g); j < hi; j += 8) {
        \\    const size_t key = size_t(sparse ? ids[j] : j) * D;
        \\    float kv[PER], vv[PER];
        \\    for (int i = 0; i < PER; i++) { kv[i] = float(kb[key + i]); vv[i] = float(vb[key + i]); }
        \\    for (int hh = 0; hh < HPT; hh++) {
        \\      float sc = 0.0f;
        \\      for (int i = 0; i < PER; i++) sc = fma(q[hh][i], kv[i], sc);
        \\      sc = simd_sum(sc);
        \\      const float mn = metal::max(m[hh], sc);
        \\      const float f = metal::exp(m[hh] - mn), e = metal::exp(sc - mn);
        \\      l[hh] = fma(l[hh], f, e);
        \\      for (int i = 0; i < PER; i++) o[hh][i] = fma(e, vv[i], o[hh][i] * f);
        \\      m[hh] = mn;
        \\    }
        \\  }
        \\
    },
    .{
        \\  if (lane == 0) { ms[g] = m; ls[g] = l; }
        \\  threadgroup_barrier(mem_flags::mem_threadgroup);
        \\  float top = -INFINITY;
        \\  for (int k = 0; k < 8; k++) top = metal::max(top, ms[k]);
        \\  const float mine = m == -INFINITY ? 0.0f : metal::exp(m - top);
        \\  for (int i = 0; i < PER; i++) tile[g][lane * PER + i] = o[i] * mine;
        \\  threadgroup_barrier(mem_flags::mem_threadgroup);
        \\  const size_t at = (size_t(r) * H + h) * P + part;
        \\  for (int d = int(thread_position_in_threadgroup.x); d < D; d += 256) {
        \\    float acc = 0.0f;
        \\    for (int k = 0; k < 8; k++) acc += tile[k][d];
        \\    PO[at * D + d] = acc;
        \\  }
        \\  if (thread_position_in_threadgroup.x == 0) {
        \\    float total = 0.0f;
        \\    for (int k = 0; k < 8; k++) total += ms[k] == -INFINITY ? 0.0f : ls[k] * metal::exp(ms[k] - top);
        \\    PM[at * 2] = top;
        \\    PM[at * 2 + 1] = total;
        \\  }
        \\
        ,
        \\  for (int hh = 0; hh < HPT; hh++) {
        \\    if (hh > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    if (lane == 0) { ms[g] = m[hh]; ls[g] = l[hh]; }
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    float top = -INFINITY;
        \\    for (int k = 0; k < 8; k++) top = metal::max(top, ms[k]);
        \\    const float mine = m[hh] == -INFINITY ? 0.0f : metal::exp(m[hh] - top);
        \\    for (int i = 0; i < PER; i++) tile[g][lane * PER + i] = o[hh][i] * mine;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    const size_t at = (size_t(r) * H + h0 + hh) * P + part;
        \\    for (int d = int(thread_position_in_threadgroup.x); d < D; d += 256) {
        \\      float acc = 0.0f;
        \\      for (int k = 0; k < 8; k++) acc += tile[k][d];
        \\      PO[at * D + d] = acc;
        \\    }
        \\    if (thread_position_in_threadgroup.x == 0) {
        \\      float total = 0.0f;
        \\      for (int k = 0; k < 8; k++) total += ms[k] == -INFINITY ? 0.0f : ls[k] * metal::exp(ms[k] - top);
        \\      PM[at * 2] = top;
        \\      PM[at * 2 + 1] = total;
        \\    }
        \\  }
        \\
    },
};

/// `role`'s recorded one-row kernel at two heads a threadgroup, or null when it does not match the recorded one here.
pub fn init(r: *Run, role: []const u8) !?*Variant {
    var key: [96]u8 = undefined;
    const s = r.roles.get(try std.fmt.bufPrint(&key, "{s}|1", .{role})) orelse return error.NoSite;
    const text = try patch(r.arena, try Run.variantText(r.arena, s.v), &edits);
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, text, mtl.CompileOptions.mlx()));
    const nv = try r.arena.create(Variant);
    nv.* = s.v.*;
    nv.pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, try std.fmt.allocPrintSentinel(r.arena, "{s}", .{s.v.name}, 0), false));
    nv.text = text;
    if (try same(r, s.v.pipe, nv.pipe)) return nv;
    std.log.warn("attention parts at two heads a threadgroup differ from the recorded kernel here; recorded kept", .{});
    return null;
}

/// `role` at the run's rows on `v`: the recorded launch at half its width.
pub fn call(r: *Run, role: []const u8, v: *Variant, ins: []const Buf, outs: []const Buf) !void {
    if (r.skip & Run.class(role) != 0) return;
    var key: [96]u8 = undefined;
    const s = r.roles.get(try std.fmt.bufPrint(&key, "{s}|{d}", .{ role, r.rows })) orelse return error.NoSite;
    try r.bindV(v, ins, outs);
    var grid = s.grid;
    grid.width /= heads;
    r.enc.dispatchThreads(grid, s.tg);
    if (!r.serial) r.enc.barrier();
}

/// Both kernels on random queries and caches at 4 rows, rows 1 and 3 sparse: the same PO and PM bytes.
fn same(r: *Run, recorded: mtl.Pipeline, two: mtl.Pipeline) !bool {
    const rows = 4;
    const cap = 512;
    const h = 24;
    const d = 256;
    const p = 16;
    var prng = std.Random.DefaultPrng.init(0x7e59);
    const rnd = prng.random();
    const q = try r.buffer(rows * h * d * 2);
    const kc = try r.buffer(2 * cap * d * 2);
    const vc = try r.buffer(2 * cap * d * 2);
    for ([_]mtl.Buffer{ q, kc, vc }) |b| for (b.slice(u16, b.length() / 2)) |*v| {
        v.* = @truncate(@as(u32, @bitCast(rnd.float(f32) * 4 - 2)) >> 16);
    };
    const ints = try r.buffer((4 + 2 + rows * cap + 2 * rows) * 4);
    const iv = ints.slice(i32, 4 + 2 + rows * cap + 2 * rows);
    @memcpy(iv[0..6], &[_]i32{ 1, 2, cap, d, rows, cap });
    const ids = iv[6 .. 6 + rows * cap];
    for (ids, 0..) |*v, i| v.* = @intCast((i * 197 + 31) % cap);
    @memcpy(iv[6 + rows * cap ..][0 .. 2 * rows], &[_]i32{ 300, 301, 200, 303, 0, 1, 0, 1 });
    const scale = try r.buffer(16);
    scale.slice(f32, 1)[0] = 0.0625;
    const n_out = rows * h * p;
    var outs: [2][2]mtl.Buffer = undefined;
    for (&outs) |*o| o.* = .{ try r.buffer(n_out * d * 4), try r.buffer(n_out * 2 * 4) };
    defer for ([_]mtl.Buffer{ q, kc, vc, ints, scale }) |b| r.release(b);
    defer for (outs) |o| for (o) |b| r.release(b);
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    for ([_]mtl.Pipeline{ recorded, two }, outs, [_]usize{ 1, heads }) |pipe, o, per| {
        e.setPipeline(pipe);
        const bind = [_]Buf{
            .{ .b = q },                                 .{ .b = kc },                                       .{ .b = ints },
            .{ .b = vc },                                .{ .b = ints, .off = 24 },                          .{ .b = ints, .off = 16 },
            .{ .b = ints, .off = (6 + rows * cap) * 4 }, .{ .b = ints, .off = (6 + rows * cap + rows) * 4 }, .{ .b = scale },
            .{ .b = o[0] },                              .{ .b = o[1] },
        };
        for (bind, 0..) |b, j| e.setBuffer(b.b, b.off, j);
        e.dispatchThreads(mtl.Size.of(h / per * 256, rows, p), mtl.Size.of(256, 1, 1));
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    for (0..2) |j| if (!std.mem.eql(u8, outs[0][j].contents()[0..outs[0][j].length()], outs[1][j].contents()[0..outs[1][j].length()])) return false;
    return true;
}

test "the two-head edits apply once each to the recorded attention parts kernel" {
    const ks = @import("kernel_sources");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var seen: usize = 0;
    for (ks.flashnext_gen.sources) |src| {
        if (!std.mem.startsWith(u8, src.name, "q4_attn_parts")) continue;
        seen += 1;
        const out = try patch(arena.allocator(), src.text, &edits);
        try std.testing.expect(std.mem.indexOf(u8, out, "HPT") != null);
    }
    try std.testing.expect(seen > 0);
}
