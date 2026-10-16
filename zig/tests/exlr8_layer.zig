//! A real EXLR8 MoE layer as projection jobs for the Metal and CUDA benches: nine experts a set, FP64 of the outputs.

const std = @import("std");
const ex = @import("exlr8");
const ref = @import("exlr8_ref");
const codec = ex.codec;
const L = ex.experts;
const Job = ex.job.Job;
const a = std.heap.page_allocator;
/// Eight routed experts and the shared one, as one token's MoE layer reads them.
pub const slots = 9;
pub const rows = 16;

pub const Set = struct {
    experts: [slots]u16,
    up: []Job,
    down: []Job,
    up_bytes: usize,
    down_bytes: usize,
    /// Each slot's 48 down block scales, then its 2048 down channel scales, in `halves`.
    sv: [slots]u32,
    suh: [slots]u32,
};

fn width(widths: *const ex.manifest.LayerWidths, e: u16, p: L.Proj, h: usize) u8 {
    if (e == L.shared) return switch (p) {
        .gate => widths.shared_gate[h],
        .up => widths.shared_up[h],
        .down => widths.shared_down,
    };
    return switch (p) {
        .gate => widths.gate[e][h],
        .up => widths.up[e][h],
        .down => widths.down[e],
    };
}

/// Every set's halves in one weight array, the tile-run offsets, fp16 scales and the layer's signs (in, then out).
pub const Host = struct {
    weights: std.ArrayList(u8) = .empty,
    bases: std.ArrayList(u64) = .empty,
    halves: std.ArrayList(f16) = .empty,
    signs: [2 * L.hidden]i8 = undefined,
    sets: []Set = &.{},

    fn scales(b: *Host, v: []align(1) const f16) !u32 {
        const at: u32 = @intCast(b.halves.items.len);
        for (v) |x| try b.halves.append(a, x);
        return at;
    }

    /// `count` sets of distinct experts at the manifest's widths; gate/up halves are jobs, down is 48 raw jobs each.
    pub fn build(layer: *const L.Layer, widths: *const ex.manifest.LayerWidths, count: usize) !Host {
        var b: Host = .{};
        @memcpy(b.signs[0..L.hidden], layer.signs_in);
        @memcpy(b.signs[L.hidden..], layer.signs_out);
        b.sets = try a.alloc(Set, count);
        for (b.sets, 0..) |*s, si| {
            var up: std.ArrayList(Job) = .empty;
            var down: std.ArrayList(Job) = .empty;
            s.up_bytes = 0;
            s.down_bytes = 0;
            for (0..slots) |slot| {
                const e: u16 = if (slot == slots - 1) L.shared else @intCast((si * 8 + slot * 31 + 7) % 256);
                s.experts[slot] = e;
                for ([_]L.Proj{ .gate, .up }, 0..) |p, pi| for (0..L.halves) |h| {
                    const k = width(widths, e, p, h);
                    const half = try layer.half(e, p, h, k);
                    const at = b.weights.items.len;
                    try b.weights.appendSlice(a, half.trellis);
                    s.up_bytes += half.trellis.len;
                    const base: u32 = @intCast(b.bases.items.len);
                    for (0..48) |ib| try b.bases.append(a, at + ib * 64 * codec.tileBytes(k));
                    try up.append(a, .{ .k = k, .blocks = 48, .base = base, .in_scale = try b.scales(half.block), .out_scale = try b.scales(half.chan), .y_col = @intCast(slot * 4096 + pi * 2048 + h * 128) });
                };
                const kd = width(widths, e, .down, 0);
                var at: [L.halves]usize = undefined;
                var chan: [2048]f16 = undefined;
                for (0..L.halves) |h| {
                    const half = try layer.half(e, .down, h, kd);
                    at[h] = b.weights.items.len;
                    try b.weights.appendSlice(a, half.trellis);
                    s.down_bytes += half.trellis.len;
                    for (half.chan, 0..) |c, i| chan[h * 128 + i] = c;
                    if (h == 0) s.sv[slot] = try b.scales(half.block);
                }
                s.suh[slot] = try b.scales(&chan);
                for (0..48) |ob| {
                    const base: u32 = @intCast(b.bases.items.len);
                    for (0..L.halves) |h| try b.bases.append(a, at[h] + ob * 64 * codec.tileBytes(kd));
                    try down.append(a, .{ .k = kd, .blocks = L.halves, .base = base, .x_col = @intCast(slot * 2048), .y_col = @intCast(slot * L.hidden + ob * 128), .raw = 1 });
                }
            }
            s.up = try up.toOwnedSlice(a);
            s.down = try down.toOwnedSlice(a);
        }
        return b;
    }
};

/// One token's stage arguments for `n` rows: the input rotation, the down inputs' rotation and the fan-in.
pub fn rotateIn(n: u32) ex.job.Rotate {
    return .{ .rows = n, .x_stride = L.hidden, .r_stride = L.hidden, .sign = 0 };
}
pub fn rotateDown(n: u32) ex.job.Rotate {
    return .{ .rows = n, .x_stride = slots * 2048, .r_stride = slots * 2048, .slot_cols = 2048, .scales = 1 };
}
pub fn combine(n: u32) ex.job.Combine {
    return .{ .rows = n, .slots = slots, .z_stride = slots * L.hidden, .z_slot = L.hidden, .y_stride = L.hidden, .y_f32 = 1, .sign = L.hidden };
}

/// Inputs for 16 rows: bf16 activations, the experts' bf16 down inputs and routing weights.
pub const Inputs = struct {
    x: []u16,
    act: []u16,
    w: []f32,

    pub fn init() !Inputs {
        var prng = std.Random.DefaultPrng.init(0xE8B);
        const rnd = prng.random();
        const in: Inputs = .{ .x = try a.alloc(u16, rows * L.hidden), .act = try a.alloc(u16, rows * slots * 2048), .w = try a.alloc(f32, rows * slots) };
        for (in.x) |*v| v.* = ref.bf16Of(@as(f32, @floatFromInt(rnd.intRangeLessThan(i32, -1000, 1000))) * 0.0021);
        for (in.act) |*v| v.* = ref.bf16Of(@as(f32, @floatFromInt(rnd.intRangeLessThan(i32, -1000, 1000))) * 0.0013);
        for (in.w) |*v| v.* = @as(f32, @floatFromInt(rnd.intRangeLessThan(i32, 1, 400))) / 512.0;
        return in;
    }
};

/// FP64 of every gate/up output (rows x 9 x 4096) and every combined down output (rows x 6144) of 16 rows.
pub fn fp64(layer: *const L.Layer, widths: *const ex.manifest.LayerWidths, s: Set, in: Inputs, gu: []const f32, y: []const f32, check: *ref.Check) !void {
    const book = try a.create(codec.Codebook);
    defer a.destroy(book);
    book.init();
    const codes = try a.alloc(u8, L.hidden * 128);
    defer a.free(codes);
    var suh: [L.hidden]f32 = undefined;
    var svh: [128]f32 = undefined;
    var x: [L.hidden]f64 = undefined;
    var x_abs: [L.hidden]f64 = undefined;
    var block: [48]f32 = undefined;
    const want = try a.alloc(f64, 2 * rows * L.hidden);
    defer a.free(want);
    const mag = want[rows * L.hidden ..];
    @memset(want, 0);
    for (s.experts, 0..) |e, slot| {
        for ([_]L.Proj{ .gate, .up }, 0..) |p, pi| for (0..L.halves) |h| {
            const half = try layer.half(e, p, h, width(widths, e, p, h));
            try codec.decodeRun(book, half.k, .out_major, half.trellis, L.hidden, 128, codes);
            half.suh(&suh);
            half.svh(&svh);
            for (&block, half.block) |*v, c| v.* = c;
            const exact = try ref.Exact.init(a, codes, 48, &suh, &svh);
            defer exact.deinit(a);
            for (0..rows) |r| {
                for (&x, &x_abs, in.x[r * L.hidden ..][0..L.hidden]) |*v, *va, bits| {
                    v.* = ref.bf16Value(bits);
                    va.* = @abs(v.*);
                }
                var out: [128]f64 = undefined;
                const m = exact.row(&x, &x_abs, &block, &out);
                for (0..128) |c| try check.add(gu[r * slots * 4096 + slot * 4096 + pi * 2048 + h * 128 + c], out[c], @abs(svh[c]) / 128 * m);
            }
        };
        const kd = width(widths, e, .down, 0);
        const ones: [16]f32 = @splat(1);
        for (0..48) |ob| {
            var sb: f32 = 0;
            for (0..L.halves) |h| {
                const half = try layer.half(e, .down, h, kd);
                const tb = 64 * codec.tileBytes(kd);
                try codec.decodeRun(book, kd, .out_major, half.trellis[ob * tb ..][0..tb], 128, 128, codes[h * 128 * 128 ..][0 .. 128 * 128]);
                for (half.chan, 0..) |c, i| suh[h * 128 + i] = c;
                sb = half.block[ob];
            }
            for (&svh, 0..) |*v, c| v.* = @as(f32, @floatFromInt(layer.signs_out[ob * 128 + c])) * sb;
            const exact = try ref.Exact.init(a, codes[0 .. 2048 * 128], 16, suh[0..2048], &svh);
            defer exact.deinit(a);
            for (0..rows) |r| {
                for (x[0..2048], x_abs[0..2048], in.act[(r * slots + slot) * 2048 ..][0..2048], suh[0..2048]) |*v, *va, bits, sc| {
                    v.* = ref.bf16Value(bits);
                    va.* = @abs(v.* * sc);
                }
                var out: [128]f64 = undefined;
                const m = exact.row(x[0..2048], x_abs[0..2048], &ones, &out);
                const w: f64 = in.w[r * slots + slot];
                for (0..128) |c| {
                    want[r * L.hidden + ob * 128 + c] += w * out[c];
                    mag[r * L.hidden + ob * 128 + c] += @abs(w * svh[c]) / 128 * m;
                }
            }
        }
    }
    for (y[0 .. rows * L.hidden], want[0 .. rows * L.hidden], mag) |got, w, m| try check.add(got, w, m);
}
