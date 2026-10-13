//! EXLR8 projection cases for the Metal and CUDA checks: synthetic weights, the job tables, FP64 references.

const std = @import("std");
const ex = @import("exlr8");
const ref = @import("exlr8_ref");
const codec = ex.codec;
const Job = ex.job.Job;
const a = std.heap.page_allocator;
pub const rows_max = ex.job.max_rows;

/// `slots` experts, each `blocks` input blocks of 128 at width k into `outs` output blocks; down cases fan in.
pub const Case = struct {
    name: []const u8,
    k: [3]u8,
    slots: usize,
    blocks: usize,
    outs: usize,
    down: bool,
    y_f32: bool,
};

pub const cases = [_]Case{
    .{ .name = "gate/up K2 f32", .k = .{ 2, 0, 0 }, .slots = 1, .blocks = 48, .outs = 1, .down = false, .y_f32 = true },
    .{ .name = "gate/up K3 f32", .k = .{ 3, 0, 0 }, .slots = 1, .blocks = 48, .outs = 1, .down = false, .y_f32 = true },
    .{ .name = "gate/up K4 bf16", .k = .{ 4, 0, 0 }, .slots = 1, .blocks = 48, .outs = 1, .down = false, .y_f32 = false },
    .{ .name = "down K2/K3/K4 combine f32", .k = .{ 2, 3, 4 }, .slots = 3, .blocks = 16, .outs = 8, .down = true, .y_f32 = true },
    .{ .name = "down K3/K4/K2 combine bf16", .k = .{ 3, 4, 2 }, .slots = 3, .blocks = 16, .outs = 8, .down = true, .y_f32 = false },
};

pub const Data = struct {
    c: Case,
    tiles: []u8,
    x: []u16,
    signs: []i8,
    scales: []f16,
    weights: []f32,

    pub fn init(c: Case, seed: u64) !Data {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        var bytes: usize = 0;
        for (0..c.slots) |e| bytes += c.blocks * c.outs * 64 * codec.tileBytes(c.k[e]);
        const d: Data = .{
            .c = c,
            .tiles = try a.alloc(u8, bytes),
            .x = try a.alloc(u16, c.slots * rows_max * c.blocks * 128),
            .signs = try a.alloc(i8, c.blocks * 128 + c.outs * 128),
            .scales = try a.alloc(f16, c.slots * (c.blocks * 128 + c.outs * 128 + 2 * 48)),
            .weights = try a.alloc(f32, rows_max * c.slots),
        };
        r.bytes(d.tiles);
        for (d.x, 0..) |*v, i| v.* = if (i % 37 == 0) 0 else ref.bf16Of(@as(f32, @floatFromInt(r.intRangeLessThan(i32, -2000, 2000))) * 0.00153);
        for (d.signs) |*v| v.* = if (r.boolean()) 1 else -1;
        for (d.scales) |*v| v.* = @floatCast(@as(f32, @floatFromInt(r.intRangeLessThan(i32, 200, 2000))) * 0.000731 * (if (r.boolean()) @as(f32, 1) else -1));
        for (d.weights) |*v| v.* = @as(f32, @floatFromInt(r.intRangeLessThan(i32, 1, 1000))) / 997.0;
        return d;
    }
    pub fn deinit(d: Data) void {
        a.free(d.tiles);
        a.free(d.x);
        a.free(d.signs);
        a.free(d.scales);
        a.free(d.weights);
    }
    fn slotStart(d: Data, e: usize) usize {
        var at: usize = 0;
        for (0..e) |i| at += d.c.blocks * d.c.outs * 64 * codec.tileBytes(d.c.k[i]);
        return at;
    }
    /// Where input block b's 64 tiles for output block o of slot e start.
    pub fn run(d: Data, e: usize, b: usize, o: usize) usize {
        return d.slotStart(e) + (b * d.c.outs + o) * 64 * codec.tileBytes(d.c.k[e]);
    }
    /// Slot e's fp16 table: k_in channel scales (down's suh), n_out output scales, 48 input-block, 48 output-block.
    pub fn scaleBase(d: Data, e: usize) usize {
        return e * (d.c.blocks * 128 + d.c.outs * 128 + 2 * 48);
    }
    pub fn kIn(d: Data) usize {
        return d.c.blocks * 128;
    }
    pub fn nOut(d: Data) usize {
        return d.c.outs * 128;
    }
    /// Input and output row widths: the slots' inputs side by side; down fans in to one output.
    pub fn xWidth(d: Data) usize {
        return d.c.slots * d.kIn();
    }
    pub fn yWidth(d: Data) usize {
        return if (d.c.down) d.nOut() else d.c.slots * d.nOut();
    }
};

/// The device tables a case's jobs read: tile-run offsets, jobs, per-slot scale offsets and block-scale offsets.
pub const Plan = struct {
    bases: []u64,
    jobs: []Job,
    suh_of: []u32,
    sv_of: []u32,

    pub fn init(d: Data) !Plan {
        const c = d.c;
        var bases: std.ArrayList(u64) = .empty;
        var jobs: std.ArrayList(Job) = .empty;
        for (0..c.slots) |e| for (0..c.outs) |o| {
            const base: u32 = @intCast(bases.items.len);
            for (0..c.blocks) |b| try bases.append(a, d.run(e, b, o));
            const sb: u32 = @intCast(d.scaleBase(e));
            const y_col: u32 = @intCast(e * d.nOut() + o * 128);
            try jobs.append(a, if (c.down)
                .{ .k = c.k[e], .blocks = @intCast(c.blocks), .base = base, .x_col = @intCast(e * d.kIn()), .y_col = y_col, .raw = 1 }
            else
                .{ .k = c.k[e], .blocks = @intCast(c.blocks), .base = base, .in_scale = sb + @as(u32, @intCast(d.kIn() + d.nOut())), .out_scale = sb + @as(u32, @intCast(d.kIn())), .y_col = y_col });
        };
        const p: Plan = .{ .bases = try bases.toOwnedSlice(a), .jobs = try jobs.toOwnedSlice(a), .suh_of = try a.alloc(u32, c.slots), .sv_of = try a.alloc(u32, c.slots) };
        for (p.suh_of, p.sv_of, 0..) |*s, *v, e| {
            s.* = @intCast(d.scaleBase(e));
            v.* = @intCast(d.scaleBase(e) + d.kIn() + d.nOut() + 48);
        }
        return p;
    }
    pub fn deinit(p: Plan) void {
        a.free(p.bases);
        a.free(p.jobs);
        a.free(p.suh_of);
        a.free(p.sv_of);
    }
    pub fn rotate(d: Data, rows: u32) ex.job.Rotate {
        return .{ .rows = rows, .x_stride = @intCast(d.xWidth()), .r_stride = @intCast(d.xWidth()), .sign = if (d.c.down) ex.job.none else 0, .slot_cols = @intCast(d.kIn()), .scales = @intFromBool(d.c.down) };
    }
    pub fn combine(d: Data, rows: u32) ex.job.Combine {
        return .{ .rows = rows, .slots = @intCast(d.c.slots), .z_stride = @intCast(d.c.slots * d.nOut()), .z_slot = @intCast(d.nOut()), .y_stride = @intCast(d.yWidth()), .y_f32 = @intFromBool(d.c.y_f32), .sign = @intCast(d.kIn()) };
    }
};

/// Every window width 1..16's outputs in order, and each row's output alone.
pub const Outputs = struct {
    windows: []u8,
    alone: []u8,
    width: usize,
    elem: usize,

    pub fn init(d: Data) !Outputs {
        const elem: usize = if (d.c.y_f32) 4 else 2;
        return .{ .windows = try a.alloc(u8, 136 * d.yWidth() * elem), .alone = try a.alloc(u8, rows_max * d.yWidth() * elem), .width = d.yWidth(), .elem = elem };
    }
    pub fn deinit(o: Outputs) void {
        a.free(o.windows);
        a.free(o.alone);
    }
};

/// Every row of a window equals the same row computed alone, byte for byte; returns the pairs compared.
pub fn rowsEqual(o: Outputs) !usize {
    var at: usize = 0;
    var pairs: usize = 0;
    const row = o.width * o.elem;
    for (1..rows_max + 1) |m| {
        for (0..m) |r| {
            if (!std.mem.eql(u8, o.windows[at + r * row ..][0..row], o.alone[r * row ..][0..row])) {
                std.debug.print("row {d} of a {d}-row window differs from the row alone\n", .{ r, m });
                return error.RowInvariantMismatch;
            }
            pairs += 1;
        }
        at += m * row;
    }
    return pairs;
}

/// The rows-alone outputs against FP64 of x W with W exact, within the kernels' first-order fp32 bound.
pub fn fp64(book: *const codec.Codebook, d: Data, o: Outputs, check: *ref.Check) !void {
    const c = d.c;
    const k_in = d.kIn();
    const n_out = d.nOut();
    const codes = try a.alloc(u8, k_in * 128);
    defer a.free(codes);
    const suh = try a.alloc(f32, k_in);
    defer a.free(suh);
    const want = try a.alloc(f64, 2 * rows_max * o.width);
    defer a.free(want);
    const mag = want[rows_max * o.width ..];
    @memset(want, 0);
    const in = try a.alloc(f64, 2 * k_in);
    defer a.free(in);
    const in_abs = in[k_in..];
    var svh: [128]f32 = undefined;
    var scales: [64]f32 = undefined;
    for (0..c.slots) |e| for (0..c.outs) |ob| {
        const tb = 64 * codec.tileBytes(c.k[e]);
        for (0..c.blocks) |b| try codec.decodeRun(book, c.k[e], .out_major, d.tiles[d.run(e, b, ob)..][0..tb], 128, 128, codes[b * 128 * 128 ..][0 .. 128 * 128]);
        const sb = d.scaleBase(e);
        const blocks = sb + k_in + n_out;
        for (suh, 0..) |*v, i| v.* = if (c.down) d.scales[sb + i] else @as(f32, @floatFromInt(d.signs[i])) * d.scales[blocks + i / 128];
        for (&svh, 0..) |*v, i| v.* = if (c.down) @as(f32, @floatFromInt(d.signs[k_in + ob * 128 + i])) * d.scales[blocks + 48 + ob] else d.scales[sb + k_in + ob * 128 + i];
        for (scales[0..c.blocks], 0..) |*v, b| v.* = if (c.down) 1 else d.scales[blocks + b];
        const exact = try ref.Exact.init(a, codes, c.blocks, suh, &svh);
        defer exact.deinit(a);
        for (0..rows_max) |r| {
            for (in[0..k_in], in_abs, 0..) |*v, *va, i| {
                v.* = ref.bf16Value(d.x[r * c.slots * k_in + e * k_in + i]);
                va.* = if (c.down) @abs(v.* * suh[i]) else @abs(v.*);
            }
            var y: [128]f64 = undefined;
            const m = exact.row(in[0..k_in], in_abs, scales[0..c.blocks], &y);
            const w: f64 = if (c.down) d.weights[r * c.slots + e] else 1;
            const col0 = if (c.down) ob * 128 else e * n_out + ob * 128;
            for (0..128) |q| {
                want[r * o.width + col0 + q] += w * y[q];
                mag[r * o.width + col0 + q] += @abs(w * svh[q]) / 128 * m;
            }
        }
    };
    for (0..rows_max) |r| for (0..o.width) |q| {
        const i = r * o.width + q;
        const got: f32 = if (o.elem == 4) @bitCast(std.mem.readInt(u32, o.alone[4 * i ..][0..4], .little)) else ref.bf16Value(std.mem.readInt(u16, o.alone[2 * i ..][0..2], .little));
        // bf16 outputs add their own rounding: 2^-9 of the value, inside 256 gamma |want|.
        try check.add(got, want[i], if (o.elem == 4) mag[i] else mag[i] + 256 * @abs(want[i]));
    };
}
