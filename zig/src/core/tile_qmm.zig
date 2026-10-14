//! Prompt matmuls before the M5 (kernels/metal/core/tile_qmm.metal): 32x32 tiles, dense or over rows sorted by expert.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// The code widths the tiles unpack (groups of 64).
pub const bit_widths = [_]u8{ 4, 6, 8 };
/// Rows and columns a tile.
pub const tile = 32;

pub const Args = extern struct { k: i32, n: i32, r: i32, one: f32 = 1.0 };

pub const Pipelines = struct {
    dense: [bit_widths.len]mtl.Pipeline,
    grouped: [bit_widths.len]mtl.Pipeline,
    grouped_glu: [bit_widths.len]mtl.Pipeline,
    twin: [bit_widths.len]mtl.Pipeline,
    twin_indexed: [bit_widths.len]mtl.Pipeline,
    twin_glu: [bit_widths.len]mtl.Pipeline,
    work: mtl.Pipeline,
    xsum: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_tile_qmm, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        inline for (bit_widths, 0..) |bits, i| {
            p.dense[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_tile_qmm_q{d}", .{bits}), false);
            p.grouped[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_tile_qmm_grouped_q{d}", .{bits}), false);
            p.grouped_glu[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_tile_qmm_grouped_glu_q{d}", .{bits}), false);
            p.twin[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_tile_qmv_q{d}", .{bits}), false);
            p.twin_indexed[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_tile_qmv_indexed_q{d}", .{bits}), false);
            p.twin_glu[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_tile_qmv_glu_q{d}", .{bits}), false);
        }
        p.work = try mtl.Pipeline.init(device, lib, "tf_tile_qmm_work", false);
        p.xsum = try mtl.Pipeline.init(device, lib, "tf_tile_xsum", false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.dense) |x| x.deinit();
        for (p.grouped) |x| x.deinit();
        for (p.grouped_glu) |x| x.deinit();
        for (p.twin) |x| x.deinit();
        for (p.twin_indexed) |x| x.deinit();
        for (p.twin_glu) |x| x.deinit();
        p.work.deinit();
        p.xsum.deinit();
    }
};

/// A checkpoint's [n, k] matrix (or experts' first): codes [n][k*bits/8], scales and biases [n][k/64].
pub const Matrix = struct { w: mtl.Buffer, w_off: usize = 0, scales: mtl.Buffer, s_off: usize = 0, biases: mtl.Buffer, b_off: usize = 0, n: usize, k: usize, bits: u8 };

fn width(m: Matrix) !usize {
    if (m.k > 64 * 192) return error.UnsupportedTileShape;
    if (m.n == 0 or m.n % tile != 0 or m.k == 0 or m.k % 64 != 0 or m.n > std.math.maxInt(i32) or m.k > std.math.maxInt(i32)) return error.UnsupportedTileShape;
    return std.mem.indexOfScalar(u8, &bit_widths, m.bits) orelse error.UnsupportedTileBits;
}

fn bind(e: mtl.ComputeEncoder, m: Matrix, args: Args, x: mtl.Buffer, x_off: usize, y: mtl.Buffer, y_off: usize) void {
    e.setBuffer(x, x_off, 0);
    e.setBuffer(m.w, m.w_off, 1);
    e.setBuffer(m.scales, m.s_off, 2);
    e.setBuffer(m.biases, m.b_off, 3);
    e.setValue(args, 4);
    e.setBuffer(y, y_off, 5);
}

/// y [rows, n] = x [rows, k] W^T; each row's sums are the same at any number of rows; `sums` from encodeSums on x.
pub fn encode(e: mtl.ComputeEncoder, p: *const Pipelines, m: Matrix, x: mtl.Buffer, x_off: usize, sums: mtl.Buffer, sums_off: usize, y: mtl.Buffer, y_off: usize, rows: usize) !void {
    const i = try width(m);
    if (rows == 0 or rows > std.math.maxInt(i32)) return error.NoRows;
    e.setPipeline(p.dense[i]);
    bind(e, m, .{ .k = @intCast(m.k), .n = @intCast(m.n), .r = @intCast(rows) }, x, x_off, y, y_off);
    e.setBuffer(sums, sums_off, 6);
    e.dispatchGroups(.{ .width = m.n / tile, .height = (rows + tile - 1) / tile }, .{ .width = 128 });
}

/// Threadgroup bytes a twin threadgroup asks for: three a core keep the M4 Pro's memory pipes fullest (measured).
const twin_tg_bytes = 10240;
/// Rows the twin takes in one call at most (decode and verify windows).
pub const twin_rows = 32;
/// Columns a twin threadgroup: 2 simdgroups of 4 columns (n is a multiple of the tile).
fn twinColumns(bits: u8) usize {
    _ = bits;
    return 8;
}

/// Each of `rows` rows' group sums (f32 [rows, k / 64] into `sums`), which the twin calls on those rows read.
pub fn encodeSums(e: mtl.ComputeEncoder, p: *const Pipelines, k: usize, x: mtl.Buffer, x_off: usize, sums: mtl.Buffer, rows: usize) void {
    e.setPipeline(p.xsum);
    e.setBuffer(x, x_off, 0);
    e.setValue(Args{ .k = @intCast(k), .n = 0, .r = @intCast(rows) }, 1);
    e.setBuffer(sums, 0, 2);
    e.dispatchThreads(.{ .width = k / 64, .height = rows }, .{ .width = @min(k / 64, 256) });
}

/// The tiles' bits for 1-32 rows (decode and windows), each row's group sums taken on the way.
pub fn encodeTwin(e: mtl.ComputeEncoder, p: *const Pipelines, m: Matrix, x: mtl.Buffer, x_off: usize, y: mtl.Buffer, y_off: usize, rows: usize) !void {
    const i = try width(m);
    if (rows == 0 or rows > twin_rows) return error.NoRows;
    e.setPipeline(p.twin[i]);
    bind(e, m, .{ .k = @intCast(m.k), .n = @intCast(m.n), .r = @intCast(rows) }, x, x_off, y, y_off);
    // 4 bits: 4 rows a threadgroup over one load of the codes; partials staged 32 groups a round
    const rt: usize = if (m.bits == 4) @min(4, rows) else 1;
    e.setThreadgroupMemory(@max(twin_tg_bytes, 2 * rt * (twinColumns(m.bits) / 2 + 1) * 32 * 4), 0);
    e.dispatchGroups(.{ .width = m.n / twinColumns(m.bits), .height = (rows + rt - 1) / rt }, .{ .width = 64 });
}

/// The twin by slot: slot y reads expert ids[y]'s matrix (`m` the first) and x row y / repeat.
pub fn encodeTwinIndexed(e: mtl.ComputeEncoder, p: *const Pipelines, m: Matrix, x: mtl.Buffer, y: mtl.Buffer, ids: mtl.Buffer, slots: usize, repeat: usize) !void {
    const i = try width(m);
    if (slots == 0 or repeat == 0) return error.NoRows;
    e.setPipeline(p.twin_indexed[i]);
    bind(e, m, .{ .k = @intCast(m.k), .n = @intCast(m.n), .r = @intCast(repeat) }, x, 0, y, 0);
    e.setBuffer(ids, 0, 6);
    e.setThreadgroupMemory(@max(twin_tg_bytes, 2 * (twinColumns(m.bits) / 2 + 1) * 32 * 4), 0);
    e.dispatchGroups(.{ .width = m.n / twinColumns(m.bits), .height = slots }, .{ .width = 64 });
}

/// encodeTwinIndexed on gate and up (one shape) into swiglu(gate, up), mlx-lm's compiled swiglu on bf16 rows.
pub fn encodeTwinGlu(e: mtl.ComputeEncoder, p: *const Pipelines, gate: Matrix, up: Matrix, x: mtl.Buffer, y: mtl.Buffer, ids: mtl.Buffer, slots: usize, repeat: usize) !void {
    const i = try width(gate);
    if (slots == 0 or repeat == 0 or up.n != gate.n or up.k != gate.k or up.bits != gate.bits) return error.NoRows;
    e.setPipeline(p.twin_glu[i]);
    bind(e, gate, .{ .k = @intCast(gate.k), .n = @intCast(gate.n), .r = @intCast(repeat) }, x, 0, y, 0);
    e.setBuffer(ids, 0, 6);
    e.setBuffer(up.w, up.w_off, 7);
    e.setBuffer(up.scales, up.s_off, 8);
    e.setBuffer(up.biases, up.b_off, 9);
    e.setThreadgroupMemory(@max(twin_tg_bytes, 2 * (twinColumns(gate.bits) / 2 + 1) * 32 * 4), 0);
    e.dispatchGroups(.{ .width = gate.n / twinColumns(gate.bits), .height = slots }, .{ .width = 64 });
}

/// Whether the twin's rows equal the tiles' for `m` on this GPU (random rows; another chip may round MMAs otherwise).
pub fn twinExact(p: *const Pipelines, device: mtl.Device, m: Matrix) !bool {
    const rows = 40;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const x = try device.buffer(rows * m.k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x7173_0001);
    for (x.slice(u16, rows * m.k)) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) * 2 - 1)) >> 16);
    const tiled = try device.buffer(rows * m.n * 2, mtl.ResourceOptions.shared);
    defer tiled.deinit();
    const twin = try device.buffer(rows * m.n * 2, mtl.ResourceOptions.shared);
    defer twin.deinit();
    const sums = try device.buffer(rows * (m.k / 64) * 4, mtl.ResourceOptions.shared);
    defer sums.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    encodeSums(e, p, m.k, x, 0, sums, rows);
    try encode(e, p, m, x, 0, sums, 0, tiled, 0, rows);
    try encodeTwin(e, p, m, x, 0, twin, 0, twin_rows);
    try encodeTwin(e, p, m, x, twin_rows * m.k * 2, twin, twin_rows * m.n * 2, rows - twin_rows);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("tile_qmm check: {s}", .{why});
        return error.TileCheckFailed;
    }
    return std.mem.eql(u16, tiled.slice(u16, rows * m.n), twin.slice(u16, rows * m.n));
}

/// Work items at most on `rows` sorted rows over `experts`: whole tiles, plus one partial tile an expert.
pub fn items(rows: usize, experts: usize) usize {
    return rows / tile + @min(experts, rows) + 1;
}

/// The work list (u32 [items + 1]) from `starts` [experts + 1], before the grouped calls on those rows.
pub fn encodeWork(e: mtl.ComputeEncoder, p: *const Pipelines, starts: mtl.Buffer, experts: usize, work: mtl.Buffer) void {
    e.setPipeline(p.work);
    e.setBuffer(starts, 0, 0);
    e.setValue(@as(u32, @intCast(experts)), 1);
    e.setBuffer(work, 0, 2);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 1 });
}

/// y = x W[e]^T on `rows` rows sorted by expert (expert e's from starts[e]); sorted row j reads x and sums row map[j].
pub fn encodeGrouped(e: mtl.ComputeEncoder, p: *const Pipelines, m: Matrix, experts: usize, x: mtl.Buffer, sums: mtl.Buffer, map: mtl.Buffer, y: mtl.Buffer, rows: usize, starts: mtl.Buffer, work: mtl.Buffer) !void {
    const i = try width(m);
    if (rows == 0 or rows > std.math.maxInt(i32)) return error.NoRows;
    e.setPipeline(p.grouped[i]);
    bind(e, m, .{ .k = @intCast(m.k), .n = @intCast(m.n), .r = @intCast(rows) }, x, 0, y, 0);
    e.setBuffer(starts, 0, 6);
    e.setBuffer(work, 0, 7);
    e.setBuffer(sums, 0, 8);
    e.setBuffer(map, 0, 12);
    e.dispatchGroups(.{ .width = m.n / tile, .height = items(rows, experts) }, .{ .width = 128 });
}

/// encodeGrouped on gate and up (the same shape) into swiglu(gate, up): a tile's halves are 16 columns of each.
pub fn encodeGroupedGlu(e: mtl.ComputeEncoder, p: *const Pipelines, gate: Matrix, up: Matrix, experts: usize, x: mtl.Buffer, sums: mtl.Buffer, map: mtl.Buffer, y: mtl.Buffer, rows: usize, starts: mtl.Buffer, work: mtl.Buffer) !void {
    const i = try width(gate);
    if (rows == 0 or rows > std.math.maxInt(i32) or up.n != gate.n or up.k != gate.k or up.bits != gate.bits) return error.NoRows;
    e.setPipeline(p.grouped_glu[i]);
    bind(e, gate, .{ .k = @intCast(gate.k), .n = @intCast(gate.n), .r = @intCast(rows) }, x, 0, y, 0);
    e.setBuffer(starts, 0, 6);
    e.setBuffer(work, 0, 7);
    e.setBuffer(sums, 0, 8);
    e.setBuffer(up.w, up.w_off, 9);
    e.setBuffer(up.scales, up.s_off, 10);
    e.setBuffer(up.biases, up.b_off, 11);
    e.setBuffer(map, 0, 12);
    e.dispatchGroups(.{ .width = gate.n / (tile / 2), .height = items(rows, experts) }, .{ .width = 128 });
}

test "work items bound every expert's tiles" {
    try std.testing.expectEqual(@as(usize, 2048 / 32 + 64 + 1), items(2048, 64));
    try std.testing.expectEqual(@as(usize, 0 + 4 + 1), items(4, 64));
}
