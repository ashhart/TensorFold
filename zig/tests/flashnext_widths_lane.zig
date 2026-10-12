//! tf-flashnext-widths, decode lanes: fused stacks packed as the engine packs them, every run at its own width.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const lib = @import("flashnext_widths_lib.zig");
const lane_w = tf.flashnext_replay.lane_w;
const pack = tf.flashnext_pack;
const st = tf.safetensors;
const Buf = tf.flashnext_replay.Buf;
const a = lib.a;

const wide = 64; // rows of the reference call: two 32-row blocks (the MTP head's 4-row windows reach 64)
const windows = [_]usize{ 1, 2, 3, 8, 16, 17, 33 }; // 17 and 33 cross the 16- and 32-row blocks

/// A stack member: rows, bits and group as the checkpoint stores it.
const Member = struct { rows: usize, bits: u8, group: usize };

fn tensor(dtype: st.DType, rows: usize, cols: usize, bytes: []const u8) st.Tensor {
    var shape: [st.max_rank]usize = @splat(0);
    shape[0] = rows;
    shape[1] = cols;
    return .{ .dtype = dtype, .rank = 2, .shape = shape, .bytes = bytes };
}

fn q(m: lib.Mat) !pack.Q {
    const kw = m.k * m.bits / 32;
    const tables = std.mem.sliceAsBytes(m.scales);
    return .{ .w = tensor(.u32, m.n, kw, m.words), .s = tensor(.bf16, m.n, m.k / m.group, tables), .b = tensor(.bf16, m.n, m.k / m.group, std.mem.sliceAsBytes(m.biases)), .rows = m.n, .k = m.k, .kw = kw, .spec = try @FieldType(pack.Q, "spec").init(m.bits, m.group) };
}

fn blob(out: *const pack.Out, name: []const u8) ?[]const u8 {
    for (out.names.items, 0..) |n, i| if (std.mem.eql(u8, n, name)) return out.blobs.items[i];
    return null;
}

/// The device lane the engine would load from these pack tensors (lane_w.of's reading, from memory).
fn wideOf(device: mtl.Device, out: *const pack.Out, k: usize, rows: usize) !lane_w.Wide {
    const uniform = [_]u32{ 6, 32, @intCast(rows) }; // a 6/32 lane writes no table
    const table = if (blob(out, "t.runs")) |b| std.mem.bytesAsSlice(u32, @as([]align(1) const u8, b)) else &uniform;
    const n_runs = table.len / 3;
    const run_rows = try a.alloc(usize, n_runs);
    for (run_rows, 0..) |*n, i| n.* = table[i * 3 + 2];
    const maps = if (n_runs > 1) blk: {
        const m = blob(out, "t.members").?;
        const members = try a.alloc(u32, m.len / 4);
        @memcpy(std.mem.sliceAsBytes(members), m);
        break :blk try lane_w.columnMaps(a, members, run_rows);
    } else null;
    const runs = try a.alloc(lane_w.Product, n_runs);
    var total: usize = 0;
    for (runs, 0..) |*p, i| {
        const base = if (n_runs == 1) "t" else try std.fmt.allocPrint(a, "t.r{d}", .{i});
        const wq = blob(out, try std.fmt.allocPrint(a, "{s}.wq", .{base})).?;
        const sbt = blob(out, try std.fmt.allocPrint(a, "{s}.sbt", .{base})).?;
        p.* = .{ .wq = .{ .b = try lib.upload(device, wq) }, .sbt = .{ .b = try lib.upload(device, sbt) }, .n = run_rows[i], .k = k, .bits = @intCast(table[i * 3]), .group = @intCast(table[i * 3 + 1]), .tiled = pack.lane.tiled(run_rows[i]) };
        if (maps) |m| p.map = .{ .b = try lib.upload(device, std.mem.sliceAsBytes(m[i])) };
        total += run_rows[i];
    }
    return .{ .runs = runs, .n = total };
}

fn mdims(device: mtl.Device, rows: usize) !Buf {
    const b = try device.buffer(32, lib.opts);
    b.slice(i32, 2)[0] = @intCast(rows);
    b.slice(i32, 2)[1] = @intCast((rows + 15) / 16 * 16);
    return .{ .b = b };
}

/// Rows [start, start + m) of x through the lane at its widths, in windows of m rows.
fn launchWindows(device: mtl.Device, queue: mtl.Queue, cache: *lane_w.Cache, w: *const lane_w.Wide, x: lib.Inputs, m: usize, y: mtl.Buffer) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    var start: usize = 0;
    while (start + m <= wide) : (start += m) {
        try lane_w.encode(e, a, device, cache, w, .{ .b = x.buf, .off = start * x.k * 2 }, try mdims(device, m), m, .{ .b = y, .off = start * w.n * 2 });
    }
    try lib.commit(cb, e);
}

/// One stack: packed by the engine's packer, run at 64 rows against float64, then every window's rows bit for bit.
fn stack(t: *lib.Tally, device: mtl.Device, queue: mtl.Queue, cache: *lane_w.Cache, members: []const Member, k: usize, seed: u64) !void {
    var mats = try a.alloc(lib.Mat, members.len);
    var qs = try a.alloc(pack.Q, members.len);
    var label: std.ArrayList(u8) = .empty;
    try label.print(a, "lane K {d}:", .{k});
    for (members, 0..) |m, i| {
        mats[i] = try lib.Mat.init(m.rows, k, m.bits, m.group, seed + i);
        qs[i] = try q(mats[i]);
        try label.print(a, " {d}x q{d}/g{d}", .{ m.rows, m.bits, m.group });
    }
    var out = pack.Out.init(a);
    try pack.lane.putLane(&out, a, "t", qs);
    var rows: usize = 0;
    for (members) |m| rows += m.rows;
    const w = try wideOf(device, &out, k, rows);
    try label.print(a, " (SK {d})", .{lane_w.splitK(w.runs[0].n, k)});
    const x = try lib.Inputs.init(device, wide, k, seed ^ 0x5eed);
    const all = try lib.poisoned(device, wide * w.n * 2);
    try launchWindows(device, queue, cache, &w, x, wide, all);
    var worst = lib.Worst{};
    var ctrl = lib.Worst{};
    const got = all.slice(u16, wide * w.n);
    var col: usize = 0;
    for (mats) |m| {
        const terms = try m.dense();
        const wrong = try m.skewed(1.01).dense();
        defer for (terms ++ wrong) |v| a.free(v);
        for (0..m.n) |j| {
            for (0..wide) |r| {
                const g = lib.fromBf(got[r * w.n + col + j]);
                const ref = lib.dot(x.row(r), terms, j);
                worst.add(col + j, g, ref[0], lib.ratio(g, ref[0], ref[1], m.group + k / m.group + 16, true));
                const cref = lib.dot(x.row(r), wrong, j);
                ctrl.add(col + j, g, cref[0], lib.ratio(g, cref[0], cref[1], m.group + k / m.group + 16, true));
            }
        }
        col += m.n;
    }
    worst.report(t, label.items);
    ctrl.control(t, label.items);
    var same = true;
    for (windows) |m| {
        const y = try lib.poisoned(device, wide * w.n * 2);
        try launchWindows(device, queue, cache, &w, x, m, y);
        same = lib.sameRows(t, label.items, m, all.slice(u8, wide * w.n * 2), y.slice(u8, wide * w.n * 2), wide / m * m, w.n * 2) and same;
    }
    if (same) std.debug.print("RESULT {s}: every row's bits equal in windows of 1, 2, 3, 8, 16, 17 and 33 rows ({d} runs)\n", .{ label.items, w.runs.len });
}

/// The generic source at 6/32 against the recorded kernel at a recorded shape: the same bits.
fn recordedTwin(t: *lib.Tally, device: mtl.Device, queue: mtl.Queue, cache: *lane_w.Cache) !void {
    const n = 2560;
    const k = 6144;
    const m = try lib.Mat.init(n, k, 6, 32, 0x632);
    var out = pack.Out.init(a);
    try pack.lane.putLane(&out, a, "t", &.{try q(m)});
    const wq: Buf = .{ .b = try lib.upload(device, blob(&out, "t.wq").?) };
    const sbt: Buf = .{ .b = try lib.upload(device, blob(&out, "t.sbt").?) };
    const x = try lib.Inputs.init(device, 16, k, 0x633);
    const md = try mdims(device, 16);
    const text = try lane_w.template(device.tensorUnits());
    const rec = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    const rec_pipe = try mtl.Pipeline.init(device, rec, try lane_w.functionName(a, text), false);
    const y_rec = try lib.poisoned(device, 16 * n * 2);
    const y_gen = try lib.poisoned(device, 16 * n * 2);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(rec_pipe);
    for ([_]Buf{ .{ .b = x.buf }, .{ .b = x.buf }, wq, sbt, md, .{ .b = y_rec } }, 0..) |b, j| e.setBuffer(b.b, b.off, j);
    e.dispatchThreads(mtl.Size.of(n / 32 * 32 * 8, 1, 1), mtl.Size.of(32 * 8, 1, 1));
    const w = lane_w.Wide{ .runs = &.{.{ .wq = wq, .sbt = sbt, .n = n, .k = k, .bits = 6, .group = 32 }}, .n = n };
    try lane_w.encode(e, a, device, cache, &w, .{ .b = x.buf }, md, 16, .{ .b = y_gen });
    try lib.commit(cb, e);
    const same = lib.sameRows(t, "lane recorded 6/32 [2560, 6144] against the generic source", 16, y_rec.slice(u8, 16 * n * 2), y_gen.slice(u8, 16 * n * 2), 16, n * 2);
    if (same) std.debug.print("RESULT lane recorded 6/32 [2560, 6144]: the generic source's bits equal the recorded kernel's\n", .{});
}

pub fn run(t: *lib.Tally, device: mtl.Device, queue: mtl.Queue) !void {
    var cache: lane_w.Cache = .empty;
    try recordedTwin(t, device, queue, &cache);
    var seed: u64 = 0x1a9e;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        for ([_][2]usize{ .{ 256, 2560 }, .{ 64, 6144 } }) |shape| {
            seed += 16;
            try stack(t, device, queue, &cache, &.{.{ .rows = shape[0], .bits = bits, .group = group }}, shape[1], seed);
        }
    };
    // the public checkpoints' mixed stacks (q, k, v, index_qk and the DeltaNet inputs), rows cut down
    const stacks = [_][]const Member{
        &.{ .{ .rows = 256, .bits = 3, .group = 64 }, .{ .rows = 32, .bits = 3, .group = 64 }, .{ .rows = 32, .bits = 8, .group = 64 }, .{ .rows = 64, .bits = 3, .group = 64 } },
        &.{ .{ .rows = 256, .bits = 6, .group = 64 }, .{ .rows = 32, .bits = 6, .group = 64 }, .{ .rows = 32, .bits = 6, .group = 128 }, .{ .rows = 64, .bits = 6, .group = 64 } },
        &.{ .{ .rows = 256, .bits = 4, .group = 64 }, .{ .rows = 64, .bits = 5, .group = 128 }, .{ .rows = 32, .bits = 5, .group = 128 }, .{ .rows = 32, .bits = 6, .group = 64 } },
        &.{ .{ .rows = 256, .bits = 3, .group = 64 }, .{ .rows = 64, .bits = 6, .group = 64 }, .{ .rows = 32, .bits = 6, .group = 64 }, .{ .rows = 32, .bits = 6, .group = 64 } },
        &.{ .{ .rows = 128, .bits = 4, .group = 32 }, .{ .rows = 64, .bits = 4, .group = 64 }, .{ .rows = 32, .bits = 2, .group = 32 } },
        // oQ4e layer 21's DeltaNet inputs: runs of 112 (5-bit z and b) and 48 (6-bit a) columns, untiled
        &.{ .{ .rows = 256, .bits = 4, .group = 64 }, .{ .rows = 64, .bits = 5, .group = 128 }, .{ .rows = 48, .bits = 5, .group = 128 }, .{ .rows = 48, .bits = 6, .group = 64 } },
        &.{.{ .rows = 48, .bits = 8, .group = 64 }},
    };
    for (stacks) |s| {
        seed += 16;
        try stack(t, device, queue, &cache, s, 2560, seed);
    }
    // K slices 1 and 2 (the vocabulary head and draft at 8/64, gdn.in), at 8 bits and a low width
    for ([_]usize{ 32768, 16384 }) |n| for ([_][2]usize{ .{ 8, 64 }, .{ 3, 64 } }) |wd| {
        seed += 16;
        try stack(t, device, queue, &cache, &.{.{ .rows = n, .bits = @intCast(wd[0]), .group = wd[1] }}, 2560, seed);
    };
}
