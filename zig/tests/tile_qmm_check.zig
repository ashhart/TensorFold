//! core/tile_qmm on synthetic 4-, 6- and 8-bit matrices: row independence, twin and grouped bits, a float64 reference.
const std = @import("std");
const mtl = @import("metal");
const tile = @import("tile_qmm");

const opts = mtl.ResourceOptions.shared;

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn fromBf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// Code `i` of a row's little-endian bit stream.
fn code(bytes: []const u8, bits: u8, i: usize) u32 {
    const off = i * @as(usize, bits);
    const lo: u32 = bytes[off / 8];
    const hi: u32 = if (off / 8 + 1 < bytes.len) bytes[off / 8 + 1] else 0;
    return ((lo | (hi << 8)) >> @intCast(off % 8)) & ((@as(u32, 1) << @intCast(bits)) - 1);
}

const n = 96;
const max_rows = 70;

/// Random experts back to back (codes, scales, biases) and input rows.
const Synth = struct {
    w: mtl.Buffer,
    s: mtl.Buffer,
    b: mtl.Buffer,
    x: mtl.Buffer,
    bits: u8,
    k: usize,
    experts: usize,

    fn init(device: mtl.Device, bits_in: u8, seed: u64, k: usize, experts: usize) !Synth {
        const bits: usize = bits_in;
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const groups = experts * n * (k / 64);
        const bytes = experts * n * k * bits / 8;
        const out = Synth{
            .w = try device.buffer(bytes, opts),
            .s = try device.buffer(groups * 2, opts),
            .b = try device.buffer(groups * 2, opts),
            .x = try device.buffer(max_rows * k * 2, opts),
            .bits = bits_in,
            .k = k,
            .experts = experts,
        };
        r.bytes(out.w.slice(u8, bytes));
        for (out.s.slice(u16, groups)) |*v| v.* = toBf(r.float(f32) * 0.02 + 0.001);
        for (out.b.slice(u16, groups)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.1);
        for (out.x.slice(u16, max_rows * k)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
        return out;
    }

    fn deinit(s: Synth) void {
        for ([_]mtl.Buffer{ s.w, s.s, s.b, s.x }) |b| b.deinit();
    }

    fn matrix(s: Synth, e: usize) tile.Matrix {
        const per = n * s.k * @as(usize, s.bits) / 8;
        const g = n * (s.k / 64) * 2;
        return .{
            .w = s.w,
            .w_off = e * per,
            .scales = s.s,
            .s_off = e * g,
            .biases = s.b,
            .b_off = e * g,
            .n = n,
            .k = s.k,
            .bits = s.bits,
        };
    }

    /// Row m of x against expert e's column c in float64, and its terms' magnitudes (the error's scale).
    fn reference(s: Synth, e: usize, m: usize, c: usize) [2]f64 {
        const k = s.k;
        const experts = s.experts;
        const groups = k / 64;
        const row_bytes = k * @as(usize, s.bits) / 8;
        const bytes = s.w.slice(u8, experts * n * row_bytes)[(e * n + c) * row_bytes ..][0..row_bytes];
        const sc = s.s.slice(u16, experts * n * groups);
        const bi = s.b.slice(u16, experts * n * groups);
        const x = s.x.slice(u16, max_rows * k);
        var total: f64 = 0;
        var scale: f64 = 0;
        for (0..groups) |g| {
            var dot: f64 = 0;
            var sum: f64 = 0;
            for (0..64) |i| {
                const xv: f64 = fromBf(x[m * k + g * 64 + i]);
                dot += xv * @as(f64, @floatFromInt(code(bytes, s.bits, g * 64 + i)));
                sum += xv;
            }
            const sd = @as(f64, fromBf(sc[(e * n + c) * groups + g])) * dot;
            const bs = @as(f64, fromBf(bi[(e * n + c) * groups + g])) * sum;
            total += sd + bs;
            scale += @abs(sd) + @abs(bs);
        }
        return .{ total, scale };
    }
};

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const tile.Pipelines,

    fn run(g: Gpu, body: anytype) !void {
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        try body.encode(e, g.p);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("GPU failed: {s}\n", .{msg});
            return error.GpuFailed;
        }
    }
};

/// The tiles on rows [first, first + rows) of x into y, with their group sums.
const Tiles = struct {
    m: tile.Matrix,
    x: mtl.Buffer,
    sums: mtl.Buffer,
    y: mtl.Buffer,
    first: usize,
    rows: usize,
    fn encode(t: Tiles, e: mtl.ComputeEncoder, p: *const tile.Pipelines) !void {
        tile.encodeSums(e, p, t.m.k, t.x, t.first * t.m.k * 2, t.sums, t.rows);
        try tile.encode(e, p, t.m, t.x, t.first * t.m.k * 2, t.sums, 0, t.y, t.first * n * 2, t.rows);
    }
};

const Twin = struct {
    m: tile.Matrix,
    x: mtl.Buffer,
    y: mtl.Buffer,
    first: usize,
    rows: usize,
    fn encode(t: Twin, e: mtl.ComputeEncoder, p: *const tile.Pipelines) !void {
        try tile.encodeTwin(e, p, t.m, t.x, t.first * t.m.k * 2, t.y, t.first * n * 2, t.rows);
    }
};

fn bf16s(b: mtl.Buffer, rows: usize) []const u16 {
    return b.slice(u16, rows * n);
}

/// Dense: R rows at once equal each row alone and the twin at any width; the worst error against float64.
fn dense(g: Gpu, s: Synth, worst: *f64, mean: *f64) !void {
    const m = s.matrix(1);
    const sums = try g.device.buffer(max_rows * (s.k / 64) * 4, opts);
    defer sums.deinit();
    const wide = try g.device.buffer(max_rows * n * 2, opts);
    defer wide.deinit();
    const alone = try g.device.buffer(max_rows * n * 2, opts);
    defer alone.deinit();
    const twin = try g.device.buffer(max_rows * n * 2, opts);
    defer twin.deinit();
    try g.run(Tiles{ .m = m, .x = s.x, .sums = sums, .y = wide, .first = 0, .rows = max_rows });
    for (0..max_rows) |r| try g.run(Tiles{ .m = m, .x = s.x, .sums = sums, .y = alone, .first = r, .rows = 1 });
    if (!std.mem.eql(u16, bf16s(wide, max_rows), bf16s(alone, max_rows))) return error.TileRowsDiffer;
    var at: usize = 0;
    for ([_]usize{ 1, 2, 3, 4, 5, 8, 15, 32 }) |w| {
        if (at + w > max_rows) break;
        try g.run(Twin{ .m = m, .x = s.x, .y = twin, .first = at, .rows = w });
        at += w;
    }
    if (!std.mem.eql(u16, bf16s(wide, at), bf16s(twin, at))) return error.TwinDiffers;
    var total: f64 = 0;
    for (0..max_rows) |r| for (0..n) |c| {
        const ref = s.reference(1, r, c);
        const err = @abs(@as(f64, fromBf(bf16s(wide, max_rows)[r * n + c])) - ref[0]) / @max(ref[1], 1e-30);
        worst.* = @max(worst.*, err);
        total += err;
    };
    mean.* = total / (max_rows * n);
}

/// The twin at the window kernels' edge widths equals the tiles on the same rows (a model's K: one or more rounds).
fn sweep(g: Gpu, s: Synth) !void {
    const m = s.matrix(0);
    const sums = try g.device.buffer(max_rows * (s.k / 64) * 4, opts);
    defer sums.deinit();
    const wide = try g.device.buffer(max_rows * n * 2, opts);
    defer wide.deinit();
    const twin = try g.device.buffer(max_rows * n * 2, opts);
    defer twin.deinit();
    try g.run(Tiles{ .m = m, .x = s.x, .sums = sums, .y = wide, .first = 0, .rows = tile.twin_rows });
    for ([_]usize{ 8, 9, 16, 17, 31, 32 }) |w| {
        try g.run(Twin{ .m = m, .x = s.x, .y = twin, .first = 0, .rows = w });
        if (!std.mem.eql(u16, bf16s(wide, w), bf16s(twin, w))) return error.TwinDiffers;
    }
}

/// 4 bits, K 64: code 7 is 1, scales 1, biases 0, x[7] = 2^-126; every path keeps bf16 0x0080 (no range change).
fn witness(g: Gpu) !void {
    const rows = tile.twin_rows;
    const w = try g.device.buffer(n * 32, opts);
    defer w.deinit();
    const sc = try g.device.buffer(n * 2, opts);
    defer sc.deinit();
    const bi = try g.device.buffer(n * 2, opts);
    defer bi.deinit();
    const x = try g.device.buffer(rows * 64 * 2, opts);
    defer x.deinit();
    for (w.slice(u8, n * 32), 0..) |*v, i| v.* = if (i % 32 == 3) 0x10 else 0;
    for (sc.slice(u16, n)) |*v| v.* = toBf(1);
    @memset(bi.slice(u16, n), 0);
    for (x.slice(u16, rows * 64), 0..) |*v, i| v.* = if (i % 64 == 7) 0x0080 else 0;
    const m = tile.Matrix{
        .w = w,
        .w_off = 0,
        .scales = sc,
        .s_off = 0,
        .biases = bi,
        .b_off = 0,
        .n = n,
        .k = 64,
        .bits = 4,
    };
    const sums = try g.device.buffer(rows * 4, opts);
    defer sums.deinit();
    const y = try g.device.buffer(rows * n * 2, opts);
    defer y.deinit();
    try g.run(Tiles{ .m = m, .x = x, .sums = sums, .y = y, .first = 0, .rows = rows });
    for (bf16s(y, rows)) |v| if (v != 0x0080) return error.WitnessTiles;
    for ([_]usize{ 1, 2, 7, 8, 16, 32 }) |width| {
        @memset(y.slice(u16, rows * n), 0);
        try g.run(Twin{ .m = m, .x = x, .y = y, .first = 0, .rows = width });
        for (bf16s(y, width)) |v| if (v != 0x0080) {
            std.debug.print("witness: twin at width {d} gives 0x{x:0>4}, the tiles 0x0080\n", .{ width, v });
            return error.WitnessTwin;
        };
    }
}

/// Picks of `tokens` rows over 4 experts, two a row.
fn spreadPicks(pick: []u32, tokens: usize) void {
    for (0..tokens) |t| {
        pick[2 * t] = @intCast((t * 7 + t / 5) % 4);
        pick[2 * t + 1] = @intCast((t * 7 + 3 + t / 5) % 4);
        if (pick[2 * t] == pick[2 * t + 1]) pick[2 * t + 1] = (pick[2 * t + 1] + 1) % 4;
    }
}

/// Picks of `tokens` rows, four a row: expert 0 on every row, then experts picked 1, 2, 3, 4, 5, 7, 1, 2... times.
fn skewedPicks(pick: []u32, tokens: usize) void {
    var e: u32 = 1;
    var left: usize = 1;
    for (1..4) |j| for (0..tokens) |t| {
        pick[t * 4 + j] = e;
        left -= 1;
        if (left == 0) {
            e += 1;
            left = if (e <= 5) e else e % 7 + 1;
        }
    };
    for (0..tokens) |t| pick[t * 4] = 0;
}

/// Grouped rows (through a row map) equal each expert's tiles; the twin by slot, by expert and gate-up swiglu too.
fn grouped(g: Gpu, s: Synth, topk: usize, pick: []const u32) !void {
    const slots = pick.len;
    const tokens = slots / topk;
    const experts = s.experts;
    const ids = try g.device.buffer(slots * 4, opts);
    defer ids.deinit();
    const map = try g.device.buffer(slots * 4, opts);
    defer map.deinit();
    const starts = try g.device.buffer((experts + 1) * 4, opts);
    defer starts.deinit();
    const work = try g.device.buffer((tile.items(slots, experts) + 1) * 4, opts);
    defer work.deinit();
    const sums = try g.device.buffer(max_rows * (s.k / 64) * 4, opts);
    defer sums.deinit();
    @memcpy(ids.slice(u32, slots), pick);
    var order_buf: [tile.window_slots]u32 = undefined; // sorted slot -> slot
    const order = order_buf[0..slots];
    const count = starts.slice(u32, experts + 1);
    var at: u32 = 0;
    for (0..experts) |e| {
        count[e] = at;
        for (pick, 0..) |p, slot| if (p == e) {
            order[at] = @intCast(slot);
            at += 1;
        };
    }
    count[experts] = at;
    for (order, 0..) |slot, j| map.slice(u32, slots)[j] = slot / @as(u32, @intCast(topk));
    const out = try g.device.buffer(slots * n * 2, opts);
    defer out.deinit();
    const twin = try g.device.buffer(slots * n * 2, opts);
    defer twin.deinit();
    const want = try g.device.buffer(max_rows * n * 2, opts);
    defer want.deinit();
    const G = struct {
        s: Synth,
        sums: mtl.Buffer,
        map: mtl.Buffer,
        out: mtl.Buffer,
        starts: mtl.Buffer,
        work: mtl.Buffer,
        tokens: usize,
        slots: usize,
        glu: bool,
        fn encode(c: @This(), e: mtl.ComputeEncoder, p: *const tile.Pipelines) !void {
            const m = c.s.matrix(0);
            const ex = c.s.experts;
            tile.encodeSums(e, p, c.s.k, c.s.x, 0, c.sums, c.tokens);
            tile.encodeWork(e, p, c.starts, ex, c.work);
            const x = c.s.x;
            if (c.glu) return tile.encodeGroupedGlu(e, p, m, m, ex, x, c.sums, c.map, c.out, c.slots, c.starts, c.work);
            try tile.encodeGrouped(e, p, m, ex, x, c.sums, c.map, c.out, c.slots, c.starts, c.work);
        }
    };
    const run = G{
        .s = s,
        .sums = sums,
        .map = map,
        .out = out,
        .starts = starts,
        .work = work,
        .tokens = tokens,
        .slots = slots,
        .glu = false,
    };
    try g.run(run);
    for (0..experts) |e| {
        if (count[e] == count[e + 1]) continue;
        try g.run(Tiles{ .m = s.matrix(e), .x = s.x, .sums = sums, .y = want, .first = 0, .rows = tokens });
        for (order, 0..) |slot, j| if (pick[slot] == e) {
            const got = bf16s(out, slots)[j * n ..][0..n];
            if (!std.mem.eql(u16, got, bf16s(want, tokens)[(slot / topk) * n ..][0..n])) return error.GroupedDiffers;
        };
    }
    const groups = try g.device.buffer(tile.groupWords(slots) * 4, opts);
    defer groups.deinit();
    const T = struct {
        s: Synth,
        ids: mtl.Buffer,
        groups: ?mtl.Buffer,
        out: mtl.Buffer,
        slots: usize,
        topk: usize,
        glu: bool,
        fn encode(c: @This(), e: mtl.ComputeEncoder, p: *const tile.Pipelines) !void {
            if (c.groups) |b| try tile.encodeTwinGroup(e, p, c.ids, c.slots, b);
            const m = c.s.matrix(0);
            if (c.glu) return tile.encodeTwinGlu(e, p, m, m, c.s.x, c.out, c.ids, c.groups, c.slots, c.topk);
            try tile.encodeTwinIndexed(e, p, m, c.s.x, c.out, c.ids, c.groups, c.slots, c.topk);
        }
    };
    for ([_]bool{ false, true }) |glu| {
        if (glu) {
            var with_glu = run;
            with_glu.glu = true;
            try g.run(with_glu);
        }
        for ([_]?mtl.Buffer{ null, groups }) |by_expert| {
            @memset(twin.contents()[0 .. slots * n * 2], 0);
            const t = T{
                .s = s,
                .ids = ids,
                .groups = by_expert,
                .out = twin,
                .slots = slots,
                .topk = topk,
                .glu = glu,
            };
            try g.run(t);
            for (order, 0..) |slot, j| {
                if (!std.mem.eql(u16, bf16s(out, slots)[j * n ..][0..n], bf16s(twin, slots)[slot * n ..][0..n])) {
                    return if (glu) error.GluTwinDiffers else error.IndexedTwinDiffers;
                }
            }
        }
    }
}

/// `--dump DIR`: each width's codes, scales, biases, inputs and tile outputs, for tile_qmm_reference.py.
fn dump(io: std.Io, dir: []const u8, s: Synth, out: mtl.Buffer) !void {
    var buf: [512]u8 = undefined;
    const k = s.k;
    const parts = .{
        .{ "w", s.w.slice(u8, n * k * @as(usize, s.bits) / 8) },
        .{ "scales", std.mem.sliceAsBytes(s.s.slice(u16, n * (k / 64))) },
        .{ "biases", std.mem.sliceAsBytes(s.b.slice(u16, n * (k / 64))) },
        .{ "x", std.mem.sliceAsBytes(s.x.slice(u16, max_rows * k)) },
        .{ "y", std.mem.sliceAsBytes(out.slice(u16, max_rows * n)) },
    };
    inline for (parts) |part| {
        const path = try std.fmt.bufPrint(&buf, "{s}/q{d}_{s}.bin", .{ dir, s.bits, part[0] });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = part[1] });
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const dir: ?[]const u8 = if (args.len == 3 and std.mem.eql(u8, args[1], "--dump")) args[2] else null;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try tile.Pipelines.load(device);
    defer p.deinit();
    const g = Gpu{ .device = device, .queue = queue, .p = &p };
    var passed: usize = 0;
    var pick: [tile.window_slots]u32 = undefined;
    try witness(g);
    std.debug.print("4-bit witness (code 7 = 1, x[7] = 2^-126): tiles and twin at widths 1-32 keep bf16 0x0080\n", .{});
    for (tile.bit_widths, 0..) |bits, i| {
        const s = try Synth.init(device, bits, 0x7173 + i, 1536, 4);
        defer s.deinit();
        var worst: f64 = 0;
        var mean: f64 = 0;
        try dense(g, s, &worst, &mean);
        if (dir) |d| {
            const sums = try device.buffer(max_rows * (s.k / 64) * 4, opts);
            defer sums.deinit();
            const out = try device.buffer(max_rows * n * 2, opts);
            defer out.deinit();
            try g.run(Tiles{ .m = s.matrix(0), .x = s.x, .sums = sums, .y = out, .first = 0, .rows = max_rows });
            try dump(init.io, d, s, out);
        }
        spreadPicks(&pick, 24);
        try grouped(g, s, 2, pick[0..48]);
        const many = try Synth.init(device, bits, 0x7183 + i, 1536, 64);
        defer many.deinit();
        for ([_]usize{ 32, 16 }) |tokens| {
            skewedPicks(&pick, tokens);
            try grouped(g, many, 4, pick[0 .. 4 * tokens]);
        }
        for ([_]usize{ 2048, 11776 }) |k| {
            const one = try Synth.init(device, bits, 0x7193 + i, k, 1);
            defer one.deinit();
            try sweep(g, one);
        }
        passed += 1;
        std.debug.print("{d}-bit {d}x{d}: rows alone, twin at widths 1-32, grouped, by slot and by expert " ++
            "with gate-up swiglu bit-equal; 64 experts at 16 and 32 rows, a top-4 pick on every row; " ++
            "twin at widths 8-32 at K 2048 and 11776; " ++
            "error vs float64 / terms' magnitude: worst {e:.2}, mean {e:.2}\n", .{ bits, n, s.k, worst, mean });
        if (worst > 1.0 / 128.0) return error.FarFromFloat64;
    }
    std.debug.print("{d} widths passed, 0 failed\n", .{passed});
}
