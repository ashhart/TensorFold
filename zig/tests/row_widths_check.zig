//! The core row projection at every MLX width and group: an fp64 reference, windows, strides and stock bytes.
const std = @import("std");
const mtl = @import("metal");
const row = @import("row_projection");
const mlx = @import("core").quant.mlx;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const a = std.heap.page_allocator;
const usage = "usage: tf-row-widths-check [STOCK.metal]  (STOCK: the unchanged row_projection.metal for the byte comparison)";

const groups_tested = [_]usize{ 32, 64, 128 };
const Shape = struct { n: usize, k: usize };
// all tail (K < 512), full passes and a tail, full passes only
const shapes = [_]Shape{ .{ .n = 64, .k = 384 }, .{ .n = 128, .k = 2688 }, .{ .n = 32, .k = 1536 } };
const rows = 16;
// a fused stack's windows reach this many rows
const x_rows = 48;
const sentinel: u16 = 0xffff;

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn fromBf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// One synthetic MLX matrix in the row layout, its inputs and its fp64 dequantization (and codes alone).
const Mat = struct {
    w: row.Weights,
    x: mtl.Buffer,
    deq: []f64,
    codes: []f64,

    fn init(device: mtl.Device, s: Shape, bits: u8, group: usize, sum: row.Sum, seed: u64) !Mat {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const words = s.n * s.k * bits / 32;
        const tables = s.n * (s.k / group);
        const m = Mat{
            .w = .{ .w = try device.buffer(words * 4, opts), .scales = try device.buffer(tables * 2, opts), .biases = try device.buffer(tables * 2, opts), .n = s.n, .k = s.k, .group = group, .bits = bits, .sum = sum },
            .x = try device.buffer(x_rows * s.k * 2, opts),
            .deq = try a.alloc(f64, s.n * s.k),
            .codes = try a.alloc(f64, s.n * s.k),
        };
        for (m.w.w.slice(u32, words)) |*v| v.* = r.int(u32);
        for (m.w.scales.slice(u16, tables)) |*v| v.* = toBf(r.float(f32) * 0.02 + 0.001);
        for (m.w.biases.slice(u16, tables)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.1);
        for (m.x.slice(u16, x_rows * s.k)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
        try mlx.reference(host(m.w, m.w.scales.slice(u16, tables), m.w.biases.slice(u16, tables)), m.deq);
        const ones = try a.alloc(u16, tables);
        defer a.free(ones);
        const zeros = try a.alloc(u16, tables);
        defer a.free(zeros);
        @memset(ones, toBf(1));
        @memset(zeros, 0);
        try mlx.reference(host(m.w, ones, zeros), m.codes);
        return m;
    }

    fn deinit(m: Mat) void {
        for ([_]mtl.Buffer{ m.w.w, m.w.scales, m.w.biases, m.x }) |b| b.deinit();
        a.free(m.deq);
        a.free(m.codes);
    }

    /// The checkpoint's view of the same words and tables, as `mlx.reference` reads it.
    fn host(w: row.Weights, scales: []const u16, biases: []const u16) mlx.Host {
        const groups = w.k / w.group;
        return .{
            .words = .{ .dtype = .i32, .rank = 2, .shape = .{ w.n, w.k * w.bits / 32, 1, 1, 1 }, .bytes = w.w.slice(u8, w.n * w.k * w.bits / 8) },
            .scales = .{ .dtype = .bf16, .rank = 2, .shape = .{ w.n, groups, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(scales) },
            .biases = .{ .dtype = .bf16, .rank = 2, .shape = .{ w.n, groups, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(biases) },
            .bits = w.bits,
            .group = @intCast(w.group),
        };
    }

    /// Row m against column c in fp64, and the fp32 sums' error bound (output rounding plus k-deep accumulation).
    fn reference(m: Mat, i: usize, c: usize) [2]f64 {
        const k = m.w.k;
        const g_n = k / m.w.group;
        const sc = m.w.scales.slice(u16, m.w.n * g_n);
        const bi = m.w.biases.slice(u16, m.w.n * g_n);
        const x = m.x.slice(u16, rows * k);
        var y: f64 = 0;
        var terms: f64 = 0;
        for (0..g_n) |g| {
            var dot: f64 = 0;
            var xs: f64 = 0;
            for (0..m.w.group) |j| {
                const kk = g * m.w.group + j;
                const xv: f64 = fromBf(x[i * k + kk]);
                y += xv * m.deq[c * k + kk];
                dot += @abs(xv * m.codes[c * k + kk]);
                xs += @abs(xv);
            }
            terms += @abs(@as(f64, fromBf(sc[c * g_n + g]))) * dot + @abs(@as(f64, fromBf(bi[c * g_n + g]))) * xs;
        }
        const u32_round = std.math.ldexp(@as(f64, 1), -24);
        const gamma = @as(f64, @floatFromInt(k + 64)) * u32_round;
        return .{ y, std.math.ldexp(@abs(y) + gamma * terms, -8) + gamma * terms };
    }
};

/// One launch: buffers x, W, S, B, dims, out at 0-5 and ids at 6 when given; grid (groups, slots).
const Launch = struct { pipeline: mtl.Pipeline, dims: [6]i32, groups: usize, slots: usize = 1, x_off: usize = 0, out_off: usize = 0, ids: ?mtl.Buffer = null };

fn run(queue: mtl.Queue, l: Launch, m: Mat, out: mtl.Buffer) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(l.pipeline);
    e.setBuffer(m.x, l.x_off, 0);
    e.setBuffer(m.w.w, 0, 1);
    e.setBuffer(m.w.scales, 0, 2);
    e.setBuffer(m.w.biases, 0, 3);
    e.setValue(l.dims, 4);
    e.setBuffer(out, l.out_off, 5);
    if (l.ids) |ids| e.setBuffer(ids, 0, 6);
    e.dispatchGroups(mtl.Size.of(l.groups, l.slots, 1), mtl.Size.of(32 * row.groups, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

fn of(c: row.Call) Launch {
    return .{ .pipeline = c.pipeline, .dims = c.dims, .groups = c.groups, .slots = c.slots };
}

/// The stock library's pipelines for the widths it already served, by name.
const Stock = struct {
    lib: mtl.Library,
    fn get(s: Stock, device: mtl.Device, kind: []const u8, bits: u8, sum: row.Sum) !mtl.Pipeline {
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "tf_row_projection{s}_q{d}_{s}", .{ kind, bits, @tagName(sum) });
        return mtl.Pipeline.init(device, s.lib, name, false);
    }
};

const Tally = struct { worst_err: f64 = 0, worst_ratio: f64 = 0, worst_col: usize = 0, values: usize = 0, exact: usize = 0, pairs: usize = 0, stock: usize = 0 };

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len > 2) {
        std.debug.print("{s}\n", .{usage});
        std.process.exit(2);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try row.Pipelines.load(device);
    defer p.deinit();
    var stock: ?Stock = null;
    if (args.len == 2) {
        const f = try mtl.MappedFile.open(try arena.dupeSentinel(u8, args[1], 0));
        defer f.deinit();
        stock = .{ .lib = try mtl.Library.fromSource(device, f.bytes[0..f.size], mtl.CompileOptions.mlx()) };
    }
    defer if (stock) |s| s.lib.deinit();
    const out = try device.buffer(rows * 512 * 2 * 4, opts);
    defer out.deinit();
    const one = try device.buffer(rows * 512 * 2 * 4, opts);
    defer one.deinit();
    const ids = try device.buffer(rows * 4, opts);
    defer ids.deinit();
    var all = Tally{};
    for (row.bit_widths) |bits| for (groups_tested) |group| {
        var t = Tally{};
        for (shapes, 0..) |s, si| for ([_]row.Sum{ .f32, .bf16 }) |sum| {
            const m = try Mat.init(device, s, bits, group, sum, 1000 * @as(u64, bits) + 10 * group + si);
            defer m.deinit();
            try check(device, queue, &p, stock, m, out, one, ids, &t);
        };
        std.debug.print("q{d} g{d}: {d} values vs fp64, worst |err| {e:.3} (col {d}), worst err/bound {d:.4}, bf16(ref) exact {d}/{d}; {d} bit-equal pairs; {d} stock bytes equal\n", .{ bits, group, t.values, t.worst_err, t.worst_col, t.worst_ratio, t.exact, t.values, t.pairs, t.stock });
        all.values += t.values;
        all.pairs += t.pairs;
        all.stock += t.stock;
        all.worst_ratio = @max(all.worst_ratio, t.worst_ratio);
    };
    all.pairs += try stack(device, queue, &p);
    if (stock == null) std.debug.print("no stock source given: the byte comparison was not run\n", .{});
    std.debug.print("PASS {d} values within the fp32-sum bound (worst err/bound {d:.4}), {d} bit-equal pairs, {d} stock outputs byte-equal\n", .{ all.values, all.worst_ratio, all.pairs, all.stock });
}

/// One matrix: the reference (f32 sums), windows, relu2, indexed and strided outputs against one row alone, then stock.
fn check(device: mtl.Device, queue: mtl.Queue, p: *const row.Pipelines, stock: ?Stock, m: Mat, out: mtl.Buffer, one: mtl.Buffer, ids: mtl.Buffer, t: *Tally) !void {
    const n = m.w.n;
    const k = m.w.k;
    // each row alone: the reference everything else is compared with
    const alone = try a.alloc(u16, rows * n);
    defer a.free(alone);
    for (0..rows) |i| {
        try run(queue, launchAt(try row.call(p, m.w, 1, false), i * k * 2, 0), m, one);
        @memcpy(alone[i * n ..][0..n], one.slice(u16, n));
    }
    if (m.w.sum == .f32) for (0..rows) |i| for (0..n) |c| {
        const ref = m.reference(i, c);
        const got: f64 = fromBf(alone[i * n + c]);
        const err = @abs(got - ref[0]);
        if (err > ref[1]) {
            std.debug.print("q{d} g{d} n {d} k {d} row {d} col {d}: GPU {d} fp64 {d} bound {e}\n", .{ m.w.bits, m.w.group, n, k, i, c, got, ref[0], ref[1] });
            return error.ReferenceMismatch;
        }
        if (err > t.worst_err) {
            t.worst_err = err;
            t.worst_col = c;
        }
        t.worst_ratio = @max(t.worst_ratio, err / ref[1]);
        if (alone[i * n + c] == toBf(@floatCast(ref[0]))) t.exact += 1;
        t.values += 1;
    };
    // windows of 2, 3, 8 and 16 rows (ending at the last row): each row as it was alone
    for ([_]usize{ 2, 3, 8, 16 }) |w| {
        const start = rows - w;
        try run(queue, launchAt(try row.call(p, m.w, w, false), start * k * 2, 0), m, out);
        for (0..w) |i| {
            if (!std.mem.eql(u16, out.slice(u16, w * n)[i * n ..][0..n], alone[(start + i) * n ..][0..n])) return error.WindowChangesBits;
            t.pairs += n;
        }
    }
    // relu2 is relu squared of the plain output
    try run(queue, of(try row.call(p, m.w, rows, true)), m, out);
    for (out.slice(u16, rows * n), alone) |r2, v| {
        const h = @max(fromBf(v), 0);
        if (r2 != toBf(h * h)) return error.Relu2DiffersFromPlain;
    }
    // indexed slots of one expert: each slot as its row alone; an id past the count is a zero row
    for (ids.slice(u32, rows), 0..) |*id, i| id.* = if (i == rows - 1) 1 else 0;
    var li = of(try row.callIndexed(p, .{ .one = m.w, .experts = 1 }, rows, false));
    li.ids = ids;
    try run(queue, li, m, out);
    for (0..rows) |i| {
        const got = out.slice(u16, rows * n)[i * n ..][0..n];
        if (i == rows - 1) {
            for (got) |v| if (v != 0) return error.MissingExpertNotZero;
        } else if (!std.mem.eql(u16, got, alone[i * n ..][0..n])) return error.IndexedDiffersFromDense;
    }
    // strided: the matrix's rows at column 16 of a wider buffer; its columns as alone, every other value untouched
    const stride = 2 * n + 24;
    @memset(out.slice(u16, rows * stride), sentinel);
    try run(queue, launchAt(try row.callStrided(p, m.w, rows, stride), 0, 16 * 2), m, out);
    for (0..rows) |i| for (0..stride) |c| {
        const v = out.slice(u16, rows * stride)[i * stride + c];
        if (c >= 16 and c < 16 + n) {
            if (v != alone[i * n + c - 16]) return error.StridedDiffersFromAlone;
            t.pairs += 1;
        } else if (v != sentinel) return error.StridedWritesOutside;
    };
    const s = stock orelse return;
    if (std.mem.indexOfScalar(u8, &.{ 2, 4, 6, 8 }, m.w.bits) == null) return;
    // the stock kernels: plain and relu2 windows of 1, 3 and 16 rows, indexed slots, byte for byte
    for ([_]usize{ 1, 3, 16 }) |w| for ([_]bool{ false, true }) |relu2| {
        const c = try row.call(p, m.w, w, relu2);
        try run(queue, of(c), m, out);
        const pipe = try s.get(device, if (relu2) "_relu2" else "", m.w.bits, m.w.sum);
        defer pipe.deinit();
        var l = of(c);
        l.pipeline = pipe;
        try run(queue, l, m, one);
        if (!std.mem.eql(u16, out.slice(u16, w * n), one.slice(u16, w * n))) return error.StockBytesDiffer;
        t.stock += w * n;
    };
    const pipe = try s.get(device, "_indexed", m.w.bits, m.w.sum);
    defer pipe.deinit();
    try run(queue, li, m, out);
    li.pipeline = pipe;
    try run(queue, li, m, one);
    if (!std.mem.eql(u16, out.slice(u16, rows * n), one.slice(u16, rows * n))) return error.StockBytesDiffer;
    t.stock += rows * n;
}

fn launchAt(c: row.Call, x_off: usize, out_off: usize) Launch {
    var l = of(c);
    l.x_off = x_off;
    l.out_off = out_off;
    return l;
}

/// A fused stack of 48-column members at mixed widths, group 64: each member's strided columns as its own call.
fn stack(device: mtl.Device, queue: mtl.Queue, p: *const row.Pipelines) !usize {
    const widths = [_]u8{ 4, 5, 4, 6, 3, 8, 2, 4 };
    const n = 48;
    const stride = widths.len * n;
    var members: [widths.len]Mat = undefined;
    for (&members, widths, 0..) |*m, bits, i| m.* = try Mat.init(device, .{ .n = n, .k = 2688 }, bits, 64, .f32, 77 + i);
    defer for (members) |m| m.deinit();
    // every member reads the same inputs
    for (members[1..]) |m| @memcpy(m.x.slice(u16, x_rows * 2688), members[0].x.slice(u16, x_rows * 2688));
    const fused = try device.buffer(x_rows * stride * 2, opts);
    defer fused.deinit();
    const own = try device.buffer(x_rows * n * 2, opts);
    defer own.deinit();
    const alone = try device.buffer(n * 2, opts);
    defer alone.deinit();
    var pairs: usize = 0;
    for ([_]usize{ 1, 2, 3, 16, 48 }) |w| {
        @memset(fused.slice(u16, w * stride), sentinel);
        for (members, 0..) |m, j| try run(queue, launchAt(try row.callStrided(p, m.w, w, stride), 0, j * n * 2), m, fused);
        for (members, 0..) |m, j| {
            try run(queue, of(try row.call(p, m.w, w, false)), m, own);
            for (0..w) |i| {
                const got = fused.slice(u16, w * stride)[i * stride + j * n ..][0..n];
                if (!std.mem.eql(u16, got, own.slice(u16, w * n)[i * n ..][0..n])) return error.StackDiffersFromMember;
                if (w == 48) {
                    try run(queue, launchAt(try row.call(p, m.w, 1, false), i * 2688 * 2, 0), m, alone);
                    if (!std.mem.eql(u16, got, alone.slice(u16, n))) return error.StackDiffersFromRowAlone;
                    pairs += n;
                }
                pairs += n;
            }
        }
        for (fused.slice(u16, w * stride)) |v| if (v == sentinel) return error.StackLeftAGap;
    }
    std.debug.print("stack of {d} members (bits 4 5 4 6 3 8 2 4, n {d}, group 64) at windows 1, 2, 3, 16, 48: each member as its own call\n", .{ widths.len, n });
    return pairs;
}
