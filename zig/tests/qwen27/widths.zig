//! The 27B's projections without tensor units at every oQ width against float64, row bits in every call, mixed stacks.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mlx = tf.quant.mlx;
const q27 = tf.qwen27;
const projection = q27.projection;
const row = tf.row_projection;
const cp = tf.checkpoint;

const opts = mtl.ResourceOptions.shared;
const a = std.heap.page_allocator;
const unit: f64 = 0x1p-24; // fp32's unit roundoff
const group = 64; // every module of both oQ checkpoints
const wide = 128; // a prompt chunk's rows
const windows = [_]usize{ 1, 2, 3, 8, 16, 17, 128 };
const bit_widths = [_]u8{ 2, 3, 4, 5, 6, 8 }; // every class at every width oQ4 and oQ2 use anywhere

/// A projection class: output rows n of input width k (gate and up share a width, so one matrix as loaded).
const Class = struct { name: []const u8, n: usize, k: usize };
const classes = [_]Class{
    .{ .name = "gdn in_proj_qkv", .n = 10240, .k = 5120 },
    .{ .name = "gdn in_proj_z", .n = 6144, .k = 5120 },
    .{ .name = "gdn in_proj_b", .n = 48, .k = 5120 },
    .{ .name = "gdn in_proj_a", .n = 48, .k = 5120 },
    .{ .name = "gdn out_proj", .n = 5120, .k = 6144 },
    .{ .name = "attn q_proj", .n = 12288, .k = 5120 },
    .{ .name = "attn k_proj", .n = 1024, .k = 5120 },
    .{ .name = "attn v_proj", .n = 1024, .k = 5120 },
    .{ .name = "attn o_proj", .n = 5120, .k = 6144 },
    .{ .name = "mlp gate_up", .n = 34816, .k = 5120 },
    .{ .name = "mlp down_proj", .n = 5120, .k = 17408 },
    .{ .name = "lm_head", .n = 248320, .k = 5120 },
};

/// A fused member: name, rows, bits.
const Member = struct { name: []const u8, n: usize, bits: u8 };
/// The mixed stacks of both checkpoints (zba and kv), k = 5120.
const stacks = [_][]const Member{
    &.{ .{ .name = "z4", .n = 6144, .bits = 4 }, .{ .name = "b5", .n = 48, .bits = 5 }, .{ .name = "a5", .n = 48, .bits = 5 } },
    &.{ .{ .name = "z4", .n = 6144, .bits = 4 }, .{ .name = "b4", .n = 48, .bits = 4 }, .{ .name = "a5", .n = 48, .bits = 5 } },
    &.{ .{ .name = "z2", .n = 6144, .bits = 2 }, .{ .name = "b2", .n = 48, .bits = 2 }, .{ .name = "a3", .n = 48, .bits = 3 } },
    &.{ .{ .name = "k5", .n = 1024, .bits = 5 }, .{ .name = "v6", .n = 1024, .bits = 6 } },
};

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn fromBf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

fn shape2(r: usize, c: usize) [5]usize {
    return .{ r, c, 1, 1, 1 };
}

/// Synthetic MLX affine weights [n, k] as the checkpoint stores them: words of random codes, bf16 scales and biases.
const Synth = struct {
    n: usize,
    k: usize,
    bits: u8,
    w: mtl.Buffer,
    s: mtl.Buffer,
    b: mtl.Buffer,

    fn init(device: mtl.Device, n: usize, k: usize, bits: u8, seed: u64) !Synth {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const sn = n * (k / group);
        const m = Synth{ .n = n, .k = k, .bits = bits, .w = try device.buffer(n * k * bits / 8, opts), .s = try device.buffer(sn * 2, opts), .b = try device.buffer(sn * 2, opts) };
        r.bytes(m.w.slice(u8, n * k * bits / 8));
        for (m.s.slice(u16, sn), m.b.slice(u16, sn)) |*s, *b| {
            s.* = toBf((r.float(f32) + 0.5) / 64 / @as(f32, @floatFromInt(@as(u32, 1) << @intCast(bits - 2))));
            b.* = toBf(-(r.float(f32) + 0.5) * 0.15);
        }
        return m;
    }

    fn deinit(m: Synth) void {
        for ([_]mtl.Buffer{ m.w, m.s, m.b }) |x| x.deinit();
    }

    fn tensor(buffer: mtl.Buffer, dtype: cp.DType, r: usize, c: usize, size: usize) cp.Tensor {
        return .{ .buffer = buffer, .offset = 0, .bytes = r * c * size, .dtype = dtype, .shape = shape2(r, c), .rank = 2 };
    }

    /// The loader's view: what gpu_weights hands projection.prepare.
    fn triple(m: Synth) projection.Triple {
        const g = m.k / group;
        return .{ .weight = tensor(m.w, .u32, m.n, m.k * m.bits / 32, 4), .scales = tensor(m.s, .bf16, m.n, g, 2), .biases = tensor(m.b, .bf16, m.n, g, 2), .bits = m.bits, .group = group };
    }

    /// Row j alone as core.quant.mlx reads it.
    fn host(m: Synth, j: usize) mlx.Host {
        const g = m.k / group;
        const wb = m.k * m.bits / 8;
        return .{
            .words = .{ .dtype = .u32, .rank = 2, .shape = shape2(1, wb / 4), .bytes = m.w.slice(u8, m.n * wb)[j * wb ..][0..wb] },
            .scales = .{ .dtype = .bf16, .rank = 2, .shape = shape2(1, g), .bytes = m.s.slice(u8, m.n * g * 2)[j * g * 2 ..][0 .. g * 2] },
            .biases = .{ .dtype = .bf16, .rank = 2, .shape = shape2(1, g), .bytes = m.b.slice(u8, m.n * g * 2)[j * g * 2 ..][0 .. g * 2] },
            .bits = m.bits,
            .group = group,
        };
    }
};

/// `wide` rows of bf16 inputs in [-1, 1).
fn inputs(device: mtl.Device, k: usize, seed: u64) !mtl.Buffer {
    var prng = std.Random.DefaultPrng.init(seed);
    const x = try device.buffer(wide * k * 2, opts);
    for (x.slice(u16, wide * k)) |*v| v.* = toBf(prng.random().float(f32) * 2 - 1);
    return x;
}

/// Every column of a matrix up to 1024 wide, else 512: the first and last 128 and 256 at an odd stride between.
fn pick(n: usize) ![]usize {
    if (n <= 1024) {
        const cols = try a.alloc(usize, n);
        for (cols, 0..) |*c, i| c.* = i;
        return cols;
    }
    const cols = try a.alloc(usize, 512);
    const step = ((n - 256) / 256 - 1) | 1;
    for (0..128) |i| {
        cols[i] = i;
        cols[511 - i] = n - 1 - i;
    }
    for (0..256) |i| cols[128 + i] = 128 + i * step;
    return cols;
}

/// y = x W^T in float64 on `cols`, the sizes the fp32 sums carry, and the scales' share (scales 1% off add 1% of it).
const Ref = struct {
    cols: []usize,
    want: []f64,
    mag: []f64,
    scaled: []f64,

    fn init(m: Synth, x: mtl.Buffer) !Ref {
        const k = m.k;
        const g_n = k / group;
        const cols = try pick(m.n);
        const c_n = cols.len;
        const r = Ref{ .cols = cols, .want = try a.alloc(f64, wide * c_n), .mag = try a.alloc(f64, wide * c_n), .scaled = try a.alloc(f64, wide * c_n) };
        const xd = try a.alloc(f64, wide * k);
        defer a.free(xd);
        const xg = try a.alloc(f64, wide * g_n); // each group's sum of |x|
        defer a.free(xg);
        @memset(xg, 0);
        for (x.slice(u16, wide * k), xd, 0..) |v, *d, i| {
            d.* = fromBf(v);
            xg[i / group] += @abs(d.*);
        }
        const deq = try a.alloc(f64, k);
        defer a.free(deq);
        const bias = m.b.slice(u16, m.n * g_n);
        for (cols, 0..) |j, c| {
            try mlx.reference(m.host(j), deq);
            for (0..wide) |i| {
                var sum: f64 = 0;
                var mag: f64 = 0;
                var scaled: f64 = 0;
                for (0..g_n) |g| {
                    const b = fromBf(bias[j * g_n + g]);
                    var qx: f64 = 0;
                    var qxs: f64 = 0;
                    for (g * group..(g + 1) * group) |e| {
                        const xv = xd[i * k + e];
                        const sq = deq[e] - b; // s q, exact
                        sum += xv * deq[e];
                        qx += sq * @abs(xv);
                        qxs += sq * xv;
                    }
                    mag += qx + @abs(b) * xg[i * g_n + g];
                    scaled += qxs;
                }
                r.want[i * c_n + c] = sum;
                r.mag[i * c_n + c] = mag;
                r.scaled[i * c_n + c] = scaled;
            }
        }
        return r;
    }

    fn deinit(r: Ref) void {
        for ([_][]f64{ r.want, r.mag, r.scaled }) |v| a.free(v);
        a.free(r.cols);
    }
};

/// Failures, values against float64, bit-equal pairs, and the largest difference as a fraction of its bound.
const Tally = struct { failed: usize = 0, values: usize = 0, pairs: usize = 0, worst: f64 = 0, ctl_min: f64 = std.math.inf(f64) };

/// `rows` bf16 rows `pitch` apart; a member's columns start at `first`.
const View = struct {
    vals: []const u16,
    pitch: usize,
    first: usize = 0,
    fn get(v: View, r: usize, j: usize) f64 {
        return fromBf(v.vals[r * v.pitch + v.first + j]);
    }
};

/// fp32 sums of a group's products, its row sum and a chain of groups (plus the kernel's tree), and one bf16 ulp.
fn bound(want: f64, mag: f64, k: usize, extra: usize) f64 {
    const terms: f64 = @floatFromInt(group + k / group + 8 + extra);
    return terms * unit * mag + @abs(want) * 0x1p-7 + 1e-30;
}

/// `rows` rows of `out` against the float64 reference: the worst column, its bound, and the 1%-off control.
fn compare(t: *Tally, name: []const u8, out: View, rows: usize, ref: Ref, k: usize, extra: usize) void {
    const c_n = ref.cols.len;
    var col: usize = 0;
    var col_abs: f64 = -1;
    var col_rel: f64 = 0;
    var frac: f64 = 0;
    var ctl: f64 = 0;
    var at: [2]usize = .{ 0, 0 };
    for (ref.cols, 0..) |j, c| {
        var abs: f64 = 0;
        var top: f64 = 0;
        for (0..rows) |r| {
            const i = r * c_n + c;
            const got = out.get(r, j);
            const d = @abs(got - ref.want[i]);
            const tol = bound(ref.want[i], ref.mag[i], k, extra);
            ctl = @max(ctl, @abs(got - ref.want[i] - 0.01 * ref.scaled[i]) / tol);
            if (!(d <= abs)) abs = d;
            top = @max(top, @abs(ref.want[i]));
            if (!(d / tol <= frac)) {
                frac = d / tol;
                at = .{ r, j };
            }
        }
        if (!(abs <= col_abs)) {
            col_abs = abs;
            col = j;
            col_rel = abs / top;
        }
    }
    t.values += rows * c_n;
    t.worst = @max(t.worst, frac);
    t.ctl_min = @min(t.ctl_min, ctl);
    const ok = frac <= 1 and ctl > 1;
    if (!ok) t.failed += 1;
    std.debug.print("RESULT {s}: {d} columns, worst column {d}: max abs {e:.3}, relative {e:.3}; {d:.4} of its bound (row {d} col {d}), " ++
        "scales 1% off {d:.1}{s}\n", .{ name, c_n, col, col_abs, col_rel, frac, at[0], at[1], ctl, if (ok) "" else " FAIL" });
}

/// Rows [0, rows) of `got` equal `want` bit for bit; the first difference printed.
fn same(t: *Tally, name: []const u8, what: []const u8, want: View, got: View, rows: usize, n: usize) bool {
    var differ: usize = 0;
    var first: ?[2]usize = null;
    var most: f64 = 0;
    for (0..rows) |r| {
        const w = want.vals[r * want.pitch + want.first ..][0..n];
        const g = got.vals[r * got.pitch + got.first ..][0..n];
        if (std.mem.eql(u16, w, g)) continue;
        differ += 1;
        if (first == null) first = .{ r, std.mem.indexOfDiff(u16, w, g).? };
        for (0..n) |j| most = @max(most, @abs(want.get(r, j) - got.get(r, j)));
    }
    if (first) |f| {
        std.debug.print("  {s}: {s} differ in {d} of {d} rows, by up to {e:.3}; first row {d} col {d}: {d} against {d}\n", .{ name, what, differ, rows, most, f[0], f[1], got.get(f[0], f[1]), want.get(f[0], f[1]) });
        return false;
    }
    t.pairs += rows * n;
    return true;
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    kern: *q27.quant_gpu.Kernels,
    pipes: *const row.Pipelines,

    fn commit(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("GPU failed: {s}\n", .{why});
            return error.GpuFailed;
        }
    }

    /// The family's product over `wide` rows in windows of `m` (a prompt chunk when m == wide), into a NaN-filled y.
    fn quant(g: Gpu, lin: projection.Linear, x: mtl.Buffer, y: mtl.Buffer, m: usize) !void {
        const n: usize = lin.n;
        @memset(y.slice(u8, wide * n * 2), 0xff);
        g.kern.prompt = m == wide;
        defer g.kern.prompt = false;
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        var start: usize = 0;
        while (start + m <= wide) : (start += m) try g.kern.quant(e, lin, .{ .buffer = x, .offset = start * lin.k * 2 }, null, .{ .buffer = x }, .{ .buffer = y, .offset = start * n * 2 }, @intCast(m));
        try commit(cb, e);
    }

    /// The row kernel alone at the matrix's width over `wide` rows, unstrided: a member's own product.
    fn rows(g: Gpu, raw: row.Weights, x: mtl.Buffer, y: mtl.Buffer) !void {
        @memset(y.slice(u8, wide * raw.n * 2), 0xff);
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        const call = try row.call(g.pipes, raw, wide, false);
        e.setPipeline(call.pipeline);
        e.setBuffer(x, 0, 0);
        e.setBuffer(raw.w, raw.w_off, 1);
        e.setBuffer(raw.scales, raw.s_off, 2);
        e.setBuffer(raw.biases, raw.b_off, 3);
        e.setValue(call.dims, 4);
        e.setBuffer(y, 0, 5);
        e.dispatchGroups(.{ .width = call.groups }, .{ .width = call.threads });
        try commit(cb, e);
    }
};

/// The kernel the family picks for a matrix before tensor units, and its tree's extra depth for the bound.
fn path(lin: projection.Linear) struct { name: []const u8, extra: usize } {
    const raw = lin.raw.?;
    if (raw.bits == 4 and raw.group == 64 and raw.n % 8 == 0) return .{ .name = "simd_qmm", .extra = std.math.log2_int(u32, row.simd.splits(raw.n)) + 1 };
    return .{ .name = "row", .extra = 0 };
}

fn release(owned: *std.ArrayList(mtl.Buffer)) void {
    for (owned.items) |b| b.deinit();
    owned.deinit(a);
}

/// One class at one width: prompt and decoded rows against float64, every window's rows against the prompt call's.
fn class(t: *Tally, g: Gpu, c: Class, bits: u8, seed: u64) !void {
    const m = try Synth.init(g.device, c.n, c.k, bits, seed);
    defer m.deinit();
    const x = try inputs(g.device, c.k, seed ^ 0x27);
    defer x.deinit();
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    defer release(&owned);
    const lin = try projection.prepare(g.device, a, &owned, &.{m.triple()}, 32);
    const p = path(lin);
    const ref = try Ref.init(m, x);
    defer ref.deinit();
    var name: [128]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "{s} [{d}, {d}] q{d} g64 on {s}", .{ c.name, c.n, c.k, bits, p.name });
    var outs: [windows.len]mtl.Buffer = undefined;
    for (&outs, windows) |*y, w| {
        y.* = try g.device.buffer(wide * c.n * 2, opts);
        try g.quant(lin, x, y.*, w);
    }
    defer for (outs) |y| y.deinit();
    const view = struct {
        fn of(y: mtl.Buffer, n: usize) View {
            return .{ .vals = y.slice(u16, wide * n), .pitch = n };
        }
    }.of;
    var tag: [160]u8 = undefined;
    compare(t, try std.fmt.bufPrint(&tag, "{s}, prompt ({d} rows)", .{ label, wide }), view(outs[windows.len - 1], c.n), wide, ref, c.k, p.extra);
    compare(t, try std.fmt.bufPrint(&tag, "{s}, decode (1 row a call)", .{label}), view(outs[0], c.n), wide, ref, c.k, p.extra);
    var ok = true;
    for (windows[0 .. windows.len - 1], outs[0 .. windows.len - 1]) |w, y| {
        var what: [48]u8 = undefined;
        ok = same(t, label, try std.fmt.bufPrint(&what, "{d}-row windows", .{w}), view(outs[windows.len - 1], c.n), view(y, c.n), wide / w * w, c.n) and ok;
    }
    if (!ok) t.failed += 1;
    std.debug.print("RESULT {s}: {s}\n", .{ label, if (ok) "every row's bits equal in calls of 1, 2, 3, 8, 16, 17 and 128 rows" else "a row's bits depend on the call FAIL" });
}

/// A mixed stack as gpu_weights.Builder.stack builds it: one product a member, each into its columns.
fn mixed(g: Gpu, owned: *std.ArrayList(mtl.Buffer), parts: []const projection.Triple, runs: []projection.Run) !projection.Linear {
    var ws: [3]q27.widths.Width = undefined;
    for (parts, 0..) |p, i| ws[i] = .{ .bits = p.bits, .group = @intCast(p.group) };
    var buf: [3][2]usize = undefined;
    const list = q27.widths.runs(ws[0..parts.len], &buf);
    if (list.len != parts.len) return error.NotMixed;
    var first: u32 = 0;
    for (list, runs[0..list.len]) |r, *run| {
        run.* = .{ .linear = try projection.prepare(g.device, a, owned, parts[r[0]..][0..r[1]], 32), .first = first };
        first += run.linear.n;
    }
    var fused = runs[0].linear;
    fused.n = first;
    fused.raw = null;
    fused.runs = runs[0..list.len];
    return fused;
}

/// A mixed stack: rows equal in every call; each member's columns within float64's bound and equal to its own product.
fn stack(t: *Tally, g: Gpu, members: []const Member, seed: u64) !void {
    const k = 5120;
    var name: [64]u8 = undefined;
    var len: usize = 0;
    for (members, 0..) |m, i| len += (try std.fmt.bufPrint(name[len..], "{s}{s}", .{ if (i == 0) "[" else " ", m.name })).len;
    const label = try std.fmt.bufPrint(name[len..], "]", .{});
    const stack_name = name[0 .. len + label.len];
    var synth: [3]Synth = undefined;
    var parts: [3]projection.Triple = undefined;
    var total: usize = 0;
    for (members, 0..) |m, i| {
        synth[i] = try Synth.init(g.device, m.n, k, m.bits, seed + i);
        parts[i] = synth[i].triple();
        total += m.n;
    }
    defer for (synth[0..members.len]) |s| s.deinit();
    const x = try inputs(g.device, k, seed ^ 0x27);
    defer x.deinit();
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    defer release(&owned);
    var runs: [3]projection.Run = undefined;
    const fused = try mixed(g, &owned, parts[0..members.len], &runs);
    var outs: [windows.len]mtl.Buffer = undefined;
    for (&outs, windows) |*y, w| {
        y.* = try g.device.buffer(wide * total * 2, opts);
        try g.quant(fused, x, y.*, w);
    }
    defer for (outs) |y| y.deinit();
    const prompt = View{ .vals = outs[windows.len - 1].slice(u16, wide * total), .pitch = total };
    var ok = true;
    for (windows[0 .. windows.len - 1], outs[0 .. windows.len - 1]) |w, y| {
        var what: [48]u8 = undefined;
        ok = same(t, stack_name, try std.fmt.bufPrint(&what, "{d}-row windows", .{w}), prompt, .{ .vals = y.slice(u16, wide * total), .pitch = total }, wide / w * w, total) and ok;
    }
    if (!ok) t.failed += 1;
    std.debug.print("RESULT stack {s}: {s}\n", .{ stack_name, if (ok) "every row's bits equal in calls of 1, 2, 3, 8, 16, 17 and 128 rows" else "a row's bits depend on the call FAIL" });
    const alone = try g.device.buffer(wide * synth[0].n * 2, opts);
    defer alone.deinit();
    const matrix = try g.device.buffer(wide * synth[0].n * 2, opts);
    defer matrix.deinit();
    for (members, runs[0..members.len], 0..) |m, run, i| {
        var tag: [128]u8 = undefined;
        const member = try std.fmt.bufPrint(&tag, "stack {s} member {s} [{d}, {d}] q{d} g64", .{ stack_name, m.name, m.n, k, m.bits });
        const col = View{ .vals = prompt.vals, .pitch = total, .first = run.first };
        const ref = try Ref.init(synth[i], x);
        defer ref.deinit();
        var line: [192]u8 = undefined;
        compare(t, try std.fmt.bufPrint(&line, "{s}, prompt ({d} rows)", .{ member, wide }), col, wide, ref, k, 0);
        compare(t, try std.fmt.bufPrint(&line, "{s}, decode (1 row a call)", .{member}), .{ .vals = outs[0].slice(u16, wide * total), .pitch = total, .first = run.first }, wide, ref, k, 0);
        try g.rows(run.linear.raw.?, x, alone);
        const own = View{ .vals = alone.slice(u16, wide * m.n), .pitch = m.n };
        if (same(t, member, "the stack's columns and the member's own row product", own, col, wide, m.n)) {
            std.debug.print("RESULT {s}: its columns equal its own row product bit for bit\n", .{member});
        } else {
            std.debug.print("RESULT {s}: its columns differ from its own row product FAIL\n", .{member});
            t.failed += 1;
        }
        // the member as a matrix of its own through the family: the row kernel, or simd_qmm at 4/64 (no output stride)
        var solo: std.ArrayList(mtl.Buffer) = .empty;
        defer release(&solo);
        const lin = try projection.prepare(g.device, a, &solo, parts[i .. i + 1], 32);
        try g.quant(lin, x, matrix, wide);
        const via = path(lin).name;
        if (same(t, member, "the stack's columns and the member as its own matrix", .{ .vals = matrix.slice(u16, wide * m.n), .pitch = m.n }, col, wide, m.n)) {
            std.debug.print("RESULT {s}: equals the member as its own matrix ({s}) bit for bit\n", .{ member, via });
        } else if (std.mem.eql(u8, via, "simd_qmm")) {
            std.debug.print("NOTE {s}: as its own matrix it takes simd_qmm (no output stride), whose bits differ\n", .{member});
        } else {
            std.debug.print("RESULT {s}: differs from the member as its own matrix ({s}) FAIL\n", .{ member, via });
            t.failed += 1;
        }
    }
}

pub fn main(_: std.process.Init) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    std.debug.print("device {s}, tensor units {}\n", .{ device.name(), device.tensorUnits() });
    if (device.tensorUnits()) {
        std.debug.print("this program checks the row path; tensor units take the lane kernels\n", .{});
        return error.TensorUnits;
    }
    var kern = q27.quant_gpu.Kernels{ .allocator = a, .device = device };
    defer kern.deinit();
    var pipes = try row.Pipelines.load(device);
    defer pipes.deinit();
    const g = Gpu{ .device = device, .queue = queue, .kern = &kern, .pipes = &pipes };
    var t = Tally{};
    var seed: u64 = 0x2727;
    for (classes) |c| for (bit_widths) |bits| {
        seed += 1;
        try class(&t, g, c, bits, seed);
    };
    for (stacks) |s| {
        seed += 16;
        try stack(&t, g, s, seed);
    }
    if (t.failed > 0) {
        std.debug.print("FAIL {d} checks; the largest difference is {d:.4} of its float64 bound\n", .{ t.failed, t.worst });
        return error.CheckFailed;
    }
    std.debug.print("PASS {d} values within {d:.4} of the float64 bound (scales 1% off: at least {d:.1} of it), {d} bit-equal pairs\n", .{ t.values, t.worst, t.ctl_min, t.pairs });
}
