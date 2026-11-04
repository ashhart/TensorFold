//! An EXL3 pack's MTP head on v1.1.0's Flash Next family (cuda_mtp.zig): the draft vocabulary rows here.

const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const projection = @import("../../cuda_projection.zig");
const fmt = core.exl3_format;

const block_n: usize = fmt.hadamard_dim;

/// lm_head as stored: trellis int16 [K/16, N/16, 8 * halves] (bytes, possibly unaligned), suh fp16 [K], svh fp16 [N].
pub const Head = struct {
    trellis: []const u8,
    suh: []const u8,
    svh: []const u8,
    k: usize,
    n: usize,
    halves: u16,

    pub fn words(h: Head) usize {
        return @as(usize, h.halves) * 8;
    }

    /// From the stored tensors; `mul1` (when present) must hold the mul1 multiplier.
    pub fn of(trellis: core.safetensors.Tensor, suh: core.safetensors.Tensor, svh: core.safetensors.Tensor, mul1: ?core.safetensors.Tensor) !Head {
        if (trellis.dtype != .i16 or trellis.rank != 3 or trellis.dim(2) % 8 != 0) return error.UnexpectedTensor;
        const h: Head = .{ .trellis = trellis.bytes, .suh = suh.bytes, .svh = svh.bytes, .k = trellis.dim(0) * 16, .n = trellis.dim(1) * 16, .halves = @intCast(trellis.dim(2) / 8) };
        if (suh.dtype != .f16 or suh.numel() != h.k or svh.dtype != .f16 or svh.numel() != h.n) return error.UnexpectedTensor;
        if (h.k % block_n != 0 or h.n % block_n != 0 or h.halves < 2 or h.halves > 16) return error.UnexpectedTensor;
        if (mul1) |m| {
            if (m.bytes.len != 4 or std.mem.readInt(u32, m.bytes[0..4], .little) != fmt.Mul1Mul) return error.UnexpectedTensor;
        }
        return h;
    }
};

/// f64 -> bf16 bits, one rounding to nearest even (no f32 step, so no double rounding).
pub fn bf16Of(v: f64) u16 {
    const near = wt.toBf16(@floatCast(v));
    if (std.math.isNan(v) or std.math.isInf(v)) return near;
    var best = near;
    var best_d = @abs(@as(f64, wt.promote(near)) - v);
    for ([_]u16{ near -% 1, near +% 1 }) |cand| {
        const cv: f64 = wt.promote(cand);
        if (std.math.isNan(cv) or std.math.isInf(cv)) continue;
        if (std.math.signbit(cv) != std.math.signbit(@as(f64, wt.promote(near))) and cv != 0) continue;
        const d = @abs(cv - v);
        if (d < best_d or (d == best_d and cand & 1 == 0)) {
            best = cand;
            best_d = d;
        }
    }
    return best;
}

const Work = struct {
    head: Head,
    ids: []const u32,
    /// Indices into `ids`, sorted by id.
    order: []const u32,
    /// Start of each block's run in `order`; `blocks` entries plus one.
    starts: []const u32,
    out: []u16,
    next: std.atomic.Value(usize) = .init(0),
    failed: ?anyerror = null,
};

fn copyF16(bytes: []const u8, out: []f16) void {
    for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
}

fn worker(w: *Work, gpa: std.mem.Allocator) void {
    work(w, gpa) catch |err| {
        w.failed = err;
    };
}

fn work(w: *Work, gpa: std.mem.Allocator) !void {
    const h = w.head;
    const tiles_k = h.k / 16;
    const tiles_n = h.n / 16;
    const per = block_n / 16;
    const words = h.words();
    const sub = try gpa.alloc(i16, tiles_k * per * words);
    defer gpa.free(sub);
    const vals = try gpa.alloc(f64, h.k * block_n);
    defer gpa.free(vals);
    const scratch = try gpa.alloc(f64, 2 * h.k * block_n);
    defer gpa.free(scratch);
    const suh = try gpa.alloc(f16, h.k);
    defer gpa.free(suh);
    copyF16(h.suh, suh);
    var svh: [block_n]f16 = undefined;
    while (true) {
        const b = w.next.fetchAdd(1, .monotonic);
        if (b + 1 >= w.starts.len) return;
        const lo = w.starts[b];
        const hi = w.starts[b + 1];
        if (lo == hi) continue;
        const nb = w.ids[w.order[lo]] / block_n;
        // the block's tiles, k-major as a [tiles_k, per] trellis of their own
        for (0..tiles_k) |tk| for (0..per) |j| {
            const src = ((tk * tiles_n) + nb * per + j) * words * 2;
            const dst = sub[(tk * per + j) * words ..][0..words];
            for (dst, 0..) |*d, i| d.* = std.mem.readInt(i16, h.trellis[src + 2 * i ..][0..2], .little);
        };
        copyF16(h.svh[nb * block_n * 2 ..][0 .. block_n * 2], &svh);
        fmt.dequantize(.{ .halves = h.halves }, .mul1, sub, tiles_k, per, suh, &svh, vals, scratch);
        for (w.order[lo..hi]) |i| {
            const col = w.ids[i] - nb * block_n;
            const row = w.out[@as(usize, i) * h.k ..][0..h.k];
            for (row, 0..) |*o, k| o.* = bf16Of(vals[k * block_n + col]);
        }
    }
}

/// Row i of `out` ([ids.len, K] bf16) is lm_head's column `ids[i]` (the weight row of token ids[i]).
pub fn draftRows(gpa: std.mem.Allocator, head: Head, ids: []const u32, out: []u16, threads: usize) !void {
    if (out.len != ids.len * head.k) return error.UnexpectedTensor;
    for (ids) |id| if (id >= head.n) return error.InvalidToken;
    const order = try gpa.alloc(u32, ids.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, ids, struct {
        fn lt(x: []const u32, a: u32, b: u32) bool {
            return x[a] < x[b];
        }
    }.lt);
    // runs of one block each
    var starts: std.ArrayList(u32) = .empty;
    defer starts.deinit(gpa);
    var prev: ?usize = null;
    for (order, 0..) |i, at| {
        const nb = ids[i] / block_n;
        if (prev == null or prev.? != nb) try starts.append(gpa, @intCast(at));
        prev = nb;
    }
    try starts.append(gpa, @intCast(order.len));
    var w: Work = .{ .head = head, .ids = ids, .order = order, .starts = starts.items, .out = out };
    const n_threads = @max(1, @min(threads, starts.items.len - 1));
    if (n_threads == 1) {
        try work(&w, gpa);
        return;
    }
    var pool: [64]std.Thread = undefined;
    var spawned: usize = 0;
    defer for (pool[0..spawned]) |t| t.join();
    for (0..@min(n_threads, pool.len)) |_| {
        pool[spawned] = try std.Thread.spawn(.{}, worker, .{ &w, gpa });
        spawned += 1;
    }
    for (pool[0..spawned]) |t| t.join();
    spawned = 0;
    if (w.failed) |err| return err;
}

/// Threads for a load: the machine's cores, at most 32 (each holds about 8 MB of float64 scratch).
pub fn loadThreads() usize {
    return @min(32, std.Thread.getCpuCount() catch 1);
}

/// The draft head of an EXL3 pack (null for any other): bf16 rows uploaded as a dense projection, or affine4 rows.
pub fn draft(a: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, ids: []const u32, affine: bool) !?wt.Q4 {
    if (!store.has("lm_head.trellis")) return null;
    const head = try Head.of(try store.hold("lm_head.trellis"), try store.hold("lm_head.suh"), try store.hold("lm_head.svh"), if (store.has("lm_head.mul1")) try store.hold("lm_head.mul1") else null);
    if (head.k != wt.dims or head.n != wt.vocab) return error.UnexpectedTensor;
    const rows = try a.alloc(u16, ids.len * wt.dims);
    defer a.free(rows);
    try draftRows(a, head, ids, rows, loadThreads());
    const bytes = std.mem.sliceAsBytes(rows);
    if (!affine) return try projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ ids.len, wt.dims, 1, 1, 1 }, .bytes = bytes });
    const words = try a.alloc(u8, ids.len * wt.dims / 2);
    defer a.free(words);
    const scales = try a.alloc(u8, ids.len * wt.dims / 16);
    defer a.free(scales);
    const biases = try a.alloc(u8, scales.len);
    defer a.free(biases);
    for (0..ids.len) |i| try @import("../../cuda_draft_affine.zig").row(bytes[i * wt.dims * 2 ..][0 .. wt.dims * 2], words[i * wt.dims / 2 ..][0 .. wt.dims / 2], scales[i * wt.dims / 16 ..][0 .. wt.dims / 16], biases[i * wt.dims / 16 ..][0 .. wt.dims / 16]);
    return try wt.uploadQ4(a, d, words, scales, biases, ids.len, wt.dims / 8);
}

// ---- tests --------------------------------------------------------------------------------------------------------

test "exl3 draft: bf16Of rounds float64 once, to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16Of(1.0));
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16Of(1.0 + 1.0 / 256.0)); // tie: even (down)
    try std.testing.expectEqual(@as(u16, 0x3F82), bf16Of(1.0 + 3.0 / 256.0)); // tie: even (up)
    // just above a tie after an f32 step would round to the tie: float64 sees it above
    const above = 1.0 + 1.0 / 256.0 + 1.0e-12;
    try std.testing.expectEqual(@as(u16, 0x3F81), bf16Of(above));
    try std.testing.expectEqual(@as(u16, 0xBF80), bf16Of(-1.0));
    try std.testing.expectEqual(@as(u16, 0), bf16Of(0));
}

fn synthHead(gpa: std.mem.Allocator, r: std.Random, k: usize, n: usize, halves: u16) !struct { t: []i16, su: []f16, sv: []f16, head: Head } {
    const t = try gpa.alloc(i16, (k / 16) * (n / 16) * halves * 8);
    for (t) |*x| x.* = @bitCast(r.int(u16));
    const su = try gpa.alloc(f16, k);
    for (su) |*x| x.* = @floatCast((r.float(f32) * 0.02 + 0.005) * (if (r.boolean()) @as(f32, 1) else -1));
    const sv = try gpa.alloc(f16, n);
    for (sv) |*x| x.* = @floatCast((r.float(f32) * 0.5 + 0.5) * (if (r.boolean()) @as(f32, 1) else -1));
    return .{ .t = t, .su = su, .sv = sv, .head = .{ .trellis = std.mem.sliceAsBytes(t), .suh = std.mem.sliceAsBytes(su), .svh = std.mem.sliceAsBytes(sv), .k = k, .n = n, .halves = halves } };
}

test "exl3 draft: rows equal the whole matrix's float64 decode rounded to bf16, any id order, any thread count" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xd4af);
    const r = prng.random();
    const k = 256;
    const n = 384;
    const s = try synthHead(gpa, r, k, n, 10);
    defer gpa.free(s.t);
    defer gpa.free(s.su);
    defer gpa.free(s.sv);
    const full = try gpa.alloc(f64, k * n);
    defer gpa.free(full);
    const scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(scratch);
    fmt.dequantize(.{ .halves = 10 }, .mul1, s.t, k / 16, n / 16, s.su, s.sv, full, scratch);
    const ids = [_]u32{ 300, 5, 127, 128, 383, 5, 0, 200, 255, 256 };
    for ([_]usize{ 1, 3 }) |threads| {
        var rows: [ids.len * k]u16 = undefined;
        try draftRows(gpa, s.head, &ids, &rows, threads);
        for (ids, 0..) |id, i| for (0..k) |kk| {
            try std.testing.expectEqual(bf16Of(full[kk * n + id]), rows[i * k + kk]);
        };
    }
    var bad: [k]u16 = undefined;
    try std.testing.expectError(error.InvalidToken, draftRows(gpa, s.head, &.{384}, &bad, 1));
}

test "exl3 draft: a draft logit is the full head's within the bf16 weight bound" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x10917);
    const r = prng.random();
    const k = 256;
    const n = 256;
    const s = try synthHead(gpa, r, k, n, 6);
    defer gpa.free(s.t);
    defer gpa.free(s.su);
    defer gpa.free(s.sv);
    const full = try gpa.alloc(f64, k * n);
    defer gpa.free(full);
    const scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(scratch);
    fmt.dequantize(.{ .halves = 6 }, .mul1, s.t, k / 16, n / 16, s.su, s.sv, full, scratch);
    var x: [k]f64 = undefined;
    for (&x) |*v| v.* = (r.float(f64) - 0.5) * 4;
    var ids: [n]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i);
    const rows = try gpa.alloc(u16, n * k);
    defer gpa.free(rows);
    try draftRows(gpa, s.head, &ids, rows, 2);
    for (0..n) |v| {
        var exact: f64 = 0;
        var mag: f64 = 0;
        var got: f64 = 0;
        for (0..k) |kk| {
            exact += x[kk] * full[kk * n + v];
            mag += @abs(x[kk] * full[kk * n + v]);
            got += x[kk] * @as(f64, wt.promote(rows[v * k + kk]));
        }
        try std.testing.expect(@abs(got - exact) <= mag / 512.0 + 1e-12);
    }
}
