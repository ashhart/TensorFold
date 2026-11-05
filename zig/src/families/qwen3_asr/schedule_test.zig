//! The shared schedules on a host binding: the tower against the float64 reference, the LM's extents, any OS.
const std = @import("std");
const cfg = @import("config.zig");
const chunks = @import("chunks.zig");
const mel = @import("mel.zig");
const ops = @import("ref_ops.zig");
const tower = @import("tower.zig");
const lm = @import("lm.zig");
const tower_ref = @import("tower_ref.zig");

/// Device memory on the host: an address is (allocation + 1) << 40 plus a byte offset, each value a bf16 slot.
const Mem = struct {
    gpa: std.mem.Allocator,
    list: std.ArrayList([]f64) = .empty,

    fn deinit(m: *Mem) void {
        for (m.list.items) |x| m.gpa.free(x);
        m.list.deinit(m.gpa);
    }

    pub fn alloc(m: *Mem, bytes: usize) !u64 {
        const v = try m.gpa.alloc(f64, (bytes + 1) / 2);
        @memset(v, std.math.nan(f64));
        try m.list.append(m.gpa, v);
        return @as(u64, m.list.items.len) << 40;
    }

    pub fn put(m: *Mem, bytes: []const u8) !u64 {
        const p = try m.alloc(bytes.len);
        const v = try m.at(p, bytes.len / 2);
        for (v, 0..) |*x, i| x.* = ops.widen(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
        return p;
    }

    pub fn taps(m: *Mem, bytes: []const u8, _: usize, _: usize) !u64 {
        return m.put(bytes);
    }

    /// `n` values at `ptr`, refused unless they lie inside one allocation.
    fn at(m: *Mem, ptr: u64, n: usize) ![]f64 {
        const id = ptr >> 40;
        const off = ptr & ((1 << 40) - 1);
        if (id == 0 or id > m.list.items.len or off % 2 != 0) return error.BadAddress;
        const v = m.list.items[id - 1];
        if (off / 2 + n > v.len) return error.OutOfBounds;
        return v[off / 2 ..][0..n];
    }
};

/// The tower's operators at torch's rounding points, the stages kept for the comparison.
const TowerHost = struct {
    m: *Mem,
    gpa: std.mem.Allocator,
    stages: std.StringHashMapUnmanaged([]f64) = .empty,

    fn deinit(t: *TowerHost) void {
        var it = t.stages.iterator();
        while (it.next()) |e| {
            t.gpa.free(e.key_ptr.*);
            t.gpa.free(e.value_ptr.*);
        }
        t.stages.deinit(t.gpa);
    }

    pub fn upload(t: *TowerHost, dst: u64, bytes: []const u8) !void {
        const v = try t.m.at(dst, bytes.len / 2);
        for (v, 0..) |*x, i| x.* = ops.widen(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
    }

    pub fn conv(t: *TowerHost, w: u64, b: u64, x: u64, y: u64, n: usize, fi: usize, ti: usize, c: usize, co: usize) !void {
        const fo = (fi - 1) / 2 + 1;
        const to = (ti - 1) / 2 + 1;
        const xs = try t.m.at(x, n * fi * ti * c);
        const ws = try t.m.at(w, co * c * 9);
        const bs = try t.m.at(b, co);
        const ys = try t.m.at(y, n * fo * to * co);
        for (0..n) |s| for (0..fo) |f| for (0..to) |tt| for (0..co) |o| {
            var acc: f64 = bs[o];
            for (0..c) |ci| for (0..3) |kh| for (0..3) |kw| {
                const sf = @as(isize, @intCast(2 * f + kh)) - 1;
                const st = @as(isize, @intCast(2 * tt + kw)) - 1;
                if (sf < 0 or st < 0 or sf >= fi or st >= ti) continue;
                acc += xs[((s * fi + @as(usize, @intCast(sf))) * ti + @as(usize, @intCast(st))) * c + ci] * ws[(o * c + ci) * 9 + kh * 3 + kw];
            };
            ys[((s * fo + f) * to + tt) * co + o] = ops.bf16(acc);
        };
    }

    pub fn dump(t: *TowerHost, stage: []const u8, ptr: u64, rows: usize, cols: usize, conv_shape: ?[3]usize) !void {
        const v = try t.m.at(ptr, rows * cols);
        if (conv_shape != null) return; // the reference taps convs per chunk, channels first: extents only here
        const got = try t.stages.getOrPut(t.gpa, stage);
        if (got.found_existing) t.gpa.free(got.value_ptr.*) else got.key_ptr.* = try t.gpa.dupe(u8, stage);
        got.value_ptr.* = try t.gpa.dupe(f64, v);
    }

    pub fn flatten(t: *TowerHost, x: u64, y: u64, n: usize, f: usize, tt: usize, c: usize) !void {
        const xs = try t.m.at(x, n * f * tt * c);
        const ys = try t.m.at(y, n * tt * c * f);
        for (0..n) |s| for (0..tt) |r| for (0..c) |ci| for (0..f) |fi| {
            ys[(s * tt + r) * c * f + ci * f + fi] = xs[((s * f + fi) * tt + r) * c + ci];
        };
    }

    pub fn linear(t: *TowerHost, x: u64, w: u64, b: ?u64, y: u64, rows: usize, n: usize, k: usize, act: tower.Act, res: ?u64) !void {
        const tmp = try t.gpa.alloc(f64, rows * n);
        defer t.gpa.free(tmp);
        ops.linear(tmp, try t.m.at(x, rows * k), try t.m.at(w, n * k), if (b) |p| try t.m.at(p, n) else null, rows, n, k);
        const r = if (res) |p| try t.m.at(p, rows * n) else null;
        const ys = try t.m.at(y, rows * n);
        for (tmp, 0..) |v0, i| {
            var v = ops.bf16(v0);
            if (act == .gelu) v = ops.bf16(ops.gelu(v));
            if (r) |rr| v = ops.bf16(v + rr[i]);
            ys[i] = v;
        }
    }

    pub fn layernorm(t: *TowerHost, x: u64, w: u64, b: u64, y: u64, rows: usize, d: usize) !void {
        const ys = try t.m.at(y, rows * d);
        @memcpy(ys, try t.m.at(x, rows * d));
        ops.layerNorm(ys, try t.m.at(w, d), try t.m.at(b, d), d);
        ops.roundAll(.torch, ys);
    }

    pub fn attention(t: *TowerHost, qkv: u64, out: u64, len: usize, heads: usize, dim: usize) !void {
        const o = try t.m.at(out, len * heads * dim);
        try ops.attention(o, try t.m.at(qkv, len * 3 * heads * dim), 0, len, heads, dim);
        ops.roundAll(.torch, o);
    }

    pub fn gelu(t: *TowerHost, x: u64, y: u64, count: usize) !void {
        const xs = try t.m.at(x, count);
        for (try t.m.at(y, count), xs) |*o, v| o.* = ops.bf16(ops.gelu(v));
    }

    pub fn sync(_: *TowerHost) !void {}
};

/// Random bf16 tensors made on first ask, the same bytes for the schedule and the reference.
const Fake = struct {
    gpa: std.mem.Allocator,
    rng: std.Random.DefaultPrng = .init(0x2105),
    map: std.StringHashMapUnmanaged([]u8) = .empty,

    fn deinit(f: *Fake) void {
        var it = f.map.iterator();
        while (it.next()) |e| {
            f.gpa.free(e.key_ptr.*);
            f.gpa.free(e.value_ptr.*);
        }
        f.map.deinit(f.gpa);
    }

    pub fn expect(f: *Fake, name: []const u8, shape: []const usize) ![]const u8 {
        if (f.map.get(name)) |b| return b;
        var n: usize = 1;
        for (shape) |s| n *= s;
        const b = try f.gpa.alloc(u8, 2 * n);
        const fan: f32 = @floatFromInt(if (shape.len > 1) n / shape[0] else 4);
        const norm = std.mem.endsWith(u8, name, "layer_norm.weight") or std.mem.endsWith(u8, name, "ln_post.weight");
        for (0..n) |i| {
            const v: f32 = if (norm) 1 + 0.1 * f.rng.random().floatNorm(f32) else f.rng.random().floatNorm(f32) / @sqrt(fan);
            std.mem.writeInt(u16, b[2 * i ..][0..2], tower.bf16Bits(v), .little);
        }
        try f.map.put(f.gpa, try f.gpa.dupe(u8, name), b);
        return b;
    }

    fn get(ctx: *anyopaque, name: []const u8) ?[]const u8 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        return f.map.get(name);
    }
};

const Ref = struct {
    gpa: std.mem.Allocator,
    stages: std.StringHashMapUnmanaged([]f64) = .empty,

    fn put(ctx: *anyopaque, stage: []const u8, offset: usize, values: []const f64) anyerror!void {
        const r: *Ref = @ptrCast(@alignCast(ctx));
        if (offset != 0 or std.mem.startsWith(u8, stage, "conv")) return;
        try r.stages.put(r.gpa, try r.gpa.dupe(u8, stage), try r.gpa.dupe(f64, values));
    }

    fn deinit(r: *Ref) void {
        var it = r.stages.iterator();
        while (it.next()) |e| {
            r.gpa.free(e.key_ptr.*);
            r.gpa.free(e.value_ptr.*);
        }
        r.stages.deinit(r.gpa);
    }
};

fn cosine(a: []const f64, b: []const f64) f64 {
    var ab: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    for (a, b) |x, y| {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    return ab / @sqrt(aa * bb);
}

const small: cfg.Audio = .{ .d = 8, .layers = 2, .heads = 2, .ffn = 16, .mels = 16, .conv = 4, .out = 6, .window = 50, .window_infer = 200, .positions = 1500, .conv_chunks = 500 };

test "tower schedule: a host binding matches the float64 reference at torch's points, two windows, a short chunk" {
    const gpa = std.testing.allocator;
    var mem: Mem = .{ .gpa = gpa };
    defer mem.deinit();
    var fake: Fake = .{ .gpa = gpa };
    defer fake.deinit();
    var w = try tower.Weights.load(gpa, small, &fake, &mem, 3); // groups of 3 over 4 chunks: one short group
    defer w.deinit(gpa);
    const frames = 350;
    const data = try gpa.alloc(f32, small.mels * frames);
    defer gpa.free(data);
    var rng: std.Random.DefaultPrng = .init(7);
    for (data) |*v| v.* = rng.random().floatNorm(f32);
    const feats: mel.Features = .{ .data = data, .mels = small.mels, .frames = frames };
    const plan = try chunks.Plan.init(small, frames);
    try std.testing.expectEqual(@as(usize, 2), plan.windows());
    const b = try tower.Buffers.alloc(small, 3, plan.rows, &mem);
    const out = try mem.alloc(2 * plan.rows * small.out);
    var host: TowerHost = .{ .m = &mem, .gpa = gpa };
    defer host.deinit();
    try std.testing.expectEqual(plan.rows, try tower.encode(&host, gpa, small, &w, b, feats, out));
    var ref: Ref = .{ .gpa = gpa };
    defer ref.deinit();
    const want = try tower_ref.run(gpa, small, .torch, .{ .ctx = &fake, .get = Fake.get }, feats, .{ .ctx = &ref, .put = Ref.put });
    defer gpa.free(want);
    var it = ref.stages.iterator();
    var seen: usize = 0;
    while (it.next()) |e| {
        const got = host.stages.get(e.key_ptr.*) orelse continue;
        seen += 1;
        try std.testing.expectEqual(e.value_ptr.len, got.len);
        const c = cosine(got, e.value_ptr.*);
        if (c < 0.9999) std.debug.print("stage {s}: cosine {d}\n", .{ e.key_ptr.*, c });
        try std.testing.expect(c >= 0.9999);
    }
    try std.testing.expect(seen >= small.layers + 4); // embed, each layer, ln_post, proj1, audio_rows
}

/// The LM's operators as extents only: every row an op touches lies inside the buffer it names.
const LmBounds = struct {
    m: *Mem,
    t: cfg.Text,
    calls: usize = 0,
    pos: usize = 0,
    rows: usize = 0,

    fn touch(b: *LmBounds, ptr: u64, values: usize) !void {
        _ = try b.m.at(ptr, values);
        b.calls += 1;
    }

    pub fn positions(b: *LmBounds, ids: []const u32, pos: usize) !void {
        b.pos = pos;
        b.rows = ids.len;
    }
    pub fn embed(b: *LmBounds, table: u64, out: u64, rows: usize, hidden: usize) !void {
        try b.touch(table, b.t.vocab * hidden);
        try b.touch(out, rows * hidden);
    }
    pub fn copy(b: *LmBounds, dst: u64, src: u64, n: usize) !void {
        try b.touch(dst, n / 2);
        try b.touch(src, n / 2);
    }
    pub fn addNorm(b: *LmBounds, x: u64, res: ?u64, w: u64, h: u64, y: u64, rows: usize) !void {
        for ([_]u64{ x, res orelse x, h, y }) |p| try b.touch(p, rows * b.t.hidden);
        try b.touch(w, b.t.hidden);
    }
    pub fn gemm(b: *LmBounds, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        try b.touch(x, rows * k);
        try b.touch(w, n * k);
        try b.touch(y, rows * n);
    }
    pub fn qkRope(b: *LmBounds, qkv: u64, qn: u64, kn: u64, layer: usize, rows: usize) !void {
        try std.testing.expect(layer < b.t.layers and rows == b.rows);
        try b.touch(qkv, rows * (b.t.heads + 2 * b.t.kv_heads) * b.t.head_dim);
        try b.touch(qn, b.t.head_dim);
        try b.touch(kn, b.t.head_dim);
    }
    pub fn attention(b: *LmBounds, layer: usize, out: u64, rows: usize) !void {
        try std.testing.expect(layer < b.t.layers and rows == b.rows);
        try b.touch(out, rows * b.t.heads * b.t.head_dim);
    }
    pub fn swiglu(b: *LmBounds, gu: u64, out: u64, rows: usize, i: usize) !void {
        try b.touch(gu, rows * 2 * i);
        try b.touch(out, rows * i);
    }
    pub fn dump(b: *LmBounds, _: []const u8, ptr: u64, rows: usize, cols: usize) !void {
        try b.touch(ptr, rows * cols);
    }
    pub fn greedy(b: *LmBounds, row: u64, embed_table: u64) !u32 {
        try b.touch(row, b.t.hidden);
        try b.touch(embed_table, b.t.vocab * b.t.hidden);
        return 7;
    }
};

test "lm schedule: every op inside its buffers, the splice inside the pass, the head tied" {
    const gpa = std.testing.allocator;
    const t: cfg.Text = .{ .hidden = 16, .layers = 3, .heads = 4, .kv_heads = 2, .head_dim = 8, .intermediate = 24, .vocab = 32, .eps = 1e-6, .theta = 1e6, .context = 64 };
    var mem: Mem = .{ .gpa = gpa };
    defer mem.deinit();
    var fake: Fake = .{ .gpa = gpa };
    defer fake.deinit();
    _ = try fake.expect("thinker.lm_head.weight", &.{ t.vocab, t.hidden });
    try std.testing.expectError(error.UntiedHead, lm.Weights.load(gpa, t, &fake, &mem));
    const head = fake.map.getPtr("thinker.lm_head.weight").?;
    @memcpy(head.*, try fake.expect("thinker.model.embed_tokens.weight", &.{ t.vocab, t.hidden }));
    var w = try lm.Weights.load(gpa, t, &fake, &mem);
    defer w.deinit(gpa);
    const b = try lm.Buffers.alloc(t, 10, &mem);
    const rows = try mem.alloc(2 * 4 * t.hidden);
    var bounds: LmBounds = .{ .m = &mem, .t = t };
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try std.testing.expectEqual(@as(u32, 7), try lm.forward(&bounds, t, &w, b, &ids, 0, .{ .ptr = rows, .at = 3, .n = 4 }));
    try std.testing.expectEqual(@as(u32, 7), try lm.forward(&bounds, t, &w, b, ids[0..1], 10, null));
    try std.testing.expectError(error.SpliceOutsidePass, lm.forward(&bounds, t, &w, b, &ids, 0, .{ .ptr = rows, .at = 8, .n = 4 }));
    try std.testing.expectError(error.PassRows, lm.forward(&bounds, t, &w, b, &(ids ++ ids), 0, null));
}
