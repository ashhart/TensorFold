//! window_rows.cu at Gemma 4's shapes and values: outputs in float64's range, a row's bits its own (batch, chunk).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const wc = @import("window_rows_check.zig");

const wa = cuda.window_attention;
const ar = core.attention_range;
const bfr = core.bf16_round;

/// Device buffers freed together.
pub const Bufs = struct {
    gpu: check.Gpu,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn deinit(b: *Bufs) void {
        for (b.list.items) |*x| x.free();
        b.list.deinit(b.gpu.gpa);
    }

    pub fn up(b: *Bufs, bytes: []const u8) !u64 {
        const x = try cuda.DeviceBuffer.fromHost(b.gpu.d, bytes);
        try b.list.append(b.gpu.gpa, x);
        return x.ptr;
    }

    /// `n` bytes of 0xff (an output no launch writes reads as NaN).
    pub fn out(b: *Bufs, s: cuda.Stream, n: usize) !u64 {
        const x = try cuda.DeviceBuffer.alloc(b.gpu.d, @max(n, 16));
        try b.list.append(b.gpu.gpa, x);
        try x.fill8(0xff, s.handle);
        return x.ptr;
    }

    pub fn get(b: *Bufs, ptr: u64, out_: []u16) !void {
        const x: cuda.DeviceBuffer = .{ .d = b.gpu.d, .ptr = ptr, .len = out_.len * 2 };
        try x.download(0, std.mem.sliceAsBytes(out_));
    }
};

/// Gemma 4's attention inputs: q rows of RMS 0.95-1.05, k 0.06-0.13 (the norms' constant weights), v unit RMS.
pub fn fill(rng: std.Random, q: []u16, kc: []u16, vc: []u16, d: usize) void {
    const cq = wc.q_rms[0] + (wc.q_rms[1] - wc.q_rms[0]) * rng.float(f32);
    const ck = wc.k_rms[0] + (wc.k_rms[1] - wc.k_rms[0]) * rng.float(f32);
    var i: usize = 0;
    while (i < q.len) : (i += d) wc.row(rng, q[i..][0..d], cq);
    i = 0;
    while (i < kc.len) : (i += d) wc.row(rng, kc[i..][0..d], ck);
    i = 0;
    while (i < vc.len) : (i += d) wc.row(rng, vc[i..][0..d], 1);
}

/// Key `key` of (row r, key head kh) set to r's first query head of kh at k's RMS: one sharp weight for that head.
pub fn sharpen(c: wc.Call, kc: []u16, r: usize, key: usize, kh: usize) void {
    const w = c.at(r);
    const src = c.q[(r * c.h + kh * (c.h / c.hk)) * c.d ..][0..c.d];
    const dst = kc[(c.cacheRow(w.base, key) * c.hk + kh) * c.d ..][0..c.d];
    for (dst, src) |*x, y| x.* = bfr.bits(bfr.round32(bfr.value(y) * 0.1));
}

fn report(what: []const u8, st: ar.Stats) void {
    check.pass("{s}: {d} outputs bit-equal to float64's value, {d} within their range, {d} non-finite as float64", .{ what, st.decided, st.ranged, st.nonfinite });
}

const DCase = struct { name: []const u8, h: usize, hk: usize, d: usize, win: usize, ring: usize, streams: usize, top: usize, rows: usize = 64 };

/// Decode at 1, 3, 17 (those under c.rows) and c.rows rows: outputs in range, a row's bits at any row count and cache.
fn decode1(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const wa.WindowAttention, rng: std.Random, c: DCase, total: *ar.Stats) !void {
    var list: [4]usize = undefined;
    var nc: usize = 0;
    for ([_]usize{ 1, 3, 17 }) |m| if (m < c.rows) {
        list[nc] = m;
        nc += 1;
    };
    list[nc] = c.rows;
    const counts = list[0 .. nc + 1];
    const most = c.rows;
    var b: Bufs = .{ .gpu = gpu };
    defer b.deinit();
    const q = try a.alloc(u16, most * c.h * c.d);
    const kc = try a.alloc(u16, c.streams * c.ring * c.hk * c.d);
    const vc = try a.alloc(u16, kc.len);
    fill(rng, q, kc, vc, c.d);
    const pos = try a.alloc(i32, most);
    const slot = try a.alloc(i32, most);
    for (pos, slot, 0..) |*p, *sl, i| {
        p.* = @intCast(if (i == most - 1) c.top - 1 else if (i % 4 == 0) i % 40 else (i * 7919) % c.top);
        sl.* = @intCast(i % c.streams);
    }
    var call: wc.Call = .{ .h = c.h, .hk = c.hk, .d = c.d, .win = c.win, .ring = c.ring, .scale = 1, .prompt = false, .rows = most, .q = q, .kc = kc, .vc = vc, .pos = pos, .slot = slot };
    for (0..most) |r| if (r % 3 == 0) { // a sharp key in every third row's window, for each key head
        const w = call.at(r);
        const lo = call.first(w.p);
        for (0..c.hk) |kh| sharpen(call, kc, r, lo + (r * 31 + kh * 7) % (w.p + 1 - lo), kh);
    };
    const dq = try b.up(std.mem.sliceAsBytes(q));
    const dk = try b.up(std.mem.sliceAsBytes(kc));
    const dv = try b.up(std.mem.sliceAsBytes(vc));
    const dp = try b.up(std.mem.sliceAsBytes(pos));
    const ds = try b.up(std.mem.sliceAsBytes(slot));
    const w: wa.Window = .{ .h = c.h, .hk = c.hk, .d = c.d, .win = c.win, .ring = c.ring, .scale = 1 };
    const scratch = try b.out(s, wa.WindowAttention.scratchBytes(w, most));
    var all: [list.len]u64 = undefined;
    const outs = all[0..counts.len];
    for (counts, outs) |m, *o| {
        o.* = try b.out(s, m * c.h * c.d * 2);
        try g.decode(s, dq, dk, dv, o.*, dp, ds, w, scratch, m);
    }
    try s.synchronize();
    const wide = try a.alloc(u16, most * c.h * c.d);
    try b.get(outs[counts.len - 1], wide);
    var st: ar.Stats = .{};
    var msg: [240]u8 = undefined;
    if (try wc.verify(a, call, wide, &st, &msg)) |why| return check.expect(false, "window rows decode {s}: {s}", .{ c.name, why });
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, o| {
        const got = try a.alloc(u16, m * c.h * c.d);
        try b.get(o, got);
        try check.sameBytes("window rows decode: a row count's bits equal the widest call's", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(wide[0 .. m * c.h * c.d]));
    }
    if (c.win == 0 and c.streams == 1 and c.ring <= 32768) { // a cache declared 16x longer: the same bits
        var w2 = w;
        w2.ring = 16 * c.ring;
        const o2 = try b.out(s, most * c.h * c.d * 2);
        try g.decode(s, dq, dk, dv, o2, dp, ds, w2, try b.out(s, wa.WindowAttention.scratchBytes(w2, most)), most);
        try s.synchronize();
        const got = try a.alloc(u16, most * c.h * c.d);
        try b.get(o2, got);
        try check.sameBytes("window rows decode: empty chunks past a row's keys leave its bits", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(wide));
    }
    report(c.name, st);
    total.decided += st.decided;
    total.ranged += st.ranged;
    total.nonfinite += st.nonfinite;
}

const PCase = struct { name: []const u8, h: usize, hk: usize, d: usize, win: usize, ring: usize, p0: usize, n: usize, p32: bool, image: ?[2]usize = null };

/// Prompt rows p0 .. p0 + n, an image span's rows seeing its end: every output in range, the same bits in two chunks.
fn prompt1(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const wa.WindowAttention, rng: std.Random, c: PCase, total: *ar.Stats) !void {
    var b: Bufs = .{ .gpu = gpu };
    defer b.deinit();
    const q = try a.alloc(u16, c.n * c.h * c.d);
    const crow = if (c.ring > 0) c.ring else c.p0 + c.n;
    const kc = try a.alloc(u16, crow * c.hk * c.d);
    const vc = try a.alloc(u16, kc.len);
    fill(rng, q, kc, vc, c.d);
    const hi = try a.alloc(i32, c.n);
    for (hi, 0..) |*x, i| x.* = @intCast(c.p0 + i);
    if (c.image) |sp| for (sp[0]..sp[1]) |i| {
        hi[i] = @intCast(c.p0 + sp[1] - 1);
    };
    var call: wc.Call = .{ .h = c.h, .hk = c.hk, .d = c.d, .win = c.win, .ring = c.ring, .scale = 1, .prompt = true, .p32 = c.p32, .rows = c.n, .q = q, .kc = kc, .vc = vc, .p0 = c.p0, .hi = if (c.image != null) hi else null };
    var r: usize = 0;
    while (r < c.n) : (r += 5) {
        const w = call.at(r);
        const lo = call.first(w.p);
        for (0..c.hk) |kh| sharpen(call, kc, r, lo + (r * 31 + kh * 7) % (w.last + 1 - lo), kh);
    }
    const dq = try b.up(std.mem.sliceAsBytes(q));
    const dk = try b.up(std.mem.sliceAsBytes(kc));
    const dv = try b.up(std.mem.sliceAsBytes(vc));
    const dh = try b.up(std.mem.sliceAsBytes(hi));
    const w: wa.Window = .{ .h = c.h, .hk = c.hk, .d = c.d, .win = c.win, .ring = c.ring, .scale = 1, .p32 = c.p32 };
    const hp: ?u64 = if (c.image != null) dh else null;
    const full = try b.out(s, c.n * c.h * c.d * 2);
    try g.promptRows(s, dq, dk, dv, full, c.p0, c.n, w, hp);
    // the chunk in two at a cut no image span crosses
    var cut = c.n / 2;
    if (c.image) |sp| if (cut > sp[0] and cut < sp[1]) {
        cut = sp[1];
    };
    const row_bytes = c.h * c.d * 2;
    const two = try b.out(s, c.n * row_bytes);
    try g.promptRows(s, dq, dk, dv, two, c.p0, cut, w, hp);
    if (cut < c.n) try g.promptRows(s, dq + cut * row_bytes, dk, dv, two + cut * row_bytes, c.p0 + cut, c.n - cut, w, if (hp) |p| p + cut * 4 else null);
    try s.synchronize();
    const got = try a.alloc(u16, c.n * c.h * c.d);
    try b.get(full, got);
    var st: ar.Stats = .{};
    var msg: [240]u8 = undefined;
    if (try wc.verify(a, call, got, &st, &msg)) |why| return check.expect(false, "window rows prompt {s}: {s}", .{ c.name, why });
    const split = try a.alloc(u16, got.len);
    try b.get(two, split);
    try check.sameBytes("window rows prompt: a chunk split in two gives each row's bits", std.mem.sliceAsBytes(split), std.mem.sliceAsBytes(got));
    report(c.name, st);
    total.decided += st.decided;
    total.ranged += st.ranged;
    total.nonfinite += st.nonfinite;
}

/// Gemma 4's sliding layers (12B: 16 query heads over 8, 31B: 32 over 16; head dim 256, window 1024, ring 1536).
pub const sliding_decode = [_]DCase{
    .{ .name = "12B sliding decode (16/8 heads of 256, window 1024, ring 1536, 3 streams)", .h = 16, .hk = 8, .d = 256, .win = 1024, .ring = 1536, .streams = 3, .top = 40000 },
    .{ .name = "31B sliding decode (32/16 heads of 256, window 1024, ring 1536)", .h = 32, .hk = 16, .d = 256, .win = 1024, .ring = 1536, .streams = 2, .top = 3000 },
    .{ .name = "no window over a linear cache (16/4 heads of 256, 3000 positions)", .h = 16, .hk = 4, .d = 256, .win = 0, .ring = 3000, .streams = 1, .top = 3000 },
};

/// Gemma 4's global layers: head dim 512, no window over a linear cache (12B 16 query heads over 1, 31B 32 over 4).
pub const global_decode = [_]DCase{
    .{ .name = "12B global decode (16/1 heads of 512, linear cache of 262144, a row at its end)", .h = 16, .hk = 1, .d = 512, .win = 0, .ring = 262144, .streams = 1, .top = 262144, .rows = 4 },
    .{ .name = "12B global decode (16/1 heads of 512, linear cache of 20000)", .h = 16, .hk = 1, .d = 512, .win = 0, .ring = 20000, .streams = 1, .top = 20000, .rows = 17 },
    .{ .name = "31B global decode (32/4 heads of 512, linear caches of 40000, 2 streams)", .h = 32, .hk = 4, .d = 512, .win = 0, .ring = 40000, .streams = 2, .top = 40000, .rows = 17 },
    .{ .name = "GQA 5 global decode (10/2 heads of 512, linear cache of 20000)", .h = 10, .hk = 2, .d = 512, .win = 0, .ring = 20000, .streams = 1, .top = 20000, .rows = 17 },
};

pub const global_prompt = [_]PCase{
    .{ .name = "12B global prompt, 128 rows at 8000, P32", .h = 16, .hk = 1, .d = 512, .win = 0, .ring = 0, .p0 = 8000, .n = 128, .p32 = true },
    .{ .name = "12B global prompt, 16 rows at 130000, P32", .h = 16, .hk = 1, .d = 512, .win = 0, .ring = 0, .p0 = 130000, .n = 16, .p32 = true },
    .{ .name = "31B global prompt, 300 rows at 0, P32", .h = 32, .hk = 4, .d = 512, .win = 0, .ring = 0, .p0 = 0, .n = 300, .p32 = true },
    .{ .name = "31B global prompt, 96 rows at 3000, P as bf16", .h = 32, .hk = 4, .d = 512, .win = 0, .ring = 0, .p0 = 3000, .n = 96, .p32 = false },
    .{ .name = "GQA 5 global prompt (10/2 heads), 96 rows at 3000, P32", .h = 10, .hk = 2, .d = 512, .win = 0, .ring = 0, .p0 = 3000, .n = 96, .p32 = true },
    .{ .name = "GQA 5 global prompt (10/2 heads), 80 rows at 700, P as bf16", .h = 10, .hk = 2, .d = 512, .win = 0, .ring = 0, .p0 = 700, .n = 80, .p32 = false },
};

pub const sliding_prompt = [_]PCase{
    .{ .name = "12B sliding prompt, first chunk, P32", .h = 16, .hk = 8, .d = 256, .win = 1024, .ring = 1536, .p0 = 0, .n = 512, .p32 = true },
    .{ .name = "12B sliding prompt, the ring wraps, P32", .h = 16, .hk = 8, .d = 256, .win = 1024, .ring = 1536, .p0 = 5000, .n = 512, .p32 = true },
    .{ .name = "12B sliding prompt, an image's rows 40-300 see each other, P32", .h = 16, .hk = 8, .d = 256, .win = 1024, .ring = 1536, .p0 = 1300, .n = 512, .p32 = true, .image = .{ 40, 300 } },
    .{ .name = "31B sliding prompt, 333 rows, P as bf16", .h = 32, .hk = 16, .d = 256, .win = 1024, .ring = 1536, .p0 = 700, .n = 333, .p32 = false },
    .{ .name = "no window over a linear cache (16/4 heads of 256), P32", .h = 16, .hk = 4, .d = 256, .win = 0, .ring = 0, .p0 = 900, .n = 300, .p32 = true },
};

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try wa.WindowAttention.load(gpu.d);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(0x6a77);
    var total: ar.Stats = .{};
    for (sliding_decode ++ global_decode) |c| {
        _ = arena.reset(.retain_capacity);
        try decode1(gpu, arena.allocator(), s, &g, prng.random(), c, &total);
    }
    for (sliding_prompt ++ global_prompt) |c| {
        _ = arena.reset(.retain_capacity);
        try prompt1(gpu, arena.allocator(), s, &g, prng.random(), c, &total);
    }
    report("window rows (all)", total);
}
