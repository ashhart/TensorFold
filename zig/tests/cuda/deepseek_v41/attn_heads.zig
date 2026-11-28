//! `tf-dsv41-test attn-heads`: decode attention at 24 heads == a 32-head launch's first 24, every form (synthetic).

// The 32-head launch's last 8 heads hold nonzero q and sinks (a two-rank rank's), so a share can't depend on them.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const Gpu = check.Gpu;
const attn = dsv41.ops.attn;

const D = 512;
const NOPE = 448;
const VS = 584; // compressed and SWA row strides in bytes (the production pools' 584)
const SB = 8; // scale bytes a row
const ring = 256;
const comp_rows = 1024;
const sel = 512; // the selection's stride, NCOMP * CH
const rope_rows = 4096;
const slots = 3;
const guard = 4096; // bytes after the share's output (0xA5) and after each scratch buffer (0x7f)

const share_heads = 24;
const full_heads = 32;

fn bf16(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Bytes on the device, freed together.
const Bufs = struct {
    gpu: Gpu,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn deinit(b: *Bufs) void {
        for (b.list.items) |*x| x.free();
        b.list.deinit(b.gpu.gpa);
    }
    fn up(b: *Bufs, bytes: []const u8) !u64 {
        const buf = try cuda.DeviceBuffer.fromHost(b.gpu.d, if (bytes.len == 0) &.{ 0, 0, 0, 0 } else bytes);
        try b.list.append(b.gpu.gpa, buf);
        return buf.ptr;
    }
    fn filled(b: *Bufs, len: usize, byte: u8) !cuda.DeviceBuffer {
        const buf = try cuda.DeviceBuffer.alloc(b.gpu.d, @max(len, 4));
        try buf.fill8(byte, null);
        try b.list.append(b.gpu.gpa, buf);
        return buf;
    }
};

/// One row count's inputs: the KV pools, selections, windows and positions every case of that row count reads.
const Rows = struct {
    R: usize,
    q_share: u64,
    q_full: u64,
    q_full_host: []u16, // the planted control edits one element of a share head
    sink_share: u64,
    sink_full: u64,
    cv: u64,
    csc: u64,
    tok: u64,
    cnt: u64,
    sv: u64,
    ssc: u64,
    lo: u64,
    hi: u64,
    pos32: u64,
    pos64: u64,
    sl: u64,
    cs: u64,
};

fn rows(a: std.mem.Allocator, b: *Bufs, rng: std.Random, R: usize) !Rows {
    // q: the share's heads, then the full launch's copy of them followed by 8 more nonzero heads
    const q_share = try a.alloc(u16, R * share_heads * D);
    const q_full = try a.alloc(u16, R * full_heads * D);
    for (0..R) |r| for (0..full_heads) |h| for (0..D) |d| {
        const v = bf16(rng.float(f32) * 2 - 1);
        q_full[(r * full_heads + h) * D + d] = v;
        if (h < share_heads) q_share[(r * share_heads + h) * D + d] = v;
    };
    const sink_full = try a.alloc(f32, full_heads);
    for (sink_full, 0..) |*s, h| s.* = if (h == 5) -std.math.inf(f32) else rng.float(f32) * 4 - 2;
    // pools: e4m3 bytes (no NaN encodings) for the 448 nope columns, bf16 for the 64 RoPE columns, 2^-7..1 scales
    const pool = struct {
        fn fill(r: std.Random, v: []u8, s: []u8, n: usize) void {
            for (0..n) |row| {
                for (0..NOPE) |c| {
                    var x = r.int(u8);
                    if (x & 0x7f == 0x7f) x ^= 1;
                    v[row * VS + c] = x;
                }
                for (0..64) |c| std.mem.writeInt(u16, v[row * VS + NOPE + 2 * c ..][0..2], bf16(r.float(f32) * 2 - 1), .little);
                for (0..SB) |c| s[row * SB + c] = 120 + r.uintLessThan(u8, 8);
            }
        }
    };
    const cv = try a.alloc(u8, comp_rows * VS);
    @memset(cv, 0);
    const csc = try a.alloc(u8, comp_rows * SB);
    pool.fill(rng, cv, csc, comp_rows);
    const sv = try a.alloc(u8, slots * ring * VS);
    @memset(sv, 0);
    const ssc = try a.alloc(u8, slots * ring * SB);
    pool.fill(rng, sv, ssc, slots * ring);
    // selections: every chunk count from one to four, a padded entry, rows of the pool in any order
    const tok = try a.alloc(i32, R * sel);
    const cnt = try a.alloc(i32, R);
    for (0..R) |r| {
        const n: usize = ([_]usize{ 512, 300, 129, 64, 1 })[r % 5];
        cnt[r] = @intCast(n);
        for (0..sel) |i| tok[r * sel + i] = if (i >= n) -1 else if (i == 7) -1 else @intCast(rng.uintLessThan(u32, comp_rows));
    }
    // positions (rows mode: one each; position mode: 1000 + r), window starts and ends, slots
    const pos32 = [_]i32{1000};
    const pos64 = try a.alloc(i64, R);
    const lo = try a.alloc(i32, R);
    const hi = try a.alloc(i32, R);
    const sl = try a.alloc(i64, R);
    for (0..R) |r| {
        pos64[r] = 900 + 37 * @as(i64, @intCast(r));
        sl[r] = @intCast(r % slots);
        // window starts at or below every row's position in both modes, binding in some rows of each
        lo[r] = 870 + 20 * @as(i32, @intCast(r % 4));
        hi[r] = 1000 + @as(i32, @intCast(r)) - @as(i32, @intCast(r % 2));
    }
    const cs = try a.alloc(f32, rope_rows * 64);
    for (0..rope_rows) |p| for (0..32) |i| {
        const th = @as(f32, @floatFromInt(p)) * std.math.pow(f32, 10000, -@as(f32, @floatFromInt(i)) / 32);
        cs[p * 64 + i] = @cos(th);
        cs[p * 64 + 32 + i] = @sin(th);
    };
    return .{
        .R = R,
        .q_share = try b.up(std.mem.sliceAsBytes(q_share)),
        .q_full = try b.up(std.mem.sliceAsBytes(q_full)),
        .q_full_host = q_full,
        .sink_share = try b.up(std.mem.sliceAsBytes(sink_full[0..share_heads])),
        .sink_full = try b.up(std.mem.sliceAsBytes(sink_full)),
        .cv = try b.up(cv),
        .csc = try b.up(csc),
        .tok = try b.up(std.mem.sliceAsBytes(tok)),
        .cnt = try b.up(std.mem.sliceAsBytes(cnt)),
        .sv = try b.up(sv),
        .ssc = try b.up(ssc),
        .lo = try b.up(std.mem.sliceAsBytes(lo)),
        .hi = try b.up(std.mem.sliceAsBytes(hi)),
        .pos32 = try b.up(std.mem.sliceAsBytes(&pos32)),
        .pos64 = try b.up(std.mem.sliceAsBytes(pos64)),
        .sl = try b.up(std.mem.sliceAsBytes(sl)),
        .cs = try b.up(std.mem.sliceAsBytes(cs)),
    };
}

const Form = struct { kw: u32, rows: bool, hi: bool, split: u32, qrope: bool, comp: bool };

/// One launch of `H` heads over `x`'s inputs into fresh scratch, a zero ticket and a guarded output; its output bytes.
fn launch(gpu: Gpu, b: *Bufs, k: *const dsv41.Kernels, s: cuda.Stream, x: *const Rows, f: Form, H: i64, q: u64, sink: u64) ![]u8 {
    const R: usize = x.R;
    const HS: usize = @intCast(attn.scratchHeads(H));
    const out_len = R * @as(usize, @intCast(H)) * D * 2;
    const out = try b.filled(out_len + guard, 0xA5);
    const scratch = [_]usize{ R * 5 * HS * D * 4, R * 5 * HS * 4, R * 5 * HS * 4 };
    var parts: [3]cuda.DeviceBuffer = undefined;
    for (&parts, scratch) |*p, n| p.* = try b.filled(n + guard, 0x7f);
    const po, const pm, const pl = parts;
    const ticket_len = R * @as(usize, @intCast(attn.tiles(H))) * attn.MAXP * 4;
    const ticket = try b.filled(ticket_len, 0);
    const args: attn.Args = .{
        .q = q,
        .has_comp = @intFromBool(f.comp),
        .cv = if (f.comp) x.cv else 0,
        .csc = if (f.comp) x.csc else 0,
        .cvs = if (f.comp) VS else 0,
        .css = if (f.comp) SB else 0,
        .tok = if (f.comp) x.tok else 0,
        .ts = if (f.comp) sel else 0,
        .cnt = x.cnt,
        .sv = x.sv,
        .ssc = x.ssc,
        .svs = VS,
        .sss = SB,
        .lo = x.lo,
        .hi = if (f.hi) x.hi else 0,
        .pos = if (f.rows) x.pos64 else x.pos32,
        .sl = if (f.rows) x.sl else 0,
        .sink = sink,
        .cs = x.cs,
        .csst = 64,
        .po = po.ptr,
        .pm = pm.ptr,
        .pl = pl.ptr,
        .out = out.ptr,
        .ticket = ticket.ptr,
        .R = @intCast(R),
        .H = @intCast(H),
        .ring = ring,
        .qrope = @intFromBool(f.qrope),
    };
    try k.others(s).attention(args, f.kw, f.rows, f.hi, f.split);
    try s.synchronize();
    // the ticket cleans itself for the next layer's launch; the scratch stays within whole tiles
    const words = try gpu.gpa.alloc(u8, @max(ticket_len, guard));
    defer gpu.gpa.free(words);
    try ticket.download(0, words[0..ticket_len]);
    try check.expect(std.mem.allEqual(u8, words[0..ticket_len], 0), "{d} heads: the ticket is not clean after the launch", .{H});
    for (parts, scratch) |p, n| {
        try p.download(n, words[0..guard]);
        try check.expect(std.mem.allEqual(u8, words[0..guard], 0x7f), "{d} heads: a write past the scratch's whole tiles", .{H});
    }
    const got = try gpu.gpa.alloc(u8, out_len + guard);
    try out.download(0, got);
    return got;
}

/// The share's rows of a full launch's output: heads [0, 24) of each row.
fn shareOf(a: std.mem.Allocator, full: []const u8, R: usize) ![]u8 {
    const row = share_heads * D * 2;
    const out = try a.alloc(u8, R * row);
    for (0..R) |r| @memcpy(out[r * row ..][0..row], full[r * full_heads * D * 2 ..][0..row]);
    return out;
}

/// Every bf16 finite and some nonzero: the comparison is over real attention outputs.
fn sane(bytes: []const u8) bool {
    var nonzero = false;
    var i: usize = 0;
    while (i + 2 <= bytes.len) : (i += 2) {
        const v = std.mem.readInt(u16, bytes[i..][0..2], .little);
        if (v & 0x7f80 == 0x7f80) return false;
        if (v & 0x7fff != 0) nonzero = true;
    }
    return nonzero;
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels) !void {
    if (k.ops.attn_part == null) return error.PartKernelsMissing;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var b: Bufs = .{ .gpu = gpu };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(0x2470_24);
    const rng = prng.random();
    var cases: usize = 0;
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_]usize{ 1, 2, 5 }) |R| {
        const x = try rows(a, &b, rng, R);
        for ([_]u32{ 2, 4 }) |kw| for ([_]bool{ false, true }) |rm| for ([_]bool{ false, true }) |hi| for ([_]bool{ true, false }) |comp| {
            for ([_][2]u32{ .{ 0, 0 }, .{ 1, 0 }, .{ 1, 1 }, .{ 4, 0 }, .{ 4, 1 } }) |sp| {
                const f: Form = .{ .kw = kw, .rows = rm, .hi = hi, .split = sp[0], .qrope = sp[1] == 1, .comp = comp };
                const got = try launch(gpu, &b, k, stream, &x, f, share_heads, x.q_share, x.sink_share);
                defer gpu.gpa.free(got);
                const full = try launch(gpu, &b, k, stream, &x, f, full_heads, x.q_full, x.sink_full);
                defer gpu.gpa.free(full);
                const want = try shareOf(a, full, R);
                var name_buf: [160]u8 = undefined;
                const what = try std.fmt.bufPrint(&name_buf, "attn-heads R {d} kw {d} rows {} hi {} split {d} qrope {} comp {}", .{ R, kw, rm, hi, f.split, f.qrope, comp });
                try check.expect(sane(want), "{s}: the 32-head launch's share is not finite and nonzero", .{what});
                try check.sameBytes(what, got[0..want.len], want);
                for (got[want.len..]) |g| try check.expect(g == 0xA5, "{s}: a write past the share's {d} heads", .{ what, share_heads });
                digest.update(want);
                cases += 1;
            }
        };
        b.deinit();
        b = .{ .gpu = gpu };
    }
    // control: one element of share head 23 changed in the full launch's q must change that head's bytes and only it
    {
        const x = try rows(a, &b, rng, 2);
        const f: Form = .{ .kw = 4, .rows = true, .hi = true, .split = 4, .qrope = false, .comp = true };
        const before = try launch(gpu, &b, k, stream, &x, f, full_heads, x.q_full, x.sink_full);
        defer gpu.gpa.free(before);
        x.q_full_host[(1 * full_heads + 23) * D + 3] ^= 0x0100;
        const planted = try b.up(std.mem.sliceAsBytes(x.q_full_host));
        const after = try launch(gpu, &b, k, stream, &x, f, full_heads, planted, x.sink_full);
        defer gpu.gpa.free(after);
        const head = D * 2;
        for (0..2) |r| for (0..full_heads) |h| {
            const at = (r * full_heads + h) * head;
            const same = std.mem.eql(u8, before[at..][0..head], after[at..][0..head]);
            try check.expect(same == !(r == 1 and h == 23), "control: row {d} head {d} {s}", .{ r, h, if (same) "unchanged" else "changed" });
        };
    }
    var sum: [32]u8 = undefined;
    digest.final(&sum);
    check.pass("BITEXACT attn-heads: {d} forms of 24 heads == a 32-head launch's first 24 (no write past them); control caught; sha256 {x}", .{ cases, sum });
}
