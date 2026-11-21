//! Test support: CUDA-format state encoded as the kernels write it (`kv_norm_store`), with an independent expectation.
const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const plan = @import("plan.zig");
const ex = @import("export.zig");

pub const Rng = struct {
    p: std.Random.DefaultPrng,
    pub fn init(seed: u64) Rng {
        return .{ .p = .init(seed) };
    }
    pub fn norm(r: *Rng) f32 {
        return r.p.random().floatNorm(f32);
    }
    pub fn below(r: *Rng, n: u32) u32 {
        return r.p.random().uintLessThan(u32, n);
    }
};

/// The value of e4m3fn code `k` (0..126 positive finite) from its fields, in f64 and by pow.
pub fn e4Ref(code: u8) f64 {
    const e: i32 = (code >> 3) & 15;
    const m: f64 = @floatFromInt(code & 7);
    const mag = if (e == 0) m / 8 * std.math.pow(f64, 2, -6) else (1 + m / 8) * std.math.pow(f64, 2, @as(f64, @floatFromInt(e - 7)));
    return if (code & 0x80 != 0) -mag else mag;
}

const e4table: [127]f64 = blk: {
    @setEvalBranchQuota(100_000);
    var t: [127]f64 = undefined;
    for (&t, 0..) |*v, k| v.* = e4Ref(k);
    break :blk t;
};

/// The nearest e4m3 code, ties to the even code, saturating at 448 (a search over `e4Ref`'s values).
pub fn e4Encode(x: f32) u8 {
    const a: f64 = @min(@abs(@as(f64, x)), 448);
    var best: u8 = 0;
    var dist: f64 = std.math.inf(f64);
    for (0..127) |k| {
        const d = @abs(a - e4table[k]);
        if (d < dist or (d == dist and k % 2 == 0)) {
            best = @intCast(k);
            dist = d;
        }
    }
    return best | if (std.math.signbit(x)) @as(u8, 0x80) else 0;
}

/// A bf16 row as the CUDA kernel stores it: 576 value bytes then 8 scale bytes.
pub fn encodeRecord(row: *const [512]u16, rec: *[g.cuda_record_bytes]u8) void {
    @memset(rec, 0);
    const floor: f32 = @bitCast(@as(u32, 0x38d1b717));
    for (0..g.cuda_tiles) |t| {
        var peak: f32 = 0;
        for (t * 64..t * 64 + 64) |d| peak = @max(peak, @abs(c.bf16ToF32(row[d])));
        const bits: u32 = @bitCast(@max(peak, floor));
        const e: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 135 + @intFromBool(bits & 0x7fffff > 0x600000);
        const inv: f32 = @bitCast(@as(u32, @intCast(127 - e)) << 23);
        for (t * 64..t * 64 + 64) |d| rec[d] = e4Encode(c.bf16ToF32(row[d]) * inv);
        rec[g.cuda_values_bytes + t] = @intCast(e + 127);
    }
    for (0..64) |j| std.mem.writeInt(u16, rec[g.cuda_nope + 2 * j ..][0..2], row[g.cuda_nope + j], .little);
}

/// What a correct decoder reads from `rec`, from the byte fields alone.
pub fn recordRef(rec: *const [g.cuda_record_bytes]u8) [512]u16 {
    var out: [512]u16 = undefined;
    for (0..g.cuda_nope) |d| {
        const s = std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, rec[g.cuda_values_bytes + d / 64]) - 127)));
        out[d] = c.f32ToBf16(@floatCast(e4Ref(rec[d]) * s));
    }
    for (0..64) |j| out[g.cuda_nope + j] = std.mem.readInt(u16, rec[g.cuda_nope + 2 * j ..][0..2], .little);
    return out;
}

/// An index key (bf16 values) as `_index_k` stores it with KFP8: e4m3 codes and one power-of-two fp32 scale.
pub fn encodeIndex(x: *const [128]u16, row: *[g.cuda_index_fp8_bytes]u8) [128]u16 {
    var amax: f32 = 0;
    for (x) |v| amax = @max(amax, @abs(c.bf16ToF32(v)));
    const s: f32 = std.math.pow(f32, 2, @ceil(std.math.log2(@max(amax, 1e-4) / 448)));
    var ref: [128]u16 = undefined;
    for (x, 0..) |v, d| {
        row[d] = e4Encode(c.bf16ToF32(v) / s);
        ref[d] = c.f32ToBf16(@floatCast(e4Ref(row[d]) * s));
    }
    std.mem.writeInt(u32, row[128..][0..4], @bitCast(s), .little);
    return ref;
}

pub fn genRow(rng: *Rng, out: *[512]u16) void {
    for (0..8) |t| {
        const k: i32 = @as(i32, @intCast(rng.below(9))) - 4;
        for (t * 64..t * 64 + 64) |d| out[d] = c.f32ToBf16(rng.norm() * std.math.pow(f32, 2, @floatFromInt(k)));
    }
}

pub fn genIndex(rng: *Rng, out: *[128]u16) void {
    const k: i32 = @as(i32, @intCast(rng.below(7))) - 3;
    for (out) |*v| v.* = c.f32ToBf16(rng.norm() * std.math.pow(f32, 2, @floatFromInt(k)));
}

/// `rows` rows of `row_bytes` as pool pages, replicated (one rank) or split over `world` ranks; unwritten rows 0xA5.
pub fn paged(arena: std.mem.Allocator, rows: []const []const u8, row_bytes: usize, page_rows: u32, world: u32, split: bool) ![]const []const u8 {
    const pages = (rows.len + page_rows - 1) / page_rows;
    const ranks = try arena.alloc([]u8, if (split) world else 1);
    for (ranks, 0..) |*r, i| {
        const owned = if (split) (pages + world - 1 - i) / world else pages;
        r.* = try arena.alloc(u8, owned * page_rows * row_bytes);
        @memset(r.*, 0xA5);
    }
    for (rows, 0..) |row, n| {
        const pg = n / page_rows;
        const r = if (split) pg % world else 0;
        const local = if (split) pg / world else pg;
        @memcpy(ranks[r][(local * page_rows + n % page_rows) * row_bytes ..][0..row_bytes], row);
    }
    return ranks;
}

pub const Truth = struct {
    comp: [g.release.sources.len][]u16,
    idx: [g.release.sources.len][]u16,
    swa: [40][]u16,
    ds: [3][]u16,
};

pub const Options = struct { upto: u64, world: u32 = 1, split: bool = false, index: ex.IndexFormat = .fp8, ds_rows: u32 = 0, seed: u64 = 1 };

pub const World = struct { cuda: ex.Cuda, truth: Truth };

fn ringOf(arena: std.mem.Allocator, rng: *Rng, upto: u64, n: u32, ring_rows: u32, expect: *[]u16) !ex.Ring {
    const values = try arena.alloc(u8, ring_rows * g.cuda_values_bytes);
    const scales = try arena.alloc(u8, ring_rows * g.cuda_scale_bytes);
    @memset(values, 0x5A);
    @memset(scales, 0x5A);
    expect.* = try arena.alloc(u16, n * 512);
    for (0..n) |i| {
        var row: [512]u16 = undefined;
        var rec: [g.cuda_record_bytes]u8 = undefined;
        genRow(rng, &row);
        encodeRecord(&row, &rec);
        const slot = (upto - n + i) % ring_rows;
        @memcpy(values[slot * 576 ..][0..576], rec[0..576]);
        @memcpy(scales[slot * 8 ..][0..8], rec[576..584]);
        @memcpy(expect.*[i * 512 ..][0..512], &recordRef(&rec));
    }
    return .{ .values = values, .scales = scales };
}

pub fn make(arena: std.mem.Allocator, o: Options) !World {
    const geo = g.release;
    var rng = Rng.init(o.seed);
    var w: World = undefined;
    const sources = try arena.alloc(ex.SourceState, geo.sources.len);
    const carries = try arena.alloc([]const u8, geo.sources.len);
    for (geo.sources, 0..) |s, si| {
        const rows: usize = @intCast(o.upto / s.ratio);
        const comp_rows = try arena.alloc([]const u8, rows);
        const idx_rows = try arena.alloc([]const u8, rows);
        w.truth.comp[si] = try arena.alloc(u16, rows * 512);
        w.truth.idx[si] = try arena.alloc(u16, rows * 128);
        for (0..rows) |r| {
            var row: [512]u16 = undefined;
            genRow(&rng, &row);
            const rec = try arena.create([g.cuda_record_bytes]u8);
            encodeRecord(&row, rec);
            comp_rows[r] = rec;
            @memcpy(w.truth.comp[si][r * 512 ..][0..512], &recordRef(rec));
            var k: [128]u16 = undefined;
            genIndex(&rng, &k);
            if (o.index == .fp8) {
                const ir = try arena.create([g.cuda_index_fp8_bytes]u8);
                @memcpy(w.truth.idx[si][r * 128 ..][0..128], &encodeIndex(&k, ir));
                idx_rows[r] = ir;
            } else {
                const ir = try arena.alloc(u8, 256);
                c.storeRow(128, &k, ir);
                @memcpy(w.truth.idx[si][r * 128 ..][0..128], &k);
                idx_rows[r] = ir;
            }
        }
        const pr = geo.page / s.ratio;
        const ib: usize = if (o.index == .fp8) g.cuda_index_fp8_bytes else g.cuda_index_bf16_bytes;
        sources[si] = .{ .layer = s.layer, .comp = .{ .split = o.split, .ranks = try paged(arena, comp_rows, g.cuda_record_bytes, pr, o.world, o.split) }, .index = .{ .split = o.split, .ranks = try paged(arena, idx_rows, ib, pr, o.world, o.split) } };
        if (s.ratio == 2) {
            const cb = try arena.alloc(u8, 4096);
            for (0..1024) |i| std.mem.writeInt(u32, cb[4 * i ..][0..4], @bitCast(rng.norm()), .little);
            carries[si] = cb;
        } else carries[si] = &.{};
    }
    const wrows: u32 = @intCast(@min(o.upto, geo.window));
    const swa = try arena.alloc(?ex.Ring, geo.layers);
    for (swa, 0..) |*r, l| r.* = try ringOf(arena, &rng, o.upto, wrows, 256, &w.truth.swa[l]);
    const ds = try arena.alloc(ex.Ring, 3);
    for (ds, 0..) |*r, i| r.* = try ringOf(arena, &rng, o.upto, o.ds_rows, 256, &w.truth.ds[i]);
    w.cuda = .{ .index_format = o.index, .sources = sources, .swa = swa, .carries = carries, .drafter = if (o.ds_rows > 0) ds else &.{} };
    return w;
}
