//! Engram table rows from packed per-rank shards on local NVMe, through io_uring, AIO or reader threads, LRU-cached.

const std = @import("std");

pub const magic: u64 = 0x31344E4531565344; // "DSV41EN1"; file `<dir>/engram-l<layer>-r<rank>of<world>.bin`
pub const header_bytes: u64 = 4096; // 6 little-endian u64: magic, layer, lo, hi, total rows, row bytes
pub const sector: u64 = 512;
pub const row_bytes: u64 = 264; // 256 e4m3 values and 8 UE8M0 scales (one a 32)
pub const values = 256;
pub const merge_gap: u64 = 4096; // holes up to this merged into one extent
pub const page = 4096;

pub const Header = struct {
    layer: u64,
    lo: u64,
    hi: u64,
    total: u64,
    row_bytes: u64,

    /// The header's six words, checked against the file's size.
    pub fn parse(b: *const [48]u8, size: u64) !Header {
        var w: [6]u64 = undefined;
        for (&w, 0..) |*x, i| x.* = std.mem.readInt(u64, b[8 * i ..][0..8], .little);
        if (w[0] != magic) return error.NotEngramShard;
        const h: Header = .{ .layer = w[1], .lo = w[2], .hi = w[3], .total = w[4], .row_bytes = w[5] };
        if (h.row_bytes != row_bytes or h.hi < h.lo or h.hi > h.total) return error.BadEngramShard;
        const payload = std.math.mul(u64, h.hi - h.lo, h.row_bytes) catch return error.BadEngramShard;
        const expected = std.math.add(u64, header_bytes, payload) catch return error.BadEngramShard;
        if (size != expected) return error.BadEngramShard;
        return h;
    }
};

/// torch's float8_e4m3fn -> float32 (c10 fp8e4m3fn_to_fp32_value, bit for bit).
pub fn e4m3(b: u8) f32 {
    const w: u32 = @as(u32, b) << 24;
    const sign = w & 0x8000_0000;
    const nonsign = w & 0x7FFF_FFFF;
    var shift: u32 = @clz(nonsign);
    shift = if (shift > 4) shift - 4 else 0;
    const inf_nan: u32 = @bitCast((@as(i32, @bitCast(nonsign +% 0x0100_0000)) >> 8) & 0x7F80_0000);
    const zero: u32 = @bitCast(@as(i32, @bitCast(nonsign -% 1)) >> 31);
    const r = sign | ((((nonsign << @intCast(shift)) >> 4) +% ((0x78 -% shift) << 23) | inf_nan) & ~zero);
    return @bitCast(r);
}

/// float32 -> bfloat16 as torch's CPU vector path: round to nearest even, NaN -> 0xFFFF (no packed table holds a NaN).
pub fn bf16(x: f32) u16 {
    if (std.math.isNan(x)) return 0xFFFF;
    const u: u32 = @bitCast(x);
    return @intCast((u +% (0x7FFF + ((u >> 16) & 1))) >> 16);
}

/// The table [scale << 8 | byte] -> bf16 bits of e4m3(byte) x 2^(scale - 127) (scale 0: 2^-127, 255: NaN).
pub fn fillTable(t: *[65536]u16) void {
    for (0..256) |s| {
        const mul = scale(@intCast(s));
        for (0..256) |b| t[s << 8 | b] = bf16(e4m3(@intCast(b)) * mul);
    }
}

pub const Rows = if (@import("builtin").os.tag == .linux)
    @import("engram_rows_linux.zig").Rows
else
    @import("engram_rows_portable.zig").Rows;

/// UE8M0, including its minimum exponent and reserved NaN, as the #826 reference equations.
pub fn scale(code: u8) f32 {
    if (code == 255) return std.math.nan(f32);
    if (code == 0) return 0x1p-127;
    return @bitCast(@as(u32, code) << 23);
}
/// One backend-neutral separate table view. Resident Metal and packed CUDA shards share this geometry.
pub const Table = struct {
    weights: []const u8,
    scales: []const u8,
    rows: usize,
    weight_stride: usize = 256,
    scale_stride: usize = 8,
    scale_block: usize = 32,
    pub fn validate(t: Table) !void {
        if (t.rows == 0 or t.weight_stride < 256 or t.scale_stride < 8 or t.scale_block != 32)
            return error.InvalidEngramTable;
        const w = std.math.mul(usize, t.rows - 1, t.weight_stride) catch return error.InvalidEngramTable;
        const s = std.math.mul(usize, t.rows - 1, t.scale_stride) catch return error.InvalidEngramTable;
        if (w > t.weights.len or 256 > t.weights.len - w or s > t.scales.len or 8 > t.scales.len - s)
            return error.InvalidEngramTable;
    }
    pub fn row(t: Table, id: usize, out: *[256]u16) !void {
        try t.validate();
        if (id >= t.rows) return error.EngramRowOutsideShard;
        for (out, 0..) |*v, j| v.* = bf16(e4m3(t.weights[id * t.weight_stride + j]) * scale(t.scales[id * t.scale_stride + j / 32]));
    }
};
// Linux's two allocations are tested independently of Linux syscalls on every host.
pub fn storage(comptime T: type, a: std.mem.Allocator, count: usize) !struct { tables: []T, lut: *[65536]u16 } {
    const tables = try a.alloc(T, count);
    errdefer a.free(tables);
    const lut = try a.create([65536]u16);
    return .{ .tables = tables, .lut = lut };
}
test "Linux row storage releases tables if LUT allocation fails" {
    const check = struct {
        fn run(a: std.mem.Allocator) !void {
            const owned = try storage(u64, a, 2);
            defer a.free(owned.tables);
            defer a.destroy(owned.lut);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, check.run, .{});
}
