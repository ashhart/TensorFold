//! Borrowed FP8 n-gram rows (E4M3 codes times one table scale, rounded once to BF16), and a ModelOpt shard's open.
const std = @import("std");
const core = @import("core");
const ple = @import("cuda_ple.zig");
const ple_reader = @import("cuda_ple_reader.zig");
const quant = core.quant;
const Tensor = core.safetensors.Tensor;

pub const Error = error{ BadFp8Table, BadFp8Rows, AliasedFp8Rows };

pub const Table = struct {
    codes: []const u8,
    rows: usize,
    width: usize,
    scale: f32,

    /// Borrow exact row-major E4M3 codes and the table's scale; the caller retains the map.
    pub fn init(weight: Tensor, scale: f32) Error!Table {
        if (weight.rank != 2 or weight.dtype != .f8_e4m3 or weight.dim(0) == 0 or weight.dim(1) == 0)
            return error.BadFp8Table;
        const size = std.math.mul(usize, weight.dim(0), weight.dim(1)) catch return error.BadFp8Table;
        if (weight.bytes.len != size or !std.math.isFinite(scale)) return error.BadFp8Table;
        return .{ .codes = weight.bytes, .rows = weight.dim(0), .width = weight.dim(1), .scale = scale };
    }

    /// Token-major requested rows, with FP32 products rounded once to BF16; malformed requests write nothing.
    pub fn gather(t: Table, ids: []const usize, out: []u16) Error!void {
        const count = std.math.mul(usize, ids.len, t.width) catch return error.BadFp8Rows;
        if (out.len != count) return error.BadFp8Rows;
        for (ids) |id| if (id >= t.rows) return error.BadFp8Rows;
        const bytes = std.mem.sliceAsBytes(out);
        if (overlaps(bytes, t.codes) or overlaps(bytes, std.mem.sliceAsBytes(ids))) return error.AliasedFp8Rows;
        for (ids, 0..) |id, row| {
            for (t.codes[id * t.width ..][0..t.width], out[row * t.width ..][0..t.width]) |code, *o| {
                const v: f32 = @floatCast(quant.fp8.e4m3(code));
                o.* = quant.convert.bf16(v * t.scale);
            }
        }
    }
};

fn overlaps(a: []const u8, b: []const u8) bool {
    if (a.len == 0 or b.len == 0) return false;
    const x = @intFromPtr(a.ptr);
    const y = @intFromPtr(b.ptr);
    return if (x <= y) y - x < a.len else x - y < b.len;
}

/// A ModelOpt table's row format, from shard 0's weight dtype (U8: NVFP4, F8_E4M3: FP8).
pub fn format(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, wm: std.json.ObjectMap, name: []const u8) !ple.Format {
    const file = wm.get(name) orelse return error.MissingTensor;
    const path = try std.fs.path.join(gpa, &.{ dir, file.string });
    defer gpa.free(path);
    var opened = try ple.Open.load(gpa, io, path);
    defer opened.close(gpa);
    const t = opened.file.get(name) orelse return error.MissingTensor;
    return switch (t.dtype) {
        .u8 => .nvfp4,
        .f8_e4m3 => .fp8,
        else => error.UnexpectedTensor,
    };
}

/// One shard's ModelOpt table (NVFP4 with E4M3 scales a 16 values, or FP8 rows) and its positional reader.
pub fn openShard(g: *ple.Ngram, file: *const ple.Open, shard: usize, nested: bool, w: Tensor) !void {
    if (g.format == .fp8) {
        const table = try Table.init(w, g.scale);
        if (table.width != g.hash.dh) return error.UnexpectedTensor;
        g.tables8[shard] = table;
        g.readers[shard] = ple_reader.Reader.initFp8(&file.file, file.io, table);
        return;
    }
    var key: [180]u8 = undefined;
    const sc = try file.tensor(try ple.partNameFor(&key, shard, nested, "weight_scale", true), .f8_e4m3);
    const table = try core.nvfp4_rows.Table.init(w, sc, g.scale);
    if (table.width != g.hash.dh) return error.UnexpectedTensor;
    g.tables[shard] = table;
    g.readers[shard] = ple_reader.Reader.init(&file.file, file.io, table);
}

test {
    _ = @import("cuda_ple_fp8_test.zig");
}
