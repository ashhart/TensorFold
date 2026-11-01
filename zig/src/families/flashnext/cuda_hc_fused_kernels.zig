//! Fixed-shape BF16 HC writeback and normalization in the flashnext_hc image.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");

pub const Mode = enum(u32) { sums = 0, moe = 2, slices = 4 };

pub const Writeback = struct {
    h: u64,
    pss: u64,
    branch: u64 = 0,
    inj: u64 = 0,
    y: u64 = 0,
    wts: u64 = 0,
    stride: usize = 0,
    slices: usize = 1,
    mode: Mode,
};

pub const Kernels = struct {
    norm: cuda.Function,

    pub fn resolve(m: cuda.Module) !Kernels {
        return .{ .norm = try m.function("fn_hc_writeback_norm") };
    }

    pub fn writeback(k: Kernels, s: cuda.Stream, in: Writeback, norm: u64, gamma: u64, rows: usize) !void {
        if (rows == 0 or in.slices == 0) return error.RowShape;
        var a: cuda.Args = .{};
        for ([_]u64{ in.h, in.pss, in.branch, in.inj, in.y, in.wts, gamma, norm }) |p| a.add(p);
        a.add(@as(i64, @intCast(in.stride)));
        a.add(@as(c_int, @intCast(in.slices)));
        a.add(@as(c_int, @intCast(@backingInt(in.mode))));
        try cuda.launch.launch(k.norm, .{ .grid = .{ .x = @intCast(rows * wt.streams) }, .block = .{ .x = 256 } }, s, &a);
    }
};
