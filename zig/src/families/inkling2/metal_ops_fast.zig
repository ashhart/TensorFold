//! Fast-path Metal kernels for Inkling 2: fused multi-row GEMV and NAX/decode attention dispatch.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("../../core/exl3_projection.zig");
const frags = @import("../../core/frags.zig");

pub const View = exl3.View;

/// One existing weight and its existing output. The fused launch writes `y` and allocates nothing.
pub const Fused = struct { w: View, y: View, n: u32 };

const FuseArgs = extern struct {
    k: u32,
    rows: u32,
    count: u32,
    pad: u32 = 0,
    n0: u32,
    n1: u32,
    n2: u32,
    n3: u32,
    n4: u32,
    n5: u32,
    n6: u32,
    n7: u32,
};

/// Multi-row GEMVs (1..8 rows/TG) sharing `x` and reusing weight loads across rows; bit-exact across `rows`.
pub fn linearFusedRows(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, x: View, rows: u32, k: u32, ins: []const Fused) !void {
    if (ins.len == 0 or ins.len > 8 or rows == 0 or k == 0) return error.InvalidInkling2Op;
    var n = [8]u32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var total: u32 = 0;
    for (ins, 0..) |in, i| {
        if (in.n == 0) return error.InvalidInkling2Op;
        n[i] = in.n;
        total += in.n;
    }
    e.setPipeline(pipe);
    e.setBuffer(x.buffer, x.offset, 0);
    for (0..8) |i| {
        const src = if (i < ins.len) ins[i].w else x;
        const dst = if (i < ins.len) ins[i].y else x;
        e.setBuffer(src.buffer, src.offset, @intCast(1 + i));
        e.setBuffer(dst.buffer, dst.offset, @intCast(9 + i));
    }
    e.setValue(FuseArgs{
        .k = k,
        .rows = rows,
        .count = @intCast(ins.len),
        .n0 = n[0],
        .n1 = n[1],
        .n2 = n[2],
        .n3 = n[3],
        .n4 = n[4],
        .n5 = n[5],
        .n6 = n[6],
        .n7 = n[7],
    }, 17);
    e.dispatchGroups(mtl.Size.of(total, (rows + 7) / 8, 1), mtl.Size.of(32, 1, 1));
}

/// Fast attention pipelines compiled with NAX fragment support (`inkling2_attn_nax` and `inkling2_attn_decode_fast`).
pub const FastAttn = struct {
    nax: mtl.Pipeline,
    decode: mtl.Pipeline,

    pub fn load(device: mtl.Device, text: []const u8) !FastAttn {
        const alloc = std.heap.page_allocator;
        const raw = try std.mem.concat(alloc, u8, &.{ "#define TF_INKLING2_ATTN_NAX 1\n", text });
        defer alloc.free(raw);
        const src = try frags.source(device, alloc, raw);
        defer alloc.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const nax = try mtl.Pipeline.init(device, lib, "inkling2_attn_nax", false);
        errdefer nax.deinit();
        const decode = try mtl.Pipeline.init(device, lib, "inkling2_attn_decode_fast", false);
        return .{ .nax = nax, .decode = decode };
    }

    pub fn deinit(self: *FastAttn) void {
        self.nax.deinit();
        self.decode.deinit();
    }

    pub fn encode(self: *const FastAttn, e: mtl.ComputeEncoder, a: anytype, args: anytype) bool {
        const grid = if (a.rows >= 64) blk: {
            e.setPipeline(self.nax);
            break :blk mtl.Size.of((a.rows + 63) / 64, a.heads, 1);
        } else if (a.heads >= 24) blk: {
            e.setPipeline(self.decode);
            break :blk mtl.Size.of(a.heads, a.rows, 1);
        } else return false;
        const views = [_]View{ a.q, a.rel, a.sink, a.gate, a.kcache, a.vcache, a.proj, a.out };
        for (views, 0..) |v, i| e.setBuffer(v.buffer, v.offset, i);
        e.setValue(args, 8);
        e.dispatchGroups(grid, mtl.Size.of(128, 1, 1));
        return true;
    }
};
