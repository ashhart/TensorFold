//! MLX 4-bit prefill GEMM (qwen_mlx4_pf.cl), R >= 16 rows: decode W to fp16 DPAS fragments, then GEMM; chunk invariant.
const std = @import("std");
const rt = @import("xpu").rt;

const spv = @import("xpu").kernels.qwen_mlx4_pf;
const spv_gemm = @import("xpu").kernels.ggml_pfgemm; // the 2D-block-load GEMM shared with the ggml path

pub const Pf = struct {
    dec_row: rt.Kernel,
    dec_blk: rt.Kernel,
    prep: rt.Kernel,
    gemm: rt.Kernel,
    gemm_module: rt.Module,
    wv: ?rt.Buffer = null,
    xt: ?rt.Buffer = null,

    pub fn init(r: *rt.Runtime) !Pf {
        var m = try r.module(spv);
        var mg = try r.moduleWith(spv_gemm, "-cl-intel-256-GRF-per-thread"); // 128 accumulator registers a lane
        return .{
            .gemm_module = mg,
            .dec_row = try m.kernel("pfdec_mlx_row", .{ 16, 8, 1 }),
            .dec_blk = try m.kernel("pfdec_mlx_blk", .{ 16, 8, 1 }),
            .prep = try m.kernel("pf_prep", .{ 64, 1, 1 }),
            .gemm = try mg.kernel("pfgemm", .{ 16, 1, 1 }),
        };
    }

    fn grow(r: *rt.Runtime, slot: *?rt.Buffer, bytes: usize) !rt.Buffer {
        if (slot.*) |b| {
            if (b.len >= bytes) return b;
            var old = b;
            old.free();
        }
        slot.* = try r.alloc(bytes);
        return slot.*.?;
    }

    /// Needs rows % 16 == 0 and in % 512 == 0. block: weights in the MLX4_BLOCK layout.
    pub fn run(self: *Pf, block: bool, w: rt.Buffer, s: rt.Buffer, b: rt.Buffer, x: rt.Buffer, R: u32, y: rt.Buffer, in: u32, y_off: u32, rows: u32) !void {
        if (rows % 16 != 0 or in % 512 != 0) return error.Invalid;
        const r = self.gemm.rt;
        const rp = (R + 63) / 64 * 64; // tokens padded to whole 64-token blocks (zeros)
        const wv = try grow(r, &self.wv, @as(usize, in) * rows * 2);
        const xt = try grow(r, &self.xt, @as(usize, rp) * in * 2);
        var p = &self.prep;
        try p.setBuffer(0, x);
        try p.setBuffer(1, xt);
        try p.setU32(2, R);
        try p.setU32(3, in);
        try p.setU32(4, rp);
        try p.launch(.{ rp / 64, in / 8, 1 });
        var d = if (block) &self.dec_blk else &self.dec_row;
        try d.setBuffer(0, w);
        try d.setBuffer(1, s);
        try d.setBuffer(2, b);
        try d.setBuffer(3, wv);
        try d.setU32(4, in);
        try d.setU32(5, rows);
        try d.launch(.{ rows / 16, in / 512, 1 }); // work-groups of 8 sub-groups along the input groups
        var g = &self.gemm;
        try g.setBuffer(0, wv);
        try g.setBuffer(1, xt);
        try g.setBuffer(2, y);
        try g.setU32(3, R);
        try g.setU32(4, rp);
        try g.setU32(5, in);
        try g.setU32(6, rows);
        try g.setU32(7, y_off);
        try g.launch(.{ rp / 64, (rows + 31) / 32, 1 });
    }
};
