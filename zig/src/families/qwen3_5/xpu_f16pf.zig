//! Prefill GEMM for the small fp16 projections (in_proj_a / in_proj_b) on ggml_pfgemm.cl; chunk invariant per token.

const std = @import("std");
const rt = @import("xpu").rt;

const spv_prep = @import("xpu").kernels.qwen_f16pf;
const spv_gemm = @import("xpu").kernels.ggml_pfgemm;

pub const Pf = struct {
    prep: rt.Kernel,
    gemm: rt.Kernel,
    xt: ?rt.Buffer = null,

    pub fn init(r: *rt.Runtime) !Pf {
        var mp = try r.module(spv_prep);
        var mg = try r.moduleWith(spv_gemm, "-cl-intel-256-GRF-per-thread");
        return .{ .prep = try mp.kernel("pf_prep", .{ 64, 1, 1 }), .gemm = try mg.kernel("pfgemm", .{ 16, 1, 1 }) };
    }

    pub fn ok(rows: u32, in: u32) bool {
        return rows % 16 == 0 and in % 32 == 0;
    }

    pub fn run(self: *Pf, r: *rt.Runtime, w: rt.Buffer, x: rt.Buffer, R: u32, y: rt.Buffer, in: u32, y_off: u32, rows: u32) !void {
        if (!ok(rows, in)) return error.Invalid;
        const rp = (R + 63) / 64 * 64;
        const need = @as(usize, rp) * in * 2;
        if (self.xt == null or self.xt.?.len < need) {
            if (self.xt) |*old| {
                try r.sync();
                old.free();
            }
            self.xt = try r.alloc(need);
        }
        const xt = self.xt.?;
        var p = &self.prep;
        try p.setBuffer(0, x);
        try p.setBuffer(1, xt);
        try p.setU32(2, R);
        try p.setU32(3, in);
        try p.setU32(4, rp);
        try p.launch(.{ rp / 64, in / 8, 1 });
        var g = &self.gemm;
        try g.setBuffer(0, w);
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
