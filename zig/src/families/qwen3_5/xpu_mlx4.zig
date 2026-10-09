//! Multi-row MLX 4-bit affine (group 64) matvec, m = 1..16 rows; each row equals the decode kernel's bits.
const std = @import("std");
const rt = @import("xpu").rt;

const spv = @import("xpu").kernels.qwen_mlx4;

pub const Family = enum { exact, fast };

/// Rows of weights a sub-group handles in the exact family, by variant R (the width tuned for the big shapes).
fn wFor(r: u32) u32 {
    return switch (r) {
        2 => 4,
        4 => 4,
        8 => 2,
        else => 1,
    };
}

pub const Rows = struct {
    r: *rt.Runtime,
    m: rt.Module,
    k1: rt.Kernel, // qmv4x
    kr: [5]rt.Kernel, // qmv4r_R_W for R = 1, 2, 4, 8, 16
    kgu: [5]rt.Kernel, // qmv4x_gateup, qmv4r_gu_R for R = 2, 4, 8, 16

    pub fn init(r: *rt.Runtime, max_in: u32, max_rows: u32) !Rows {
        _ = .{ max_in, max_rows };
        var m = try r.module(spv);
        var nm: [32]u8 = undefined;
        var kr: [5]rt.Kernel = undefined;
        const rs = [_]u32{ 1, 2, 4, 8, 16 };
        for (rs, 0..) |rr, i| kr[i] = try m.kernel(try std.fmt.bufPrintSentinel(&nm, "qmv4r_{d}_{d}", .{ rr, wFor(rr) }, 0), .{ 16, 1, 1 });
        var kgu: [5]rt.Kernel = undefined;
        kgu[0] = try m.kernel("qmv4x_gateup", .{ 16, 1, 1 });
        for (rs[1..], 1..) |rr, i| kgu[i] = try m.kernel(try std.fmt.bufPrintSentinel(&nm, "qmv4r_gu_{d}", .{rr}, 0), .{ 16, 1, 1 });
        return .{
            .r = r,
            .m = m,
            .kgu = kgu,
            .k1 = try m.kernel("qmv4x", .{ 16, 1, 1 }),
            .kr = kr,
        };
    }

    pub fn deinit(self: *Rows) void {
        _ = self;
    }

    /// y[r * rows + y_off + row] (bf16, or fp32 when f32out) for r < m. `in` a multiple of 64 .
    pub fn matvec(self: *Rows, fam: Family, w: rt.Buffer, s: rt.Buffer, b: rt.Buffer, x: rt.Buffer, y: rt.Buffer, in: u32, rows: u32, m: u32, y_off: u32, f32out: bool) !void {
        if (m == 0 or m > 16) return error.Invalid;
        const fo: u32 = @intFromBool(f32out);
        _ = fam;
        {
            if (m == 1) {
                var k = &self.k1;
                try k.setBuffer(0, w);
                try k.setBuffer(1, s);
                try k.setBuffer(2, b);
                try k.setBuffer(3, x);
                try k.setBuffer(4, y);
                try k.setU32(5, in);
                try k.setU32(6, 0);
                try k.setU32(7, y_off);
                try k.setU32(8, rows);
                try k.setU32(9, fo);
                return k.launch(.{ rows, 1, 1 });
            }
            const idx: usize = if (m <= 2) 1 else if (m <= 4) 2 else if (m <= 8) 3 else 4;
            const rr: u32 = @as(u32, 1) << @intCast(idx);
            var k = &self.kr[idx];
            try k.setBuffer(0, w);
            try k.setBuffer(1, s);
            try k.setBuffer(2, b);
            try k.setBuffer(3, x);
            try k.setBuffer(4, y);
            try k.setU32(5, in);
            try k.setU32(6, y_off);
            try k.setU32(7, rows);
            try k.setU32(8, rows);
            try k.setU32(9, m);
            try k.setU32(10, fo);
            return k.launch(.{ (rows + wFor(rr) - 1) / wFor(rr), 1, 1 });
        }
    }

    /// act [m][rows] = bf16(silu(gate x) * up x): the MLP SwiGLU in one launch, row invariant (as qmv4x_gateup).
    pub fn gateUp(self: *Rows, wg: rt.Buffer, sg: rt.Buffer, bg: rt.Buffer, wu: rt.Buffer, su: rt.Buffer, bu: rt.Buffer, x: rt.Buffer, act: rt.Buffer, in: u32, rows: u32, m: u32) !void {
        if (m == 0 or m > 16) return error.Invalid;
        const idx: usize = if (m == 1) 0 else if (m <= 2) 1 else if (m <= 4) 2 else if (m <= 8) 3 else 4;
        var k = &self.kgu[idx];
        try k.setBuffer(0, wg);
        try k.setBuffer(1, sg);
        try k.setBuffer(2, bg);
        try k.setBuffer(3, wu);
        try k.setBuffer(4, su);
        try k.setBuffer(5, bu);
        try k.setBuffer(6, x);
        try k.setBuffer(7, act);
        try k.setU32(8, in);
        try k.setU32(9, rows);
        if (idx > 0) try k.setU32(10, m);
        try k.launch(.{ rows, 1, 1 });
    }
};
