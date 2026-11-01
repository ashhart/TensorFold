//! Dense prompt projections on the tensor units through Flash Next's width-general tiles (MLX's bf16 dequantization).
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const row = @import("../../core/row_projection.zig");

const Args = extern struct { k: i32, n: i32, m: i32, ld: i32 = 0 };

/// The shared tile kernels for one weight format, compiled as the engine does.
pub fn source(device: mtl.Device, a: std.mem.Allocator, bits: u8, group: u16) ![]u8 {
    const body = try frags.source(device, a, ks.flashnext_qmmw);
    return std.fmt.allocPrint(a, "#define WB {d}\n#define WG {d}\n{s}", .{ bits, group, body });
}

pub const Candidate = struct {
    pipe: mtl.Pipeline,
    bits: u8,
    group: u16,

    pub fn init(device: mtl.Device, bits: u8, group: u16) !?Candidate {
        if (!device.tensorUnits()) return null;
        if (std.mem.indexOfScalar(u8, &.{ 4, 6, 8 }, bits) == null or (group != 32 and group != 64)) return error.UnsupportedLfmNaxFormat;
        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer arena.deinit();
        const text = try source(device, arena.allocator(), bits, group);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "tf_qmmw_t_nax_win", false);
        errdefer pipe.deinit();
        if (pipe.simdWidth() != 32 or pipe.maxThreads() < 128) return error.Reflection;
        return .{ .pipe = pipe, .bits = bits, .group = group };
    }

    pub fn deinit(self: *Candidate) void {
        self.pipe.deinit();
    }

    /// Scales and biases load eight groups a row at once: k spans whole windows, their buffers are 16 byte aligned.
    pub fn supports(self: *const Candidate, w: row.Weights, rows: usize) bool {
        const window = 8 * @as(usize, self.group);
        return rows > 0 and w.bits == self.bits and w.group == self.group and w.n % 64 == 0 and w.k % window == 0 and
            w.s_off % 16 == 0 and w.b_off % 16 == 0;
    }

    pub fn encode(self: *const Candidate, e: mtl.ComputeEncoder, w: row.Weights, x: mtl.Buffer, x_off: usize, y: mtl.Buffer, y_off: usize, rows: usize) void {
        e.setPipeline(self.pipe);
        e.setBuffer(w.w, w.w_off, 0);
        e.setBuffer(w.scales, w.s_off, 1);
        e.setBuffer(w.biases, w.b_off, 2);
        e.setBuffer(x, x_off, 3);
        e.setValue(Args{ .k = @intCast(w.k), .n = @intCast(w.n), .m = @intCast(rows) }, 4);
        e.setBuffer(y, y_off, 5);
        e.dispatchGroups(.of(w.n / 64, (rows + 63) / 64, 1), .of(128, 1, 1));
    }
};
