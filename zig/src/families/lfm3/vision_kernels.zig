//! LFM3-VL BF16 image operators with FP32 reductions and no tensor-unit requirement.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const Ref = @import("vision_weights.zig").Ref;

pub const names = [_][]const u8{
    "l3v_dense",
    "l3v_rms",
    "l3v_add",
    "l3v_conv",
    "l3v_qk",
    "l3v_attention",
    "l3v_swiglu",
    "l3v_layer_norm",
    "l3v_project",
    "l3v_gelu",
};

pub const Kernels = struct {
    p: [names.len]mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
        _ = gpa;
        const library = try mtl.Library.fromSource(device, sources.lfm3_vision, mtl.CompileOptions.mlx());
        defer library.deinit();
        var out: Kernels = undefined;
        var loaded: usize = 0;
        errdefer for (out.p[0..loaded]) |p| p.deinit();
        for (names, 0..) |name, i| {
            out.p[i] = try mtl.Pipeline.init(device, library, name, false);
            loaded += 1;
        }
        return out;
    }

    pub fn deinit(self: *Kernels) void {
        for (self.p) |p| p.deinit();
    }

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        inline for (names, 0..) |name, i| if (comptime std.mem.eql(u8, name, key)) return self.p[i];
        @compileError("no LFM3 vision pipeline " ++ key);
    }

    /// Dense BF16 x W^T with FP32 sums; rows and columns reuse operands in blocks of four.
    pub fn linear(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows: usize, n: usize, k: usize) void {
        e.setPipeline(self.get("l3v_dense"));
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setBuffer(y.buf, y.off, 2);
        e.setValue(extern struct { m: u32, n: u32, k: u32 }{ .m = @intCast(rows), .n = @intCast(n), .k = @intCast(k) }, 3);
        e.dispatchGroups(mtl.Size.of((n + 3) / 4, (rows + 3) / 4, 1), mtl.Size.of(32, 1, 1));
    }
};
