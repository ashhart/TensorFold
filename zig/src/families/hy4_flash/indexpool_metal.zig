//! Metal launch for Hy4-flash's FP32 sparse-index key pooling.
const mtl = @import("metal");
const groups = @import("index_groups.zig");
const sources = @import("kernel_sources");

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };
pub const Bindings = struct { values: Ref, gates: Ref, position_bias: Ref, pooled: Ref };

pub const Pool = struct {
    pipeline: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pool {
        const library = try mtl.Library.fromSource(device, sources.hy4_flash_indexpool, mtl.CompileOptions.mlx());
        defer library.deinit();
        return fromLibrary(device, library);
    }

    pub fn fromLibrary(device: mtl.Device, library: mtl.Library) !Pool {
        return .{ .pipeline = try mtl.Pipeline.init(device, library, "tf_hy4_pool_groups", false) };
    }

    pub fn deinit(self: *Pool) void {
        self.pipeline.deinit();
    }

    pub fn run(self: Pool, encoder: mtl.ComputeEncoder, refs: Bindings, shape: groups.PoolShape) void {
        if (shape.groups == 0 or shape.width == 0) return;
        encoder.setPipeline(self.pipeline);
        for ([_]Ref{ refs.values, refs.gates, refs.position_bias, refs.pooled }, 0..) |ref, i| {
            encoder.setBuffer(ref.buffer, ref.offset, i);
        }
        encoder.setValue(shape, 4);
        encoder.dispatchThreads(mtl.Size.of(shape.width, shape.groups, 1), mtl.Size.of(32, 1, 1));
    }
};
