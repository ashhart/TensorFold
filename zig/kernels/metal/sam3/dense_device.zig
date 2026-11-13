pub const source = @embedFile("dense_device.metal");
pub const names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device", "sam_dense_device_fallback" };
pub const gelu_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_gelu", "sam_dense_device_gelu_fallback" };
pub const gelu_linear_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_gelu_linear", "sam_dense_device_fallback" };
pub const compensated_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_compensated", "sam_dense_device_fallback" };
pub const nn_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_nn", "sam_dense_device_nn_fallback" };
pub const nn_compensated_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_nn_compensated", "sam_dense_device_nn_fallback" };

pub fn encode(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [5]mtl.Pipeline, refs: anytype, shape: [3]u32, fallback: bool) void {
    const rows: usize = shape[0];
    const cols: usize = shape[1];
    const depth: usize = shape[2];
    e.setPipeline(pipes[0]);
    e.setBuffer(refs[4].buffer, refs[4].offset, 0);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes[1]);
    for ([_]usize{ 0, 1, 2, 4 }, 0..) |index, slot| e.setBuffer(refs[index].buffer, refs[index].offset, slot);
    e.setValue(shape, 4);
    e.dispatchThreads(mtl.Size.of(@min((rows + cols) * depth + cols, 65536), 1, 1), mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes[2]);
    e.setBuffer(refs[4].buffer, refs[4].offset, 0);
    e.setValue(shape, 1);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes[3]);
    for (0..4) |i| e.setBuffer(refs[i].buffer, refs[i].offset, i);
    e.setValue(shape, 4);
    e.setBuffer(refs[4].buffer, refs[4].offset, 5);
    e.dispatchIndirect(refs[4].buffer, refs[4].offset + 4, mtl.Size.of(128, 1, 1));
    e.barrier();
    if (fallback) {
        e.setPipeline(pipes[4]);
        e.dispatchIndirect(refs[4].buffer, refs[4].offset + 16, mtl.Size.of(128, 1, 1));
        e.barrier();
    }
}
