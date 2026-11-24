pub const source = @embedFile("dense_device.metal") ++ @embedFile("dense_wide.metal");
pub const names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device", "sam_dense_device_fallback" };
pub const half_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_half_domain", "sam_dense_device_plan", "sam_dense_half_weight", "sam_dense_half_fallback" };
pub const gelu_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_gelu", "sam_dense_device_gelu_fallback" };
pub const gelu_linear_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_gelu_linear", "sam_dense_device_fallback" };
pub const compensated_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_compensated", "sam_dense_device_fallback" };
pub const nn_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_nn", "sam_dense_device_nn_fallback" };
pub const nn_compensated_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_device_domain", "sam_dense_device_plan", "sam_dense_device_nn_compensated", "sam_dense_device_nn_fallback" };
pub const wide_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_wide_domain", "sam_dense_device_plan", "sam_dense_wide_plain", "sam_dense_device_fallback" };
pub const wide_compensated_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_wide_domain", "sam_dense_device_plan", "sam_dense_wide_compensated", "sam_dense_device_fallback" };
pub const wide_gelu_names = [_][]const u8{ "sam_dense_device_clear", "sam_dense_wide_domain", "sam_dense_device_plan", "sam_dense_wide_gelu", "sam_dense_device_gelu_fallback" };

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

pub const operands_name = "sam_dense_wide_operands";
pub const cached_names = [_][]const u8{ "sam_dense_wide_seed", "sam_dense_wide_rows" };
pub const wide_plan_name = "sam_dense_wide_plan";
pub const wide_gelu_flag_names = [_][]const u8{ "sam_dense_wide_plan", "sam_dense_wide_gelu_flag", "sam_dense_device_gelu_fallback" };
pub const wide_plain_residual_names = [_][]const u8{ "sam_dense_wide_plan", "sam_dense_wide_plain_residual", "sam_dense_wide_residual_fallback" };
pub const wide_plain_tiled_names = [_][]const u8{ "sam_dense_wide_plan", "sam_dense_wide_plain_tiled", "sam_dense_wide_tiled_fallback" };
pub const wide_comp_residual_names = [_][]const u8{ "sam_dense_wide_plan", "sam_dense_wide_compensated_residual", "sam_dense_wide_residual_fallback" };

/// Wide layout with cached weight flags: pipes seed, rows, plan, product, fallback; refs x, w, b, out, domain, flags.
pub fn encodeCached(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [5]mtl.Pipeline, refs: anytype, shape: [3]u32) void {
    encodeCachedExtra(mtl, e, pipes, refs, shape, null);
}

/// Cached wide layout with optional buffer(6) (residual or out_flag) bound on product and fallback.
pub fn encodeCachedExtra(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [5]mtl.Pipeline, refs: anytype, shape: [3]u32, extra: anytype) void {
    const count: u32 = shape[0] * shape[2];
    e.setPipeline(pipes[0]);
    e.setBuffer(refs[4].buffer, refs[4].offset, 0);
    e.setBuffer(refs[5].buffer, refs[5].offset, 1);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
    e.barrier();
    e.setPipeline(pipes[1]);
    e.setBuffer(refs[0].buffer, refs[0].offset, 0);
    e.setBuffer(refs[4].buffer, refs[4].offset, 1);
    e.setValue(count, 2);
    e.dispatchThreads(mtl.Size.of(@min(@max(count / 4, 4), 1 << 18), 1, 1), mtl.Size.of(256, 1, 1));
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
    if (@TypeOf(extra) != @TypeOf(null)) {
        if (extra) |ref| e.setBuffer(ref.buffer, ref.offset, 6);
    }
    e.dispatchIndirect(refs[4].buffer, refs[4].offset + 4, mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes[4]);
    e.dispatchIndirect(refs[4].buffer, refs[4].offset + 16, mtl.Size.of(128, 1, 1));
    e.barrier();
}

/// Precomputed weight+input flags: pipes plan, product, fallback; refs x, w, b, out, domain, wflag, iflag, extra.
pub fn encodeReady(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [3]mtl.Pipeline, refs: anytype, shape: [3]u32, emit_flag: bool, bind_extra: bool) void {
    e.setPipeline(pipes[0]);
    e.setBuffer(refs[4].buffer, refs[4].offset, 0);
    e.setBuffer(refs[5].buffer, refs[5].offset, 1);
    e.setBuffer(refs[6].buffer, refs[6].offset, 2);
    e.setBuffer(refs[7].buffer, refs[7].offset, 3);
    e.setValue(shape, 4);
    e.setValue(@as(u32, @intFromBool(emit_flag)), 5);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes[1]);
    for (0..4) |i| e.setBuffer(refs[i].buffer, refs[i].offset, i);
    e.setValue(shape, 4);
    e.setBuffer(refs[4].buffer, refs[4].offset, 5);
    if (bind_extra) e.setBuffer(refs[7].buffer, refs[7].offset, 6);
    e.dispatchIndirect(refs[4].buffer, refs[4].offset + 4, mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes[2]);
    e.dispatchIndirect(refs[4].buffer, refs[4].offset + 16, mtl.Size.of(128, 1, 1));
    e.barrier();
}
