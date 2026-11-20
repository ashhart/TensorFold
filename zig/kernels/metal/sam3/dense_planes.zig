pub const source = @embedFile("dense_planes.metal");
pub const names = [_][]const u8{ "sam_dense_planes_clear", "sam_dense_planes_pack", "sam_dense_planes_domain", "sam_dense_planes_plan", "sam_dense_planes_restore", "sam_dense_half_fallback", "sam_dense_planes_finish", "sam_dense_planes_gelu" };
pub const tile = [2]u32{ 32, 128 };

// Halves in a planes buffer for n values: three planes, n restore floats and eight flag words.
pub fn planesCount(n: usize) usize {
    return 2 * ((3 * n + 1) / 2) + 2 * n + 16;
}

// Byte offset of the FP32 restore area in a planes buffer for n values.
pub fn restoreOffset(n: usize) usize {
    return 4 * ((3 * n + 1) / 2);
}

pub const Epilogue = enum(u32) { none, gelu_planes, residual, tile_residual };

// pipes: names in order, then the projection; refs: input, weight, bias, output, planes, plan, residual, projected.
// The projection writes refs[7] for gelu_planes and refs[3] otherwise; the fallback always writes refs[7].
// A fallback first rebuilds every FP32 input in the planes buffer's restore area.
pub fn encode(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [9]mtl.Pipeline, refs: anytype, shape: [3]u32, packing: bool, epilogue: Epilogue) void {
    const rows: usize = shape[0];
    const cols: usize = shape[1];
    const depth: usize = shape[2];
    const bind = struct {
        fn buffers(encoder: mtl.ComputeEncoder, tensors: anytype, comptime indices: anytype) void {
            inline for (indices, 0..) |index, slot| encoder.setBuffer(tensors[index].buffer, tensors[index].offset, slot);
        }
    };
    e.setPipeline(pipes[0]);
    bind.buffers(e, refs, .{ 4, 5, 3 });
    e.setValue(shape, 3);
    e.setValue([2]u32{ @intFromBool(packing), @intFromBool(epilogue == .gelu_planes) }, 4);
    e.dispatchThreads(mtl.Size.of(16, 1, 1), mtl.Size.of(16, 1, 1));
    e.barrier();
    if (packing) {
        e.setPipeline(pipes[1]);
        bind.buffers(e, refs, .{ 0, 1, 2, 4, 5 });
        e.setValue(shape, 5);
        e.dispatchThreads(mtl.Size.of(@min(rows * depth, 1 << 20), 1, 1), mtl.Size.of(128, 1, 1));
    } else {
        e.setPipeline(pipes[2]);
        bind.buffers(e, refs, .{ 1, 2, 5 });
        e.setValue(shape, 3);
        e.dispatchThreads(mtl.Size.of(@min(cols * depth + cols, 65536), 1, 1), mtl.Size.of(128, 1, 1));
    }
    e.barrier();
    e.setPipeline(pipes[3]);
    bind.buffers(e, refs, .{ 4, 5 });
    e.setValue(shape, 2);
    e.setValue(tile, 3);
    e.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    e.barrier();
    e.setPipeline(pipes[8]);
    if (epilogue == .gelu_planes) bind.buffers(e, refs, .{ 4, 1, 2, 7 }) else bind.buffers(e, refs, .{ 4, 1, 2, 3 });
    e.setValue(shape, 4);
    for ([_]usize{ 5, 6 }) |index| e.setBuffer(refs[index].buffer, refs[index].offset, index);
    e.dispatchIndirect(refs[5].buffer, refs[5].offset + 4, mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes[4]);
    bind.buffers(e, refs, .{4});
    e.setValue(shape, 1);
    e.dispatchIndirect(refs[5].buffer, refs[5].offset + 28, mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes[5]);
    e.setBuffer(refs[4].buffer, refs[4].offset + restoreOffset(rows * depth), 0);
    for ([_]usize{ 1, 2 }) |index| e.setBuffer(refs[index].buffer, refs[index].offset, index);
    e.setBuffer(refs[7].buffer, refs[7].offset, 3);
    e.setValue(shape, 4);
    e.setBuffer(refs[5].buffer, refs[5].offset, 5);
    e.dispatchIndirect(refs[5].buffer, refs[5].offset + 16, mtl.Size.of(128, 1, 1));
    e.barrier();
    switch (epilogue) {
        .none => {},
        .gelu_planes => {
            e.setPipeline(pipes[7]);
            bind.buffers(e, refs, .{ 7, 3 });
            e.setValue(@as(u32, @intCast(rows * cols)), 2);
            e.dispatchThreads(mtl.Size.of(@min(rows * cols, 1 << 20), 1, 1), mtl.Size.of(128, 1, 1));
            e.barrier();
        },
        .residual, .tile_residual => {
            e.setPipeline(pipes[6]);
            bind.buffers(e, refs, .{ 7, 3, 6 });
            e.setValue(shape, 3);
            e.setValue(@as(u32, @intFromBool(epilogue == .tile_residual)), 4);
            e.dispatchIndirect(refs[5].buffer, refs[5].offset + 40, mtl.Size.of(128, 1, 1));
            e.barrier();
        },
    }
}
