pub const source = @embedFile("dense_pair.metal");

pub fn pairCount(n: usize) usize {
    return 4 * n + 16;
}

pub fn weightPairCount(nw: usize) usize {
    return 2 * nw + 16;
}

pub fn restoreOffset(n: usize) usize {
    return 4 * n;
}

pub const Epilogue = enum(u32) { none, gelu_pair, residual, tile_residual };

pub fn mainName(epilogue: Epilogue) []const u8 {
    return switch (epilogue) {
        .none => "sam_dense_pair_plain",
        .gelu_pair => "sam_dense_pair_gelu_pair",
        .residual => "sam_dense_pair_residual",
        .tile_residual => "sam_dense_pair_tiled",
    };
}

pub fn mainTile(epilogue: Epilogue) [2]u32 {
    return if (epilogue == .none) .{ 64, 128 } else .{ 32, 128 };
}

pub fn mainThreads(epilogue: Epilogue) usize {
    return if (epilogue == .none) 256 else 128;
}

pub const helper_names = [_][]const u8{
    "sam_dense_pair_clear",
    "sam_dense_pair_pack",
    "sam_dense_pair_plan",
    "sam_dense_pair_restore",
    "sam_dense_device_fallback",
    "sam_dense_planes_finish",
    "sam_dense_pair_gelu_fallback",
};

// refs: 0=x(f32 or pair), 1=w_pair, 2=w_f32, 3=bias, 4=out, 5=x_pair, 6=plan, 7=residual, 8=scratch.
pub fn encode(comptime mtl: type, e: mtl.ComputeEncoder, helpers: [7]mtl.Pipeline, main: mtl.Pipeline, refs: anytype, shape: [3]u32, packing: bool, epilogue: Epilogue) void {
    const rows: usize = shape[0];
    const cols: usize = shape[1];
    const depth: usize = shape[2];
    const nx: u32 = @intCast(rows * depth);
    const outputs: u32 = @intCast(rows * cols);
    if (packing) {
        e.setPipeline(helpers[0]);
        e.setBuffer(refs[5].buffer, refs[5].offset, 0);
        e.setValue(nx, 1);
        e.dispatchThreads(mtl.Size.of(8, 1, 1), mtl.Size.of(8, 1, 1));
        e.barrier();
        e.setPipeline(helpers[1]);
        e.setBuffer(refs[0].buffer, refs[0].offset, 0);
        e.setBuffer(refs[5].buffer, refs[5].offset, 1);
        e.setValue(nx, 2);
        e.dispatchThreads(mtl.Size.of(@min((rows * depth + 3) / 4, 1 << 18), 1, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
    }
    e.setPipeline(helpers[2]);
    e.setBuffer(refs[5].buffer, refs[5].offset, 0);
    e.setBuffer(refs[1].buffer, refs[1].offset, 1);
    e.setBuffer(refs[6].buffer, refs[6].offset, 2);
    e.setBuffer(refs[4].buffer, refs[4].offset, 3);
    e.setValue(shape, 4);
    e.setValue(mainTile(epilogue), 5);
    e.setValue(@as(u32, @intFromBool(epilogue == .gelu_pair)), 6);
    e.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    e.barrier();
    e.setPipeline(main);
    e.setBuffer(refs[5].buffer, refs[5].offset, 0);
    e.setBuffer(refs[1].buffer, refs[1].offset, 1);
    e.setBuffer(refs[3].buffer, refs[3].offset, 2);
    e.setBuffer(refs[4].buffer, refs[4].offset, 3);
    e.setValue(shape, 4);
    e.setBuffer(refs[6].buffer, refs[6].offset, 5);
    e.setBuffer(refs[7].buffer, refs[7].offset, 6);
    e.dispatchIndirect(refs[6].buffer, refs[6].offset + 4, mtl.Size.of(mainThreads(epilogue), 1, 1));
    e.barrier();
    e.setPipeline(helpers[3]);
    e.setBuffer(refs[5].buffer, refs[5].offset, 0);
    e.setValue(nx, 1);
    e.dispatchIndirect(refs[6].buffer, refs[6].offset + 28, mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(helpers[4]);
    e.setBuffer(refs[5].buffer, refs[5].offset + restoreOffset(rows * depth), 0);
    e.setBuffer(refs[2].buffer, refs[2].offset, 1);
    e.setBuffer(refs[3].buffer, refs[3].offset, 2);
    const fb_dst = if (epilogue == .none or epilogue == .gelu_pair) refs[4] else refs[8];
    const fb_off = if (epilogue == .gelu_pair) fb_dst.offset + restoreOffset(rows * cols) else fb_dst.offset;
    e.setBuffer(fb_dst.buffer, fb_off, 3);
    e.setValue(shape, 4);
    e.setBuffer(refs[6].buffer, refs[6].offset, 5);
    e.dispatchIndirect(refs[6].buffer, refs[6].offset + 16, mtl.Size.of(128, 1, 1));
    e.barrier();
    switch (epilogue) {
        .none => {},
        .gelu_pair => {
            e.setPipeline(helpers[6]);
            e.setBuffer(refs[4].buffer, refs[4].offset + restoreOffset(rows * cols), 0);
            e.setBuffer(refs[4].buffer, refs[4].offset, 1);
            e.setValue(outputs, 2);
            e.dispatchIndirect(refs[6].buffer, refs[6].offset + 40, mtl.Size.of(128, 1, 1));
            e.barrier();
        },
        .residual, .tile_residual => {
            e.setPipeline(helpers[5]);
            e.setBuffer(refs[8].buffer, refs[8].offset, 0);
            e.setBuffer(refs[4].buffer, refs[4].offset, 1);
            e.setBuffer(refs[7].buffer, refs[7].offset, 2);
            e.setValue(shape, 3);
            e.setValue(@as(u32, @intFromBool(epilogue == .tile_residual)), 4);
            e.dispatchIndirect(refs[6].buffer, refs[6].offset + 40, mtl.Size.of(128, 1, 1));
            e.barrier();
        },
    }
}
