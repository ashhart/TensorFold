pub const source = @embedFile("attention_device.metal");
pub const names = [_][]const u8{ "sam_attention_device_clear", "sam_attention_device_domain", "sam_attention_device_center", "sam_attention_device64", "sam_attention_device_fallback", "sam_attention_device_rescale" };
pub const names32 = [_][]const u8{ "sam_attention_device_clear", "sam_attention_device_domain", "sam_attention_device_center", "sam_attention_device32", "sam_attention_device32_fallback", "sam_attention_device32_rescale" };

pub fn encode(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [6]mtl.Pipeline, refs: anytype, p: [25]i32, fallback: bool) void {
    const batch: usize = @intCast(p[0]);
    const heads: usize = @intCast(p[1]);
    const rows: usize = @intCast(p[2]);
    const keys: usize = @intCast(p[3]);
    const nq: usize = @intCast(p[5]);
    const window = p[13] == 64 and heads == 16 and rows == 576 and keys == 576;
    e.setPipeline(pipes[0]);
    e.setBuffer(refs[4].buffer, refs[4].offset, 0);
    e.setValue(@as(u32, @intCast(1 + nq * heads * batch)), 1);
    e.dispatchThreads(mtl.Size.of(1 + nq * heads * batch, 1, 1), mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes[1]);
    for ([_]usize{ 0, 1, 2, 4 }, 0..) |index, slot| e.setBuffer(refs[index].buffer, refs[index].offset, slot);
    e.setValue(p, 4);
    e.dispatchThreads(mtl.Size.of(@min(batch * heads * (rows + 2 * keys) * 64, 65536), 1, 1), mtl.Size.of(128, 1, 1));
    e.barrier();
    if (window) {
        e.setPipeline(pipes[2]);
        for ([_]usize{ 1, 2, 5, 4 }, 0..) |index, slot| e.setBuffer(refs[index].buffer, refs[index].offset, slot);
        e.setValue(p, 4);
        e.dispatchGroups(mtl.Size.of(batch * heads, 1, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
    }
    e.setPipeline(pipes[3]);
    for (0..4) |i| e.setBuffer(refs[i].buffer, refs[i].offset, i);
    e.setValue(p, 4);
    e.setBuffer(refs[4].buffer, refs[4].offset, 5);
    e.setBuffer(refs[5].buffer, refs[5].offset, 6);
    e.dispatchGroups(mtl.Size.of(nq * (if (window) @as(usize, 2) else 1), heads, batch), mtl.Size.of(128, 1, 1));
    e.barrier();
    if (fallback) {
        e.setPipeline(pipes[4]);
        e.dispatchGroups(mtl.Size.of(nq, heads, batch), mtl.Size.of(128, 1, 1));
        e.barrier();
        e.setPipeline(pipes[5]);
        e.dispatchGroups(mtl.Size.of(nq, heads, batch), mtl.Size.of(128, 1, 1));
        e.barrier();
    }
}
