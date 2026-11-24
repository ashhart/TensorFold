pub const source = @embedFile("attention_device.metal");
pub const names = [_][]const u8{ "sam_attention_device_clear", "sam_attention_device_domain", "sam_attention_device64", "sam_attention_device_fallback", "sam_attention_device_rescale" };
pub const names32 = [_][]const u8{ "sam_attention_device_clear", "sam_attention_device_domain", "sam_attention_device32", "sam_attention_device32_fallback", "sam_attention_device32_rescale" };

pub fn encode(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [5]mtl.Pipeline, refs: anytype, p: [25]i32, fallback: bool) void {
    encodeWith(mtl, e, pipes, refs, p, fallback, false);
}

pub fn encodePrechecked(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [5]mtl.Pipeline, refs: anytype, p: [25]i32, fallback: bool) void {
    encodeWith(mtl, e, pipes, refs, p, fallback, true);
}

fn encodeWith(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [5]mtl.Pipeline, refs: anytype, p: [25]i32, fallback: bool, prechecked: bool) void {
    const batch: usize = @intCast(p[0]);
    const heads: usize = @intCast(p[1]);
    const rows: usize = @intCast(p[2]);
    const keys: usize = @intCast(p[3]);
    const nq: usize = @intCast(p[5]);
    const dim: usize = @intCast(p[13]);
    if (!prechecked) {
        e.setPipeline(pipes[0]);
        e.setBuffer(refs[4].buffer, refs[4].offset, 0);
        e.setValue(@as(u32, @intCast(1 + nq * heads * batch)), 1);
        e.dispatchThreads(mtl.Size.of(1 + nq * heads * batch, 1, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
        e.setPipeline(pipes[1]);
        for ([_]usize{ 0, 1, 2, 4 }, 0..) |index, slot| e.setBuffer(refs[index].buffer, refs[index].offset, slot);
        e.setValue(p, 4);
        const quads = batch * heads * (rows + 2 * keys) * (dim / 4);
        e.dispatchThreads(mtl.Size.of(@min(quads, 1 << 18), 1, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
    }
    e.setPipeline(pipes[2]);
    for (0..4) |i| e.setBuffer(refs[i].buffer, refs[i].offset, i);
    e.setValue(p, 4);
    e.setBuffer(refs[4].buffer, refs[4].offset, 5);
    e.dispatchGroups(mtl.Size.of(nq, heads, batch), mtl.Size.of(128, 1, 1));
    e.barrier();
    if (fallback) {
        e.setPipeline(pipes[3]);
        e.dispatchGroups(mtl.Size.of(nq, heads, batch), mtl.Size.of(128, 1, 1));
        e.barrier();
        e.setPipeline(pipes[4]);
        e.dispatchGroups(mtl.Size.of(nq, heads, batch), mtl.Size.of(128, 1, 1));
        e.barrier();
    }
}
