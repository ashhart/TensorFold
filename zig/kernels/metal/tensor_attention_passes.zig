pub const source = @embedFile("tensor_attention_passes.metal");

pub fn encode(comptime mtl: type, e: mtl.ComputeEncoder, scores: mtl.Pipeline, values: mtl.Pipeline, refs: anytype, p: [24]i32, scale: f32) void {
    const batch: usize = @intCast(p[0]);
    const heads: usize = @intCast(p[1]);
    const nq: usize = @intCast(p[5]);
    e.setPipeline(scores);
    for ([_]usize{ 0, 1, 3, 4 }, 0..) |at, i| e.setBuffer(refs[at].buffer, refs[at].offset, i);
    e.setValue(p, 4);
    e.setValue([4]f32{ scale, 0, 0, 0 }, 5);
    e.dispatchThreads(mtl.Size.of(nq * 32, heads * 4, batch), mtl.Size.of(32, 4, 1));
    e.barrier();
    e.setPipeline(values);
    for ([_]usize{ 2, 3, 4, 5 }, 0..) |at, i| e.setBuffer(refs[at].buffer, refs[at].offset, i);
    e.setValue(p, 4);
    e.dispatchThreads(mtl.Size.of(nq * 32, heads * 4, batch * 2), mtl.Size.of(32, 4, 1));
    e.barrier();
}
