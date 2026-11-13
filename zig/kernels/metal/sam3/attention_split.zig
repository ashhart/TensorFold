pub const value_parts = 4;
pub const source = @import("std").fmt.comptimePrint("#define SAM_SPLIT_VALUE_PARTS {d}\n", .{value_parts}) ++ @embedFile("attention_split.metal");
pub const chunk_blocks = 32;

pub fn encode(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [3]mtl.Pipeline, refs: anytype, p: [27]i32) void {
    const batch: usize = @intCast(p[0]);
    const heads: usize = @intCast(p[1]);
    const rows: usize = @intCast(p[2]);
    const nq: usize = @intCast(p[5]);
    const splits: usize = @intCast(p[24]);
    e.setPipeline(pipes[0]);
    for ([_]usize{ 0, 1, 3, 4 }, 0..) |at, index| e.setBuffer(refs[at].buffer, refs[at].offset, index);
    e.setValue(p, 4);
    e.dispatchThreads(mtl.Size.of(nq * 32, heads * 4, batch * splits), mtl.Size.of(32, 4, 1));
    e.barrier();
    e.setPipeline(pipes[1]);
    for ([_]usize{ 2, 3, 4, 5 }, 0..) |at, index| e.setBuffer(refs[at].buffer, refs[at].offset, index);
    e.setValue(p, 4);
    e.dispatchThreads(mtl.Size.of(nq * 32, heads * 4, batch * splits * value_parts), mtl.Size.of(32, 4, 1));
    e.barrier();
    e.setPipeline(pipes[2]);
    for ([_]usize{ 5, 4, 6 }, 0..) |at, index| e.setBuffer(refs[at].buffer, refs[at].offset, index);
    e.setValue(p, 3);
    e.dispatchThreads(mtl.Size.of(rows * 32, heads, batch), mtl.Size.of(128, 1, 1));
    e.barrier();
}
