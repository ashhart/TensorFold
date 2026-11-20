pub const source = @embedFile("attention_split32.metal");
pub const chunk_blocks = 16;
pub const names = [_][]const u8{ "sam_attention_device_clear", "sam_attention_device_domain", "sam_attention_split32", "sam_attention_split32_merge", "sam_attention_device32_fallback", "sam_attention_device32_rescale" };
pub const names64 = [_][]const u8{ "sam_attention_device_clear", "sam_attention_device_domain", "sam_attention_split64", "sam_attention_split32_merge", "sam_attention_device_fallback", "sam_attention_device_rescale" };

pub fn encode(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [6]mtl.Pipeline, refs: anytype, p: [27]i32) void {
    inline for (0..6) |phase| encodePhase(mtl, e, pipes, refs, p, phase);
}

pub fn encodePhase(comptime mtl: type, e: mtl.ComputeEncoder, pipes: [6]mtl.Pipeline, refs: anytype, p: [27]i32, phase: usize) void {
    const batch: usize = @intCast(p[0]);
    const heads: usize = @intCast(p[1]);
    const rows: usize = @intCast(p[2]);
    const keys: usize = @intCast(p[3]);
    const nq: usize = @intCast(p[5]);
    const splits: usize = @intCast(p[24]);
    const dim: usize = @intCast(p[13]);
    var whole: [25]i32 = undefined;
    @memcpy(whole[0..24], p[0..24]);
    whole[24] = p[26];
    e.setPipeline(pipes[phase]);
    switch (phase) {
        0 => {
            e.setBuffer(refs[6].buffer, refs[6].offset, 0);
            e.setValue(@as(u32, @intCast(1 + nq * heads * batch)), 1);
            e.dispatchThreads(mtl.Size.of(1 + nq * heads * batch, 1, 1), mtl.Size.of(128, 1, 1));
        },
        1 => {
            for ([_]usize{ 0, 1, 2, 6 }, 0..) |at, i| e.setBuffer(refs[at].buffer, refs[at].offset, i);
            e.setValue(whole, 4);
            e.dispatchThreads(mtl.Size.of(@min(batch * heads * (rows + 2 * keys) * dim, 65536), 1, 1), mtl.Size.of(128, 1, 1));
        },
        2 => {
            for ([_]usize{ 0, 1, 2, 3, 4, 6 }, 0..) |at, i| e.setBuffer(refs[at].buffer, refs[at].offset, i);
            e.setValue(p, 6);
            e.dispatchGroups(mtl.Size.of(nq, heads, batch * splits), mtl.Size.of(128, 1, 1));
        },
        3 => {
            for ([_]usize{ 3, 4, 5, 6 }, 0..) |at, i| e.setBuffer(refs[at].buffer, refs[at].offset, i);
            e.setValue(p, 4);
            e.dispatchThreads(mtl.Size.of(rows * dim, heads, batch), mtl.Size.of(128, 1, 1));
        },
        4, 5 => {
            for ([_]usize{ 0, 1, 2, 5 }, 0..) |at, i| e.setBuffer(refs[at].buffer, refs[at].offset, i);
            e.setValue(whole, 4);
            e.setBuffer(refs[6].buffer, refs[6].offset, 5);
            e.dispatchGroups(mtl.Size.of(nq, heads, batch), mtl.Size.of(128, 1, 1));
        },
        else => unreachable,
    }
    e.barrier();
}
