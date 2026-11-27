//! Independent plain decode launches: one token row, pair-order IDs, six COUNT=1 experts, RPS4/SG2.
const fixture = @import("nemotron_experts_fixture.zig");
pub fn encode(p: anytype, e: anytype, pipe: anytype, out: anytype) void {
    const f = p.f.?;
    const k = f.shape.k;
    const n = f.shape.n;
    const pairs = fixture.top_k;
    const order = [8]i32{ 0, 1, 2, 3, 4, 5, 0, 0 };
    const ones = [8]i32{ 1, 1, 1, 1, 1, 1, 0, 0 };
    for (0..f.shape.rows) |row| {
        e.setPipeline(pipe);
        e.setBuffer(p.bufs[0].buf, p.bufs[0].off + f.xRow(row * pairs) * k * 2, 0);
        e.setBuffer(p.plain_ids.buf, p.plain_ids.off + row * pairs * 4, 1);
        e.setValue(order, 2);
        e.setValue(ones, 3);
        e.setValue(order, 4);
        e.setValue(@as(i32, pairs), 5);
        for (p.bufs[6..9], 6..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setBuffer(out.buf, out.off + row * pairs * n * 2, 9);
        e.dispatchThreads(.{ .width = 64, .height = n / 8, .depth = pairs }, .{ .width = 64, .height = 1, .depth = 1 });
    }
}
