//! The MTP head drafting a tree of lanes: the root step, then one step a depth over every lane with children.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const layers = @import("layers.zig");
const tree = @import("tree.zig");
const mtp = @import("mtp.zig");

const Buffer = mtl.Buffer;
const Shape = lanes.shape.Shape;

/// Each lane's place: the head row and top-k slot of its step (lanes with children, by depth), else none.
const Plan = struct {
    n: usize,
    stepped: usize = 0,
    step: [st.max_rows]i32 = @splat(-1), // stepped lanes' order j: head row M0 + 1 + j, top-k slot 1 + j
    first: [lanes.shape.max_depth + 1]usize = @splat(0), // the first j at each depth
    count: [lanes.shape.max_depth + 1]usize = @splat(0),
    deepest: usize = 0,

    fn of(sh: Shape) Plan {
        var p = Plan{ .n = sh.parents.len };
        var parent_of_any: [st.max_rows]bool = @splat(false);
        for (sh.parents, sh.depths) |q, d| {
            if (q >= 0) parent_of_any[@intCast(q)] = true;
            p.deepest = @max(p.deepest, d);
        }
        for (0..p.deepest + 1) |d| {
            p.first[d] = p.stepped;
            for (sh.depths, 0..) |ld, i| if (ld == d and parent_of_any[i]) {
                p.step[i] = @intCast(p.stepped);
                p.stepped += 1;
            };
            p.count[d] = p.stepped - p.first[d];
        }
        return p;
    }

    /// The top-k slot whose ranks a lane's token comes from: its parent's step (the root's is slot 0).
    fn from(p: Plan, sh: Shape, i: usize) i32 {
        const q = sh.parents[i];
        return if (q < 0) 0 else 1 + p.step[@intCast(q)];
    }
};

/// Draft `sh` into `out` (lane i at out_off + 4 i) from hidden row `h` and token `ids`; the cache keeps the root's row.
pub fn draft(head: *mtp.Head, e: *fwd.Enc, cache: *st.Cache, sh: Shape, h: Buffer, h_off: usize, ids: Buffer, ids_off: usize, out: Buffer, out_off: usize) void {
    const f = head.forward();
    const s = &head.scratch;
    const c = head.c;
    const d_bytes = c.hidden * 2;
    const p = Plan.of(sh);
    std.debug.assert(p.n >= 1 and p.n < st.max_rows and 1 + p.stepped < st.max_levels);
    const m0 = cache.mtp_len;

    // the root: the chain's first step, its output row the depth-0 lanes' hidden input
    head.stepAt(e, cache, h, h_off, ids, ids_off, .greedy, s.ids, 0, 0);
    copyRows(head, e, head.hid, 0, head.hids, 0, 1);

    for (0..p.deepest + 1) |d| {
        // this depth's lanes take their parent's ranked tokens (into the window, and the stepped ones' inputs)
        var entries: [st.max_rows][4]i32 = undefined;
        var n: usize = 0;
        for (sh.depths, sh.ranks, 0..) |ld, r, i| if (ld == d) {
            entries[n] = .{ p.from(sh, i), r, @intCast(out_off / 4 + i), p.step[i] };
            n += 1;
        };
        e.pipe(head.k.get("tf_lane_tokens"));
        e.buf(cache.topk, 0, 0);
        e.e.setBytes(std.mem.sliceAsBytes(entries[0..@max(n, 2)]), 1);
        e.buf(out, 0, 2);
        e.buf(head.pass_ids, 0, 3);
        e.run(.{ n, 1, 1 }, .{ 64, 1, 1 });

        const rows = p.count[d];
        if (rows == 0) continue;
        const j0 = p.first[d];
        // each stepped lane's hidden input: its parent step's output row
        var from_rows: [st.max_rows]u32 = undefined;
        var at: usize = 0;
        for (sh.depths, 0..) |ld, i| if (ld == d and p.step[i] >= 0) {
            from_rows[at] = @intCast(p.from(sh, i));
            at += 1;
        };
        tree.gather(f, e, head.hids, 0, from_rows[0..rows], head.pass_h);
        // ancestor tables by logical position after the root's row (offset 0): the root, the lane's ancestors, itself
        const paths = s.tpaths.slice(i32, st.max_rows * tree.max_depth);
        const depths = s.tdepths.slice(i32, st.max_rows);
        for (sh.depths, 0..) |ld, i| if (ld == d and p.step[i] >= 0) {
            const j: usize = @intCast(p.step[i]);
            depths[j] = @intCast(d + 1);
            var lane: i32 = @intCast(i);
            var k = d + 1;
            while (lane >= 0) : (lane = sh.parents[@intCast(lane)]) {
                paths[j * tree.max_depth + k] = 1 + p.step[@intCast(lane)];
                k -= 1;
            }
            paths[j * tree.max_depth] = 0;
        };
        head.prep(e, rows, head.pass_h, 0, head.pass_ids, j0 * 4);
        const kvs = [_]layers.Kvs{.{ .kv = cache.mtp.?, .len = m0, .rows = rows, .tree = true, .deepest = d + 1, .write = m0 + 1 + j0, .tables = j0 }};
        layers.attention(f, e, head.w.attention, &kvs, rows);
        f.norm(e, s.delta, head.w.norm2, rows, c.eps);
        layers.moe(f, e, head.w.moe, head.w.final, rows, c.eps);
        copyRows(head, e, s.x, 0, head.hids, (1 + j0) * d_bytes, rows);
        f.coop(e, "draft", head.w.draft, s.x, 0, s.xs, s.logits, rows);
        head.topRows(e, cache, rows, 1 + j0);
    }
    cache.mtp_len = m0 + 1;

    // the trunk for grafts: the rank-0 path's steps (the root, then each chain lane with a chain child)
    var chain: [lanes.shape.max_depth]usize = undefined;
    const levels = lanes.shape.chainLanes(sh, &chain);
    cache.trunk[0] = 0;
    for (chain[0..levels -| 1], 1..) |lane, k| cache.trunk[k] = @intCast(1 + p.step[lane]);
    cache.levels = levels;
}

fn copyRows(head: *mtp.Head, e: *fwd.Enc, src: Buffer, src_off: usize, dst: Buffer, dst_off: usize, rows: usize) void {
    e.pipe(head.k.get("tf_copy_rows"));
    e.buf(src, src_off, 0);
    e.buf(dst, dst_off, 1);
    e.run(.{ head.c.hidden, rows, 1 }, .{ 256, 1, 1 });
}

test "a tree's stepped lanes by depth, and each lane's ranks from its parent's step" {
    var parents = [_]i32{ -1, 0, -1, 1, 0 };
    var ranks = [_]u8{ 0, 0, 1, 0, 1 };
    var depths = [_]u8{ 0, 1, 0, 2, 1 };
    const sh = Shape{ .parents = &parents, .ranks = &ranks, .depths = &depths, .expected = 0 };
    const p = Plan.of(sh);
    try std.testing.expectEqual(@as(usize, 2), p.stepped);
    try std.testing.expectEqual(@as(i32, 0), p.step[0]);
    try std.testing.expectEqual(@as(i32, 1), p.step[1]);
    try std.testing.expectEqual(@as(i32, -1), p.step[2]);
    try std.testing.expectEqual(@as(i32, 0), p.from(sh, 2));
    try std.testing.expectEqual(@as(i32, 1), p.from(sh, 4));
    try std.testing.expectEqual(@as(i32, 2), p.from(sh, 3));
}
