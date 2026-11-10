//! Extensible proposal pools use the shared scoring callbacks and preserve the existing best-first order.
const std = @import("std");

pub fn Api(comptime Candidates: type, comptime lp_top: usize, comptime branch: usize, comptime chance: anytype, comptime words: anytype, comptime draw: anytype, comptime temperature: f64) type {
    return struct {
        pub const chancesAfter = chance;
        pub const ids = words;
        pub const drawn = draw;
        pub const scale = temperature;

        pub const Choices = struct { tokens: [lp_top + 4]u32 = undefined, logp: [lp_top + 4]f64 = undefined, len: usize = 0 };

        pub fn baseChoices(c: Candidates, row: usize, pred: u32) Choices {
            var out: Choices = .{ .len = c.topk };
            var lc: [lp_top]f64 = undefined;
            chancesAfter(c, row, pred, &lc);
            @memcpy(out.tokens[0..c.topk], ids(c, row)[c.layout.ids..][0..c.topk]);
            @memcpy(out.logp[0..c.topk], lc[0..c.topk]);
            return out;
        }

        pub fn growWith(comptime most_window: usize, provider: anytype, pending: u32, rows: usize, most: usize, tokens: []u32, parents: []i32, own: []f64) usize {
            const Open = struct { cost: f64, parent: i32, token: u32, row: usize, own: f64 };
            var open: [branch * (most_window + 1)]Open = undefined;
            var n_open: usize = 0;
            var count: usize = 0;
            var pred = pending;
            var parent: i32 = -1;
            var row: usize = 0;
            var path: f64 = 0;
            while (true) {
                // the node just taken (or the pending token) opens its best children on the next row
                if (row < rows) {
                    const next = provider.choices(row, pred);
                    var order: [lp_top + 4]usize = undefined;
                    for (order[0..next.len], 0..) |*o, i| o.* = i;
                    std.mem.sort(usize, order[0..next.len], &next, struct {
                        fn more(v: *const Choices, a: usize, b: usize) bool {
                            return v.logp[a] > v.logp[b] or (v.logp[a] == v.logp[b] and a < b);
                        }
                    }.more);
                    for (order[0..@min(branch, next.len)]) |j| {
                        if (!std.math.isFinite(next.logp[j]) or n_open == open.len) break;
                        open[n_open] = .{ .cost = path - next.logp[j], .parent = parent, .token = next.tokens[j], .row = row, .own = next.logp[j] };
                        n_open += 1;
                    }
                }
                if (count == @min(most, tokens.len) or n_open == 0) break;
                var best: usize = 0;
                for (open[1..n_open], 1..) |o, i| {
                    const b = open[best];
                    if (o.cost < b.cost or (o.cost == b.cost and (o.parent < b.parent or (o.parent == b.parent and (o.token < b.token or (o.token == b.token and o.row < b.row)))))) best = i;
                }
                const node = open[best];
                open[best] = open[n_open - 1];
                n_open -= 1;
                tokens[count] = node.token;
                parents[count] = node.parent;
                own[count] = node.own;
                pred = node.token;
                parent = @intCast(count);
                row = node.row + 1;
                path = node.cost;
                count += 1;
            }
            return count;
        }
    };
}
