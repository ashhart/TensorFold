//! kvt_plan's fixtures on the host: the served calls reach each branch of the kernel, and every model is whole.
const std = @import("std");
const core = @import("core");
const plan = @import("contract_flashnext_exl3_kvt_plan.zig");

const g = core.contract.gates;

test "kvt plan contract: the served calls reach each branch of the kernel, and every model is whole" {
    const a = std.testing.allocator;
    var assigned = false;
    var overflow = false;
    var all_hit = false;
    var ties = false;
    var wrapped = false;
    var idle = false;
    var sticky = false;
    for (plan.served) |shape| {
        var pinned = false; // some call needs the last block before the window, so the control's early window shows
        for (0..14) |seed| {
            var cs: plan.Case = .{ .a = a, .rig = undefined, .f = undefined };
            defer cs.free();
            for ([_]g.Fill{ .random, .nonfinite }) |fill| {
                try std.testing.expect(try cs.build(shape, fill, seed));
                const mo = &cs.model;
                const cap = @max(shape.n, 2048);
                try std.testing.expect(mo.need.len <= cap and mo.miss.len <= cap);
                var used: usize = 0;
                for (mo.stamp) |t| used += @intFromBool(t == mo.epoch and mo.epoch != 0);
                var hits: usize = 0;
                for (mo.hit) |h| hits += @intFromBool(h);
                if (mo.assigned > 0) {
                    assigned = true;
                    try std.testing.expectEqual(hits + mo.assigned, used);
                    try std.testing.expectEqual(mo.assigned, mo.miss.len);
                    try std.testing.expectEqual(@as(u32, @intCast(mo.assigned)), mo.ctl[1]);
                    for (mo.victims[1..], mo.victims[0 .. mo.victims.len - 1]) |hi, lo| ties = ties or hi == lo;
                }
                overflow = overflow or (mo.ctl[2] == 1 and mo.epoch != 0 and mo.miss.len > 0 and mo.assigned == 0 and cs.ctl[2] == 0);
                all_hit = all_hit or (mo.need.len > 0 and mo.miss.len == 0);
                wrapped = wrapped or mo.epoch == 0;
                sticky = sticky or (cs.ctl[2] == 1 and mo.epoch != 0 and mo.need.len == 0);
                idle = idle or (mo.epoch != 0 and cs.ctl[2] == 0 and mo.need.len == 0);
                pinned = pinned or std.mem.indexOfScalar(u32, mo.need, plan.before - 1) != null;
                try std.testing.expect(mo.ctl[0] == mo.epoch or mo.epoch == 0);
            }
        }
        try std.testing.expect(pinned);
    }
    try std.testing.expect(assigned and overflow and all_hit and ties and wrapped and idle and sticky);
}
