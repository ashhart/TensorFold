const std = @import("std");
const config = @import("loop_config.zig");
const schedule = @import("loop_schedule.zig");
const t = std.testing;

const json =
    \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2]},
    \\ "mhc":{"start":4,"end":28,"streams":2,"iterations":20,"fixed_identity":false,"pre_scale":1.0}}
;

test "loop config and execution schedule" {
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const c = try config.parse(parsed.value.object, 32);
    var linear: [32]bool = @splat(true);
    for (0..32) |i| if (i % 4 == 3) {
        linear[i] = false;
    };
    const s = try schedule.build(&linear, c);
    try t.expectEqual(@as(usize, 56), s.len);
    try t.expectEqual(@as(usize, 42), s.deltas);
    try t.expectEqual(@as(usize, 14), s.attentions);
    var di: usize = 0;
    var ai: usize = 0;
    for (s.slice(), 0..) |step, i| {
        const layer = if (i < 4) i else if (i < 52) 4 + (i - 4) % 24 else i - 24;
        try t.expectEqual(layer, step.layer);
        try t.expectEqual(layer >= 4 and layer < 28, step.inside);
        try t.expectEqual(layer == 4, step.enter);
        try t.expectEqual(layer == 27, step.leave);
        if (linear[layer]) {
            try t.expectEqual(@as(?usize, di), step.delta);
            try t.expectEqual(@as(?usize, null), step.attention);
            di += 1;
        } else {
            try t.expectEqual(@as(?usize, ai), step.attention);
            try t.expectEqual(@as(?usize, null), step.delta);
            ai += 1;
        }
    }
    const identity = try schedule.build(&linear, null);
    try t.expectEqual(@as(usize, 32), identity.len);
    for (identity.slice(), 0..) |step, i| try t.expectEqual(i, step.layer);
    try t.expectEqual(@as(usize, 24), identity.deltas);
    try t.expectEqual(@as(usize, 8), identity.attentions);
    try t.expectError(error.BadLoops, c.withLoops(0));
    try t.expectError(error.BadLoops, c.withLoops(5));
    try t.expectEqual(@as(usize, 4), (try c.withLoops(4)).count);
}

test "loop config rejects malformed values" {
    for ([_][]const u8{
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[1]}}
        ,
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2]},"mhc":{"start":3,"end":5,"streams":2,"iterations":20,"fixed_identity":false}}
        ,
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2]},"mhc":{"start":4,"end":28,"streams":4,"iterations":20,"fixed_identity":false}}
        ,
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2],"unknown":0}}
    }) |input| {
        const doc = try std.json.parseFromSlice(std.json.Value, t.allocator, input, .{});
        defer doc.deinit();
        try t.expectError(error.BadLoopConfig, config.parse(doc.value.object, 32));
    }
}
