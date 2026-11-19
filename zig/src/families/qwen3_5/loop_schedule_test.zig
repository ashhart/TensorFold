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
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2]},"mhc":{"start":4,"end":28,"streams":3,"iterations":20,"fixed_identity":false}}
        ,
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2],"use_gate":true}}
        ,
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2],"use_gate":null}}
        ,
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2]},"mhc":{"start":4,"end":28,"streams":2,"iterations":20,"fixed_identity":false,"unknown":0}}
    }) |input| {
        const doc = try std.json.parseFromSlice(std.json.Value, t.allocator, input, .{});
        defer doc.deinit();
        try t.expectError(error.BadLoopConfig, config.parse(doc.value.object, 32));
    }
}

test "loop config accepts use_gate false, ignores other loop keys and allows 4 streams" {
    for ([_][]const u8{
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2],"use_gate":false}}
        ,
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2],"unknown":0,"note":{"a":1}}}
        ,
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2]},"mhc":{"start":4,"end":28,"streams":4,"iterations":20,"fixed_identity":false}}
    }) |input| {
        const doc = try std.json.parseFromSlice(std.json.Value, t.allocator, input, .{});
        defer doc.deinit();
        _ = try config.parse(doc.value.object, 32);
    }
}

test "loop config keeps any iteration count and pre_scale" {
    const doc = try std.json.parseFromSlice(std.json.Value, t.allocator,
        \\{"loop_settings":{"start":4,"end":28,"eval_count":2,"counts":[2]},"mhc":{"start":4,"end":28,"streams":2,"iterations":7,"fixed_identity":false,"pre_scale":1.75}}
    , .{});
    defer doc.deinit();
    const m = (try config.parse(doc.value.object, 32)).mhc.?;
    try t.expectEqual(@as(usize, 7), m.iterations);
    try t.expectEqual(@as(f64, 1.75), m.pre_scale);
}

test "mHC decisions follow every checkpoint step and plain identity" {
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const c = try config.parse(parsed.value.object, 32);
    var linear: [32]bool = @splat(true);
    for (0..32) |i| if (i % 4 == 3) {
        linear[i] = false;
    };
    const s = try schedule.build(&linear, c);
    var pending = false;
    for (s.slice()) |st| {
        const d = schedule.decide(st, pending);
        try t.expectEqual(st.enter, d.enter);
        try t.expectEqual(st.inside, d.route);
        try t.expectEqual(if (st.inside) false else pending, d.first_add);
        try t.expectEqual(!st.inside, d.second_add);
        try t.expectEqual(st.inside, d.merge);
        try t.expectEqual(st.leave, d.leave);
        const got = d.actions();
        var expected: [10]schedule.Action = undefined;
        var n: usize = 0;
        if (st.enter) {
            expected[n] = .enter;
            n += 1;
        }
        if (st.inside) {
            for ([_]schedule.Action{ .route, .norm_plain, .mixer, .merge, .route, .norm_plain, .mlp, .merge }) |item| {
                expected[n] = item;
                n += 1;
            }
        } else {
            for ([_]schedule.Action{ if (pending) .norm_add else .norm_plain, .mixer, .norm_add, .mlp }) |item| {
                expected[n] = item;
                n += 1;
            }
        }
        if (st.leave) {
            expected[n] = .leave;
            n += 1;
        }
        try t.expectEqualSlices(schedule.Action, expected[0..n], got.slice());
        pending = d.pending;
        try t.expectEqual(!st.inside, pending);
    }
    const plain = try schedule.build(&linear, null);
    pending = false;
    for (plain.slice(), 0..) |st, i| {
        const d = schedule.decide(st, pending);
        try t.expectEqual(i != 0, d.first_add);
        try t.expect(d.second_add and !d.route and !d.enter and !d.leave and !d.merge);
        const got = d.actions();
        const expected = [_]schedule.Action{ if (i == 0) .norm_plain else .norm_add, .mixer, .norm_add, .mlp };
        try t.expectEqualSlices(schedule.Action, &expected, got.slice());
        pending = d.pending;
    }
}
