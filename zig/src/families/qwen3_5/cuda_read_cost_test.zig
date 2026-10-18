//! Recording controls for the output work a hidden-row learner does not consume.
const std = @import("std");
const more = @import("cuda_train_more.zig");
const engine = @import("cuda_engine.zig");
const t = std.testing;
test "dense read cost: keys omit logits but target and loss scoring still need them" {
    const keys = more.keepFor(.keys);
    try t.expect(!std.meta.eql(keys, @as(engine.Keep, .logits)));
    {
        try t.expectEqual(engine.Keep.logits, more.keepFor(.target));
        try t.expectEqual(engine.Keep.logits, more.keepFor(.loss));
    }
}

const Recorder = struct {
    g: struct { hidden: usize = 64 } = .{},
    learning: ?*const @import("cuda_learn_hooks.zig").Keep = null,
    h: u64 = 100,
    x: u64 = 200,
    logits: u64 = 300,
    final_norm: u64 = 400,
    head: u64 = 500,
    stream: @import("cuda").Stream = undefined,
    norms: usize = 0,
    heads: usize = 0,
    src: u64 = 0,
    pub fn norm(r: *@This(), residual: bool, weight: u64, rows: usize) !void {
        try t.expect(residual and weight == r.final_norm and rows == 3);
        r.norms += 1;
    }
    pub fn proj(r: *@This(), weight: u64, src: u64, dst: u64, rows: usize, prompt: bool) !void {
        try t.expect(weight == r.head and dst == r.logits and rows == 1 and !prompt);
        r.heads += 1;
        r.src = src;
    }
};
test "dense read cost: hidden taps skip only the output tail and a scoring pass retains it" {
    var r: Recorder = .{};
    try @import("cuda_forward.zig").finish(&r, 3, 1, true);
    try t.expectEqual(@as(usize, 0), r.norms);
    try t.expectEqual(@as(usize, 0), r.heads);
    try @import("cuda_forward.zig").finish(&r, 3, 1, false);
    try t.expectEqual(@as(usize, 1), r.norms);
    try t.expectEqual(@as(usize, 1), r.heads);
    try t.expectEqual(@as(u64, 456), r.src);
}
