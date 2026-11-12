//! TF_QWEN27_METAL_KV_PAGES's saved, restored and moved slots against main's: the same digests, KV rows included.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const graph = @import("synthetic_graph.zig");

const Result = struct { resumed: [32]u8, moved: [32]u8, fresh: [32]u8 };

fn run(a: std.mem.Allocator, model: *q.model.Model, paged: bool, first: []const u32, other: []const u32) !Result {
    var r = try if (paged) q.decode_round.Runner.initPaged(a, model, 2, 1024) else q.decode_round.Runner.init(a, model, 2, 1024);
    defer r.deinit();
    if (r.caches.len == 0) return error.NoAttentionLayer;
    const s: q.session.Session = .{ .runner = &r };
    try r.reset(0);
    try s.prefill(first, 128);
    const st = try q.snapshot.save(a, &r, null, @intCast(first.len));
    defer q.snapshot.drop(a, st);
    try st.landed();
    var out: Result = undefined;
    // the slot is overwritten, then the kept state comes back and decodes on
    try r.reset(0);
    try s.prefill(other, 128);
    try q.snapshot.restore(a, &r, null, st);
    for ([_]u32{ 7, 8, 9 }) |t| try s.step(t);
    out.resumed = try s.fingerprint();
    // it moves to slot 1 and back past another prompt in slot 0
    try q.snapshot.moveSlot(a, &r, 0, null, 1, null);
    try r.reset(0);
    try s.prefill(other, 128);
    try q.snapshot.moveSlot(a, &r, 1, null, 0, null);
    out.moved = try s.fingerprint();
    // the same prompt and steps without a kept state
    try r.reset(0);
    try s.prefill(first, 128);
    for ([_]u32{ 7, 8, 9 }) |t| try s.step(t);
    out.fresh = try s.fingerprint();
    return out;
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const autorelease = mtl.objc.Pool.push();
    defer autorelease.pop();
    const model = try graph.createLayers(a, 5);
    defer model.deinit();
    var first: [600]u32 = undefined; // past two pages of 256 rows
    var other: [300]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 37 + 11) % 96);
    for (&other, 0..) |*t, i| t.* = @intCast((i * 53 + 5) % 96);
    const main_ = try run(a, model, false, &first, &other);
    const pages = try run(a, model, true, &first, &other);
    for ([_]struct { what: []const u8, x: [32]u8, y: [32]u8 }{
        .{ .what = "restored digest, main and pages", .x = main_.resumed, .y = pages.resumed },
        .{ .what = "moved digest, main and pages", .x = main_.moved, .y = pages.moved },
        .{ .what = "fresh digest, main and pages", .x = main_.fresh, .y = pages.fresh },
        .{ .what = "pages: restored and fresh", .x = pages.resumed, .y = pages.fresh },
    }) |c| if (!std.mem.eql(u8, &c.x, &c.y)) {
        std.debug.print("FAIL {s} differ\n", .{c.what});
        return error.Differs;
    };
    std.debug.print("PASS kept states through pages: restored equal to fresh, and restored, moved and fresh slots the same digests as main's\n", .{});
}
