//! E5 pre-registered proposal durations, actual recording CUDA frames and default-off identity.
const std = @import("std");
const q = @import("qwen35");
const lanes = @import("lanes");
const h = q.draft.context_hold;
const frame = @import("qwen35_draft_frame_host.zig");
const t = std.testing;

test "Context hold: requested shallow and deep durations differ" {
    var s: lanes.Stream = undefined;
    s.cached = 4096;
    var item: lanes.drafter.Hold = .{ .stream = &s, .pending = 1, .position = 4097, .depth = 7 };
    try t.expectEqual(@as(usize, 8), h.rows(&.{item}, 16));
    item.depth = 8;
    try t.expectEqual(@as(usize, 16), h.rows(&.{item}, 16));
}

test "Context hold: cached threshold is exact and applies to every shared member" {
    var streams: [8]lanes.Stream = undefined;
    var items: [8]lanes.drafter.Hold = undefined;
    for (&streams, &items) |*s, *item| {
        s.cached = 0;
        item.* = .{ .stream = s, .pending = 1, .position = 16384, .depth = 3 };
    }
    for ([_]usize{ 1, 4, 8 }) |n| {
        try t.expectEqual(@as(usize, 8), h.rows(items[0..n], 16));
        items[n - 1].position = 16385;
        try t.expectEqual(@as(usize, 16), h.rows(items[0..n], 16));
        items[n - 1].position = 16384;
    }
    try t.expectEqual(@as(usize, 16), h.rows(&.{}, 16));
    items[0].position = 0;
    try t.expectEqual(@as(usize, 16), h.rows(items[0..1], 16));
    items[0].position = 16384;
    items[0].lanes = @ptrFromInt(0x1000);
    try t.expectEqual(@as(usize, 16), h.rows(items[0..1], 16));
}

test "Context hold: actual N15 frame below threshold runs eight rows at C1 C4 C8" {
    try frame.contextHoldFrame(16383, 3, 8, true);
}

test "Context hold: actual frame at threshold and deep hold keep sixteen rows" {
    try frame.contextHoldFrame(16384, 3, 16, true);
    try frame.contextHoldFrame(4096, 8, 16, true);
}

test "Context hold: disabled adapter keeps original frame below threshold" {
    try frame.contextHoldFrame(4096, 3, 16, false);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "Context hold: exactly one default-off strict setting and original callbacks" {
    const name = "TF_Q35_DFLASH_CONTEXT_HOLD";
    const e = h.environment;
    try t.expectEqual(@as(usize, 331), e.rules.len);
    try t.expectEqual(@as(?u64, null), try e.parse(name, null));
    try t.expectEqual(@as(?u64, 0), try e.parse(name, "0"));
    try t.expectEqual(@as(?u64, 1), try e.parse(name, "1"));
    try t.expectError(error.InvalidEnvironment, e.parse(name, "true"));
    _ = unsetenv(name);
    defer _ = unsetenv(name);
    var d: q.draft.Draft = undefined;
    d.c = frame.config();
    d.c.block = 16;
    d.tree_nodes = 15;
    try t.expect(!h.enabled());
    const off = d.drafter();
    try t.expectEqual(@as(*anyopaque, &d), off.ptr);
    try t.expect(off.vtable == &q.draft.Draft.drafter_vtable);
    inline for (@typeInfo(lanes.drafter.Drafter.VTable).@"struct".field_names) |field| {
        try t.expectEqual(@field(q.draft.Draft.drafter_vtable, field), @field(off.vtable, field));
    }
    _ = setenv(name, "0", 1);
    try t.expect(d.drafter().vtable == off.vtable);
    _ = setenv(name, "1", 1);
    const on = d.drafter();
    try t.expect(on.vtable.hold != off.vtable.hold);
    inline for (@typeInfo(lanes.drafter.Drafter.VTable).@"struct".field_names) |field| {
        if (comptime !std.mem.eql(u8, field, "hold")) try t.expectEqual(@field(off.vtable, field), @field(on.vtable, field));
    }
    d.tree_nodes = 24;
    try t.expect(d.drafter().vtable == off.vtable);
    d.tree_nodes = 15;
    d.c.kind = .dspine;
    try t.expect(d.drafter().vtable == off.vtable);
    // Disabled adaptation does not inspect a draft, allocator or driver at all.
    const untouched = h.adapt(@ptrFromInt(@alignOf(q.draft.Draft)), off, false);
    try t.expect(untouched.ptr == off.ptr and untouched.vtable == off.vtable);
}

test "Context hold: fresh 129K context is long even with zero restored prompt tokens" {
    var stream: lanes.Stream = undefined;
    stream.cached = 0;
    const item: lanes.drafter.Hold = .{ .stream = &stream, .pending = 1, .position = 132097, .depth = 3 };
    try t.expectEqual(@as(usize, 16), h.rows(&.{item}, 16));
}
