//! Host tests of the EXL3 graph key: its state-binding identity and the rendezvous signature.

const std = @import("std");
const state_binding = @import("../kv/exl3_state_binding.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const Key = gkey.Key;

test "exl3: state binding normalized identity preserves all nonstate key guards" {
    const original: Key = .{ .kind = 2, .rows = 2, .scratch = 11, .scratch_cap = 16, .scores = 12, .output = 13, .keep = 14, .keep_nonstate = 15, .streams = 16, .plan = 17 };
    var flipped = original;
    flipped.seq[0][4] = 100;
    flipped.keep = 101;
    try std.testing.expect(!std.meta.eql(original, flipped));
    try std.testing.expectEqualDeep(state_binding.normalized(original), state_binding.normalized(flipped));
    inline for (@typeInfo(Key).@"struct".field_names) |field| {
        if (comptime !std.mem.eql(u8, field, "seq") and !std.mem.eql(u8, field, "paging") and
            !std.mem.eql(u8, field, "paging_logical") and
            !std.mem.eql(u8, field, "chunks") and !std.mem.eql(u8, field, "score_tiles") and !std.mem.eql(u8, field, "keep"))
        {
            var changed = original;
            @field(changed, field) += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        }
    }
    for (0..gkey.max_lanes) |lane| {
        for (0..5) |component| {
            if (lane == 0 and component == 4) continue;
            var changed = original;
            changed.seq[lane][component] += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        }
        for (0..6) |component| {
            var changed = original;
            changed.paging[lane][component] += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        }
        var changed = original;
        changed.chunks[lane] += 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        changed = original;
        changed.score_tiles[lane] += 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        changed = original;
        changed.paging_logical = @splat(@splat(0));
        changed.paging_logical.?[lane][0] = 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        changed.paging_logical.?[lane][1] = 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
    }
}

test "exl3: rendezvous signatures compare topology and paging origins without device addresses" {
    var key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    key.seq[0] = .{ 1, 2, 100, 4096, 3 };
    key.paging[0] = .{ 10, 20, 30, 40, 50, 60 };
    const signature = gkey.graphSignature(key, 4096);
    key.seq[0][0] = 1000;
    key.seq[0][2] = 1001;
    key.paging[0][0] = 1002;
    key.output = 1003;
    try std.testing.expectEqual(signature, gkey.graphSignature(key, 4096));
    try std.testing.expect(signature != gkey.graphSignature(key, 4097));
    key.paging[0][5] += 1;
    try std.testing.expect(signature != gkey.graphSignature(key, 4096));
}

test "exl3: shared paging rendezvous ignores physical ownership but local graph keys retain it" {
    var first: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    first.seq[0] = .{ 1, 2, 3, 4096, 4 };
    first.paging[0] = .{ 10, 20, 30, 40, 50, 60 };
    first.paging_logical = @splat(@splat(0));
    first.paging_logical.?[0] = .{ 100, 200 };
    var second = first;
    second.paging[0] = .{ 1000, 2000, 3000, 4000, 5000, 60 };
    try std.testing.expect(!std.meta.eql(first, second));
    try std.testing.expect(!std.meta.eql(state_binding.normalized(first), state_binding.normalized(second)));
    try std.testing.expectEqual(gkey.graphSignature(first, 4096), gkey.graphSignature(second, 4096));
    second.paging_logical.?[0][0] += 1;
    try std.testing.expect(gkey.graphSignature(first, 4096) != gkey.graphSignature(second, 4096));
    second = first;
    second.paging_logical.?[0][1] += 1;
    try std.testing.expect(gkey.graphSignature(first, 4096) != gkey.graphSignature(second, 4096));
    second = first;
    second.seq[0][3] += 1;
    try std.testing.expect(gkey.graphSignature(first, 4096) != gkey.graphSignature(second, 4096));
}
