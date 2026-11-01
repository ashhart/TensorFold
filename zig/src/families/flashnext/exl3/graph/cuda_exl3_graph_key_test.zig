//! Host tests of the EXL3 graph key: its state-binding identity and the rendezvous signature.

const std = @import("std");
const state_binding = @import("../kv/exl3_state_binding.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const Key = gkey.Key;

test "exl3: paged graph identities bind tables while rank agreement compares page layout" {
    const Pg = @import("../../cuda_glue.zig").Pg;
    var first: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    first.pages[0] = Key.pageOf(Pg{ .pt = 100, .ps = 4096, .ptok = 256, .ips = 1024, .ipt = 200 });
    var other = first;
    other.pages[0][0] += 1000;
    other.pages[0][4] += 1000;
    try std.testing.expect(!std.meta.eql(first, other));
    try std.testing.expect(gkey.sameShape(first, other));
    try std.testing.expectEqual(gkey.graphSignature(first, 0), gkey.graphSignature(other, 0));
    for ([_]usize{ 1, 2, 3 }) |field| {
        other = first;
        other.pages[0][field] += 1;
        try std.testing.expect(!gkey.sameShape(first, other));
        if (field == 2) {
            try std.testing.expect(gkey.graphSignature(first, 0) != gkey.graphSignature(other, 0));
        } else {
            try std.testing.expectEqual(gkey.graphSignature(first, 0), gkey.graphSignature(other, 0));
        }
    }
    other = first;
    other.pages[0] = @splat(0);
    try std.testing.expect(!gkey.sameShape(first, other));
    try std.testing.expect(gkey.graphSignature(first, 0) != gkey.graphSignature(other, 0));
}

test "exl3: state binding normalized identity preserves all nonstate key guards" {
    const original: Key = .{ .kind = 2, .rows = 2, .scratch = 11, .scratch_cap = 16, .scores = 12, .output = 13, .keep = 14, .keep_nonstate = 15, .streams = 16, .plan = 17 };
    var flipped = original;
    flipped.seq[0][4] = 100;
    flipped.keep = 101;
    try std.testing.expect(!std.meta.eql(original, flipped));
    try std.testing.expectEqualDeep(state_binding.normalized(original), state_binding.normalized(flipped));
    inline for (@typeInfo(Key).@"struct".field_names) |field| {
        if (comptime !std.mem.eql(u8, field, "seq") and
            !std.mem.eql(u8, field, "pages") and !std.mem.eql(u8, field, "tiers") and !std.mem.eql(u8, field, "chunks") and !std.mem.eql(u8, field, "score_tiles") and !std.mem.eql(u8, field, "keep"))
        {
            var changed = original;
            @field(changed, field) += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        }
    }
    for (0..gkey.max_lanes) |lane| {
        for (0..5) |component| {
            var changed = original;
            changed.pages[lane][component] += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
            changed = original;
            changed.tiers[lane][component] += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        }
        for (0..5) |component| {
            if (lane == 0 and component == 4) continue;
            var changed = original;
            changed.seq[lane][component] += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        }
        var changed = original;
        changed.chunks[lane] += 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        changed = original;
        changed.score_tiles[lane] += 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
    }
}

test "exl3: rendezvous signatures compare topology and capacity without device addresses" {
    var key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    key.seq[0] = .{ 1, 2, 100, 4096, 3 };
    const signature = gkey.graphSignature(key, 4096);
    key.seq[0][0] = 1000;
    key.seq[0][2] = 1001;
    key.output = 1003;
    try std.testing.expectEqual(signature, gkey.graphSignature(key, 4096));
    try std.testing.expect(signature != gkey.graphSignature(key, 4097));
    key.seq[0][3] += 1;
    try std.testing.expect(signature != gkey.graphSignature(key, 4096));
}

test "exl3: tier frame staging runs before replay and unwinds failed preparation" {
    const page_graph = @import("cuda_exl3_graph_pages.zig");
    const Selection = struct {
        graph_staged: bool = false,
        calls: usize = 0,
        fail_at: ?usize = null,
        pub fn begin(s: *@This(), _: void, layer: usize, position: usize, rows: usize) !void {
            try std.testing.expect(!s.graph_staged);
            try std.testing.expectEqual(s.calls + 2, layer);
            try std.testing.expectEqual(@as(usize, 2048), position);
            try std.testing.expectEqual(@as(usize, 4), rows);
            if (s.fail_at == s.calls) return error.OutOfDeviceMemory;
            s.calls += 1;
        }
    };
    var selection: Selection = .{};
    try page_graph.stage(&selection, {}, 2, 3, 2048, 4);
    try std.testing.expect(selection.graph_staged and selection.calls == 3);
    try std.testing.expectError(error.NestedPageGraphStage, page_graph.stage(&selection, {}, 2, 3, 2048, 4));
    page_graph.finish(&selection);
    try std.testing.expect(!selection.graph_staged);
    selection = .{ .fail_at = 1 };
    try std.testing.expectError(error.OutOfDeviceMemory, page_graph.stage(&selection, {}, 2, 3, 2048, 4));
    try std.testing.expect(!selection.graph_staged);
}

test "exl3: tier graph keys distinguish sparse split topology and mutable frame windows" {
    const Selection = struct { win0: usize = 0, npages: usize = 129, nslots: usize = 576 };
    var selection: Selection = .{};
    var first: Key = .{ .kind = 2, .rows = 4, .scratch = 1, .scores = 2, .output = 3 };
    first.tiers[0] = Key.tierOf(&selection, 2048, 4);
    var other = first;
    other.tiers[0] = Key.tierOf(&selection, 2052, 4);
    try std.testing.expect(!gkey.sameShape(first, other));
    selection.win0 = 256;
    other.tiers[0] = Key.tierOf(&selection, 2048, 4);
    try std.testing.expect(!std.meta.eql(first, other));
    try std.testing.expect(!gkey.sameShape(first, other));
    try std.testing.expectEqual(gkey.graphSignature(first, 2048), gkey.graphSignature(other, 2048));
    selection.nslots += 64;
    other.tiers[0] = Key.tierOf(&selection, 2048, 4);
    try std.testing.expect(gkey.graphSignature(first, 2048) != gkey.graphSignature(other, 2048));
}
