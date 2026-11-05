//! DeepSeek-V4.1's start-up line: the resolved serving settings, the same on both ranks (issue #1395).
const std = @import("std");

/// Whether a setting was left to the engine's default or set by the environment.
pub const Source = enum { default, set };

/// A resolved setting: the engine's on/off state and where it came from.
pub const Value = struct { on: bool, source: Source };

/// The settings a repeated-prompt A/B compares, as the engine resolved them.
pub const Settings = struct {
    pieces_floor: Value, // TF_DSV41_PIECES_FLOOR: adaptive prefill rows and the memory floor
    roce: Value, // the RoCE path the session accepted (TF_ROCE asks for the small gathers)
    graphs: Value, // TF_DSV41_ROUND_GRAPH: the decode round's launches as a graph
    side_streams: Value, // TF_DSV41_BRANCHES: the indexer/compressor on a side stream
};

/// The mapping the start-up call and the test share: a resolved on/off and whether its source was set.
pub fn resolved(pieces_floor: bool, pieces_floor_set: bool, roce: bool, roce_set: bool, graphs: bool, graphs_set: bool, side_streams: bool, side_streams_set: bool) Settings {
    return .{
        .pieces_floor = .{ .on = pieces_floor, .source = source(pieces_floor_set) },
        .roce = .{ .on = roce, .source = source(roce_set) },
        .graphs = .{ .on = graphs, .source = source(graphs_set) },
        .side_streams = .{ .on = side_streams, .source = source(side_streams_set) },
    };
}

fn source(set: bool) Source {
    return if (set) .set else .default;
}

/// One line from resolved settings; a pure function, so both ranks print the same words for the same settings.
/// Off and unset differ: `0(default)` is the engine's default, `0(set)` an explicit off.
pub fn line(buf: []u8, s: Settings) []const u8 {
    return std.fmt.bufPrint(buf, "dsv41 serve: pieces_floor={s} roce={s} graphs={s} side_streams={s}", .{
        value(s.pieces_floor),
        value(s.roce),
        value(s.graphs),
        value(s.side_streams),
    }) catch "dsv41 serve: settings";
}

fn value(v: Value) []const u8 {
    return if (v.on) if (v.source == .set) "1(set)" else "1(default)" else if (v.source == .set) "0(set)" else "0(default)";
}

test "start-up line: off and unset differ, set values print their state, both ranks print the same" {
    var a: [192]u8 = undefined;
    var b: [192]u8 = undefined;
    var c: [192]u8 = undefined;
    const unset = line(&a, resolved(false, false, false, false, false, false, false, false));
    for ([_][]const u8{ "pieces_floor=0(default)", "roce=0(default)", "graphs=0(default)", "side_streams=0(default)" }) |word| {
        try std.testing.expect(std.mem.indexOf(u8, unset, word) != null);
    }
    const off = line(&b, resolved(false, true, false, true, false, true, false, true));
    for ([_][]const u8{ "pieces_floor=0(set)", "roce=0(set)", "graphs=0(set)", "side_streams=0(set)" }) |word| {
        try std.testing.expect(std.mem.indexOf(u8, off, word) != null);
    }
    try std.testing.expect(!std.mem.eql(u8, unset, off)); // an explicit off is not the default
    const on = line(&c, resolved(true, true, true, false, true, true, true, false));
    for ([_][]const u8{ "pieces_floor=1(set)", "roce=1(default)", "graphs=1(set)", "side_streams=1(default)" }) |word| {
        try std.testing.expect(std.mem.indexOf(u8, on, word) != null);
    }
    var again: [192]u8 = undefined;
    try std.testing.expectEqualStrings(on, line(&again, resolved(true, true, true, false, true, true, true, false)));
}
