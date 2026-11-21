//! DeepSeek-V4.1's startup settings, reported by each rank after its model loads.
const std = @import("std");

/// Whether a setting was left to the engine's default or set by the environment.
pub const Source = enum { default, set };
pub const Comm = enum { nccl, mailbox, roce };
pub const Get = *const fn ([:0]const u8) ?[]const u8;

/// A resolved setting: the engine's on/off state and where it came from.
pub const Value = struct { on: bool, source: Source };

/// The settings a repeated-prompt A/B compares, as the engine resolved them.
pub const Settings = struct {
    pieces_floor: Value,
    comm: struct { kind: Comm, source: Source },
    graphs: Value,
    side_streams: Value,
    ignored_roce: bool,
};

/// State comes from the loaded engine; only its provenance comes from the environment.
pub fn resolved(pieces_floor: bool, comm: Comm, graphs: bool, side_streams: bool, get: Get) Settings {
    return .{
        .pieces_floor = .{ .on = pieces_floor, .source = source(get("TF_DSV41_PIECES_FLOOR") != null) },
        .comm = .{ .kind = comm, .source = source(get("TF_COMM_BACKEND") != null) },
        .graphs = .{ .on = graphs, .source = source(get("TF_DSV41_ROUND_GRAPH") != null) },
        .side_streams = .{ .on = side_streams, .source = source(get("TF_DSV41_BRANCHES") != null) },
        .ignored_roce = get("TF_ROCE") != null,
    };
}

fn source(set: bool) Source {
    return if (set) .set else .default;
}

/// Each rank prints its accepted transport and resolved state, with the source of each setting.
pub fn line(buf: []u8, s: Settings) []const u8 {
    return std.fmt.bufPrint(buf, "dsv41 serve: pieces_floor={s} comm={t}({t}) graphs={s} side_streams={s}", .{
        value(s.pieces_floor), s.comm.kind, s.comm.source, value(s.graphs), value(s.side_streams),
    }) catch "dsv41 serve: settings";
}

/// Called once at startup on each rank; the notice describes an ignored variable, not a transport choice.
pub fn print(s: Settings, output: anytype) void {
    var buf: [192]u8 = undefined;
    output(line(&buf, s));
    if (s.ignored_roce) output("dsv41: TF_ROCE is ignored; DeepSeek uses TF_COMM_BACKEND for its transport");
}

fn value(v: Value) []const u8 {
    return if (v.on) if (v.source == .set) "1(set)" else "1(default)" else if (v.source == .set) "0(set)" else "0(default)";
}

const Fixture = struct {
    var pairs: []const struct { [:0]const u8, []const u8 } = &.{};
    fn get(name: [:0]const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "startup line: resolved state, source mapping and identical rank settings" {
    defer Fixture.pairs = &.{};
    var a: [192]u8 = undefined;
    var b: [192]u8 = undefined;
    Fixture.pairs = &.{};
    const unset = line(&a, resolved(false, .nccl, false, false, Fixture.get));
    try std.testing.expectEqualStrings("dsv41 serve: pieces_floor=0(default) comm=nccl(default) graphs=0(default) side_streams=0(default)", unset);
    Fixture.pairs = &.{ .{ "TF_DSV41_PIECES_FLOOR", "0" }, .{ "TF_COMM_BACKEND", "nccl" }, .{ "TF_DSV41_ROUND_GRAPH", "0" }, .{ "TF_DSV41_BRANCHES", "0" } };
    const off = line(&b, resolved(false, .nccl, false, false, Fixture.get));
    try std.testing.expectEqualStrings("dsv41 serve: pieces_floor=0(set) comm=nccl(set) graphs=0(set) side_streams=0(set)", off);
    Fixture.pairs = &.{ .{ "TF_DSV41_PIECES_FLOOR", "1" }, .{ "TF_DSV41_ROUND_GRAPH", "1" } };
    const on = resolved(true, .roce, true, true, Fixture.get);
    try std.testing.expectEqualStrings("dsv41 serve: pieces_floor=1(set) comm=roce(default) graphs=1(set) side_streams=1(default)", line(&a, on));
    try std.testing.expectEqualStrings(line(&a, on), line(&b, resolved(true, .roce, true, true, Fixture.get)));
    // The accepted transport can differ from the requested backend after a session fallback.
    Fixture.pairs = &.{.{ "TF_COMM_BACKEND", "roce" }};
    try std.testing.expectEqualStrings("dsv41 serve: pieces_floor=0(default) comm=nccl(set) graphs=0(default) side_streams=0(default)", line(&a, resolved(false, .nccl, false, false, Fixture.get)));
    try std.testing.expectEqual(Comm.mailbox, resolved(false, .mailbox, false, false, Fixture.get).comm.kind);
}

test "startup line: TF_ROCE alone never marks the accepted transport as set" {
    defer Fixture.pairs = &.{};
    for ([_][]const u8{ "0", "1", "" }) |raw| {
        Fixture.pairs = &.{.{ "TF_ROCE", raw }};
        const s = resolved(false, .roce, false, false, Fixture.get);
        try std.testing.expectEqual(Source.default, s.comm.source);
        try std.testing.expect(s.ignored_roce);
        var buf: [192]u8 = undefined;
        try std.testing.expect(std.mem.indexOf(u8, line(&buf, s), "comm=roce(default)") != null);
    }
    Fixture.pairs = &.{ .{ "TF_ROCE", "0" }, .{ "TF_COMM_BACKEND", "nccl" } };
    try std.testing.expectEqual(Source.set, resolved(false, .nccl, false, false, Fixture.get).comm.source);
}

test "startup line: one ignored-variable notice per startup, none when absent" {
    const Capture = struct {
        var settings: usize = 0;
        var notices: usize = 0;
        fn output(text: []const u8) void {
            if (std.mem.startsWith(u8, text, "dsv41 serve:")) settings += 1;
            if (std.mem.indexOf(u8, text, "TF_ROCE is ignored") != null) notices += 1;
        }
    };
    defer Fixture.pairs = &.{};
    Capture.settings = 0;
    Capture.notices = 0;
    Fixture.pairs = &.{};
    print(resolved(false, .nccl, false, false, Fixture.get), Capture.output);
    try std.testing.expectEqual(@as(usize, 1), Capture.settings);
    try std.testing.expectEqual(@as(usize, 0), Capture.notices);
    Fixture.pairs = &.{.{ "TF_ROCE", "0" }};
    print(resolved(false, .roce, false, false, Fixture.get), Capture.output);
    try std.testing.expectEqual(@as(usize, 2), Capture.settings);
    try std.testing.expectEqual(@as(usize, 1), Capture.notices);
}
