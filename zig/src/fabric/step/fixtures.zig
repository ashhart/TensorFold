//! Golden bytes of step messages, good and bad, for every user of the header to test against (fixtures/).
const std = @import("std");
const header = @import("header.zig");
const section = @import("section.zig");
const message = @import("message.zig");

/// What every fixture is judged against: a receiver's limits and the round and group it expects.
pub const limits: message.Limits = .{ .bytes = 512, .streams = 4, .rows = 8, .context = 64 };
pub const context: header.Context = .{ .round = 3, .group = 0x0A0B_0C0D };

fn head(kind: header.Kind) header.Header {
    return .{ .kind = kind, .round = context.round, .group = context.group };
}

/// The ring fixtures' round (fabric/stage/fixtures.zig): one stream of each op but cancel, 7 rows, a cut at row 4.
pub const streams = [_]section.Stream{
    .{ .id = 3, .op = .open, .first = 0, .rows = 4 },
    .{ .id = 7, .op = .@"continue", .first = 10, .rows = 1, .accepted = 2, .generation = 1 },
    .{ .id = 9, .op = .@"resume", .first = 32, .rows = 2, .resume_at = 32, .generation = 2 },
    .{ .id = 12, .op = .close, .first = 20, .rows = 0, .generation = 1 },
};
pub const cuts = [_]u32{4};
const tokens = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 90, 91 }; // the round's 7 ids, then a 2-id tail

const round: message.Message = .{ .header = head(.round), .streams = &streams, .cuts = &cuts };
const round_tokens: message.Message = .{ .header = head(.round), .streams = &streams, .tokens = &tokens, .cuts = &cuts };
const keepalive: message.Message = .{ .header = head(.round) };
const begin_pages = [_]u32{ 4, 5, 6 };
const begin_sampling = [_]section.Sampling{.{ .stream = 5, .top_k = 40, .seed = 42, .temperature = 0.7, .top_p = 0.9, .min_p = 0.05, .has = .{ .seed = true } }};
const begin: message.Message = .{
    .header = withFamily(head(.state), 1),
    .streams = &.{.{ .id = 5, .op = .open, .first = 0, .rows = 0 }},
    .sampling = &begin_sampling,
    .pages = &.{.{ .stream = 5, .first = 0, .kept = 1, .ids = &begin_pages }},
};
const drops: message.Message = .{ .header = head(.state), .snapshots = &.{
    .{ .op = .drop, .id = 11 },
    .{ .op = .drop, .id = 12 },
    .{ .op = .drop, .id = 0x1_0000_0003 },
} };
const restore_chunk: message.Message = .{
    .header = withMode(head(.round), .prompt),
    .streams = &.{.{ .id = 6, .op = .@"resume", .first = 16, .rows = 4, .resume_at = 16, .generation = 1 }},
    .tokens = &.{ 20, 21, 22, 23, 24 },
    .snapshots = &.{.{ .op = .restore, .stream = 6, .id = 21, .at = 16 }},
};
const compact: message.Message = .{ .header = head(.round), .rows = &.{ .{ .id = 1, .rows = 1 }, .{ .id = 2, .rows = 1 }, .{ .id = 4, .rows = 1 } }, .tokens = &.{ 7, 8, 9 } };
const copy_page = [_]u32{9};
const copies: message.Message = .{ .header = head(.state), .copies = &.{.{ .from = 7, .to = 9 }}, .pages = &.{.{ .stream = 2, .first = 3, .ids = &copy_page }} };
const family: message.Message = .{ .header = withFamily(head(.family), 7), .family = "fivex" };
const stop: message.Message = .{ .header = head(.stop) };
const sample: message.Message = .{
    .header = withFamily(head(.family), 20),
    .sampling = &.{
        // DeepSeek's 7-word form
        .{ .stream = 0, .top_k = 50, .temperature = 0.8, .top_p = 0.95, .row0 = 3, .rows = 2 },
        // 9 words
        .{ .stream = 0, .seed = 99, .temperature = 1.0, .rows = 1, .pos0 = 12, .has = .{ .seed = true, .pos0 = true } },
    },
};

fn withFamily(h: header.Header, f: u32) header.Header {
    var x = h;
    x.family = f;
    return x;
}

fn withMode(h: header.Header, m: header.Mode) header.Header {
    var x = h;
    x.mode = m;
    return x;
}

const Build = *const fn (out: []u8) anyerror![]u8;

pub const Fixture = struct { name: []const u8, build: Build, want: ?anyerror };

fn fx(name: []const u8, build: Build, want: ?anyerror) Fixture {
    return .{ .name = name, .build = build, .want = want };
}

fn good(comptime m: message.Message) Build {
    return struct {
        fn b(out: []u8) anyerror![]u8 {
            return message.write(m, out);
        }
    }.b;
}

/// `m` with one edit to the message before it is written.
fn with(comptime m: message.Message, comptime edit: fn (*message.Message) void) Build {
    return struct {
        fn b(out: []u8) anyerror![]u8 {
            var x = m;
            edit(&x);
            return message.write(x, out);
        }
    }.b;
}

/// `m`'s bytes with byte `at` set to `v`.
fn poke(comptime m: message.Message, comptime at: usize, comptime v: u8) Build {
    return struct {
        fn b(out: []u8) anyerror![]u8 {
            const p = try message.write(m, out);
            p[at] = v;
            return p;
        }
    }.b;
}

/// `m` with stream `i` replaced (the ring's round only).
fn stream(comptime i: usize, comptime s: section.Stream) fn (*message.Message) void {
    return struct {
        const list = blk: {
            var l = streams;
            l[i] = s;
            break :blk l;
        };
        fn f(x: *message.Message) void {
            x.streams = &list;
        }
    }.f;
}

const big: [520]u8 = @splat(0x5A);
const six_tokens = [_]u32{ 1, 2, 3, 4, 5, 6 };
const late_cut = [_]u32{7};

pub const all = [_]Fixture{
    fx("round.bin", good(round), null),
    fx("round_tokens.bin", good(round_tokens), null),
    fx("keepalive.bin", good(keepalive), null),
    fx("begin.bin", good(begin), null),
    fx("drops.bin", good(drops), null),
    fx("restore_chunk.bin", good(restore_chunk), null),
    fx("compact.bin", good(compact), null),
    fx("copies.bin", good(copies), null),
    fx("family.bin", good(family), null),
    fx("stop.bin", good(stop), null),
    fx("sample.bin", good(sample), null),
    fx("bad_magic.bin", poke(round, 0, 0), error.NotStepHeader),
    fx("bad_version.bin", poke(round, 4, 2), error.BadVersion),
    fx("bad_kind.bin", poke(round, 6, 9), error.BadKind),
    fx("bad_mode.bin", poke(begin, 7, 1), error.BadMode),
    fx("bad_keepalive_mode.bin", poke(keepalive, 7, 1), error.BadMode),
    fx("bad_reserved_header.bin", poke(round, 50, 1), error.BadReserved),
    fx("too_large.bin", good(.{ .header = head(.family), .family = &big }), error.TooLarge),
    fx("bad_length_bytes.bin", poke(round, 32, 136 + 8), error.BadLength),
    fx("bad_section_version.bin", poke(round, 65, 2), error.BadSectionVersion),
    fx("bad_reserved_prefix.bin", poke(round, 66, 1), error.BadReserved),
    fx("unknown_section.bin", poke(round, 64, 9), error.UnknownSection),
    fx("unordered_sections.bin", poke(round_tokens, 232, 2), error.UnorderedSections),
    fx("bad_padding.bin", poke(round_tokens, 228, 1), error.BadReserved),
    fx("bad_sections.bin", poke(round, 20, 0b1110), error.BadSections),
    fx("both_row_maps.bin", good(.{ .header = head(.round), .streams = &streams, .rows = &.{.{ .id = 1, .rows = 1 }} }), error.BothRowMaps),
    fx("bad_streams.bin", poke(round, 24, 3), error.BadStreams),
    fx("bad_op.bin", poke(round, 64 + 8 + 4, 9), error.BadOp),
    fx("bad_open.bin", with(round, stream(0, .{ .id = 3, .op = .open, .first = 1, .rows = 4 })), error.BadOpen),
    fx("bad_resume.bin", with(round, stream(2, .{ .id = 9, .op = .@"resume", .first = 32, .rows = 2, .resume_at = 33, .generation = 2 })), error.BadResume),
    fx("rows_on_close.bin", with(round, stream(3, .{ .id = 12, .op = .close, .first = 20, .rows = 1, .generation = 1 })), error.RowsOnClose),
    fx("past_context.bin", with(round, stream(2, .{ .id = 9, .op = .@"resume", .first = 63, .rows = 2, .resume_at = 32, .generation = 2 })), error.PastContext),
    fx("too_many_rows.bin", with(round, stream(0, .{ .id = 3, .op = .open, .first = 0, .rows = 6 })), error.TooManyRows),
    fx("bad_total.bin", poke(round, 28, 6), error.BadTotal),
    fx("rows_off_round.bin", good(.{ .header = head(.state), .streams = &.{.{ .id = 5, .op = .open, .first = 0, .rows = 2 }} }), error.RowsOffRound),
    fx("sections_off_round.bin", good(.{ .header = head(.stop), .family = "x" }), error.SectionsOffRound),
    fx("bad_tokens.bin", good(.{ .header = head(.round), .streams = &streams, .tokens = &six_tokens, .cuts = &cuts }), error.BadTokens),
    fx("bad_cut.bin", good(.{ .header = head(.round), .streams = &streams, .cuts = &late_cut }), error.BadCut),
    fx("bad_sampling.bin", with(begin, struct {
        fn f(x: *message.Message) void {
            x.sampling = &.{.{ .stream = 5, .top_k = 40, .seed = 42, .temperature = 0.7 }}; // a seed with its bit clear
        }
    }.f), error.BadSampling),
    fx("bad_pages.bin", with(begin, struct {
        fn f(x: *message.Message) void {
            x.pages = &.{.{ .stream = 5, .first = 0, .kept = 4, .ids = &begin_pages }};
        }
    }.f), error.BadPages),
    fx("bad_snapshot_op.bin", poke(drops, 72, 9), error.BadSnapshotOp),
    fx("bad_snapshot.bin", good(.{ .header = head(.state), .snapshots = &.{.{ .op = .drop, .stream = 5, .id = 11 }} }), error.BadSnapshot),
    fx("other_round.bin", good(.{ .header = .{ .kind = .round, .round = 4, .group = context.group } }), error.OtherRound),
    fx("other_group.bin", good(.{ .header = .{ .kind = .round, .round = context.round, .group = 1 } }), error.OtherGroup),
};

/// What a receiver makes of a fixture's bytes: null when it takes them, else the error it refuses them with.
pub fn judge(bytes: []const u8) anyerror!void {
    const s = try message.decode(bytes, limits);
    try header.expect(s.header, context);
}
