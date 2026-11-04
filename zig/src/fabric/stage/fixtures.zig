//! Golden bytes of every stage protocol message, good and bad, for a CUDA or Metal stage to test against (fixtures/).
const std = @import("std");
const payload = @import("payload.zig");
const boot = @import("boot.zig");
const plan = @import("plan.zig");
const frame = @import("frame.zig");

/// The fixtures' toy model: 8 layers in two stages, 2 planes of 4 bf16 values a row, a drafter of width 2.
pub const model: boot.Model = .{
    .version = "tensorfold stage-v0 test",
    .kv_dtype = .bf16,
    .layers = 8,
    .boundary = .{ .hidden = 4, .row_dtype = .bf16, .planes = 2, .draft_width = 2 },
    .shapes = .{ .prompt_chunk = 4, .max_rows = 8, .parallel = 4, .context = 64 },
    .taps = &.{ 1, 5 },
};

/// Rank 0's plan: rank 0 on CUDA with layers 0..4, rank 1 on Metal with layers 4..8.
pub const expects = [2]boot.Expect{
    .{ .model = &model, .rank = 0, .world = 2, .backend = .cuda, .range = .{ .first = 0, .end = 4 } },
    .{ .model = &model, .rank = 1, .world = 2, .backend = .metal, .range = .{ .first = 4, .end = 8 } },
};

/// The description rank `rank` sends when it matches `expects[rank]`.
pub fn description(rank: u32) boot.Description {
    const e = expects[rank];
    return .{
        .rank = rank,
        .world = 2,
        .version = boot.Text.init(model.version) catch unreachable,
        .backend = e.backend,
        .kv_dtype = model.kv_dtype,
        .range = e.range,
        .boundary = model.boundary,
        .shapes = model.shapes,
        .taps = boot.Taps.init(model.taps[rank .. rank + 1]) catch unreachable,
        .kv_rows = if (rank == 0) 256 else 128,
        .states = 4,
    };
}

/// The round every plan and rows fixture is cut from: one stream of each op but cancel, 7 rows, a cut at row 4.
pub const streams = [_]plan.Stream{
    .{ .id = 3, .op = .open, .first = 0, .rows = 4 },
    .{ .id = 7, .op = .@"continue", .first = 10, .rows = 1, .accepted = 2, .generation = 1 },
    .{ .id = 9, .op = .@"resume", .first = 32, .rows = 2, .resume_at = 32, .generation = 2 },
    .{ .id = 12, .op = .close, .first = 20, .rows = 0, .generation = 1 },
};
pub const cuts = [_]u32{4};

pub const limits = plan.Limits.of(model.shapes);

pub const Kind = enum { describe, ready, plan, rows };
const Build = *const fn (out: []u8) anyerror![]u8;

pub const Fixture = struct { name: []const u8, kind: Kind, build: Build, want: ?anyerror };

fn descWith(comptime rank: u32, comptime edit: fn (*boot.Description) void) Build {
    return struct {
        fn b(out: []u8) anyerror![]u8 {
            var d = description(rank);
            edit(&d);
            return boot.encodeDescription(&d, out);
        }
    }.b;
}

fn same(_: *boot.Description) void {}

fn planWith(comptime edit: fn (*[streams.len]plan.Stream, *[cuts.len]u32) void) Build {
    return struct {
        fn b(out: []u8) anyerror![]u8 {
            var s = streams;
            var c = cuts;
            edit(&s, &c);
            return plan.write(&s, &c, out);
        }
    }.b;
}

/// `base`'s bytes with byte `at` set to `v`.
fn poke(comptime base: Build, comptime at: usize, comptime v: u8) Build {
    return struct {
        fn b(out: []u8) anyerror![]u8 {
            const p = try base(out);
            p[at] = v;
            return p;
        }
    }.b;
}

/// `base`'s bytes without the last `n`.
fn short(comptime base: Build, comptime n: usize) Build {
    return struct {
        fn b(out: []u8) anyerror![]u8 {
            const p = try base(out);
            return p[0 .. p.len - n];
        }
    }.b;
}

fn ready(comptime r: boot.Ready) Build {
    return struct {
        fn b(out: []u8) anyerror![]u8 {
            return boot.encodeReady(r, out);
        }
    }.b;
}

fn keepAlive(out: []u8) anyerror![]u8 {
    return plan.write(&.{}, &.{}, out);
}

fn tooManyStreams(out: []u8) anyerror![]u8 {
    var s: [5]plan.Stream = undefined;
    for (&s, 0..) |*x, i| x.* = .{ .id = @intCast(i), .op = .@"continue", .first = 1, .rows = 1 };
    return plan.write(&s, &.{}, out);
}

/// A rows frame for the round above: its 32-byte header with `rows` and `bytes`, then `bytes` of a fixed pattern.
fn rowsFrame(comptime rows: u32, comptime bytes: u64) Build {
    return struct {
        fn b(out: []u8) anyerror![]u8 {
            const n = frame.size + bytes;
            if (out.len < n) return error.NoRoom;
            out[0..frame.size].* = frame.encode(.{ .kind = .rows, .round = 1, .rows = rows, .bytes = bytes });
            for (out[frame.size..n], 0..) |*x, i| x.* = @truncate(i * 37 + 11);
            return out[0..n];
        }
    }.b;
}

const round_bytes = 2 * 7 * 4 * 2 + 7 * 2 * 4; // 2 planes x 7 rows x 4 bf16, then 7 rows x 2 fp32 tap sums

fn nop(_: *[streams.len]plan.Stream, _: *[cuts.len]u32) void {}

const d0 = descWith(0, same);
const d1 = descWith(1, same);
const r1 = ready(.{ .rank = 1, .loaded = true, .warm = true });
const round = planWith(nop);

fn fx(comptime name: []const u8, comptime build: Build, comptime want: ?anyerror) Fixture {
    const kind: Kind = for (std.meta.tags(Kind)) |k| {
        if (std.mem.startsWith(u8, name, @tagName(k))) break k;
    } else @compileError("a fixture's name starts with its message: " ++ name);
    return .{ .name = name ++ ".bin", .kind = kind, .build = build, .want = want };
}

/// Every fixture, its file name under fixtures/, and the error a stage refuses it with (null: accepted).
pub const all = [_]Fixture{
    fx("describe_rank0", d0, null),
    fx("describe_rank1", d1, null),
    fx("describe_bad_magic", poke(d1, 0, 0), error.NotStagePayload),
    fx("describe_bad_payload_version", poke(d1, 4, 2), error.BadPayloadVersion),
    fx("describe_bad_length", short(d1, 1), error.BadLength),
    fx("describe_bad_reserved", poke(d1, 22, 1), error.BadReserved),
    fx("describe_bad_backend", poke(d1, 16, 9), error.BadBackend),
    fx("describe_bad_row_dtype", poke(d1, 18, 5), error.BadRowDtype),
    fx("describe_bad_version_text", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.version.bytes[3] = 0x07;
        }
    }.f), error.BadVersionText),
    fx("describe_bad_taps", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.taps.ids[0] = 2;
        }
    }.f), error.BadTaps),
    fx("describe_other_rank", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.world = 3;
        }
    }.f), error.OtherRank),
    fx("describe_other_version", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.version.bytes[d.version.len - 1] = 'X';
        }
    }.f), error.OtherVersion),
    fx("describe_other_backend", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.backend = .cuda;
        }
    }.f), error.OtherBackend),
    fx("describe_other_kv_dtype", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.kv_dtype = .f8e4m3;
        }
    }.f), error.OtherKvDtype),
    fx("describe_other_range", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.range.first = 3;
        }
    }.f), error.OtherRange),
    fx("describe_other_width", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.boundary.hidden = 8;
        }
    }.f), error.OtherWidth),
    fx("describe_other_dtype", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.boundary.row_dtype = .f16;
        }
    }.f), error.OtherDtype),
    fx("describe_other_planes", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.boundary.planes = 1;
        }
    }.f), error.OtherPlanes),
    fx("describe_other_draft_width", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.boundary.draft_width = 0;
        }
    }.f), error.OtherDraftWidth),
    fx("describe_other_shapes", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.shapes.context = 128;
        }
    }.f), error.OtherShapes),
    fx("describe_other_taps", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.taps.ids[0] = 6;
        }
    }.f), error.OtherTaps),
    fx("describe_short_capacity", descWith(1, struct {
        fn f(d: *boot.Description) void {
            d.kv_rows = 32;
        }
    }.f), error.ShortCapacity),
    fx("ready_rank1", r1, null),
    fx("ready_cold", ready(.{ .rank = 1, .loaded = true, .warm = false }), error.NotReady),
    fx("ready_bad_flags", poke(r1, 12, 7), error.BadFlags),
    fx("ready_wrong_message", poke(r1, 6, 3), error.WrongMessage),
    fx("plan_keepalive", keepAlive, null),
    fx("plan_round", round, null),
    fx("plan_bad_length", short(round, 4), error.BadLength),
    fx("plan_bad_reserved", poke(keepAlive, 20, 1), error.BadReserved),
    fx("plan_too_many_streams", tooManyStreams, error.TooManyStreams),
    fx("plan_unordered", planWith(struct {
        fn f(s: *[streams.len]plan.Stream, _: *[cuts.len]u32) void {
            s[1].id = 2;
        }
    }.f), error.UnorderedStreams),
    fx("plan_bad_op", poke(round, plan.plan_fixed + 4, 6), error.BadOp),
    fx("plan_bad_open", planWith(struct {
        fn f(s: *[streams.len]plan.Stream, _: *[cuts.len]u32) void {
            s[0].first = 5;
        }
    }.f), error.BadOpen),
    fx("plan_open_generation", planWith(struct {
        fn f(s: *[streams.len]plan.Stream, _: *[cuts.len]u32) void {
            s[0].generation = 1;
        }
    }.f), error.BadOpen),
    fx("plan_bad_resume", planWith(struct {
        fn f(s: *[streams.len]plan.Stream, _: *[cuts.len]u32) void {
            s[2].resume_at = 40;
        }
    }.f), error.BadResume),
    fx("plan_rows_on_close", planWith(struct {
        fn f(s: *[streams.len]plan.Stream, _: *[cuts.len]u32) void {
            s[3].rows = 1;
        }
    }.f), error.RowsOnClose),
    fx("plan_past_context", planWith(struct {
        fn f(s: *[streams.len]plan.Stream, _: *[cuts.len]u32) void {
            s[1].first = 64;
        }
    }.f), error.PastContext),
    fx("plan_too_many_rows", planWith(struct {
        fn f(s: *[streams.len]plan.Stream, _: *[cuts.len]u32) void {
            s[2].rows = 4;
        }
    }.f), error.TooManyRows),
    fx("plan_bad_total", poke(round, 16, 6), error.BadTotal),
    fx("plan_bad_cut", planWith(struct {
        fn f(_: *[streams.len]plan.Stream, c: *[cuts.len]u32) void {
            c[0] = 7;
        }
    }.f), error.BadCut),
    fx("rows_round", rowsFrame(7, round_bytes), null),
    fx("rows_other_count", rowsFrame(6, round_bytes), error.OtherRowCount),
    fx("rows_other_bytes", rowsFrame(7, round_bytes - 1), error.OtherRowBytes),
};

/// What a stage makes of a fixture's bytes: null when it takes them, else the error it refuses them with.
pub fn judge(kind: Kind, bytes: []const u8) anyerror!void {
    switch (kind) {
        .describe => {
            const d = try boot.decodeDescription(bytes);
            try boot.check(expects[@min(d.rank, 1)], &d);
        },
        .ready => if (!(try boot.decodeReady(bytes)).ready()) return error.NotReady,
        .plan => _ = try plan.decode(bytes, limits),
        .rows => {
            if (bytes.len < frame.size) return error.BadLength;
            const h = try frame.decode(bytes[0..frame.size], 1 << 20);
            var buf: [plan.planSize(streams.len, cuts.len)]u8 = undefined;
            const p = try plan.decode(try plan.write(&streams, &cuts, &buf), limits);
            _ = try plan.checkRows(h, p, model.boundary);
            if (bytes.len - frame.size != h.bytes) return error.BadLength;
        },
    }
}
