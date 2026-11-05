//! Image spans for the paged cache contract, through the production media plan and launch wrappers.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const State = std.meta.Child(@FieldType(flash.prompt.Seq, "media"));
pub const Plan = @FieldType(@FieldType(State, "image"), "plan");
pub const Span = std.meta.Child(@FieldType(Plan, "runs"));
pub const View = std.meta.Child(@FieldType(flash.batch_state.SeqState, "media"));

pub const Spec = struct {
    n: [2]usize,
    spans: [2][]const Span,

    pub fn plans(spec: Spec, a: std.mem.Allocator) ![2]Plan {
        var first = try Plan.init(a, spec.n[0], spec.spans[0], .{});
        errdefer first.deinit();
        return .{ first, try Plan.init(a, spec.n[1], spec.spans[1], .{}) };
    }
};

/// Prompt, verify across the prompt's end, sparse decode after it, and a fused prompt with multiple images.
pub const specs = [_]Spec{
    .{ .n = .{ 59, 57 }, .spans = .{ &.{.{ .at = 27, .gh = 8, .gw = 12 }}, &.{ .{ .at = 3, .gh = 10, .gw = 14 }, .{ .at = 41, .gh = 4, .gw = 6 } } } },
    .{ .n = .{ 70, 135 }, .spans = .{ &.{.{ .at = 59, .gh = 4, .gw = 10 }}, &.{.{ .at = 119, .gh = 4, .gw = 14 }} } },
    .{ .n = .{ 2000, 2003 }, .spans = .{ &.{.{ .at = 1920, .gh = 8, .gw = 16 }}, &.{.{ .at = 1955, .gh = 6, .gw = 14 }} } },
    .{ .n = .{ 270, 385 }, .spans = .{ &.{ .{ .at = 121, .gh = 8, .gw = 18 }, .{ .at = 253, .gh = 4, .gw = 8 } }, &.{ .{ .at = 250, .gh = 10, .gw = 16 }, .{ .at = 350, .gh = 6, .gw = 10 } } } },
};

/// The contract exercises attention and pooling only; features, src and splice are never consumed by these launches.
pub fn at(d: *const cuda.Driver, f: anytype, rope: u64, p: Plan) View {
    return .{
        .prep = f.prep_media,
        .pool = f.pool_media,
        .prep_paged = f.prep_media_paged,
        .pool_paged = f.pool_media_paged,
        .planes = .{ .prep = f.prep_media_planes, .pool = f.pool_media_planes },
        .image = .{ .features = 0, .src = 0, .rope = rope, .n = p.src.len, .delta = p.delta, .width = 5120, .splice_fn = .{ .d = d, .handle = null } },
    };
}

pub fn view(rig: anytype, f: anytype, p: Plan) !View {
    return at(rig.gpu.d, f, try rig.input(std.mem.sliceAsBytes(p.pos)), p);
}

test "paged media contract: real plans retain image axes, text suffixes and a negative decode delta" {
    for (specs) |spec| {
        var plans = try spec.plans(std.testing.allocator);
        defer for (&plans) |*p| p.deinit();
        for (plans, spec.spans, spec.n) |p, spans, n| {
            try std.testing.expect(p.delta < 0);
            for (spans) |span| {
                const start = span.at;
                const width = span.gw / 2;
                try std.testing.expect(p.src[start] >= 0 and p.src[start + span.tokens() - 1] >= 0);
                try std.testing.expect(p.pos[start] != p.pos[2 * n + start + 1]);
                try std.testing.expect(p.pos[n + start + width] == p.pos[start] + 1);
                try std.testing.expectEqual(@as(i32, -1), p.src[start - 1]);
            }
            try std.testing.expectEqual(@as(i32, @intCast(n - 1)) + p.delta, p.pos[n - 1]);
        }
    }
}
