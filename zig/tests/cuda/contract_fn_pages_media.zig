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

const Capture = struct {
    var calls: usize = 0;
    var symbols: [2]usize = undefined;
    var pages: [2]flash.Pg = undefined;
    var configs: [2][7]u32 = undefined;
    var rope: [2]u64 = undefined;
    var n: [2]i32 = undefined;
    var delta: [2]i32 = undefined;

    fn launch(f: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, sm: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const prep = @intFromPtr(f) == 3;
        const i: usize = if (prep) 18 else 10;
        symbols[calls] = @intFromPtr(f);
        rope[calls] = @as(*const u64, @ptrCast(@alignCast(args.?[i].?))).*;
        n[calls] = @as(*const i32, @ptrCast(@alignCast(args.?[i + 1].?))).*;
        delta[calls] = @as(*const i32, @ptrCast(@alignCast(args.?[i + 2].?))).*;
        pages[calls] = @as(*const flash.Pg, @ptrCast(@alignCast(args.?[i + 3].?))).*;
        configs[calls] = .{ gx, gy, gz, bx, by, bz, sm };
        calls += 1;
        return 0;
    }
};

test "paged media contract: production wrappers select both media symbols with full Pages and image arguments" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const f = .{
        .prep_media = cuda.Function{ .d = &d, .handle = @ptrFromInt(1) },
        .pool_media = cuda.Function{ .d = &d, .handle = @ptrFromInt(2) },
        .prep_media_paged = cuda.Function{ .d = &d, .handle = @ptrFromInt(3) },
        .pool_media_paged = cuda.Function{ .d = &d, .handle = @ptrFromInt(4) },
    };
    var plans = try specs[1].plans(std.testing.allocator);
    defer for (&plans) |*p| p.deinit();
    const pg: flash.Pg = .{ .pt = 0x7000, .ps = 13 * 64 * 1184, .ptok = 64 };
    for (plans) |p| {
        Capture.calls = 0;
        const v = at(&d, f, 0x8000, p);
        try v.attn(s, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1e-6, 16, pg);
        try v.pooled(s, 1, 2, 3, 4, 5, 1e-6, 16, pg);
        try std.testing.expectEqualSlices(usize, &.{ 3, 4 }, &Capture.symbols);
        for (0..2) |i| {
            try std.testing.expectEqual(pg, Capture.pages[i]);
            try std.testing.expectEqual(@as(u64, 0x8000), Capture.rope[i]);
            try std.testing.expectEqual(@as(i32, @intCast(p.src.len)), Capture.n[i]);
            try std.testing.expectEqual(p.delta, Capture.delta[i]);
        }
        try std.testing.expectEqual([7]u32{ 16, 31, 1, 256, 1, 1, 0 }, Capture.configs[0]);
        try std.testing.expectEqual([7]u32{ 6, 1, 1, 128, 1, 1, 0 }, Capture.configs[1]);
    }
}
