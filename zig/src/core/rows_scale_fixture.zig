//! Test-owned valid state corpora, with short images and no shared blocks, for startup and allocation faults.
const std = @import("std");
const sys = @import("sys");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const t = std.testing;
const a = t.allocator;

pub const Fixture = struct {
    root: [:0]u8,
    f: fake.Fake = .{ .gpa = a },

    pub fn init(tag: []const u8) !Fixture {
        const tmp = if (sys.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
        return .{ .root = try std.fmt.allocPrintSentinel(a, "{s}/tf-rows-scale-{s}-{d}", .{ tmp, tag, sys.c.getpid() }, 0) };
    }

    pub fn close(x: *Fixture) void {
        @import("prompt_imprint_test.zig").rmTree(x.root);
        a.free(x.root);
    }

    pub fn options(x: *Fixture) rows.Options {
        return .{ .root = x.root, .identity = 7, .cap = 1 << 40, .floor = 0, .daily = 1 << 40, .lookahead = 1 };
    }

    pub fn seed(x: *Fixture, count: usize) !void {
        const r = try rows.Rows.open(a, x.f.snapshots(), x.options());
        defer r.deinit();
        var parts: std.ArrayList(rows.image.Part) = .empty;
        defer parts.deinit(a);
        try fake.Fake.layout(64, &parts, a);
        var ranges: std.ArrayList(rows.image.Range) = .empty;
        defer ranges.deinit(a);
        try rows.image.ownRanges(parts.items, 64, rows.block_tokens, &ranges, a);
        const payload = try a.alloc(u8, @intCast(rows.image.total(ranges.items)));
        defer a.free(payload);
        var tokens: [65]u32 = @splat(17);
        for (0..count) |i| {
            tokens[0] = @intCast(i + 1);
            x.f.prompt = &tokens;
            x.f.at = 64;
            x.f.build(64, payload);
            const name = rows.chain.state(r.keying, &tokens, &.{}, &.{}, 64);
            const stem = std.fmt.bytesToHex(rows.chain.stem(name), .lower);
            var w = try rows.file.Writer.open(r.dir, &stem, ".st");
            errdefer w.abort();
            try w.append(payload);
            const counts: [3]u32 = .{ 65, 0, 0 };
            _ = try w.finish(.{ .kind = @backingInt(rows.file.Kind.state), .payload = 0, .payload_hash = 0, .name = name, .at = 64, .block = rows.block_tokens }, &.{ std.mem.asBytes(&counts), std.mem.sliceAsBytes(&tokens) }, false);
        }
        x.f.prompt = &.{};
    }

    pub fn files(x: *Fixture) !usize {
        const r = try rows.Rows.open(a, x.f.snapshots(), x.options());
        defer r.deinit();
        return r.states.items.len;
    }
};
