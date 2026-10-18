//! The lane backend a CUDA family hands the lane core, each entry called with the device context current.
const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;

/// `v` (the family's entries, held as `inner` by `H`) behind `H.bind`: every entry the family gives, and only those.
pub fn vtable(comptime H: type, v: be.Backend.VTable) be.Backend.VTable {
    return .{
        .prefill = struct {
            fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                const x = H.bind(p);
                return x.inner.vtable.prefill(x.inner.ptr, s);
            }
        }.f,
        .prefill_step = if (v.prefill_step != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream) anyerror!bool {
                const x = H.bind(p);
                return x.inner.vtable.prefill_step.?(x.inner.ptr, s);
            }
        }.f else null,
        .prefill_many = if (v.prefill_many != null) struct {
            fn f(p: *anyopaque, ss: []const *lanes.Stream) anyerror!bool {
                const x = H.bind(p);
                return x.inner.vtable.prefill_many.?(x.inner.ptr, ss);
            }
        }.f else null,
        .many_rows = v.many_rows,
        .first = struct {
            fn f(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
                const x = H.bind(p);
                return x.inner.vtable.first(x.inner.ptr, s, position);
            }
        }.f,
        .queue = struct {
            fn f(p: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
                const x = H.bind(p);
                return x.inner.vtable.queue(x.inner.ptr, s, feed, position);
            }
        }.f,
        .read = struct {
            fn f(p: *anyopaque, handle: u64) anyerror!u32 {
                const x = H.bind(p);
                return x.inner.vtable.read(x.inner.ptr, handle);
            }
        }.f,
        .verify = struct {
            fn f(p: *anyopaque, w: []const be.Window, out: []be.Verified) anyerror!void {
                const x = H.bind(p);
                return x.inner.vtable.verify(x.inner.ptr, w, out);
            }
        }.f,
        .keep = struct {
            fn f(p: *anyopaque, w: []const be.Window, paths: []const []const u32) anyerror!void {
                const x = H.bind(p);
                return x.inner.vtable.keep(x.inner.ptr, w, paths);
            }
        }.f,
        .draft = struct {
            fn f(p: *anyopaque, r: []const be.DraftRequest) anyerror!void {
                const x = H.bind(p);
                return x.inner.vtable.draft(x.inner.ptr, r);
            }
        }.f,
        .unspeculate = if (v.unspeculate != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                const x = H.bind(p);
                return x.inner.vtable.unspeculate.?(x.inner.ptr, s);
            }
        }.f else null,
        .probabilities = if (v.probabilities != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
                const x = H.bind(p);
                return x.inner.vtable.probabilities.?(x.inner.ptr, s, out);
            }
        }.f else null,
        .tree = if (v.tree != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
                const x = H.bind(p);
                return x.inner.vtable.tree.?(x.inner.ptr, s, gpa);
            }
        }.f else null,
        .alternatives = if (v.alternatives != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, out: []be.Alternative) anyerror!usize {
                const x = H.bind(p);
                return x.inner.vtable.alternatives.?(x.inner.ptr, s, out);
            }
        }.f else null,
        .prepare_features = if (v.prepare_features != null) struct {
            fn f(p: *anyopaque, taps: []const u32) anyerror!void {
                const x = H.bind(p);
                return x.inner.vtable.prepare_features.?(x.inner.ptr, taps);
            }
        }.f else null,
        .features = if (v.features != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
                const x = H.bind(p);
                return x.inner.vtable.features.?(x.inner.ptr, s, taps, start, count);
            }
        }.f else null,
        .first_row = if (v.first_row != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream) anyerror!lanes.LogprobRow {
                const x = H.bind(p);
                return x.inner.vtable.first_row.?(x.inner.ptr, s);
            }
        }.f else null,
        .first_masked = if (v.first_masked != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
                const x = H.bind(p);
                return x.inner.vtable.first_masked.?(x.inner.ptr, s, position, mask);
            }
        }.f else null,
        .held = if (v.held != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
                const x = H.bind(p);
                return x.inner.vtable.held.?(x.inner.ptr, s, out);
            }
        }.f else null,
        .prompt_rows = v.prompt_rows,
        .reports_fill = v.reports_fill, // the family's fill-progress reporting reaches the CUDA wrapper
        .release = struct {
            fn f(p: *anyopaque, s: *lanes.Stream) void {
                const x = H.bind(p);
                x.inner.vtable.release(x.inner.ptr, s);
            }
        }.f,
    };
}

test "every entry the family's backend gives is forwarded, and none it leaves out (a stepped prompt pass included)" {
    const H = struct {
        inner: be.Backend,
        pub fn bind(p: *anyopaque) *@This() {
            return @ptrCast(@alignCast(p));
        }
    };
    var fake: lanes.fake.Fake = .{ .gpa = std.testing.allocator, .prefill_chunks = 3 };
    defer fake.deinit();
    for ([_]be.Backend{ fake.backend(), fake.stepped(), fake.batchedSteps() }) |b| {
        const w = vtable(H, b.vtable.*);
        const info = @typeInfo(be.Backend.VTable).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (@typeInfo(T) == .optional) try std.testing.expectEqual(@field(b.vtable, name) != null, @field(w, name) != null);
            if (T == bool or @typeInfo(T) == .int) try std.testing.expectEqual(@field(b.vtable, name), @field(w, name)); // e.g. many_rows
        }
    }
    try std.testing.expect(vtable(H, fake.stepped().vtable.*).prefill_step != null);
}

test "each forwarded entry calls the family's entry of the same name, once" {
    const S = lanes.Stream;
    const Rec = struct {
        name: []const u8 = "",
        calls: usize = 0,
        fn hit(p: *anyopaque, name: []const u8) void {
            const r: *@This() = @ptrCast(@alignCast(p));
            r.name = name;
            r.calls += 1;
        }
        fn once(r: *@This(), name: []const u8) !void {
            try std.testing.expectEqualStrings(name, r.name);
            try std.testing.expectEqual(@as(usize, 1), r.calls);
            r.* = .{};
        }
    };
    const spy: be.Backend.VTable = .{
        .prefill = struct {
            fn f(p: *anyopaque, _: *S) anyerror!void {
                Rec.hit(p, "prefill");
            }
        }.f,
        .prefill_step = struct {
            fn f(p: *anyopaque, _: *S) anyerror!bool {
                Rec.hit(p, "prefill_step");
                return true;
            }
        }.f,
        .first = struct {
            fn f(p: *anyopaque, _: *S, _: u64) anyerror!u64 {
                Rec.hit(p, "first");
                return 0;
            }
        }.f,
        .queue = struct {
            fn f(p: *anyopaque, _: *S, _: be.Feed, _: u64) anyerror!u64 {
                Rec.hit(p, "queue");
                return 0;
            }
        }.f,
        .read = struct {
            fn f(p: *anyopaque, _: u64) anyerror!u32 {
                Rec.hit(p, "read");
                return 0;
            }
        }.f,
        .verify = struct {
            fn f(p: *anyopaque, _: []const be.Window, _: []be.Verified) anyerror!void {
                Rec.hit(p, "verify");
            }
        }.f,
        .keep = struct {
            fn f(p: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {
                Rec.hit(p, "keep");
            }
        }.f,
        .draft = struct {
            fn f(p: *anyopaque, _: []const be.DraftRequest) anyerror!void {
                Rec.hit(p, "draft");
            }
        }.f,
        .unspeculate = struct {
            fn f(p: *anyopaque, _: *S) anyerror!void {
                Rec.hit(p, "unspeculate");
            }
        }.f,
        .probabilities = struct {
            fn f(p: *anyopaque, _: *S, _: []f64) anyerror!bool {
                Rec.hit(p, "probabilities");
                return true;
            }
        }.f,
        .tree = struct {
            fn f(p: *anyopaque, _: *S, _: std.mem.Allocator) anyerror!?lanes.stream.Held {
                Rec.hit(p, "tree");
                return null;
            }
        }.f,
        .alternatives = struct {
            fn f(p: *anyopaque, _: *S, _: []be.Alternative) anyerror!usize {
                Rec.hit(p, "alternatives");
                return 0;
            }
        }.f,
        .prepare_features = struct {
            fn f(p: *anyopaque, _: []const u32) anyerror!void {
                Rec.hit(p, "prepare_features");
            }
        }.f,
        .features = struct {
            fn f(p: *anyopaque, _: *S, _: []const u32, _: u64, _: u32) anyerror!be.Features {
                Rec.hit(p, "features");
                return undefined;
            }
        }.f,
        .first_row = struct {
            fn f(p: *anyopaque, _: *S) anyerror!lanes.LogprobRow {
                Rec.hit(p, "first_row");
                return undefined;
            }
        }.f,
        .first_masked = struct {
            fn f(p: *anyopaque, _: *S, _: u64, _: []const u32) anyerror!u64 {
                Rec.hit(p, "first_masked");
                return 0;
            }
        }.f,
        .held = struct {
            fn f(p: *anyopaque, _: *S, _: []u32) anyerror!void {
                Rec.hit(p, "held");
            }
        }.f,
        .release = struct {
            fn f(p: *anyopaque, _: *S) void {
                Rec.hit(p, "release");
            }
        }.f,
    };
    const H = struct {
        inner: be.Backend,
        pub fn bind(p: *anyopaque) *@This() {
            return @ptrCast(@alignCast(p));
        }
    };
    var rec: Rec = .{};
    var h: H = .{ .inner = .{ .ptr = &rec, .vtable = &spy } };
    const w = vtable(H, spy);
    const p: *anyopaque = &h;
    var s: S = undefined;
    const feed: be.Feed = undefined;
    var verified: [0]be.Verified = .{};
    var probs: [0]f64 = .{};
    var alts: [0]be.Alternative = .{};
    var out: [0]u32 = .{};
    try w.prefill(p, &s);
    try rec.once("prefill");
    _ = try w.prefill_step.?(p, &s);
    try rec.once("prefill_step");
    _ = try w.first(p, &s, 0);
    try rec.once("first");
    _ = try w.queue(p, &s, feed, 0);
    try rec.once("queue");
    _ = try w.read(p, 0);
    try rec.once("read");
    try w.verify(p, &.{}, &verified);
    try rec.once("verify");
    try w.keep(p, &.{}, &.{});
    try rec.once("keep");
    try w.draft(p, &.{});
    try rec.once("draft");
    try w.unspeculate.?(p, &s);
    try rec.once("unspeculate");
    _ = try w.probabilities.?(p, &s, &probs);
    try rec.once("probabilities");
    _ = try w.tree.?(p, &s, std.testing.allocator);
    try rec.once("tree");
    _ = try w.alternatives.?(p, &s, &alts);
    try rec.once("alternatives");
    try w.prepare_features.?(p, &.{});
    try rec.once("prepare_features");
    _ = try w.features.?(p, &s, &.{}, 0, 0);
    try rec.once("features");
    _ = try w.first_row.?(p, &s);
    try rec.once("first_row");
    _ = try w.first_masked.?(p, &s, 0, &.{});
    try rec.once("first_masked");
    try w.held.?(p, &s, &out);
    try rec.once("held");
    w.release(p, &s);
    try rec.once("release");
}
