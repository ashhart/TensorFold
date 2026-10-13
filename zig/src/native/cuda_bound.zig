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
    for ([_]be.Backend{ fake.backend(), fake.stepped() }) |b| {
        const w = vtable(H, b.vtable.*);
        const info = @typeInfo(be.Backend.VTable).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (@typeInfo(T) == .optional) try std.testing.expectEqual(@field(b.vtable, name) != null, @field(w, name) != null);
            if (T == bool) try std.testing.expectEqual(@field(b.vtable, name), @field(w, name));
        }
    }
    try std.testing.expect(vtable(H, fake.stepped().vtable.*).prefill_step != null);
}
