//! Disk restore's operative state bytes against a fresh host state through real plans, leases and metadata restore.
const std = @import("std");
const cuda = @import("cuda");
const image = @import("cuda_disk_image.zig");
const fixture = @import("cuda_disk_image_test.zig");
const helper = @import("cuda_disk_half_test.zig");
const layout = @import("cuda_snapshot_layout.zig");
const snaps = @import("cuda_snapshots.zig");
const types = @import("cuda_engine_types.zig");
const State = @import("cuda_state.zig").State;
const t = std.testing;

const Fake = struct {
    gpa: std.mem.Allocator = t.allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    c: @import("config.zig").Config,
    cur: struct { st: State, in_head: usize = 0 },
    capacity: usize = helper.cap,
    mtp: bool,
    share_pages: bool = false,
    snaps: std.AutoHashMapUnmanaged(u64, types.Snap) = .empty,
    snap_pages: ?snaps.pages.Pool = null,

    fn deinit(e: *Fake) void {
        var it = e.snaps.valueIterator();
        while (it.next()) |s| snaps.free(e, s);
        e.snaps.deinit(e.gpa);
        e.snap_pages.?.deinit();
        e.cur.st.deinit();
    }
};

fn fill32(ptr: u64, value: c_uint, n: usize, _: cuda.abi.Stream) callconv(.c) c_int {
    const out = @as([*]u32, @ptrFromInt(ptr))[0..n];
    @memset(out, value);
    return 0;
}

fn bytes(ptr: u64, n: usize) []u8 {
    return @as([*]u8, @ptrFromInt(ptr))[0..n];
}

test "GLM live-half disk: restored operative bytes and metadata equal fresh across two ranks and parities" {
    var d = fixture.H.driver();
    d.api.cuMemsetD32Async = fill32;
    for ([_]bool{ false, true }) |half| {
        const mode = try helper.Mode.init(half);
        defer mode.deinit();
        for (0..2) |rank| for ([_]u1{ 0, 1 }) |h| for ([_]bool{ false, true }) |head| {
            var m: fixture.Model = undefined;
            try helper.model(&m, rank);
            var fresh = try State.initFormat(t.allocator, &d, &m.w, helper.cap, false);
            defer fresh.deinit();
            @memset(fresh.cur, h);
            const at: u32 = 32;
            const mtp: u32 = if (head) 31 else 0;
            const sh = try image.Shape.of(&fresh, &m.w.config, helper.cap, head);
            const plan = try layout.Plan.init(&fresh, &m.w.config, at, if (head) mtp else null, 0);
            for (plan.parts[0..plan.count], 0..) |part, j| for (bytes(part.live, part.len), 0..) |*b, i| {
                b.* = @truncate(j *% 17 +% i *% 31 +% 7);
            };
            const fresh_bits: [2]u1 = @splat(h);
            try fresh.restoreHost(.{ .d = &d, .handle = null }, at, mtp, &fresh_bits, .{ 0, 0 });
            const packed_image = try helper.packImage(t.allocator, &d, &fresh, &m.w.config, at, mtp, sh, if (half) h else null);
            defer t.allocator.free(packed_image);
            var e: Fake = .{ .d = &d, .stream = .{ .d = &d, .handle = null }, .c = m.w.config, .mtp = head, .cur = .{ .st = try State.initFormat(t.allocator, &d, &m.w, helper.cap, false) } };
            e.snap_pages = try snaps.pages.Pool.init(t.allocator, &d, 8 << 20, false);
            defer e.deinit();
            try e.snaps.ensureTotalCapacity(t.allocator, 1);
            @memset(bytes(e.cur.st.arena.buf.ptr, e.cur.st.arena.buf.len), 0xa9);
            e.cur.st.cur[0] = 1 - h;
            e.cur.st.cur[1] = h;
            try snaps.blankAs(&e, 7, at);
            var words: [image.Words.count]u64 = undefined;
            @memcpy(std.mem.sliceAsBytes(&words), packed_image[0..image.Words.bytes]);
            const decoded = try image.Words.decodeForm(words, at, 2, .unsplit, sh.form);
            try snaps.seal(&e, 7, decoded.mtp_len, .{ decoded.base, decoded.mtp_base }, decoded.cur);
            var segs: std.ArrayList(image.Seg) = .empty;
            defer segs.deinit(t.allocator);
            try image.segments(sh, at, mtp, &segs, t.allocator);
            const saved = e.snaps.getPtr(7).?;
            var incoming: [image.Words.count]u64 = undefined;
            try (image.Copy{ .d = &d, .pool = &e.snap_pages.?, .lease = &saved.lease, .segs = segs.items, .words = &incoming }).run(0, packed_image, false);
            try snaps.restore(&e, 7);
            const restored = try layout.Plan.initFor(&e.cur.st, &e.c, at, if (head) mtp else null, 0, fresh.cur);
            try t.expectEqual(plan.len, restored.len);
            try t.expectEqual(@as(usize, at), e.cur.st.pos);
            try t.expectEqual(@as(usize, mtp), e.cur.st.mtp_len);
            try t.expectEqualSlices(u1, fresh.cur, e.cur.st.cur);
            for (plan.parts[0..plan.count], restored.parts[0..restored.count]) |a, b| {
                try t.expectEqual(a.len, b.len);
                try t.expectEqualSlices(u8, bytes(a.live, a.len), bytes(b.live, b.len));
            }
            if (half) {
                const dead = e.cur.st.rec + @as(u64, 1 - h) * e.cur.st.kda_layers * e.cur.st.rec_layer;
                for (bytes(dead, e.cur.st.kda_layers * e.cur.st.rec_layer)) |b| try t.expectEqual(@as(u8, 0xa9), b);
            }
        };
    }
}
