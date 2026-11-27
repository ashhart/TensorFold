//! Live-half disk geometry and canonical bytes over the real state layout and host-memory CUDA copies.
const std = @import("std");
const cuda = @import("cuda");
const sys = @import("sys");
const image = @import("cuda_disk_image.zig");
const fixture = @import("cuda_disk_image_test.zig");
const layout = @import("cuda_snapshot_layout.zig");
const State = @import("cuda_state.zig").State;
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const pages = @import("cuda_snapshot_pages.zig");
const t = std.testing;
pub const cap = 64;

pub const Mode = struct {
    saved: ?[:0]u8,
    pub fn init(on: bool) !Mode {
        const old = sys.getenv("TF_GLM_KV_LIVE_HALF");
        const m: Mode = .{ .saved = if (old) |s| try std.fmt.allocPrintSentinel(t.allocator, "{s}", .{s}, 0) else null };
        _ = sys.env.setenv("TF_GLM_KV_LIVE_HALF", if (on) "1" else "0", 1);
        return m;
    }
    pub fn deinit(m: Mode) void {
        if (m.saved) |s| {
            _ = sys.env.setenv("TF_GLM_KV_LIVE_HALF", s, 1);
            t.allocator.free(s);
        } else _ = sys.env.unsetenv("TF_GLM_KV_LIVE_HALF");
    }
};

pub fn model(m: *fixture.Model, rank: usize) !void {
    try m.init();
    m.w.config.hidden = 16;
    m.w.config.kda_heads = 3;
    m.w.config.kv_lora = 8;
    m.w.config.i_dim = 8;
    m.w.share = try shard.Share.of(&m.w.config, rank, 2);
}

fn bytes(ptr: u64, n: usize) []u8 {
    return @as([*]u8, @ptrFromInt(ptr))[0..n];
}

pub fn packImage(a: std.mem.Allocator, d: *cuda.Driver, st: *const State, c: *const @import("config.zig").Config, at: u32, mtp: u32, sh: image.Shape, half: ?u1) ![]u8 {
    const plan = try layout.Plan.withHalf(st, c, at, if (sh.head) mtp else null, 0, half);
    var pool = try pages.Pool.init(a, d, 8 << 20, false);
    defer pool.deinit();
    var lease = try pool.take(plan.len);
    defer pool.give(&lease);
    try pool.move(.{ .d = d, .handle = null }, &lease, &plan, true);
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(a);
    try image.segments(sh, at, if (sh.head) mtp else 0, &segs, a);
    const out = try a.alloc(u8, @intCast(image.imageBytes(sh, at)));
    errdefer a.free(out);
    var cur: u64 = 0;
    for (st.cur, 0..) |b, i| cur |= @as(u64, b) << @intCast(i);
    var w = (image.Words{ .at = at, .mtp_len = if (sh.head) mtp else 0, .base = 0, .mtp_base = 0, .cur = cur, .world = 2, .form = sh.form }).encode();
    try (image.Copy{ .d = d, .pool = &pool, .lease = &lease, .segs = segs.items, .words = &w }).run(0, out, true);
    return out;
}

test "GLM live-half disk: both rank prefixes and windows remove exactly the dead KDA half" {
    var d = fixture.H.driver();
    for (0..2) |rank| {
        var m: fixture.Model = undefined;
        try model(&m, rank);
        for ([_]bool{ false, true }) |fp8| for ([_]usize{ 64, 6144 }) |capacity| {
            if (fp8) m.w.config.kv_lora = 512;
            var st = try State.initFormat(t.allocator, &d, &m.w, capacity, fp8);
            defer st.deinit();
            for ([_]bool{ false, true }) |head| {
                const whole = try image.Shape.withForm(&st, &m.w.config, capacity, head, .whole);
                const half = try image.Shape.withForm(&st, &m.w.config, capacity, head, .live_half);
                const dead = st.kda_layers * st.rec_layer;
                try t.expect(dead > 0);
                try t.expectEqual(whole.prefix - dead, half.prefix);
                try t.expectEqual(whole.window - dead, half.window);
                try t.expectEqual(whole.prefix - whole.window, half.prefix - half.window);
                for ([_]u32{ 4, 32, @intCast(@min(2048, capacity)), @intCast(@min(4096, capacity)), @intCast(capacity) }) |at| for ([_]u1{ 0, 1 }) |h| {
                    const mtp = @min(at + 7, capacity);
                    const p = try layout.Plan.withHalf(&st, &m.w.config, at, if (head) mtp else null, 0, h);
                    try t.expectEqual(@as(u64, p.len), image.leaseBytes(half, at, if (head) mtp else 0));
                    try t.expectEqual(image.imageBytes(whole, at) - dead, image.imageBytes(half, at));
                };
            }
        };
    }
}

test "GLM live-half disk: the first header word names the form and wrong forms refuse by name" {
    const old: image.Words = .{ .at = 32, .mtp_len = 31, .base = 0, .mtp_base = 0, .cur = 3, .world = 2 };
    const original = [_]u64{ 1, 32, 31, 0, 0, 3, 2, 0, std.math.maxInt(u64) };
    try t.expectEqualSlices(u64, &original, &old.encode());
    var half = old;
    half.form = .live_half;
    const x = half.encode();
    try t.expectEqual(@as(u64, 2), x[0]);
    try t.expectEqualSlices(u64, original[1..], x[1..]);
    try t.expectError(error.DiskKdaForm, image.Words.decode(x, 32, 2, .unsplit));
    try t.expectError(error.DiskKdaForm, image.Words.decodeForm(original, 32, 2, .unsplit, .live_half));
    try t.expectEqual(half, try image.Words.decodeForm(x, 32, 2, .unsplit, .live_half));
    try t.expectError(error.DiskWords, image.Words.checkForm(3, .whole));
}

test "GLM live-half disk: canonical packed bytes do not depend on which half was live" {
    var d = fixture.H.driver();
    var m: fixture.Model = undefined;
    try model(&m, 0);
    var st = try State.initFormat(t.allocator, &d, &m.w, cap, false);
    defer st.deinit();
    const sh = try image.Shape.withForm(&st, &m.w.config, cap, true, .live_half);
    const n = st.kda_layers * st.rec_layer;
    var keep: ?[]u8 = null;
    defer if (keep) |p| t.allocator.free(p);
    for ([_]u1{ 0, 1 }) |h| {
        @memset(st.cur, h);
        @memset(bytes(st.arena.buf.ptr, st.arena.buf.len), 0x31);
        @memset(bytes(st.rec + @as(u64, 1 - h) * n, n), 0xe7);
        const out = try packImage(t.allocator, &d, &st, &m.w.config, 32, 31, sh, h);
        if (keep) |p| {
            defer t.allocator.free(out);
            try t.expectEqualSlices(u8, p[image.Words.bytes..], out[image.Words.bytes..]);
        } else keep = out;
    }
}

test "GLM live-half disk: two uneven rank prefixes route shared bytes and unwritten windows at the half boundary" {
    var d = fixture.H.driver();
    var shapes: [2]image.Shape = undefined;
    for (&shapes, 0..) |*sh, rank| {
        var m: fixture.Model = undefined;
        try model(&m, rank);
        var st = try State.initFormat(t.allocator, &d, &m.w, cap, false);
        defer st.deinit();
        sh.* = try image.Shape.withForm(&st, &m.w.config, cap, true, .live_half);
    }
    const prefixes = [_]u64{ shapes[0].prefix, shapes[1].prefix };
    const windows = [_]u64{ shapes[0].window, shapes[1].window };
    try t.expect(prefixes[0] != prefixes[1]);
    const own = image.imageBytes(shapes[0], 32);
    var pieces: std.ArrayList(image.Piece) = .empty;
    defer pieces.deinit(t.allocator);
    try image.route(shapes[0], &prefixes, 32, image.Words.bytes + prefixes[0] - 1, 2, &pieces, t.allocator);
    try t.expectEqual(@as(usize, 2), pieces.items.len);
    try t.expect(!pieces.items[0].shared and pieces.items[1].shared);
    pieces.clearRetainingCapacity();
    try image.route(shapes[0], &prefixes, 32, own, prefixes[1], &pieces, t.allocator);
    try t.expectEqual(image.Words.bytes, pieces.items[0].off);
    try t.expectEqual(@as(u32, 1), pieces.items[0].rank);
    var loose: std.ArrayList(image.Span) = .empty;
    defer loose.deinit(t.allocator);
    try image.unwritten(shapes[0], &prefixes, &windows, 32, 31, &loose, t.allocator);
    try t.expectEqual(image.Words.bytes + windows[0], loose.items[0].off);
    try t.expectEqual(own + windows[1], loose.items[loose.items.len - 1].off);
}

test "GLM live-half disk: paged sequences use the same half-aware prefix and window counts" {
    var d = fixture.H.driver();
    for ([_]bool{ false, true }) |half| {
        const mode = try Mode.init(half);
        defer mode.deinit();
        for (0..2) |rank| {
            var m: fixture.Model = undefined;
            try model(&m, rank);
            var st = try State.initPaged(t.allocator, &d, &m.w, 4096, .{ .kv = 0x200000000, .pages = 8, .page_bytes = @import("cuda_state.zig").pageBytes(&m.w) });
            defer st.deinit();
            const sh = try image.Shape.of(&st, &m.w.config, 4096, true);
            try t.expect(st.table != 0);
            for ([_]u32{ 2048, 4096 }) |at| {
                const n = try layout.Plan.size(&st, &m.w.config, 4096, at, at, 0);
                try t.expectEqual(@as(u64, n), image.leaseBytes(sh, at, at));
            }
        }
    }
}
