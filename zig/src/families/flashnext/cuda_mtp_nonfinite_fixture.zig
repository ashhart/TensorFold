//! Recording CUDA row outputs and completion boundaries; no device or checkpoint.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mtp = @import("cuda_mtp.zig");
const draw = @import("cuda_mtp_draw.zig");
const wt = @import("cuda_prompt_wt.zig");

pub const Device = struct {
    pub var batch_host: [@import("cuda_draw_batch.zig").bytes]u8 align(8) = undefined;
    pub var best: [4]f32 = undefined;
    pub var reductions: []const [4]f32 = &.{};
    pub var vals: []const u16 = &.{ 0x3f80, 0 };
    pub var cols: []const i32 = &.{ 0, 1 };
    pub var head_row: []const u16 = &.{ 0x3f80, 0, 0xbf80 };
    pub var verify_rows: []const u16 = &.{};
    pub var launches: usize = 0;
    pub var waits: usize = 0;
    pub var grids: [512]u32 = undefined;
    pub var functions: [512]usize = undefined;
    pub var sources: [mtp.max_rows]u64 = undefined;
    pub var best_calls: usize = 0;
    pub var best_bytes: usize = 0;
    pub var pending: bool = false;
    pub var top_k: usize = 0;
    pub var single_vals: ?[]const u16 = null;
    pub var single_cols: ?[]const i32 = null;
    pub var reads: usize = 0;
    pub var copies: usize = 0;
    pub var copy_src: [512]u64 = undefined;
    pub var copy_dst: [512]u64 = undefined;
    pub var copy_bytes: [512]usize = undefined;
    pub const Event = enum { best, topk, wait, read_best, read_vals, read_cols, read_row, read_batch };
    pub var events: [512]Event = undefined;
    pub var events_n: usize = 0;

    fn event(e: Event) void {
        if (events_n < events.len) {
            events[events_n] = e;
            events_n += 1;
        }
    }
    pub fn reset() void {
        launches = 0;
        waits = 0;
        best_calls = 0;
        best_bytes = 0;
        pending = false;
        reads = 0;
        copies = 0;
        events_n = 0;
    }
    fn launch(f: cuda.abi.Function, gx: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        grids[launches] = gx;
        functions[launches] = @intFromPtr(f);
        launches += 1;
        pending = true;
        if (@intFromPtr(f) == 1) {
            event(.best);
            sources[best_calls] = std.mem.bytesToValue(u64, @as([*]const u8, @ptrCast(args.?[0].?))[0..8]);
            best_calls += 1;
        } else {
            event(.topk);
            top_k = @intCast(std.mem.bytesToValue(c_int, @as([*]const u8, @ptrCast(args.?[3].?))[0..4]));
        }
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) c_int {
        waits += 1;
        pending = false;
        event(.wait);
        return 0;
    }
    fn read(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) c_int {
        reads += 1;
        std.debug.assert(!pending);
        const bytes = @as([*]u8, @ptrCast(dst.?))[0..n];
        const tv = 0x1000 + mtp.max_rows * 16;
        const tc = tv + mtp.max_rows * draw.topk_max * 2;
        if (src == 0x9000) {
            event(.read_batch);
            const b = @import("cuda_draw_batch.zig");
            std.debug.assert(n == b.bytes);
            @memset(bytes, 0);
            for (0..b.rows_max) |i| @memcpy(bytes[i * 16 ..][0..16], std.mem.asBytes(if (reductions.len > i) &reductions[i] else &best));
            @memcpy(bytes[b.vals_at..][0 .. vals.len * 2], std.mem.sliceAsBytes(vals));
            @memcpy(bytes[b.cols_at..][0 .. cols.len * 4], std.mem.sliceAsBytes(cols));
        } else if (src == 0x1000 and n > mtp.max_rows * 16) {
            event(.read_batch);
            @memset(bytes, 0);
            const rows = mtp.max_rows;
            for (0..rows) |i| @memcpy(bytes[i * 16 ..][0..16], std.mem.asBytes(if (reductions.len > i) &reductions[i] else &best));
            const va = rows * 16;
            const ca = va + rows * draw.topk_max * 2;
            @memcpy(bytes[va..][0..@min(vals.len * 2, ca - va)], std.mem.sliceAsBytes(vals)[0..@min(vals.len * 2, ca - va)]);
            @memcpy(bytes[ca..][0..@min(cols.len * 4, n - ca)], std.mem.sliceAsBytes(cols)[0..@min(cols.len * 4, n - ca)]);
        } else if (src == 0x1000) {
            event(.read_best);
            best_bytes = n;
            std.debug.assert(n % 16 == 0);
            for (0..n / 16) |i| @memcpy(bytes[i * 16 ..][0..16], std.mem.asBytes(if (reductions.len > i) &reductions[i] else &best));
        } else if (src == tv) {
            event(.read_vals);
            const v = if (single_vals) |v| (if (n <= v.len * 2) v else vals) else vals;
            std.debug.assert(n <= v.len * 2);
            @memcpy(bytes, std.mem.sliceAsBytes(v)[0..n]);
        } else if (src == tc) {
            event(.read_cols);
            const v = if (single_cols) |v| (if (n <= v.len * 4) v else cols) else cols;
            std.debug.assert(n <= v.len * 4);
            @memcpy(bytes, std.mem.sliceAsBytes(v)[0..n]);
        } else if (src == 0x3000 and n <= head_row.len * 2) {
            event(.read_row);
            @memcpy(bytes, std.mem.sliceAsBytes(head_row)[0..n]);
        } else if (src >= 0x4000 and src - 0x4000 + n <= verify_rows.len * 2) {
            event(.read_row);
            @memcpy(bytes, std.mem.sliceAsBytes(verify_rows)[src - 0x4000 ..][0..n]);
        } else @memset(bytes, 0);
        return 0;
    }
    fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        copy_dst[copies] = dst;
        copy_src[copies] = src;
        copy_bytes[copies] = n;
        copies += 1;
        return 0;
    }
    fn context(out: *c_int) callconv(.c) c_int {
        out.* = 0;
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) c_int {
        out.* = 0;
        return 0;
    }
    pub fn driver() cuda.Driver {
        cuda.download_stage.forget();
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuLaunchKernel = launch;
        d.api.cuStreamSynchronize = wait;
        d.api.cuMemcpyDtoH_v2 = read;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuCtxGetDevice = context;
        d.api.cuDeviceGetAttribute = attribute;
        single_vals = null;
        single_cols = null;
        reductions = &.{};
        vals = &.{ 0x3f80, 0 };
        cols = &.{ 0, 1 };
        head_row = &.{ 0x3f80, 0, 0xbf80 };
        verify_rows = &.{};
        best = reduction(1, 2, 0);
        reset();
        return d;
    }
};

pub fn reduction(value: f32, lse: f32, col: i32) [4]f32 {
    return .{ value, lse, @bitCast(col), 0 };
}
pub fn draft(d: *cuda.Driver, host: []u16) mtp.Draft {
    var x: mtp.Draft = undefined;
    x.label = "fixture-stream";
    x.len = 2051;
    x.cap = 4096;
    x.pg = .{};
    x.drafted = 0;
    x.pooled_lane = false;
    x.host = host;
    x.rows = 1;
    x.fed0 = 2050;
    x.best = .{ .d = d, .ptr = 0x1000, .len = @import("cuda_mtp_draft.zig").best_bytes };
    x.out = .{ .d = d, .ptr = 0x2000, .len = wt.wide * 2 };
    x.dlogits = .{ .d = d, .ptr = 0x3000, .len = 6 };
    x.logits = .{ .d = d, .ptr = 0x4000, .len = wt.vocab * 2 };
    return x;
}
pub fn head(d: *cuda.Driver) !mtp.Head {
    return headIds(d, &.{ 0, 3, 7 });
}
pub fn headIds(d: *cuda.Driver, ids: []const u32) !mtp.Head {
    const gpa = std.testing.allocator;
    var h: mtp.Head = undefined;
    h.artifact = try core.proposal_vocab.legacy.artifact(gpa, ids, .{ .vocab = wt.vocab, .hidden = wt.dims, .format = "bf16", .group = 0, .scale_recipe = "fixture" });
    h.ids = h.artifact.ids();
    h.service = try @import("lanes").proposal_head.Service.legacy(gpa, &h.artifact);
    h.draw_batch = .{ .scores = .{ .d = d, .ptr = 0xA000, .len = @import("cuda_draft_draw.zig").scoreBytes(ids.len) }, .device = .{ .d = d, .ptr = 0x9000, .len = @import("cuda_draw_batch.zig").bytes }, .host = &Device.batch_host, .on = true };
    h.best_fn = .{ .d = d, .handle = @ptrFromInt(1) };
    h.topk_fn = .{ .d = d, .handle = @ptrFromInt(2) };
    return h;
}
