//! Recording CUDA row outputs and completion boundaries; no device or checkpoint.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mtp = @import("cuda_mtp.zig");
const draw = @import("cuda_mtp_draw.zig");
const wt = @import("cuda_prompt_wt.zig");

pub const Device = struct {
    pub var best: [4]f32 = undefined;
    pub var reductions: []const [4]f32 = &.{};
    pub var vals: []const u16 = &.{ 0x3f80, 0 };
    pub var cols: []const i32 = &.{ 0, 1 };
    pub var head_row: []const u16 = &.{ 0x3f80, 0, 0xbf80 };
    pub var verify_rows: []const u16 = &.{};
    pub var launches: usize = 0;
    pub var waits: usize = 0;
    pub var sources: [mtp.max_rows]u64 = undefined;
    pub var best_calls: usize = 0;
    pub var best_bytes: usize = 0;
    pub var pending: bool = false;
    pub const Event = enum { best, topk, wait, read_best, read_vals, read_cols, read_row };
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
        events_n = 0;
    }
    fn launch(f: cuda.abi.Function, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        launches += 1;
        pending = true;
        if (@intFromPtr(f) == 1) {
            event(.best);
            sources[best_calls] = std.mem.bytesToValue(u64, @as([*]const u8, @ptrCast(args.?[0].?))[0..8]);
            best_calls += 1;
        } else event(.topk);
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) c_int {
        waits += 1;
        pending = false;
        event(.wait);
        return 0;
    }
    fn read(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) c_int {
        std.debug.assert(!pending);
        const bytes = @as([*]u8, @ptrCast(dst.?))[0..n];
        const tv = 0x1000 + mtp.max_rows * 16;
        const tc = tv + mtp.max_rows * draw.topk_max * 2;
        if (src == 0x1000) {
            event(.read_best);
            best_bytes = n;
            std.debug.assert(n % 16 == 0);
            for (0..n / 16) |i| @memcpy(bytes[i * 16 ..][0..16], std.mem.asBytes(if (reductions.len > i) &reductions[i] else &best));
        } else if (src == tv) {
            event(.read_vals);
            std.debug.assert(n <= vals.len * 2);
            @memcpy(bytes, std.mem.sliceAsBytes(vals)[0..n]);
        } else if (src == tc) {
            event(.read_cols);
            std.debug.assert(n <= cols.len * 4);
            @memcpy(bytes, std.mem.sliceAsBytes(cols)[0..n]);
        } else if (src == 0x3000 and n <= head_row.len * 2) {
            event(.read_row);
            @memcpy(bytes, std.mem.sliceAsBytes(head_row)[0..n]);
        } else if (src >= 0x4000 and src - 0x4000 + n <= verify_rows.len * 2) {
            event(.read_row);
            @memcpy(bytes, std.mem.sliceAsBytes(verify_rows)[src - 0x4000 ..][0..n]);
        } else @memset(bytes, 0);
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
        d.api.cuCtxGetDevice = context;
        d.api.cuDeviceGetAttribute = attribute;
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
    const gpa = std.testing.allocator;
    var h: mtp.Head = undefined;
    h.artifact = try core.proposal_vocab.legacy.artifact(gpa, &.{ 0, 3, 7 }, .{ .vocab = wt.vocab, .hidden = wt.dims, .format = "bf16", .group = 0, .scale_recipe = "fixture" });
    h.ids = h.artifact.ids();
    h.service = try @import("lanes").proposal_head.Service.legacy(gpa, &h.artifact);
    h.best_fn = .{ .d = d, .handle = @ptrFromInt(1) };
    h.topk_fn = .{ .d = d, .handle = @ptrFromInt(2) };
    return h;
}
