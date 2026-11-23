//! LFM2.5-24B-A2B on CUDA behind lanes.Backend: liquid's lane backend over this family's model and forward.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const liquid = @import("liquid_cuda");
const snap = liquid.snap;

const B = liquid.lanes.Backend(struct {
    pub const cfg = @import("config.zig");
    pub const fwd = @import("cuda_forward.zig");
    pub const mdl = @import("cuda_model.zig");
    pub const kern = @import("cuda_kernels.zig");
    pub const snap = @import("cuda_snap.zig");
    pub const text_only = error.LfmTextOnly;
    pub const request_starts = true;
});

pub const Liquid = B.Liquid;
pub const prompt_rows = B.prompt_rows;
pub const round_rows = B.round_rows;
pub const max_window = B.max_window;
pub const pagesFor = B.pagesFor;

const Waits = struct {
    var n: usize = 0;
    var on: usize = 0;
    var of: usize = 0;
    fn wait(st: cuda.abi.Stream, ev: cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        n += 1;
        on = @intFromPtr(st);
        of = @intFromPtr(ev);
        return 0;
    }
};

test "fenced: the prompt stream waits for the keeps' done once a keep is queued, and not again until the next" {
    var d: cuda.Driver = undefined;
    d.api.cuStreamWaitEvent = Waits.wait;
    var x: Liquid = undefined;
    x.s = .{ .d = &d, .handle = @ptrFromInt(1) };
    x.snaps = null;
    Waits.n = 0;
    try std.testing.expectEqual(x.s.handle, (try x.fenced()).handle); // no prompt cache: nothing to wait for
    const side: snap.Side = .{ .stream = .{ .d = &d, .handle = @ptrFromInt(2) }, .ready = .{ .d = &d, .handle = @ptrFromInt(3) }, .done = .{ .d = &d, .handle = @ptrFromInt(4) }, .pending = true };
    x.snaps = .{ .pages = try kv.Pool.init(std.testing.allocator, 1, snap.page), .arena = null, .side = side };
    defer x.snaps.?.pages.deinit();
    try std.testing.expectEqual(@as(usize, 0), Waits.n);
    try std.testing.expectEqual(x.s.handle, (try x.fenced()).handle);
    try std.testing.expect(Waits.n == 1 and Waits.on == 1 and Waits.of == 4);
    _ = try x.fenced(); // nothing queued since: no second wait
    try std.testing.expectEqual(@as(usize, 1), Waits.n);
}
