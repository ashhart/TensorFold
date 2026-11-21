const cuda = @import("cuda");

pub const Fixture = struct {
    pub const Call = enum { query, wait, sync, copy };
    pub var calls: [128]Call = undefined;
    pub var count: usize = 0;
    pub var query_result: cuda.abi.Result = cuda.abi.error_not_ready;
    pub var wait_result: cuda.abi.Result = cuda.abi.success;
    pub var sync_result: cuda.abi.Result = cuda.abi.success;
    pub var last_event: cuda.abi.Event = null;

    fn note(call: Call, event: cuda.abi.Event) void {
        calls[count] = call;
        count += 1;
        if (event != null) last_event = event;
    }

    fn query(event: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        note(.query, event);
        return query_result;
    }

    fn wait(_: cuda.abi.Stream, event: cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        note(.wait, event);
        return wait_result;
    }

    fn sync(event: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        note(.sync, event);
        return sync_result;
    }

    fn copy(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        note(.copy, null);
        return cuda.abi.success;
    }

    pub fn init(driver: *cuda.Driver, ring: anytype, epochs: []u64, epoch: u64) void {
        count = 0;
        query_result = cuda.abi.error_not_ready;
        wait_result = cuda.abi.success;
        sync_result = cuda.abi.success;
        last_event = null;
        driver.* = undefined;
        driver.api.cuEventQuery = query;
        driver.api.cuStreamWaitEvent = wait;
        driver.api.cuEventSynchronize = sync;
        driver.api.cuMemcpyHtoDAsync_v2 = copy;
        ring.* = undefined;
        ring.copy = .{ .d = driver, .handle = @ptrFromInt(1) };
        ring.epoch_of = epochs;
        ring.epoch = epoch;
        ring.copy_ordered_epoch = 0;
        ring.stats = .{};
        for (&ring.rev, 0..) |*event, index| event.* = .{ .d = driver, .handle = @ptrFromInt(index + 1) };
    }
};
