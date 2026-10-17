//! Track wrapper captures by driver, context and stream across submitting threads.
const custody_sync = @import("copy_source_sync.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const Scope = struct { driver: *const Driver, context: abi.Context, stream: abi.Stream, mode: abi.CaptureMode };
var scopes: [64]?Scope = @splat(null);
var mutex: custody_sync.Mutex = .{};
pub fn begin(stream: Stream, context: abi.Context, mode: abi.CaptureMode) error{ CopySourceLimit, UnownedCaptureSource }!void {
    mutex.lock();
    defer mutex.unlock();
    for (scopes) |scope| if (scope) |s| {
        if (s.driver == stream.d and s.context == context and s.stream == stream.handle) return error.UnownedCaptureSource;
    };
    for (&scopes) |*scope| if (scope.* == null) {
        scope.* = .{ .driver = stream.d, .context = context, .stream = stream.handle, .mode = mode };
        return;
    };
    return error.CopySourceLimit;
}
pub fn end(stream: Stream, context: abi.Context) void {
    mutex.lock();
    defer mutex.unlock();
    for (&scopes) |*scope| if (scope.*) |s| {
        if (s.driver == stream.d and s.context == context and s.stream == stream.handle) {
            scope.* = null;
            return;
        }
    };
}
pub fn active(d: *const Driver, context: abi.Context) bool {
    mutex.lock();
    defer mutex.unlock();
    for (scopes) |scope| if (scope) |s| {
        if (s.driver == d and (s.context == context or s.mode == .global)) return true;
    };
    return false;
}
