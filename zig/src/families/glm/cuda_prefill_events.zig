//! Poll-only, same-stream timing; ownership never inserts a wait or a synchronization.
const std = @import("std");
const abi = @import("cuda").abi;
pub const Pair = struct {
    first: abi.Event,
    last: abi.Event,
    pub fn init(api: *const abi.Api, stream: abi.Stream) ?Pair {
        var p: Pair = undefined;
        if (api.cuEventCreate(&p.first, 0) != 0) return null;
        if (api.cuEventCreate(&p.last, 0) != 0) {
            _ = api.cuEventDestroy_v2(p.first);
            return null;
        }
        if (api.cuEventRecord(p.first, stream) != 0) {
            p.deinit(api);
            return null;
        }
        return p;
    }
    pub fn stop(p: Pair, api: *const abi.Api, stream: abi.Stream) bool {
        return api.cuEventRecord(p.last, stream) == 0;
    }
    pub fn elapsed(p: Pair, api: *const abi.Api) ?u64 {
        if (api.cuEventQuery(p.last) != 0) return null;
        var ms: f32 = 0;
        if (api.cuEventElapsedTime(&ms, p.first, p.last) != 0) return null;
        if (!std.math.isFinite(ms) or ms < 0 or ms > 1e12) return null;
        return @intFromFloat(@as(f64, ms) * 1e6);
    }
    pub const Measurement = struct { elapsed_ns: u64, begin_ns: u64, end_ns: u64 };
    fn since(api: *const abi.Api, origin: abi.Event, event: abi.Event) ?u64 {
        if (origin == event) return 0;
        var ms: f32 = 0;
        if (api.cuEventElapsedTime(&ms, origin, event) != 0) return null;
        if (!std.math.isFinite(ms) or ms < 0 or ms > 1e12) return null;
        return @intFromFloat(@as(f64, ms) * 1e6);
    }
    pub fn measure(p: Pair, api: *const abi.Api, origin: abi.Event) ?Measurement {
        const ns = p.elapsed(api) orelse return null;
        const begin = since(api, origin, p.first) orelse return null;
        const end = since(api, origin, p.last) orelse return null;
        if (end < begin) return null;
        return .{ .elapsed_ns = ns, .begin_ns = begin, .end_ns = end };
    }
    pub fn deinit(p: Pair, api: *const abi.Api) void {
        _ = api.cuEventDestroy_v2(p.last);
        _ = api.cuEventDestroy_v2(p.first);
    }
};
