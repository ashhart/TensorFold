//! Shared attribution event pairs: poll completion, never wait or insert a dependency into application work.
const abi = @import("abi.zig");
pub const Pair = struct {
    begin: abi.Event,
    end: abi.Event,
    pub fn init(api: *const abi.Api, stream: abi.Stream) ?Pair {
        var p: Pair = .{ .begin = null, .end = null };
        if (api.cuEventCreate(&p.begin, 0) != 0) return null;
        if (api.cuEventCreate(&p.end, 0) != 0) {
            _ = api.cuEventDestroy_v2(p.begin);
            return null;
        }
        if (api.cuEventRecord(p.begin, stream) != 0) {
            p.deinit(api);
            return null;
        }
        return p;
    }
    pub fn stop(p: Pair, api: *const abi.Api, stream: abi.Stream) bool {
        return api.cuEventRecord(p.end, stream) == 0;
    }
    pub fn elapsed(p: Pair, api: *const abi.Api) ?u64 {
        if (api.cuEventQuery(p.end) != 0) return null;
        var ms: f32 = 0;
        if (api.cuEventElapsedTime(&ms, p.begin, p.end) != 0 or !@import("std").math.isFinite(ms) or ms < 0 or ms > 1e12) return null;
        return @intFromFloat(@as(f64, ms) * 1e6);
    }
    pub fn deinit(p: Pair, api: *const abi.Api) void {
        _ = api.cuEventDestroy_v2(p.end);
        _ = api.cuEventDestroy_v2(p.begin);
    }
};
