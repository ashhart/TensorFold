//! Flash Next's SSD residency flags are refused before a device or checkpoint opens wherever they cannot run.
const std = @import("std");
const Open = @import("engine_api.zig").Open;

pub fn refusal(o: Open, backend: []const u8) ?[]const u8 {
    if (o.ssd_experts_gib == null and !o.ple_on_ssd and o.fsr_trace == null) return null;
    if (!std.mem.eql(u8, backend, "metal") or !std.mem.eql(u8, o.model_type, "qwen4_exp"))
        return "--ssd-experts, --ple-on-ssd and --fsr-route-trace require Flash Next (qwen4_exp) on Metal";
    if (o.speed_up != null) return "FSR SSD residency requires one Metal device; --speed-up is unsupported";
    if (o.ssd_experts_gib) |g| if (!std.math.isFinite(g) or g <= 0 or g >= 0x1p33)
        return "--ssd-experts requires a positive finite GiB budget below 2^33";
    return null;
}
