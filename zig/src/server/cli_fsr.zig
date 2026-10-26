//! Exact Metal residency options, parsed before the native adapter checks family and backend support.
const std = @import("std");
pub fn apply(out: anytype, name: []const u8, value: []const u8) !bool {
    if (std.mem.eql(u8, name, "--ssd-experts")) {
        const g = std.fmt.parseFloat(f64, value) catch return error.ExpectedPositiveExpertGiB;
        if (!std.math.isFinite(g) or g <= 0 or g >= 0x1p33) return error.ExpectedPositiveExpertGiB;
        out.ssd_experts_gib = g;
    } else if (std.mem.eql(u8, name, "--ple-on-ssd")) {
        out.ple_on_ssd = true;
    } else if (std.mem.eql(u8, name, "--fsr-route-trace")) {
        if (value.len == 0 or std.mem.indexOfScalar(u8, value, 0) != null) return error.ExpectedLocalRouteTracePath;
        out.fsr_trace = value;
    } else return false;
    return true;
}
