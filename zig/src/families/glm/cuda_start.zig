//! Connect rank liveness before resolving page settings or constructing the engine.
const std = @import("std");

/// Every rank resolves its page mode within the same watched failure boundary as engine construction.
pub fn open(io: std.Io, p: anytype, text: ?[]const u8, initial: anytype, watch: anytype, args: anytype, comptime after: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    if (p.world > 1) try watch.start(io, p.rank, p.world);
    return resolved(p, text, initial, args, after) catch |err| {
        watch.fail(@errorName(err));
        return err;
    };
}

fn resolved(p: anytype, text: ?[]const u8, initial: anytype, args: anytype, comptime after: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    var settings = initial;
    settings.pages = try paging(text, p.world, settings.kv8);
    return after(args, settings);
}

/// TF_KV_PAGES keeps FP8 on whole windows; explicit pages with FP8 are refused on every rank.
pub fn paging(text: ?[]const u8, world: usize, kv8: bool) !bool {
    const v = text orelse "auto";
    if (std.mem.eql(u8, v, "auto")) return world >= 2 and !kv8;
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return false;
    if (kv8) return error.Fp8OnPages;
    return true;
}
