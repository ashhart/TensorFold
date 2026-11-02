//! Connect rank liveness before resolving page settings or constructing the engine.
const std = @import("std");

/// Every rank resolves its page mode within the same watched failure boundary as engine construction.
pub fn open(io: std.Io, p: anytype, text: ?[]const u8, initial: anytype, watch: anytype, args: anytype, comptime after: anytype, comptime report: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    if (p.world > 1) try watch.start(io, p.rank, p.world);
    return resolved(p, text, initial, args, after, report) catch |err| {
        watch.fail(@errorName(err));
        return err;
    };
}

fn resolved(p: anytype, text: ?[]const u8, initial: anytype, args: anytype, comptime after: anytype, comptime report: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    var settings = initial;
    settings.pages = try paging(text, p.world, settings.kv8);
    report(args, Report{ .text = text, .rank = p.rank, .world = p.world, .kv8 = settings.kv8, .pages = settings.pages });
    return after(args, settings);
}

pub fn log(_: anytype, selection: Report) void {
    std.log.info("glm kv selection: {f}", .{selection});
}

/// A configuration selection, before construction; the resolved boolean is the one passed to the engine.
pub const Report = struct {
    text: ?[]const u8,
    rank: usize,
    world: usize,
    kv8: bool,
    pages: bool,

    pub fn format(r: Report, writer: *std.Io.Writer) !void {
        const v = r.text orelse "auto";
        const mode = if (std.mem.eql(u8, v, "auto")) "auto" else if (std.mem.eql(u8, v, "off") or std.mem.eql(u8, v, "0")) "off" else "on";
        try writer.print("{s}; TF_KV_PAGES={s} ({s}); rank {d} of {d}; cache {s}", .{
            if (r.pages) "pool pages" else "window per stream",
            mode,
            if (r.text == null) "default" else "env",
            r.rank,
            r.world,
            if (r.kv8) "fp8" else "bf16",
        });
    }
};

/// TF_KV_PAGES keeps FP8 on whole windows; explicit pages with FP8 are refused on every rank.
pub fn paging(text: ?[]const u8, world: usize, kv8: bool) !bool {
    const v = text orelse "auto";
    if (std.mem.eql(u8, v, "auto")) return world >= 2 and !kv8;
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return false;
    if (kv8) return error.Fp8OnPages;
    return true;
}
