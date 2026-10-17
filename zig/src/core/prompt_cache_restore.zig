//! One counted restore failure, with its classification captured before releasing the final borrow.
pub const Details = struct { kept: u32, prompt: usize, expired: bool, err: anyerror };
pub const Log = struct {
    ctx: *anyopaque,
    write: *const fn (*anyopaque, u32, anyerror) void,
    details: ?*const fn (*anyopaque, Details) void = null,
};
pub fn report(log: ?Log, detail: Details) void {
    if (log) |sink| {
        if (sink.details) |write| write(sink.ctx, detail) else sink.write(sink.ctx, detail.kept, detail.err);
        return;
    }
    const why = if (detail.expired) "state already expired" else "backend refused the restore";
    @import("prompt_cache.zig").note("restore of the kept {d}-token state failed for a {d}-token prompt ({s}, {s}); prefilling from the start", .{ detail.kept, detail.prompt, why, @errorName(detail.err) });
}
