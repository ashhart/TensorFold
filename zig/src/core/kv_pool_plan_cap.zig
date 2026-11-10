//! Pool capacity accounts for owned state runs independently of the admission and memory-backing policy.
const std = @import("std");

pub fn bind(layout: anytype, budget: anytype, planned: anytype) @TypeOf(planned) {
    var result = planned;
    std.debug.assert(result.lanes > 0);
    const fit: u64 = result.lanes - 1;
    const runs = fit + @intFromBool(budget.engine_paged and layout.engine_run);
    const limit = if (budget.window > 0)
        (fit + @intFromBool(budget.engine_paged)) * layout.pages(budget.window) +
            runs * layout.runPages() + result.kept / layout.pageBytes()
    else
        std.math.maxInt(u64);
    const host_room = if (@hasField(@TypeOf(budget), "pages_room")) budget.pages_room else 0;
    const device_room = budget.room - budget.engine - fit * layout.fixed_bytes -
        result.snaps * layout.snap_bytes - result.copied;
    const available = if (host_room > 0) host_room else device_room;
    result.pages = @intCast(@min(available / layout.pageBytes(), limit));
    if (@hasField(@TypeOf(result), "host")) result.host = if (host_room > 0) result.pages else 0;
    return result;
}
