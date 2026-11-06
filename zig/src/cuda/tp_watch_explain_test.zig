//! Startup refusal wording remains independent of the liveness wire.
const std = @import("std");
const watch = @import("tp_watch.zig");

test "a liveness line that cannot start names its port, not the glue kernels" {
    for ([_]anyerror{ error.BindFailed, error.RankNeverCame, error.AcceptFailed }) |e| {
        const text = watch.explain(e).?;
        try std.testing.expect(std.mem.indexOf(u8, text, "NCCL_COMM_ID's port + 1") != null);
        try std.testing.expect(std.mem.indexOf(u8, text, "glue kernels") == null);
    }
    const bind = watch.explain(error.BindFailed).?;
    try std.testing.expect(std.mem.indexOf(u8, bind, "earlier server still exiting") != null);
    try std.testing.expect(std.mem.indexOf(u8, bind, "host may not be this machine's") != null);
    const missing = watch.explain(error.RankNeverCame).?;
    try std.testing.expect(std.mem.indexOf(u8, missing, "every rank") != null);
    try std.testing.expect(std.mem.indexOf(u8, missing, "rank 0 waited") == null);
    for ([_]anyerror{ error.OutOfMemory, error.SocketFailed, error.NoCommId, error.NcclUnavailable, error.RankNeverReady }) |e|
        try std.testing.expect(watch.explain(e) == null);
}
