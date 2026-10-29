//! What a two-Mac link that failed to connect asks of whoever starts the servers.
const std = @import("std");

/// The advice for a link that never connected; null for any other error.
pub fn problem(e: anyerror) ?[]const u8 {
    return if (e == error.ConnectFailed) "the other Mac never answered on the link (ConnectFailed). Start both ranks with their own settings file, each link's via naming the other Mac's address on the cable, and run both attached to a session or with Local Network access allowed in System Settings: macOS drops the packets of a server started detached, such as with nohup over SSH" else null;
}

test problem {
    try std.testing.expect(std.mem.indexOf(u8, problem(error.ConnectFailed).?, "Local Network") != null);
    try std.testing.expect(problem(error.OutOfMemory) == null);
}
