//! Test roots for the tensorfold CLI commands.
test {
    _ = @import("hub.zig");
    _ = @import("models.zig");
    _ = @import("info.zig");
    _ = @import("pull.zig");
    _ = @import("cli.zig");
}
