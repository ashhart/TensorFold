//! Compile tensorfold chat for Windows, where it builds only its --url path, without linking or running it.
const chat = @import("src/server/chat_cli.zig");

comptime {
    _ = &chat.command;
    _ = &chat.main;
}
