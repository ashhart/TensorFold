//! Required metadata errors retain their exact key through file close and the native CLI's error display.
const gg = @import("gguf.zig");
pub fn get(d: *const gg.Document, comptime key: []const u8) !gg.Value {
    return d.get(key) orelse @field(anyerror, "MissingGgufMetadata:" ++ key);
}
