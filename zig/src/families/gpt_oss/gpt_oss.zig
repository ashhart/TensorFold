//! Host-only gpt-oss package. It advertises no native backend or checkpoint execution capability.
pub const config = @import("config.zig");
pub const generation = @import("generation.zig");
pub const names = @import("names.zig");
pub const attention = @import("attention.zig");
pub const route = @import("route.zig");
pub const mxfp4 = @import("mxfp4");
pub const parseHarmony = @import("reply_text").parseHarmony;
