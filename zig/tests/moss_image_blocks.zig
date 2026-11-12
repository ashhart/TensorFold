//! tf-moss-image-blocks on Metal: moss_image_blocks_run.zig's gate on the system GPU.
const std = @import("std");

pub fn main(init: std.process.Init) !u8 {
    return @import("moss_image_blocks_run.zig").run(@import("moss_image.zig").Metal, init);
}
