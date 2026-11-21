//! tf-moss-image-generate on Metal: the family parser's picture from options on the system GPU.
const std = @import("std");
const Metal = @import("moss_image.zig").Metal;

pub fn main(init: std.process.Init) !u8 {
    return @import("moss_image_generate_run.zig").run(Metal, init);
}
