//! CLI for the lane host's token-map builder: --tokenizer PATH --config PATH --output PATH.
const std = @import("std");
const tf = @import("tensorfold");

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 7) {
        std.debug.print("usage: tf-dsv41-token-map --tokenizer PATH --config PATH --output PATH\n", .{});
        return error.InvalidArguments;
    }
    var tokenizer_path: ?[]const u8 = null;
    var config_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--tokenizer")) tokenizer_path = args[i + 1] else if (std.mem.eql(u8, args[i], "--config")) config_path = args[i + 1] else if (std.mem.eql(u8, args[i], "--output")) output_path = args[i + 1] else return error.InvalidArguments;
    }
    try tf.deepseek_v41_token_map.build(a, io, tokenizer_path.?, config_path.?, output_path.?);
}
