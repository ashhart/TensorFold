//! The image tests' tokenizer, shared by images.zig, image_key.zig and image_pieces.zig.
const std = @import("std");
const model_text = @import("model_text.zig");

/// A test tokenizer: "<|image|>" is token 9 unless read plainly, every other byte its own id plus 100.
pub const MarkText = struct {
    pub fn text() model_text.Text {
        return .{ .ctx = undefined, .vtable = &.{ .encode = encode, .decode = undefined, .token_id = undefined, .token_string = tokenString, .vocab_size = undefined, .eos_ids = undefined, .render = undefined, .template_source = undefined, .encode_plain = plain } };
    }
    pub fn encode(_: *anyopaque, a: std.mem.Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < s.len) {
            if (std.mem.startsWith(u8, s[i..], "<|image|>")) {
                try out.append(a, 9);
                i += 9;
            } else {
                try out.append(a, @as(u32, s[i]) + 100);
                i += 1;
            }
        }
        return out.items;
    }
    fn plain(_: *anyopaque, a: std.mem.Allocator, s: []const u8) model_text.Error![]u32 {
        const out = try a.alloc(u32, s.len);
        for (out, s) |*o, c| o.* = @as(u32, c) + 100;
        return out;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return a.dupe(u8, if (id == 9) "<|image|>" else "");
    }
};
