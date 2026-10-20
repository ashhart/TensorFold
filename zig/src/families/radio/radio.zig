//! RADIO image input for Nemotron Omni checkpoints on CUDA: the processor, the tower and projector, the prompt rows.
const std = @import("std");
const core = @import("core");

pub const config = @import("config.zig");
pub const Config = config.Config;
pub const image = @import("image.zig");
pub const video = @import("video.zig");
pub const position = @import("position.zig");
pub const weights = @import("weights.zig");
pub const Tower = @import("tower.zig").Tower;
pub const rows = @import("rows.zig");
pub const Rows = rows.Rows;
pub const Span = rows.Span;
pub const plan = rows.plan;

/// The words for a refusal the server cannot word: the JPEG library this machine lacks, or a CMYK JPEG.
pub const decoderProblem = core.picture.decoderProblem;

/// `dir`'s image side (config.json, preprocessor_config.json), or null with `problem`; error.NoVisionTower without one.
pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) !?Config {
    const text = try read(a, io, dir, "config.json", 1 << 26) orelse return error.NoConfig;
    defer a.free(text);
    const pre = try read(a, io, dir, "preprocessor_config.json", 1 << 20);
    defer if (pre) |p| a.free(p);
    return Config.parse(a, text, pre, problem);
}

fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, most: usize) !?[]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(most)) catch |e| switch (e) {
        error.FileNotFound => null,
        else => e,
    };
}

/// The reserved token a video's `<video>` stands as before encoding: the checkpoint's first spare special token.
pub const video_token = "<SPECIAL_30>";

/// The prompt's image tokens: the placeholder the template writes, and the two the processor wraps its rows in.
pub const Marks = struct {
    image: u32,
    open: u32,
    close: u32,
    /// The token a video's placeholder becomes before encoding (the tokenizer has no `<video>` token); null: none.
    video: ?u32 = null,

    /// The marks from the tokenizer's special tokens; null when one is missing or the image token is not the config's.
    pub fn resolve(t: *const core.tokenizer.Tokenizer, c: *const Config) ?Marks {
        const m: Marks = .{
            .image = t.specialTokenId("<image>") orelse return null,
            .open = t.specialTokenId("<img>") orelse return null,
            .close = t.specialTokenId("</img>") orelse return null,
            .video = t.specialTokenId(video_token),
        };
        return if (m.image == c.image_token) m else null;
    }

    /// The tokens user text may not spell (it stays text): the placeholder and its wrapper.
    pub fn markers(m: *const Marks) [3]u32 {
        return .{ m.image, m.open, m.close };
    }
};

test {
    _ = config;
    _ = image;
    _ = video;
    _ = position;
    _ = weights;
    _ = rows;
    _ = @import("tower.zig");
}

test {
    _ = @import("video_test.zig");
    _ = @import("upload_test.zig");
}
