//! LFM3-Omni's chat prompt for a system text and one spoken turn, ending on the interleaved generation marker.
const std = @import("std");
const tokenizer = @import("core").tokenizer;
const cfg = @import("config.zig");

pub const Prompt = struct {
    ids: []u32,
    /// The first of the contiguous `<|audio|>` rows the encoder's rows replace.
    audio_at: usize,
};

/// The system turn, a user turn of `n_audio` `<|audio|>` rows, then the assistant's empty think and `<|mixed_start|>`.
pub fn build(gpa: std.mem.Allocator, tok: *const tokenizer.Tokenizer, system: []const u8, n_audio: usize) !Prompt {
    const head = try std.fmt.allocPrint(gpa, "<|startoftext|><|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n<|vision_start|>", .{system});
    defer gpa.free(head);
    const a = try tok.encode(gpa, head);
    defer gpa.free(a);
    const b = try tok.encode(gpa, "<|vision_end|><|im_end|>\n<|im_start|>assistant\n<think></think>");
    defer gpa.free(b);
    const ids = try gpa.alloc(u32, a.len + n_audio + b.len + 1);
    @memcpy(ids[0..a.len], a);
    @memset(ids[a.len..][0..n_audio], cfg.audio_token_id);
    @memcpy(ids[a.len + n_audio ..][0..b.len], b);
    ids[ids.len - 1] = cfg.mixed_start_token_id;
    if (a[a.len - 1] != cfg.vision_start_token_id or b[0] != cfg.vision_end_token_id) return error.TokenizerSpecialTokens;
    return .{ .ids = ids, .audio_at = a.len };
}
