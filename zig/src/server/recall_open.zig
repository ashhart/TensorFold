//! Recall's record for the served tokenizer: its identity, its directory, and refusal of another tokenizer's record.
const std = @import("std");
const api = @import("engine_api");
const tokenizer = @import("tokenizer");
const cli = @import("cli.zig");
const hf_text = @import("hf_text.zig");
const rs = api.recall_store;
const Allocator = std.mem.Allocator;

/// The record when --recall asks for one (null: off); RecallRefused with `problem` saying why.
pub fn open(gpa: Allocator, io: std.Io, a: Allocator, args: cli.Args, text: *hf_text.HfText, served: []const u8, problem: *[]const u8) !?*rs.Store {
    if (!args.recall) return null;
    const identity: rs.Identity = .{ .digest = try tokenizer.identity(&text.tok, a), .vocab = text.text().vocabSize(), .name = served };
    const dir = args.recall_dir orelse try rs.defaultDir(a, identity);
    const cap: u64 = @intFromFloat(args.recall_gib * (1 << 30));
    return (try rs.Store.open(gpa, io, identity, .{ .dir = dir, .cap_bytes = cap, .read_only = args.recall_read_only }, problem, a)) orelse error.RecallRefused;
}
