//! Llama's prompt attention: its loader's WINDOW 0 and RING 0 fold the shared text to its old full-attention kernel.
const std = @import("std");

const shared_path = "zig/kernels/metal/core/prompt_attention_nax.metal";
const loader_path = "zig/src/families/llama/contracts.zig";
const sources_path = "zig/kernels/metal/sources.zig";

/// The mask line: at WINDOW 0 its window term drops, leaving Llama's old `col > row`.
const mask_line = "const bool out = col > row || (WINDOW && col < row - WINDOW + 1);";

/// Every code line that may name WINDOW or RING: each guarded, so 0 gives Llama's old expression.
const guarded = [_][]const u8{
    "const int lo = WINDOW ? max(0, first_pos - WINDOW + 1) : 0;",
    "const int slot = RING ? kpos % RING : kpos;",
    mask_line,
    "const device bfloat* vblk = vc + long(RING ? kp % RING : kp) * LD;",
};

fn identifier(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_';
}

fn word(text: []const u8, name: []const u8) bool {
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, text, start, name)) |at| {
        const end = at + name.len;
        if ((at == 0 or !identifier(text[at - 1])) and (end == text.len or !identifier(text[end]))) return true;
        start = at + 1;
    }
    return false;
}

/// Whether every code line naming WINDOW or RING is one of the guarded lines, the mask line among them.
fn folds(text: []const u8) bool {
    var mask = false;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (std.mem.startsWith(u8, line, "//")) continue;
        if (!word(line, "WINDOW") and !word(line, "RING")) continue;
        const known = for (guarded) |g| {
            if (std.mem.eql(u8, line, g)) break true;
        } else false;
        if (!known) return false;
        if (std.mem.eql(u8, line, mask_line)) mask = true;
    }
    return mask;
}

fn read(a: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(4 << 20));
}

test "prompt attention fold: Llama's loader defines WINDOW 0 and RING 0 over the shared text, every use guarded" {
    const a = std.testing.allocator;
    const shared = try read(a, shared_path);
    defer a.free(shared);
    try std.testing.expect(folds(shared));
    const loader = try read(a, loader_path);
    defer a.free(loader);
    const at = std.mem.indexOf(u8, loader, "pub fn attentionSourceHeads(") orelse return error.NoLoader;
    const end = std.mem.indexOfPos(u8, loader, at, "\n}\n") orelse return error.NoLoader;
    try std.testing.expect(std.mem.indexOf(u8, loader[at..end], "#define WINDOW 0\\n#define RING 0\\n") != null);
    const sources = try read(a, sources_path);
    defer a.free(sources);
    try std.testing.expect(std.mem.indexOf(u8, sources, "const prompt_attention_nax = @embedFile(\"core/prompt_attention_nax.metal\");") != null);
    try std.testing.expect(std.mem.indexOf(u8, sources, "pub const llama_attention_nax = \"#define TF_PROMPT_ATTENTION ll_prompt_attention\\n\" ++ prompt_attention_nax;") != null);
}

test "control: an unguarded window term or a missing mask line breaks the fold" {
    const a = std.testing.allocator;
    const shared = try read(a, shared_path);
    defer a.free(shared);
    const bare = try std.mem.replaceOwned(u8, a, shared, "(WINDOW && col", "(col");
    defer a.free(bare);
    try std.testing.expect(!folds(bare));
    const unmasked = try std.mem.replaceOwned(u8, a, shared, mask_line, "const bool out = col > row;");
    defer a.free(unmasked);
    try std.testing.expect(!folds(unmasked));
}
