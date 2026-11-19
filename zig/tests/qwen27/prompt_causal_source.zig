//! Frozen causal-loop baseline and complete-source reconstruction for the prompt attention byte gate.
const std = @import("std");
const q = @import("qwen27");

pub const before =
    \\  const int kbeg = c * PCK, kend = min(kbeg + PCK, args.start + args.rows);
;
pub const after =
    \\  const int last = min(args.rows * PG, (int(tg.z) + 1) * (SG / 2) * 16) - 1;
    \\  const int seen_tg = args.start + last / PG + 1;
    \\  const int kbeg = c * PCK;
    \\  int kend = min(kbeg + PCK, args.start + args.rows);
    \\  if (kbeg >= seen_tg) {
    \\    if (live && h == 0 && (lane & 9) == 0) {
    \\      const int64_t base = ((int64_t)hk * args.chunks + c) * args.rp + tile * 16;
    \\      PM[base + fm] = -INFINITY; PL[base + fm] = 0.0f;
    \\      PM[base + fm + 8] = -INFINITY; PL[base + fm + 8] = 0.0f;
    \\    }
    \\    return;
    \\  }
    \\  kend = min(kend, kbeg + (seen_tg - kbeg + PTK - 1) / PTK * PTK);
;
pub const raw = [_][]const u8{ q.runtime_sources.attention_prompt, q.runtime_sources.attention_prompt_sg };
const pins = [_][]const u8{ "67fa58af326c188ed6881226d8232e78a50129c1eb82cf59e08b2525e4ec76e5", "32f30c60796a9c1bc6c886a11f6d08c81d53b5b20aca7ea9414d62b305fe9989" };

pub fn baseline(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const count = std.mem.count(u8, text, after ++ "\n");
    if (count == 0 or count > 2) return error.CausalSource;
    return std.mem.replaceOwned(u8, a, text, after ++ "\n", before ++ "\n");
}

pub fn planted(a: std.mem.Allocator, text: []const u8, cap: bool) ![]u8 {
    const from = if (cap) "kend = min(kend, kbeg + (seen_tg - kbeg + PTK - 1) / PTK * PTK);" else "if (kbeg >= seen_tg)";
    const to = if (cap) "kend = min(kend, kbeg + max((seen_tg - kbeg) / PTK - 1, 0) * PTK);" else "if (kbeg + PCK >= seen_tg)";
    if (std.mem.count(u8, text, from) == 0) return error.CausalSource;
    return std.mem.replaceOwned(u8, a, text, from, to);
}

test "causal prompt baseline: both whole sources match the frozen parent, paged edits remain four" {
    const a = std.testing.allocator;
    for (raw, pins) |text, pin| {
        const old = try baseline(a, text);
        defer a.free(old);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(old, &digest, .{});
        try std.testing.expectEqualStrings(pin, &std.fmt.bytesToHex(digest, .lower));
        try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, text, "(r, kt, kend, q, k, v, xs"));
        const begin = std.mem.indexOf(u8, text, after ++ "\n").?;
        try std.testing.expect(std.mem.indexOf(u8, text[begin..][0..after.len], "barrier") == null);
    }
}
