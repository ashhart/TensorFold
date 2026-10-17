//! The fixed FP32 codec geometry consumed by both backends, checked before upload.
const std = @import("std");
fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn count(name: []const u8) !usize {
    var parts: [8][]const u8 = undefined;
    var it = std.mem.splitScalar(u8, name, '.');
    var n: usize = 0;
    while (it.next()) |p| {
        if (n == parts.len) return error.BadTensor;
        parts[n] = p;
        n += 1;
    }
    if (n < 2) return error.BadTensor;
    const p = parts[0..n];
    if (eq(p[0], "vq") and n == 2) {
        if (eq(p[1], "table")) return 13312 * 1024;
        if (eq(p[1], "cbn")) return 13312 * 8;
        if (eq(p[1], "in_w")) return 10 * 8 * 1024;
        if (eq(p[1], "in_b")) return 10 * 8;
    }
    if (eq(p[0], "post") or eq(p[0], "pre")) return tf(p[1..], 8);
    if (eq(p[0], "up") or eq(p[0], "down")) {
        const i = std.fmt.parseInt(usize, p[1], 10) catch return error.BadTensor;
        if (i >= 2) return error.BadTensor;
        if (n == 3) {
            if (eq(p[2], "b")) return if (eq(p[0], "up")) 2048 else 1024;
            if (eq(p[2], "wa") or eq(p[2], "w")) return 2 * 1024 * 1024;
        }
        if (n == 4 and eq(p[2], "cn")) {
            const f = p[3];
            if (eq(f, "dw")) return 1024 * 7;
            if (eq(f, "pw1") or eq(f, "pw2")) return 4096 * 1024;
            if (eq(f, "pw1_b")) return 4096;
            for ([_][]const u8{ "dw_b", "ln_w", "ln_b", "pw2_b", "gamma" }) |s| if (eq(f, s)) return 1024;
        }
    }
    const dec = eq(p[0], "dec");
    if (!dec and !eq(p[0], "enc")) return error.BadTensor;
    if (n == 2) {
        const f = p[1];
        if (eq(f, "conv0")) return if (dec) 7 * 1536 * 1024 else 7 * 64;
        if (eq(f, "conv0_b")) return if (dec) 1536 else 64;
        if (eq(f, "final")) return if (dec) 7 * 96 else 3 * 1024 * 1024;
        if (eq(f, "final_b")) return if (dec) 1 else 1024;
        if (eq(f, "final_alpha")) return if (dec) 96 else 1024;
    }
    const bi = std.fmt.parseInt(usize, p[1], 10) catch return error.BadTensor;
    if (bi >= 4 or n < 3) return error.BadTensor;
    if (eq(p[2], "tf") and !dec and bi == 3) return tf(p[3..], 4);
    const cin: usize = if (dec) @as(usize, 1536) >> @intCast(bi) else @as(usize, 64) << @intCast(bi);
    const cout = if (dec) cin / 2 else cin * 2;
    if (n == 3) {
        if (eq(p[2], "alpha")) return cin;
        const strides = if (dec) [4]usize{ 8, 8, 4, 2 } else [4]usize{ 2, 4, 8, 8 };
        if (eq(p[2], "b")) return if (dec) strides[bi] * cout else cout;
        if (eq(p[2], "wa") or eq(p[2], "wb")) return strides[bi] * cin * cout;
    }
    if (n == 4 and std.mem.startsWith(u8, p[2], "ru")) {
        const ri = std.fmt.parseInt(usize, p[2][2..], 10) catch return error.BadTensor;
        if (ri >= 3) return error.BadTensor;
        const d = if (dec) cout else cin;
        if (eq(p[3], "c7")) return 7 * d * d;
        if (eq(p[3], "c1")) return d * d;
        for ([_][]const u8{ "a1", "a2", "c7_b", "c1_b" }) |f| if (eq(p[3], f)) return d;
    }
    return error.BadTensor;
}

fn tf(p: []const []const u8, layers: usize) !usize {
    if (p.len == 1 and eq(p[0], "norm")) return 1024;
    if (p.len != 2) return error.BadTensor;
    const i = std.fmt.parseInt(usize, p[0], 10) catch return error.BadTensor;
    if (i >= layers) return error.BadTensor;
    if (eq(p[1], "wqkv")) return 3 * 1024 * 1024;
    if (eq(p[1], "wo")) return 1024 * 1024;
    if (eq(p[1], "w13")) return 2 * 3072 * 1024;
    if (eq(p[1], "w2")) return 3072 * 1024;
    for ([_][]const u8{ "attn_norm", "ffn_norm", "g_attn", "g_ffn" }) |f| if (eq(p[1], f)) return 1024;
    return error.BadTensor;
}

pub fn validate(name: []const u8, elements: usize) !void {
    if (elements != try count(name)) return error.BadTensor;
}

test "codec uploads reject undersized and unknown tensors before GPU access" {
    try validate("dec.0.wa", 8 * 768 * 1536);
    try validate("enc.3.tf.0.w13", 2 * 3072 * 1024);
    try std.testing.expectError(error.BadTensor, validate("post.0.attn_norm", 1));
    try std.testing.expectError(error.BadTensor, validate("up.2.wa", 1));
    try std.testing.expectError(error.BadTensor, validate("enc.0.tf.0.w13", 1));
}
