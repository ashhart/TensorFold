//! The fresh-prompt packet used by both CUDA ranks, checked before entering collectives.
const engine = @import("cuda_engine.zig");
const prompts = @import("cuda_prompts.zig");
pub const words = engine.prompt_rows + 2 * engine.max_together;
pub fn encode(parts: []const engine.Prompt, capacity: usize, buf: []u32) ![]u32 {
    if (!prompts.fits(parts, capacity)) return error.BadPrompts;
    var n: usize = 2 * parts.len;
    for (parts) |p| n += p.ids.len;
    if (buf.len < n) return error.BadStep;
    var at: usize = 2 * parts.len;
    for (parts, 0..) |p, i| {
        buf[2 * i] = p.slot;
        buf[2 * i + 1] = @intCast(p.ids.len);
        @memcpy(buf[at..][0..p.ids.len], p.ids);
        at += p.ids.len;
    }
    return buf[0..n];
}
pub fn decode(ids: []const u32, count: usize, capacity: usize, out: *[engine.max_together]engine.Prompt) ![]const engine.Prompt {
    if (count == 0 or count > out.len or ids.len < 2 * count) return error.BadStep;
    var at: usize = 2 * count;
    for (out[0..count], 0..) |*p, i| {
        const n = ids[2 * i + 1];
        if (n > ids.len - at) return error.BadStep;
        p.* = .{ .slot = ids[2 * i], .ids = ids[at..][0..n] };
        at += n;
    }
    if (at != ids.len or !prompts.fits(out[0..count], capacity)) return error.BadStep;
    return out[0..count];
}
