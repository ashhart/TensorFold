//! Pin ledger boundaries in the real paired bodies, including branches the CPU cannot execute numerically.
const std = @import("std");
const t = std.testing;
const pair = @embedFile("cuda_prompt_pair.zig");
const forward = @embedFile("cuda_forward_pair.zig");
fn has(text: []const u8, needle: []const u8) !void {
    try t.expect(std.mem.indexOf(u8, text, needle) != null);
}
test "GLM prefill ledger: paired MTP and both commits have actual body boundaries" {
    const mtp = pair[std.mem.indexOf(u8, pair, "fn mtpRows(").?..];
    try has(mtp, "trace.span(e.stream, .mtp)");
    try t.expect(std.mem.count(u8, pair, "trace.span(e.stream, .commit)") == 2);
    try has(@embedFile("cuda_forward.zig"), "trace.span(f.stream, .mtp_projection)");
}
test "GLM prefill ledger: joint FFN shared output stages and exchange row ranges are pinned" {
    const joint = forward[std.mem.indexOf(u8, forward, "fn pairedMoe(").?..std.mem.indexOf(u8, forward, "fn exchangeCombined(").?];
    try has(joint, "trace.ffnPart(a.stream, j, 0, ra)");
    try has(joint, "trace.ffnPart(a.stream, j, ra, rb)");
    try has(joint, "trace.span(a.stream, .shared_expert)");
    try has(joint, "trace.jointFfn(a.stream, j, ra, rb)");
    try has(joint, "const exchange_a = trace.ffnPart(a.stream, j, 0, ra)");
    try has(joint, "const exchange_b = trace.ffnPart(b.stream, j, ra, rb)");
    try has(forward, "trace.span(f.stream, .output_projection)");
}
test "GLM prefill ledger: paired image splice remains inside embedding" {
    const splice = std.mem.indexOf(u8, forward, "im.v.splice").?;
    const end = std.mem.indexOf(u8, forward, "embed_stamp.end()").?;
    try t.expect(splice < end);
}
