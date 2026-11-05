//! A bounded eight-token CPU reference consumes the same exact prompt ids the native server reports.
const std = @import("std");
const reference = @import("reference");
const Input = struct {
    prompts: []const struct { id: []const u8, prompt_ids: []const u32, device_tokens: []const u32 = &.{} },
};
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.ExpectedGgufAndCaseJson;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(1 << 20));
    const cases = try std.json.parseFromSlice(Input, a, bytes, .{ .ignore_unknown_fields = true });
    if (cases.value.prompts.len != 3) return error.ExpectedThreeFixedPrompts;
    for (cases.value.prompts) |case| {
        if (case.prompt_ids.len == 0 or case.prompt_ids.len > 256) return error.CpuReferencePromptLimit;
        if (case.device_tokens.len < 8) return error.ExpectedEightDeviceTokensPerPrompt;
    }
    var weights = try reference.source.Weights.init(init.gpa, init.io, args[1], 112 << 30);
    defer weights.deinit(init.io);
    std.debug.print("CPU_READY cached_f32_bytes={d} arithmetic=binary64 activation=bf16\n", .{weights.bytes});
    var mismatches: usize = 0;
    for (cases.value.prompts) |case| {
        var arena = std.heap.ArenaAllocator.init(init.gpa);
        defer arena.deinit();
        const temp = arena.allocator();
        var session = try reference.Session.init(temp, &weights, case.prompt_ids.len + 8);
        var logits = try session.forward(init.gpa, case.prompt_ids);
        for (0..8) |step| {
            const choice = try reference.pick(logits);
            const expected = case.device_tokens[step];
            const equal = expected == choice.token;
            mismatches += @intFromBool(!equal);
            std.debug.print("CPU_TOKEN id={s} step={d} token={d} device={d} equal={any} runner_up={d} logit={d} margin={d}\n", .{ case.id, step, choice.token, expected, equal, choice.runner_up, choice.logit, choice.margin });
            init.gpa.free(logits);
            if (step != 7) logits = try session.forward(init.gpa, &.{choice.token});
        }
    }
    std.debug.print("CPU_SUMMARY prompts=3 tokens=24 mismatches={d}\n", .{mismatches});
    if (mismatches != 0) return error.CpuReferenceTokenMismatch;
}
