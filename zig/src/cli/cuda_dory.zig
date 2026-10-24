//! `tensorfold run` on a Dory checkpoint (model_type dory): its own family and loop count, the same output line.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const dory = @import("dory");

/// The checkpoint at `dir` is a Dory one: its config.json names model_type dory.
pub fn wants(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22)) catch return false;
    defer gpa.free(text);
    const Head = struct { model_type: []const u8 = "" };
    const head = std.json.parseFromSlice(Head, gpa, text, .{ .ignore_unknown_fields = true }) catch return false;
    defer head.deinit();
    return std.mem.eql(u8, head.value.model_type, "dory");
}

pub const Options = struct { tokens: []const u32, max_tokens: usize, loops: ?usize, context: ?usize, drafts: bool, graphs: bool, stop_eos: bool, sampling: ?lanes.Sampling };

pub fn run(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options) !u8 {
    if (o.tokens.len == 0) return error.NoPromptTokens;
    const res = try dory.cuda_run.run(gpa, io, ctx, dir, .{ .tokens = o.tokens, .max_tokens = @intCast(o.max_tokens), .loops = o.loops, .context = o.context orelse 0, .drafts = o.drafts, .eager = !o.graphs, .ignore_eos = !o.stop_eos, .sampling = o.sampling });
    defer gpa.free(res.tokens);
    var digest: [32]u8 = undefined;
    const text = try core.ids_json.write(gpa, res.tokens);
    defer gpa.free(text);
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const steps = @max(1, res.tokens.len -| 1);
    const ms = res.decode_ms / @as(f64, @floatFromInt(steps));
    std.debug.print("tokens {d} sha {s} prefill {d:.4}s decode {d:.4}s {d:.3} ms/token rounds {d} accepted {d} loops {d} graphs {s}\n", .{ res.tokens.len, hex[0..12], res.prompt_ms / 1e3, res.decode_ms / 1e3, ms, res.rounds, res.accepted, res.loops, if (o.graphs) "on" else "off" });
    std.debug.print("ids", .{});
    for (res.tokens) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
    return 0;
}
