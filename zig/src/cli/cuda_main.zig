//! `tensorfold` on Linux with CUDA: a model run on token ids with no Python at all, plus the oracle checks.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const core = @import("core");
const lanes = @import("lanes");
const checks = @import("cuda_checks.zig");
const kernel_checks = @import("cuda_kernel_checks.zig");
const shared_checks = @import("cuda_shared_checks.zig");
const lanes_cli = @import("cuda_lanes.zig");
const segments_cli = @import("cuda_segments.zig");
const decode = nemotron.decode;

const usage =
    \\usage: tensorfold run MODEL --tokens ID,ID,... [--max-tokens N] [--no-drafts] [--report PATH] [--kernels DIR] [--device N]
    \\         [--temperature T] [--top-k K] [--top-p P] [--min-p M] [--seed S]   (temperature 0: greedy)
    \\         [--context N] [--ignore-eos] [--eager] [--costs MS1,...,MS16,LEVEL]
    \\         [--tokens-file PATH] [--segments N]   (N whole 2048-row prompt chunks a call as staggered segments,
    \\         1-4; default TF_CUDA_SEGMENTS, else 1)
    \\       tensorfold segments MODEL IDS_FILE... [--counts 1,2,3,4] [--repeat N] [--max-tokens N] [--profile]
    \\                (each prompt prefilled at every segment count, interleaved: state hashes, prompt and decode speed)
    \\       tensorfold check-weights MODEL DIGESTS.json [--kernels DIR]
    \\       tensorfold teacher MODEL TEACHER.json [--dump DIR] [--kernels DIR]
    \\       tensorfold prefill MODEL PROMPTS.json NAME [--dump DIR] [--kernels DIR]
    \\       tensorfold rounds MODEL --tokens ID,... [--max-tokens N]   (GPU ms a serial and a window graph round)
    \\       tensorfold check-draws MODEL FIXTURES_DIR   (MTP draws against the lane fixtures)
    \\       tensorfold shared-widths MODEL PROMPTS.json --streams N [--max-tokens N] [sampling as run]
    \\                (N streams' continuations as shared windows of varying splits against serial, 2 <= N <= 8)
    \\       tensorfold check-kernels MODEL   (lane_gemv and the forked MoE against the kernels they replace, real weights)
    \\       tensorfold widths MODEL --tokens ID,... [--max-tokens N] [--eager] [sampling as run]
    \\                (every verify width 1-16 against serial decoding, a wrong draft every third window)
    \\       tensorfold lanes MODEL PROMPTS.json [--solo] [--max-tokens N] [--no-drafts] [sampling as run] [--report PATH]
    \\                (every prompt through the lane core at once, or one at a time with --solo)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var opts = Options{ .model = args[2] };
    var positional: std.ArrayList([]const u8) = .empty;
    defer positional.deinit(gpa);
    defer gpa.free(opts.tokens);
    defer if (opts.costs) |c| gpa.free(c);
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const value = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, a, "--tokens")) {
            opts.tokens = try parseIds(gpa, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            opts.max_tokens = try std.fmt.parseInt(u32, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--temperature")) {
            opts.sampling.temperature = try std.fmt.parseFloat(f64, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--top-k")) {
            opts.sampling.top_k = try std.fmt.parseInt(u32, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--top-p")) {
            opts.sampling.top_p = try std.fmt.parseFloat(f64, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--min-p")) {
            opts.sampling.min_p = try std.fmt.parseFloat(f64, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--seed")) {
            opts.sampling.seed = try std.fmt.parseInt(u64, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--report")) {
            opts.report = value;
            i += 1;
        } else if (std.mem.eql(u8, a, "--kernels")) {
            opts.kernels = value;
            i += 1;
        } else if (std.mem.eql(u8, a, "--device")) {
            opts.device = try std.fmt.parseInt(u32, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--dump")) {
            opts.dump = value;
            i += 1;
        } else if (std.mem.eql(u8, a, "--costs")) {
            opts.costs = try parseFloats(gpa, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tokens-file")) {
            gpa.free(opts.tokens);
            opts.tokens = try segments_cli.readIds(gpa, init.io, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--segments")) {
            opts.segments = try std.fmt.parseInt(usize, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--counts")) {
            opts.counts = try parseCounts(value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--streams")) {
            opts.streams = try std.fmt.parseInt(usize, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--repeat")) {
            opts.repeat = try std.fmt.parseInt(usize, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--profile")) {
            opts.profile = true;
        } else if (std.mem.eql(u8, a, "--context")) {
            opts.context = try std.fmt.parseInt(u32, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--no-drafts")) {
            opts.drafts = false;
        } else if (std.mem.eql(u8, a, "--solo")) {
            opts.solo = true;
        } else if (std.mem.eql(u8, a, "--ignore-eos")) {
            opts.stop_eos = false;
        } else if (std.mem.eql(u8, a, "--eager")) {
            opts.graphs = false;
        } else if (std.mem.startsWith(u8, a, "--")) {
            std.debug.print("unknown option {s}\n{s}", .{ a, usage });
            return 2;
        } else try positional.append(gpa, a);
    }
    const kernels = opts.kernels orelse init.environ_map.get("TENSORFOLD_CUDA_KERNELS") orelse blk: {
        const exe = try std.process.executableDirPathAlloc(init.io, init.arena.allocator());
        break :blk try std.fs.path.join(init.arena.allocator(), &.{ exe, "..", "share", "tensorfold", "cuda", "sm121" });
    };
    var driver = try cuda.Driver.open();
    defer driver.close();
    const device = try deviceOrdinal(opts.device, init.environ_map.get("TF_CUDA_DEVICE"));
    var ctx = try cuda.Context.init(&driver, @intCast(device));
    defer ctx.deinit();
    const cmd = args[1];
    const bench = std.mem.eql(u8, cmd, "segments");
    const decoding = std.mem.eql(u8, cmd, "run") or std.mem.eql(u8, cmd, "lanes") or bench;
    const mtp = std.mem.eql(u8, cmd, "check-weights") or std.mem.eql(u8, cmd, "rounds") or std.mem.eql(u8, cmd, "check-draws") or std.mem.eql(u8, cmd, "widths") or std.mem.eql(u8, cmd, "check-kernels") or (decoding and opts.drafts);
    const widths = std.mem.eql(u8, cmd, "widths");
    const graphs = opts.graphs and (std.mem.eql(u8, cmd, "run") or std.mem.eql(u8, cmd, "rounds") or bench or widths);
    const sampling: ?lanes.Sampling = if ((std.mem.eql(u8, cmd, "run") or widths) and opts.sampling.temperature > 0) opts.sampling else null;
    const segments = opts.segments orelse if (init.environ_map.get("TF_CUDA_SEGMENTS")) |v| try std.fmt.parseInt(usize, v, 10) else 1;
    const engine = try nemotron.Engine.init(gpa, init.io, &ctx, opts.model, kernels, .{ .context = opts.context, .mtp = mtp, .graphs = graphs, .sampling = sampling, .segments = segments });
    defer engine.deinit();
    const rest = positional.items;
    if (std.mem.eql(u8, cmd, "run")) return run(gpa, init.io, engine, opts);
    if (std.mem.eql(u8, cmd, "lanes") and rest.len == 1) {
        const s: ?lanes.Sampling = if (opts.sampling.temperature > 0) opts.sampling else null;
        return lanes_cli.run(gpa, init.io, engine, opts.model, .{ .prompts = rest[0], .max_tokens = @intCast(opts.max_tokens), .sampling = s, .drafts = opts.drafts, .solo = opts.solo, .report = opts.report });
    }
    if (bench and rest.len >= 1) {
        var drafter: ?nemotron.Drafter = if (opts.drafts) try nemotron.Drafter.init(gpa, init.io, engine, opts.model, opts.graphs, opts.costs) else null;
        defer if (drafter) |*d| d.deinit();
        const counts = opts.counts.items[0..opts.counts.len];
        return segments_cli.run(gpa, init.io, engine, if (drafter) |*d| d else null, .{ .files = rest, .counts = counts, .repeat = opts.repeat, .max_tokens = opts.max_tokens, .stop_eos = opts.stop_eos, .profile = opts.profile, .report = opts.report });
    }
    if (std.mem.eql(u8, cmd, "check-weights") and rest.len == 1) return checks.weights(gpa, init.io, engine, rest[0]);
    if (std.mem.eql(u8, cmd, "teacher") and rest.len == 1) return checks.teacher(gpa, init.io, engine, rest[0], opts.dump);
    if (std.mem.eql(u8, cmd, "prefill") and rest.len == 2) return checks.prefill(gpa, init.io, engine, rest[0], rest[1], opts.dump);
    if (std.mem.eql(u8, cmd, "rounds")) return checks.rounds(engine, opts.tokens, opts.max_tokens);
    if (widths) return checks.widths(gpa, engine, opts.tokens, opts.max_tokens);
    if (std.mem.eql(u8, cmd, "check-draws") and rest.len == 1) return checks.draws(gpa, init.io, engine, rest[0]);
    if (std.mem.eql(u8, cmd, "shared-widths") and rest.len == 1) {
        const s: ?lanes.Sampling = if (opts.sampling.temperature > 0) opts.sampling else null;
        return shared_checks.widths(gpa, init.io, engine, rest[0], opts.streams, opts.max_tokens, s);
    }
    if (std.mem.eql(u8, cmd, "check-kernels")) {
        try engine.reset();
        return kernel_checks.check(gpa, engine);
    }
    std.debug.print("{s}", .{usage});
    return 2;
}

const Options = struct {
    model: []const u8,
    tokens: []u32 = &.{},
    max_tokens: usize = 256,
    sampling: lanes.Sampling = .{ .seed = 0, .temperature = 0 },
    drafts: bool = true,
    report: ?[]const u8 = null,
    kernels: ?[]const u8 = null,
    device: ?u32 = null,
    dump: ?[]const u8 = null,
    context: ?usize = null,
    stop_eos: bool = true,
    graphs: bool = true,
    solo: bool = false,
    costs: ?[]f64 = null,
    segments: ?usize = null,
    counts: Counts = .{ .items = .{ 1, 2, 3, 4 }, .len = 4 },
    repeat: usize = 3,
    streams: usize = 4,
    profile: bool = false,
};

/// Segment counts for `segments`, one to four of them.
const Counts = struct { items: [4]usize, len: usize };

fn parseCounts(text: []const u8) !Counts {
    var c: Counts = .{ .items = undefined, .len = 0 };
    var it = std.mem.tokenizeAny(u8, text, ", ");
    while (it.next()) |t| {
        if (c.len == c.items.len) return error.TooManyCounts;
        c.items[c.len] = try std.fmt.parseInt(usize, t, 10);
        c.len += 1;
    }
    return c;
}

/// The GPU ordinal `--device` or `TF_CUDA_DEVICE` picks before an engine loads (PR #354); nothing set means 0.
fn deviceOrdinal(flag: ?u32, env: ?[]const u8) !u32 {
    if (flag) |d| return d;
    const value = env orelse return 0;
    if (value.len == 0) return 0;
    return std.fmt.parseInt(u32, value, 10);
}

fn parseFloats(gpa: std.mem.Allocator, text: []const u8) ![]f64 {
    var out: std.ArrayList(f64) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", ");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseFloat(f64, t));
    return out.toOwnedSlice(gpa);
}

fn parseIds(gpa: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", ");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, t, 10));
    return out.toOwnedSlice(gpa);
}

fn run(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, o: Options) !u8 {
    if (o.tokens.len == 0) return error.NoPromptTokens;
    const room = e.max_len - o.tokens.len - nemotron.state.max_rows;
    const count = @max(1, @min(o.max_tokens, room));
    var drafter: ?nemotron.Drafter = if (o.drafts) try nemotron.Drafter.init(gpa, io, e, o.model, o.graphs, o.costs) else null;
    defer if (drafter) |*d| d.deinit();
    if (drafter) |d| {
        const v = d.rule.costs.verify;
        std.debug.print("up to 15 MTP drafts a round, each verified while it pays for its row (measured: {d:.2}/{d:.2}/{d:.2}/{d:.2}/{d:.2} ms at 1/2/4/8/16 rows, {d:.3} ms a draft)\n", .{ v[1], v[2], v[4], v[8], v[16], d.rule.costs.level });
    }
    const res = try decode.generate(gpa, io, e, if (drafter) |*d| d else null, o.tokens, count, .{ .stop_eos = o.stop_eos });
    defer gpa.free(res.tokens);
    var digest: [32]u8 = undefined;
    const text = try core.ids_json.write(gpa, res.tokens);
    defer gpa.free(text);
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const steps = @max(1, res.tokens.len - 1);
    const ms = res.decode_seconds * 1e3 / @as(f64, @floatFromInt(steps));
    std.debug.print("tokens {d} sha {s} prefill {d:.4}s decode {d:.4}s {d:.3} ms/token rounds {d} accepted {d} segments {d}\n", .{ res.tokens.len, hex[0..12], res.prefill_seconds, res.decode_seconds, ms, res.rounds, res.accepted, e.segments });
    if (o.report) |path| {
        const report = .{
            .engine = "zig-cuda",
            .prompt_tokens = o.tokens,
            .tokens = res.tokens,
            .token_sha256 = hex,
            .prefill_seconds = res.prefill_seconds,
            .decode_seconds = res.decode_seconds,
            .ms_per_token = ms,
            .rounds = res.rounds,
            .accepted_drafts = res.accepted,
            .drafted = res.drafted,
            .drafts = o.drafts,
            .sampling = e.sampling,
            .graphs = o.graphs,
            .load_seconds = e.load_seconds,
            .max_len = e.max_len,
            .segments = e.segments,
        };
        const json = try std.json.Stringify.valueAlloc(gpa, report, .{});
        defer gpa.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }
    return 0;
}

test "the device ordinal comes from --device, then TF_CUDA_DEVICE, else 0" {
    try std.testing.expectEqual(@as(u32, 1), try deviceOrdinal(1, null));
    try std.testing.expectEqual(@as(u32, 1), try deviceOrdinal(1, "7")); // the flag wins
    try std.testing.expectEqual(@as(u32, 3), try deviceOrdinal(null, "3"));
    try std.testing.expectEqual(@as(u32, 0), try deviceOrdinal(null, null));
    try std.testing.expectEqual(@as(u32, 0), try deviceOrdinal(null, ""));
    try std.testing.expectError(error.Overflow, deviceOrdinal(null, "4294967296"));
}
