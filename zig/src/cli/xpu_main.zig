//! `tensorfold-xpu` on Linux with an Intel GPU: a model run on token ids, serial decode; model_type picks the family.

const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const nemotron = @import("nemotron_xpu");
const decode = xpu.decode;

/// A panic drains the device queue (bounded wait) before the default panic runs, so no work is left queued.
pub const panic = std.debug.FullPanic(xpu.rt.panicDrain);

const usage =
    \\usage: tensorfold-xpu run MODEL --tokens ID,ID,... | --tokens-file PATH [--max-tokens N] [--no-drafts] [--ignore-eos] [--context N]
    \\         [--device N] [--force ID,ID,...] [--logits] [--report PATH]
    \\       tensorfold-xpu devices
    \\  --force   teacher forcing: feed these ids after the prompt, one step per id (the end token never stops)
    \\  --prefill ROWS  the prompt in windows of ROWS tokens (16 and up)
    \\  --logits  print each step's five largest logits ("step N pos P top5: id:logit ...")
    \\  Serial decode only: MTP drafts, sampling and the CUDA engine's graph and kernel options are not available here.
    \\
;

/// CUDA `run` options this engine cannot honour; naming one is an error, not a no-op.
const unsupported = [_][]const u8{ "--temperature", "--top-k", "--top-p", "--min-p", "--seed", "--kernels", "--dump", "--costs", "--segments", "--counts", "--repeat", "--profile", "--eager", "--solo" };

const Options = struct {
    model: []const u8,
    tokens: []u32 = &.{},
    force: ?[]u32 = null,
    max_tokens: usize = 256,
    context: usize = 4096,
    stop_eos: bool = true,
    logits: bool = false,
    device: ?u32 = null,
    report: ?[]const u8 = null,
    prefill: u32 = 0,
};


const Parsed = union(enum) { run: Options, refuse: []const u8 };

fn isUnsupported(a: []const u8) bool {
    for (unsupported) |u| if (std.mem.eql(u8, a, u)) return true;
    return false;
}

fn parseIds(gpa: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", ");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, t, 10));
    return out.toOwnedSlice(gpa);
}

/// The ids in a file (comma or space separated), for prompts too long for a command line.
fn readIds(gpa: std.mem.Allocator, path: []const u8) ![]u32 {
    const text = if (path.len > 0 and path[0] == '/') try xpu.loader.readFile(gpa, "/", path[1..]) else try xpu.loader.readFile(gpa, ".", path);
    defer gpa.free(text);
    return parseIds(gpa, std.mem.trim(u8, text, " \r\n"));
}

/// A refusal; the ids parsed so far are dropped.
fn refuse(gpa: std.mem.Allocator, o: Options, what: []const u8) Parsed {
    gpa.free(o.tokens);
    if (o.force) |f| gpa.free(f);
    return .{ .refuse = what };
}

/// The options after `run MODEL`; an unsupported or unknown one comes back as a refusal naming it.
fn parse(gpa: std.mem.Allocator, model: []const u8, rest: []const []const u8) !Parsed {
    var o: Options = .{ .model = model };
    errdefer gpa.free(o.tokens);
    errdefer if (o.force) |f| gpa.free(f);
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const a = rest[i];
        const takes = for ([_][]const u8{ "--tokens", "--tokens-file", "--max-tokens", "--context", "--ctx", "--device", "--force", "--report", "--prefill" }) |n| {
            if (std.mem.eql(u8, a, n)) break true;
        } else false;
        if (takes and i + 1 >= rest.len) return refuse(gpa, o, "missing value after the option");
        const value = if (takes) rest[i + 1] else "";
        if (takes) i += 1;
        if (std.mem.eql(u8, a, "--tokens")) {
            gpa.free(o.tokens);
            o.tokens = try parseIds(gpa, value);
        } else if (std.mem.eql(u8, a, "--tokens-file")) {
            gpa.free(o.tokens);
            o.tokens = try readIds(gpa, value);
        } else if (std.mem.eql(u8, a, "--force")) {
            if (o.force) |f| gpa.free(f);
            o.force = try parseIds(gpa, value);
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            o.max_tokens = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, a, "--context") or std.mem.eql(u8, a, "--ctx")) {
            o.context = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, a, "--device")) {
            o.device = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, a, "--report")) {
            o.report = value;
        } else if (std.mem.eql(u8, a, "--prefill")) {
            o.prefill = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, a, "--ignore-eos")) {
            o.stop_eos = false;
        } else if (std.mem.eql(u8, a, "--logits")) {
            o.logits = true;
        } else if (std.mem.eql(u8, a, "--no-drafts")) {
            // serial decode is the only mode
        } else return refuse(gpa, o, a);
    }
    return .{ .run = o };
}

pub fn main(init: std.process.Init) !u8 {
    xpu.stop.install();
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    if (std.mem.eql(u8, args[1], "devices")) return devices();
    if (!std.mem.eql(u8, args[1], "run") or args.len < 3) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const parsed = try parse(gpa, args[2], args[3..]);
    const o = switch (parsed) {
        .run => |o| o,
        .refuse => |what| {
            if (isUnsupported(what)) {
                std.debug.print("{s} is not available on the Intel GPU engine (serial greedy decode only)\n{s}", .{ what, usage });
            } else std.debug.print("unknown option or bad usage: {s}\n{s}", .{ what, usage });
            return 2;
        },
    };
    defer gpa.free(o.tokens);
    defer if (o.force) |f| gpa.free(f);
    if (o.tokens.len == 0) {
        std.debug.print("run needs --tokens\n{s}", .{usage});
        return 2;
    }
    const fam = try detect(gpa, init.io, o.model);
    var driver = try xpu.Driver.open();
    defer driver.close();
    var ctx = try xpu.Context.init(&driver, o.device);
    defer ctx.deinit();
    std.debug.print("device {s}, max single allocation {d:.2} GB\n", .{ ctx.name(), @as(f64, @floatFromInt(ctx.maxAlloc())) / 1e9 });
    if (!hasNoDrafts(args)) std.debug.print("MTP drafts are not available on the Intel GPU engine: decoding serially\n", .{});
    switch (fam) {
        .nemotron => {
            const e = nemotron.Engine.init(gpa, init.io, &ctx, o.model, .{ .context = o.context, .progress = true, .window_rows = o.prefill }) catch |err| {
                if (err == error.UnsupportedFormat) {
                    std.debug.print("{s} is not a checkpoint the Intel GPU engine reads: NVFP4 checkpoints are not supported on the XPU backend yet; use the MLX 4-bit checkpoint\n", .{o.model});
                    return 2;
                }
                return err;
            };
            defer e.deinit();
            return load(gpa, init.io, e, o);
        },
    }
}

const Family = enum { nemotron };

/// The family a config.json's model_type names; any other type is refused.
fn family(model_type: []const u8) ?Family {
    if (std.mem.eql(u8, model_type, "nemotron_h")) return .nemotron;
    return null;
}

fn detect(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Family {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22)) catch |e| {
        std.debug.print("cannot read {s}: {t}\n", .{ path, e });
        return error.NoConfig;
    };
    defer gpa.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const t = if (parsed.value == .object) parsed.value.object.get("model_type") else null;
    const name = if (t) |v| (if (v == .string) v.string else "") else "";
    return family(name) orelse {
        std.debug.print("model_type \"{s}\" in {s} is not supported on the Intel GPU engine (nemotron_h)\n", .{ name, path });
        return error.UnsupportedModel;
    };
}

fn load(gpa: std.mem.Allocator, io: std.Io, e: anytype, o: Options) !u8 {
    std.debug.print("loaded in {d:.1} s, device memory {d:.2} GB\n", .{ e.load_seconds, @as(f64, @floatFromInt(e.deviceBytes())) / 1e9 });
    return run(gpa, io, e, o) catch |err| {
        if (std.mem.eql(u8, @errorName(err), "Interrupted")) {
            std.debug.print("interrupted: the queue was drained, device memory is freed on exit\n", .{});
            return 130;
        }
        return err;
    };
}

fn hasNoDrafts(args: []const []const u8) bool {
    for (args) |a| if (std.mem.eql(u8, a, "--no-drafts")) return true;
    return false;
}

fn devices() !u8 {
    var driver = try xpu.Driver.open();
    defer driver.close();
    var all: [xpu.context.max_devices]xpu.context.Entry = undefined;
    for (try xpu.context.enumerate(&driver, &all), 0..) |*e, i| {
        const p = e.props;
        const eus = p.num_slices * p.num_subslices_per_slice * p.num_eus_per_subslice;
        std.debug.print("device {d}: {s} EUs={d} clock={d}MHz maxalloc={d}MiB\n", .{ i, xpu.context.entryName(e), eus, p.core_clock_rate, p.max_mem_alloc_size >> 20 });
    }
    return 0;
}

fn run(gpa: std.mem.Allocator, io: std.Io, e: anytype, o: Options) !u8 {
    const room = if (o.force) |f| f.len else e.max_len + 1 - @min(e.max_len + 1, o.tokens.len);
    const count = @max(1, @min(o.max_tokens, room));
    const res = try decode.generate(gpa, io, e, o.tokens, count, .{ .stop_eos = o.stop_eos, .force = o.force, .top5 = o.logits, .prefill_rows = o.prefill });
    defer gpa.free(res.tokens);
    defer gpa.free(res.top5);
    defer gpa.free(res.lpf);
    for (res.top5, 0..) |t, i| {
        std.debug.print("step {d} pos {d} top5:", .{ i, o.tokens.len - 1 + i });
        for (t) |x| std.debug.print(" {d}:{d:.4}", .{ x.id, x.v });
        std.debug.print("\n", .{});
        // forced runs: the log-probability of the forced token and of our top-1, for the reference comparisons
        if (o.force) |f| if (i < res.lpf.len) std.debug.print("step {d} lpf {d} {d:.5} {d:.5}\n", .{ i, f[i], res.lpf[i][0], res.lpf[i][1] });
    }
    if (res.interrupted) std.debug.print("interrupted: stopped after {d} tokens, device work finished\n", .{res.tokens.len});
    std.debug.print("generated ids:", .{});
    for (res.tokens) |g| std.debug.print(" {d}", .{g});
    std.debug.print("\n", .{});
    var digest: [32]u8 = undefined;
    const text = try core.ids_json.write(gpa, res.tokens);
    defer gpa.free(text);
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const steps = @max(1, res.tokens.len -| 1);
    const ms = res.decode_seconds * 1e3 / @as(f64, @floatFromInt(steps));
    std.debug.print("tokens {d} sha {s} prefill {d:.4}s decode {d:.4}s {d:.3} ms/token ({d:.1} tokens/s) rounds {d} accepted 0\n", .{ res.tokens.len, hex[0..12], res.prefill_seconds, res.decode_seconds, ms, 1e3 / ms, res.rounds });
    if (o.report) |path| {
        const report = .{
            .engine = "zig-xpu",
            .prompt_tokens = o.tokens,
            .tokens = res.tokens,
            .token_sha256 = hex,
            .prefill_seconds = res.prefill_seconds,
            .decode_seconds = res.decode_seconds,
            .ms_per_token = ms,
            .rounds = res.rounds,
            .accepted_drafts = 0,
            .drafted = 0,
            .drafts = false,
            .forced = o.force != null,
            .load_seconds = e.load_seconds,
            .max_len = e.max_len,
            .device_bytes = e.deviceBytes(),
        };
        const json = try std.json.Stringify.valueAlloc(gpa, report, .{});
        defer gpa.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }
    return if (res.interrupted) 130 else 0;
}

test "the CUDA run options this engine cannot honour are refused by name" {
    const gpa = std.testing.allocator;
    for (unsupported) |u| {
        const p = try parse(gpa, "m", &.{ "--tokens", "1,2", u });
        try std.testing.expectEqualStrings(u, p.refuse);
    }
    const p = try parse(gpa, "m", &.{ "--bogus" });
    try std.testing.expectEqualStrings("--bogus", p.refuse);
}

test "model types pick a family, others are refused" {
    try std.testing.expectEqual(Family.nemotron, family("nemotron_h").?);
    try std.testing.expect(family("llama") == null and family("") == null);
}

test "run options parse" {
    const gpa = std.testing.allocator;
    const p = try parse(gpa, "m", &.{ "--tokens", "1, 2,3", "--max-tokens", "5", "--no-drafts", "--ignore-eos", "--force", "7,8", "--device", "1", "--logits" });
    defer gpa.free(p.run.tokens);
    defer gpa.free(p.run.force.?);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, p.run.tokens);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, p.run.force.?);
    try std.testing.expectEqual(@as(usize, 5), p.run.max_tokens);
    try std.testing.expect(!p.run.stop_eos and p.run.logits);
    try std.testing.expectEqual(@as(?u32, 1), p.run.device);
}
