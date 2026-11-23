//! The `models` command: the documented chat checkpoints that fit this machine with a rough speed, and those on disk.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");
const catalog = @import("catalog.zig");
const sys = @import("sys");
const machine = sys.machine;
const Machine = machine.Machine;
const Entry = catalog.Entry;

const gib = machine.gib;
/// Room past the floor for a context and caches before a fit counts as comfortable.
const room: u64 = 4 * gib;
/// What a Mac keeps for macOS once its GPU limit is raised.
const mac_reserve: u64 = 8 * gib;

pub const Options = struct { json: bool = false, all: bool = false };

/// One listed checkpoint: the facts a line prints, and the fields of a `--json` entry.
const Listed = struct { repo: []const u8, family: []const u8, model_type: []const u8 };

pub const Fit = enum { fits, tight, raise_limit, untested, too_big, pair, other_backend, unknown };

/// A speed shown for this machine: measured on its chip, or scaled from another chip by memory bandwidth.
pub const Estimate = struct { tok_s: f64, measured: bool, from: []const u8, plain: bool };

const Disk = enum { no, yes, drafter_missing };
const Row = struct { e: *const Entry, need: u64, fit: Fit, speed: ?Estimate, disk: Disk };

/// The engine's floor: what it keeps resident plus the runtime margin `info` reports.
pub fn need(e: *const Entry) u64 {
    return e.residentBytes() + sys.memory_policy.margin;
}

pub fn fitOf(e: *const Entry, mc: *const Machine) Fit {
    if (mc.backend == .none) return .unknown;
    if (e.machines > 1) return .pair;
    if (!(if (mc.backend == .metal) e.metal else e.cuda)) return .other_backend;
    const n = need(e);
    if (n > mc.gpu_limit) return if (mc.backend == .metal and n + mac_reserve <= mc.memory) .raise_limit else .too_big;
    if (mc.backend == .metal and e.m5 == .only and !m5(mc.chip())) return .untested;
    return if (n + room <= mc.gpu_limit) .fits else .tight;
}

/// The one checkpoint `start` selects, with pinned pull specs and the snapshot directories serve must use.
pub const StartChoice = struct {
    entry: *const Entry,
    model: []const u8,
    drafter: ?[]const u8,
    model_dir: []const u8,
    drafter_dir: ?[]const u8,
    download_bytes: u64,
    cached: bool,
};

fn pinned(repo: []const u8, revision: []const u8, a: Allocator) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}@{s}", .{ repo, revision });
}

/// A pinned entry is cached only after pull publishes its exact revision through refs/main.
fn startSnapshot(a: Allocator, io: std.Io, root: []const u8, repo: []const u8, revision: []const u8) !struct { []const u8, bool } {
    const repo_dir = try std.fs.path.join(a, &.{ root, try hub.repoDirName(a, repo) });
    const dir = try std.fs.path.join(a, &.{ repo_dir, "snapshots", revision });
    var ref_buf: [128]u8 = undefined;
    const ref = std.Io.Dir.cwd().readFile(io, try std.fs.path.join(a, &.{ repo_dir, "refs", "main" }), &ref_buf) catch return .{ dir, false };
    if (!std.mem.eql(u8, std.mem.trim(u8, ref, " \r\n"), revision)) return .{ dir, false };
    return .{ dir, hub.isDir(io, dir) and @import("checkpoint_config").optional(a, io, dir) != null };
}

/// Largest comfortably fitting fully pinned `models` row that `tensorfold pull` can install; ties keep registry order.
pub fn chooseForStart(a: Allocator, io: std.Io, env: ?*const std.process.Environ.Map, override: ?[]const u8, entries: []const Entry, mc: *const Machine) !?StartChoice {
    var best: ?*const Entry = null;
    for (entries) |*e| {
        if (e.hf_dir != null) continue;
        const revision = e.revision orelse continue;
        if (!hub.hexSha(revision)) continue;
        const drafting = startUsesDrafter(e, mc);
        if (drafting and (e.drafter_revision == null or !hub.hexSha(e.drafter_revision.?))) continue;
        if (fitOf(e, mc) != .fits) continue;
        if (best == null or need(e) > need(best.?)) best = e;
    }
    const e = best orelse return null;
    const drafting = startUsesDrafter(e, mc);
    const root = try hub.cacheDir(a, env, override);
    const main = try startSnapshot(a, io, root, e.repo, e.revision.?);
    const draft = if (drafting) try startSnapshot(a, io, root, e.drafter.?, e.drafter_revision.?) else .{ @as([]const u8, ""), true };
    return .{
        .entry = e,
        .model = try pinned(e.repo, e.revision.?, a),
        .drafter = if (drafting) try pinned(e.drafter.?, e.drafter_revision.?, a) else null,
        .model_dir = main[0],
        .drafter_dir = if (drafting) draft[0] else null,
        .download_bytes = if (drafting) e.download else e.download_main orelse e.download,
        .cached = main[1] and draft[1],
    };
}

/// `.m5 = .drafts` rows serve plain on pre-M5 Metal; M5 and CUDA keep their pinned drafter.
fn startUsesDrafter(e: *const Entry, mc: *const Machine) bool {
    return e.drafter != null and !(mc.backend == .metal and e.m5 == .drafts and !m5(mc.chip()));
}

/// A measurement on this chip, else the nearest on this backend (same chip generation first) scaled by bandwidth.
pub fn estimate(e: *const Entry, mc: *const Machine) ?Estimate {
    if (mc.backend == .none) return null;
    const plain = mc.backend == .metal and e.m5 == .drafts and !m5(mc.chip());
    for (e.speeds) |s| if (s.backend == mc.backend and s.plain == plain and contains(mc.chip(), s.chip))
        return .{ .tok_s = s.tok_s, .measured = true, .from = s.chip, .plain = plain };
    if (mc.bandwidth == 0) return null;
    const here: f64 = @floatFromInt(mc.bandwidth);
    var best: ?catalog.Speed = null;
    var best_score = std.math.inf(f64);
    for (e.speeds) |s| {
        if (s.backend != mc.backend or s.plain != plain) continue;
        const bw = machine.bandwidthOf(s.chip);
        if (bw == 0) continue;
        const score = @abs(@log(@as(f64, @floatFromInt(bw)) / here)) + @as(f64, if (generation(s.chip) == generation(mc.chip())) 0 else 10);
        if (score < best_score) {
            best_score = score;
            best = s;
        }
    }
    const s = best orelse return null;
    return .{ .tok_s = s.tok_s * here / @as(f64, @floatFromInt(machine.bandwidthOf(s.chip))), .measured = false, .from = s.chip, .plain = plain };
}

/// Lists the catalog against this machine (or `given`) and the cache's checkpoints; returns the process exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, opts: Options, given: ?Machine) !u8 {
    const root = try hub.cacheDir(a, env, override);
    const listed = try cached(a, io, root);
    const mc = given orelse machine.detect(a, io);
    const rows = try a.alloc(Row, catalog.entries.len);
    for (rows, &catalog.entries) |*r, *e| r.* = .{ .e = e, .need = need(e), .fit = fitOf(e, &mc), .speed = estimate(e, &mc), .disk = try diskOf(a, io, root, env, e) };
    std.mem.sort(Row, rows, {}, struct {
        fn less(_: void, x: Row, y: Row) bool {
            if (x.fit != y.fit) return @backingInt(x.fit) < @backingInt(y.fit);
            return x.need > y.need;
        }
    }.less);
    if (opts.json) return printJson(a, out, listed, &mc, rows);
    try printMachine(out, &mc);
    const shown = runnable(rows, mc.backend == .none or opts.all);
    if (shown == 0) {
        try out.writeAll("No catalog model fits here.\n");
    } else {
        try out.print("{s}, largest first:\n", .{if (mc.backend == .none) "Every model and the memory it needs" else if (mc.backend == .metal) "Models that fit this Mac" else "Models that fit this machine"});
        try printRows(a, out, rows[0..shown], &mc, true);
    }
    if (opts.all and shown < rows.len and mc.backend != .none) {
        try out.writeAll("\nThe rest, with what each needs:\n");
        try printRows(a, out, rows[shown..], &mc, false);
    }
    try out.print("\nMemory is the weights and any drafter the engine keeps, plus a {d} GiB margin; context and caches take more.\n", .{sys.memory_policy.margin >> 30});
    try out.writeAll("Speed is one stream with drafts as shipped: measured on that chip, or estimated from another by memory bandwidth.\n");
    try out.writeAll("Quality is the share of TensorFold Web's 228-item quality suite the checkpoint gets right (greedy, thinking off);\n" ++
        "each score's receipt is in docs/receipts/quality-suite, and - is not rated yet.\n");
    if (!opts.all and shown < rows.len) try out.print("{d} more need more memory, two machines or the other backend: tensorfold models --all\n", .{rows.len - shown});
    try printCached(out, root, listed);
    return 0;
}

/// Rows sort runnable fits first; this counts them (every row when the machine is unknown).
fn runnable(rows: []const Row, every: bool) usize {
    if (every and rows.len > 0 and rows[0].fit == .unknown) return rows.len;
    var n: usize = 0;
    for (rows) |r| {
        if (@backingInt(r.fit) > @backingInt(Fit.untested)) break;
        n += 1;
    }
    return n;
}

fn printMachine(out: *std.Io.Writer, mc: *const Machine) !void {
    const name = if (mc.chip().len > 0) mc.chip() else "This machine";
    if (mc.backend == .none) return out.print("{s}: no Metal or CUDA GPU found.\n\n", .{name});
    try out.writeAll(name);
    if (mc.gpu_cores > 0) try out.print(", {d}-core GPU", .{mc.gpu_cores});
    if (mc.gpus > 1) try out.print(", {d} GPUs (one counted)", .{mc.gpus});
    if (mc.memory > 0) try out.print(", {d:.0} GiB memory", .{gibs(mc.memory)});
    try out.print(": {s} may hold {d:.1} GiB", .{ if (mc.backend == .metal) "Metal" else "CUDA", gibs(mc.gpu_limit) });
    if (mc.bandwidth > 0) try out.print(", {d} GB/s", .{mc.bandwidth / 1_000_000_000});
    try out.writeAll(".\n\n");
}

fn printRows(a: Allocator, out: *std.Io.Writer, rows: []const Row, mc: *const Machine, install: bool) !void {
    var width: usize = 0;
    for (rows) |r| width = @max(width, r.e.title.len + 2 + r.e.format.len);
    try out.splatByteAll(' ', width + 4);
    try out.print("    memory  quality  {s}\n", .{if (install) "speed, one stream" else "why not here"});
    for (rows) |r| {
        const name = try std.fmt.allocPrint(a, "{s}, {s}", .{ r.e.title, r.e.format });
        try out.print("  {s}", .{name});
        try out.splatByteAll(' ', width - name.len + 2);
        try out.print("{d:>6.1} GiB  ", .{gibs(r.need)});
        if (r.e.quality) |q| try out.print("{d:>6.0}%  ", .{share(q) * 100}) else try out.writeAll("      -  ");
        switch (r.fit) {
            .too_big => try out.writeAll("more than the GPU may hold"),
            .pair => try out.writeAll("served by two machines"),
            .other_backend => try out.writeAll(if (mc.backend == .metal) "CUDA only" else "Metal only"),
            else => try printSpeed(out, r, mc),
        }
        try out.writeAll(switch (r.disk) {
            .yes => "  [on disk]",
            .drafter_missing => "  [on disk, drafter missing]",
            .no => "",
        });
        try out.writeByte('\n');
        if (install) try out.print("      {s}\n", .{try installLine(a, r.e)}) else try out.print("      {s}\n", .{r.e.recipe});
    }
}

fn printSpeed(out: *std.Io.Writer, r: Row, mc: *const Machine) !void {
    if (r.speed) |s| {
        try out.print("{s}{d:.0} tok/s", .{ if (s.measured) "" else "~", s.tok_s });
        if (s.measured) try out.print(", measured on {s}", .{s.from}) else try out.print(", est. from {s}", .{s.from});
        if (s.plain) try out.writeAll(", no drafts before M5");
    } else if (mc.backend != .none) try out.writeAll("speed not measured yet");
    try out.writeAll(switch (r.fit) {
        .tight => "; little room for context",
        .raise_limit => "; raise iogpu.wired_limit_mb (serve names the value)",
        .untested => "; untested on M1-M4",
        else => "",
    });
}

/// The commands the supported-models guide gives: `tensorfold pull` with any drafter, or `hf download` into ~/models.
pub fn installLine(a: Allocator, e: *const Entry) ![]const u8 {
    const main = if (e.hf_dir) |d|
        try std.fmt.allocPrint(a, "hf download {s} --revision {s} --local-dir ~/models/{s}", .{ e.repo, e.revision.?, d })
    else
        try std.fmt.allocPrint(a, "tensorfold pull {s}{s}{s}", .{ e.repo, if (e.revision != null) "@" else "", e.revision orelse "" });
    const d = e.drafter orelse return main;
    return std.fmt.allocPrint(a, "{s} && tensorfold pull {s}{s}{s}", .{ main, d, if (e.drafter_revision != null) "@" else "", e.drafter_revision orelse "" });
}

fn printCached(out: *std.Io.Writer, root: []const u8, listed: []const Listed) !void {
    if (listed.len == 0) return out.print("\nNo checkpoints a native family serves are in the Hugging Face cache at {s} yet.\n", .{root});
    try out.writeAll("\nIn the Hugging Face cache:\n");
    for (listed) |m| try out.print("{s}: {s} ({s})\n", .{ m.repo, m.family, m.model_type });
}

const Found = struct { listed: Listed, snapshot: []const u8 };

/// The cache's checkpoints a Zig family serves, with their snapshot folders; null when the cache does not exist.
fn find(a: Allocator, io: std.Io, root: []const u8) !?[]const Found {
    var dir = std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var found: std.ArrayList(Found) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory) continue;
        const repo = (repoFromDirName(a, entry.name) catch continue) orelse continue;
        const snapshot = (try hub.cachedSnapshot(a, io, root, repo)) orelse continue;
        const model_type = hub.modelType(a, io, snapshot);
        const family = hub.family(model_type) orelse continue;
        try found.append(a, .{ .listed = .{ .repo = repo, .family = family.title, .model_type = family.model_type }, .snapshot = snapshot });
    }
    return found.items;
}

/// A checkpoint `models` lists, with its snapshot's size, for `tensorfold device`'s fit against this machine.
pub const OnDisk = struct { repo: []const u8, family: []const u8, model_type: []const u8, bytes: u64 };

pub fn onDisk(a: Allocator, io: std.Io, env: ?*const std.process.Environ.Map, override: ?[]const u8) ![]const OnDisk {
    const found = (try find(a, io, try hub.cacheDir(a, env, override))) orelse return &.{};
    const out = try a.alloc(OnDisk, found.len);
    for (out, found) |*o, f| o.* = .{ .repo = f.listed.repo, .family = f.listed.family, .model_type = f.listed.model_type, .bytes = try hub.sizeOf(a, io, f.snapshot) };
    return out;
}

/// The same cached checkpoints as `onDisk`, without their snapshot sizes.
fn cached(a: Allocator, io: std.Io, root: []const u8) ![]const Listed {
    const found = (try find(a, io, root)) orelse return &.{};
    const out = try a.alloc(Listed, found.len);
    for (out, found) |*o, f| o.* = f.listed;
    return out;
}

fn diskOf(a: Allocator, io: std.Io, root: []const u8, env: ?*const std.process.Environ.Map, e: *const Entry) !Disk {
    const home = if (env) |m| m.get("HOME") orelse m.get("USERPROFILE") else null;
    const main = if (e.hf_dir) |d|
        home != null and hub.isDir(io, try std.fs.path.join(a, &.{ home.?, "models", d }))
    else
        (try hub.cachedSnapshot(a, io, root, e.repo)) != null;
    if (!main) return .no;
    if (e.drafter) |d| if ((try hub.cachedSnapshot(a, io, root, d)) == null) return .drafter_missing;
    return .yes;
}

const JsonMachine = struct { chip: []const u8, backend: []const u8, gpu_cores: u32, gpus: u32, memory_bytes: u64, gpu_limit_bytes: u64, unified: bool, bandwidth_bytes_per_second: u64 };
const JsonQuality = struct { passed: u16, items: u16, share: f64, digest: []const u8, receipt: []const u8 };
const JsonEntry = struct { title: []const u8, format: []const u8, repo: []const u8, revision: ?[]const u8, drafter: ?[]const u8, need_bytes: u64, download_bytes: u64, fit: []const u8, tok_s: ?f64, tok_s_measured: ?bool, tok_s_from: ?[]const u8, quality: ?JsonQuality, on_disk: bool, install: []const u8, recipe: []const u8 };

/// The share of graded quality items passed, 0 to 1.
fn share(q: catalog.Quality) f64 {
    return @as(f64, @floatFromInt(q.passed)) / @as(f64, @floatFromInt(q.items));
}

/// `{"schema": 1, "models": [...]}` as before (the cache), plus this machine and every catalog entry against it.
fn printJson(a: Allocator, out: *std.Io.Writer, listed: []const Listed, mc: *const Machine, rows: []const Row) !u8 {
    const entries = try a.alloc(JsonEntry, rows.len);
    for (entries, rows) |*j, r| j.* = .{
        .title = r.e.title,
        .format = r.e.format,
        .repo = r.e.repo,
        .revision = r.e.revision,
        .drafter = r.e.drafter,
        .need_bytes = r.need,
        .download_bytes = r.e.download,
        .fit = @tagName(r.fit),
        .tok_s = if (r.speed) |s| @round(s.tok_s * 10) / 10 else null,
        .tok_s_measured = if (r.speed) |s| s.measured else null,
        .tok_s_from = if (r.speed) |s| s.from else null,
        .quality = if (r.e.quality) |q| .{ .passed = q.passed, .items = q.items, .share = @round(share(q) * 1000) / 1000, .digest = catalog.quality_digest, .receipt = q.receipt } else null,
        .on_disk = r.disk == .yes,
        .install = try installLine(a, r.e),
        .recipe = r.e.recipe,
    };
    const m: JsonMachine = .{ .chip = mc.chip(), .backend = @tagName(mc.backend), .gpu_cores = mc.gpu_cores, .gpus = mc.gpus, .memory_bytes = mc.memory, .gpu_limit_bytes = mc.gpu_limit, .unified = mc.unified, .bandwidth_bytes_per_second = mc.bandwidth };
    try std.json.Stringify.value(.{ .schema = 1, .models = listed, .machine = m, .catalog = entries }, .{}, out);
    try out.writeByte('\n');
    return 0;
}

fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(gib));
}

fn contains(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

/// M5 GPUs have the tensor units some drafters need.
fn m5(chip: []const u8) bool {
    return contains(chip, "M5");
}

/// An Apple chip's generation digit (M3 Ultra -> 3); 0 for other chips.
fn generation(chip: []const u8) u8 {
    var i: usize = 0;
    while (i + 1 < chip.len) : (i += 1) if (chip[i] == 'M' and std.ascii.isDigit(chip[i + 1])) return chip[i + 1] - '0';
    return 0;
}

/// models--Org--Name back to Org/Name: the first double dash separates them; other entries do not name a repo.
fn repoFromDirName(a: Allocator, name: []const u8) !?[]const u8 {
    const prefix = "models--";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const rest = name[prefix.len..];
    const sep = std.mem.indexOf(u8, rest, "--") orelse return null;
    if (sep == 0 or sep + 2 >= rest.len) return null;
    return try std.fmt.allocPrint(a, "{s}/{s}", .{ rest[0..sep], rest[sep + 2 ..] });
}

/// A cache with one checkpoint a Zig family serves (Org/Flash) and one none does (Org/Draft); the caller deletes it.
fn syntheticCache(a: Allocator, io: std.Io) ![]const u8 {
    const root = try std.fmt.allocPrint(a, ".tf-models-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    const w = std.Io.Dir.cwd();
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1" }));
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs/main" }), .data = "rev1" });
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1/config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}}" });
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2/config.json" }), .data = "{\"model_type\": \"gemma3n\"}" });
    return root;
}

fn mac(chip: []const u8, cores: u32, memory: u64) Machine {
    var m: Machine = .{ .backend = .metal, .unified = true, .gpu_cores = cores, .memory = memory, .gpu_limit = machine.defaultGpuLimit(memory) };
    m.setChip(chip);
    m.bandwidth = machine.bandwidthFor(chip, cores);
    return m;
}

fn pick(repo: []const u8, format: []const u8) *const Entry {
    for (&catalog.entries) |*e| if (std.mem.eql(u8, e.repo, repo) and std.mem.eql(u8, e.format, format)) return e;
    unreachable;
}

test "models lists serve-able checkpoints and skips families Zig cannot serve" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try syntheticCache(a, io);
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};

    var out: std.Io.Writer.Allocating = .init(a);
    const code = try run(a, io, &out.writer, null, root, .{}, mac("Apple M5 Max", 40, 128 * gib));
    try std.testing.expectEqual(@as(u8, 0), code);
    const listed = out.written();
    try std.testing.expect(has(listed, "Org/Flash: Qwen3.8 Flash Next (qwen4_exp)"));
    try std.testing.expect(std.mem.indexOf(u8, listed, "Org/Draft") == null);
}

test "an M5 Max lists what fits largest first, measured and estimated speeds, and hides the rest without --all" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const missing = try std.fmt.allocPrint(a, ".tf-models-none-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});

    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out.writer, null, missing, .{}, mac("Apple M5 Max", 40, 128 * gib)));
    const text = out.written();
    try std.testing.expect(has(text, "Apple M5 Max, 40-core GPU, 128 GiB memory: Metal may hold 107.5 GiB, 614 GB/s."));
    try std.testing.expect(has(text, "Models that fit this Mac, largest first:"));
    try std.testing.expect(has(text, "Qwen3.8-27B, MLX 4-bit + DFlash2"));
    try std.testing.expect(has(text, "64 tok/s, measured on M5 Max"));
    try std.testing.expect(has(text, "    memory  quality  speed, one stream\n"));
    try std.testing.expect(has(text, "19.3 GiB      84%  263 tok/s, measured on M5 Max")); // Lightning's receipt: 192 of 228
    try std.testing.expect(has(text, "GiB        -  speed not measured yet")); // an unrated, unmeasured checkpoint
    try std.testing.expect(has(text, "Quality is the share of TensorFold Web's 228-item quality suite"));
    try std.testing.expect(has(text, "tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit && tensorfold pull z-lab/Qwen3.8-27B-DFlash2"));
    try std.testing.expect(has(text, "est. from M5 Ultra"));
    try std.testing.expect(std.mem.indexOf(u8, text, "MLX 6-bit + MTP") == null); // 147 GiB of Flash Next does not fit
    try std.testing.expect(std.mem.indexOf(u8, text, "NVFP4") == null); // CUDA only
    try std.testing.expect(has(text, "more need more memory, two machines or the other backend: tensorfold models --all"));
    try std.testing.expect(std.mem.indexOf(u8, text, "Qwen3.8 Flash Next, MLX 4-bit") == null or
        std.mem.indexOf(u8, text, "Qwen3.8 Flash Next, MLX 4-bit").? < std.mem.indexOf(u8, text, "Qwen3.8-27B").?); // largest first

    var all: std.Io.Writer.Allocating = .init(a);
    _ = try run(a, io, &all.writer, null, missing, .{ .all = true }, mac("Apple M5 Max", 40, 128 * gib));
    try std.testing.expect(has(all.written(), "The rest, with what each needs:"));
    try std.testing.expect(has(all.written(), "MLX 6-bit + MTP"));
    try std.testing.expect(has(all.written(), "served by two machines"));
    try std.testing.expect(has(all.written(), "CUDA only"));
    try std.testing.expect(has(all.written(), "more than the GPU may hold"));
}

test "the fit rule: comfortable, tight, a raised Mac limit, too big, untested, and the other backend" {
    const small = mac("Apple M4 Pro", 0, 24 * gib); // 16 GiB for the GPU
    try std.testing.expectEqual(Fit.fits, fitOf(pick("mlx-community/Qwen3.5-2B-MLX-4bit", "MLX 4-bit"), &small));
    try std.testing.expectEqual(Fit.too_big, fitOf(pick("TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "MLX 4-bit + MTP"), &small)); // 19.3 GiB leaves macOS under 8
    try std.testing.expectEqual(Fit.too_big, fitOf(pick("TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP", "MLX 6-bit + MTP"), &small));
    try std.testing.expectEqual(Fit.other_backend, fitOf(pick("Aleph-Alpha/Kolibri-1", "block FP8"), &small));
    try std.testing.expectEqual(Fit.pair, fitOf(pick("TensorFold/GLM-5.3-Flash-MLX-4bit-MTP", "MLX 4-bit + MTP"), &mac("Apple M5 Ultra", 0, 512 * gib)));

    const m32 = mac("Apple M5 Max", 32, 32 * gib); // 21.3 GiB for the GPU; the 27B's floor is 20.5 GiB
    try std.testing.expectEqual(Fit.tight, fitOf(pick("TensorFold/Qwen3.8-27B-MLX-4bit", "MLX 4-bit + DFlash2"), &m32));
    try std.testing.expectEqual(Fit.raise_limit, fitOf(pick("mlx-community/gemma-4-31B-it-4bit", "MLX 4-bit + DFlash"), &m32)); // 22.0 GiB
    try std.testing.expectEqual(Fit.untested, fitOf(pick("RecursiveSI/flint-int4", "MLX INT4"), &mac("Apple M3 Ultra", 0, 256 * gib)));
    try std.testing.expectEqual(Fit.fits, fitOf(pick("RecursiveSI/flint-int4", "MLX INT4"), &mac("Apple M5 Max", 40, 128 * gib)));

    var spark: Machine = .{ .memory = 128 * gib };
    machine.applyNvidia(&spark, "NVIDIA GB10, [N/A]\n");
    try std.testing.expectEqual(Fit.fits, fitOf(pick("TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP", "MLX 4-bit + MTP"), &spark));
    try std.testing.expectEqual(Fit.other_backend, fitOf(pick("TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP", "MLX 6-bit + MTP"), &spark));
}

test "speeds: measured on this chip, scaled by bandwidth from the same generation, plain before M5, none unmeasured" {
    const max = mac("Apple M5 Max", 40, 128 * gib);
    const here = estimate(pick("TensorFold/Qwen3.8-27B-MLX-4bit", "MLX 4-bit + DFlash2"), &max).?;
    try std.testing.expect(here.measured and here.tok_s == 64);

    const gemma = pick("mlx-community/gemma-4-31B-it-4bit", "MLX 4-bit + DFlash");
    const scaled = estimate(gemma, &max).?; // 150 on an M5 Ultra at 1,200 GB/s, here 614
    try std.testing.expect(!scaled.measured and !scaled.plain and std.math.approxEqAbs(f64, scaled.tok_s, 150.0 * 614.0 / 1200.0, 1e-6));
    const old = estimate(gemma, &mac("Apple M3 Ultra", 0, 256 * gib)).?; // drafts need M5: the plain 51
    try std.testing.expect(old.plain and std.math.approxEqAbs(f64, old.tok_s, 51.0 * 819.0 / 1200.0, 1e-6));

    const lfm = estimate(pick("LiquidAI/LFM2.5-24B-A2B-MLX-4bit", "MLX 4-bit"), &max).?;
    try std.testing.expectEqualStrings("M5 Ultra", lfm.from); // same generation beats the nearer M3 Ultra bandwidth
    try std.testing.expectEqual(@as(?Estimate, null), estimate(pick("mlx-community/Qwen3.5-2B-MLX-4bit", "MLX 4-bit"), &max));
    try std.testing.expectEqual(@as(?Estimate, null), estimate(pick("TensorFold/Qwen3.8-27B-MLX-4bit", "MLX 4-bit + DFlash2"), &mac("Apple M9", 0, 64 * gib)));
}

test "models --json keeps schema 1's cached checkpoints and adds the machine and the catalog" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try syntheticCache(a, io);
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};

    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out.writer, null, root, .{ .json = true }, mac("Apple M5 Max", 40, 128 * gib)));
    var parsed = try std.json.parseFromSlice(std.json.Value, a, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 1), parsed.value.object.get("schema").?.integer);
    const models = parsed.value.object.get("models").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), models.len);
    try std.testing.expectEqualStrings("Org/Flash", models[0].object.get("repo").?.string);
    try std.testing.expectEqualStrings("Qwen3.8 Flash Next", models[0].object.get("family").?.string);
    try std.testing.expectEqualStrings("qwen4_exp", models[0].object.get("model_type").?.string);
    try std.testing.expectEqualStrings("metal", parsed.value.object.get("machine").?.object.get("backend").?.string);
    const items = parsed.value.object.get("catalog").?.array.items;
    try std.testing.expectEqual(catalog.entries.len, items.len);
    try std.testing.expectEqualStrings("fits", items[0].object.get("fit").?.string);
    var rated: usize = 0;
    for (items) |item| switch (item.object.get("quality").?) {
        .null => {},
        .object => |q| {
            rated += 1;
            try std.testing.expectEqualStrings(catalog.quality_digest, q.get("digest").?.string);
            if (std.mem.eql(u8, item.object.get("repo").?.string, "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit"))
                try std.testing.expectEqual(@as(i64, 192), q.get("passed").?.integer);
        },
        else => return error.TestUnexpectedResult,
    };
    try std.testing.expect(rated > 0 and rated < items.len);

    var none: std.Io.Writer.Allocating = .init(a);
    const missing = try std.fs.path.join(a, &.{ root, "missing" });
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &none.writer, null, missing, .{ .json = true }, .{}));
    var empty = try std.json.parseFromSlice(std.json.Value, a, none.written(), .{});
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty.value.object.get("models").?.array.items.len);
    try std.testing.expectEqualStrings("unknown", empty.value.object.get("catalog").?.array.items[0].object.get("fit").?.string);
}

test "onDisk sizes the checkpoints models lists, and an absent cache has none" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try syntheticCache(a, io);
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const found = try onDisk(a, io, null, root);
    try std.testing.expectEqual(@as(usize, 1), found.len);
    try std.testing.expectEqualStrings("Org/Flash", found[0].repo);
    try std.testing.expectEqualStrings("qwen4_exp", found[0].model_type);
    try std.testing.expect(found[0].bytes > 0); // its config.json
    try std.testing.expectEqual(@as(usize, 0), (try onDisk(a, io, null, try std.fs.path.join(a, &.{ root, "missing" }))).len);
}

test "start CLI: a fake registry picks the largest pullable fit and refuses a machine that fits nothing" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const missing = try std.fmt.allocPrint(a, ".tf-start-none-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    const entries = [_]Entry{
        .{ .title = "Small", .format = "4-bit", .repo = "Org/Small", .revision = "1111111111111111111111111111111111111111", .download = 2 * gib, .metal = true, .recipe = "small" },
        .{ .title = "Large", .format = "4-bit", .repo = "Org/Large", .revision = "2222222222222222222222222222222222222222", .drafter = "Org/Draft", .drafter_revision = "3333333333333333333333333333333333333333", .download = 9 * gib, .metal = true, .cuda = true, .m5 = .drafts, .recipe = "large" },
        .{ .title = "Unpinned", .format = "4-bit", .repo = "Org/Unpinned", .download = 20 * gib, .metal = true, .recipe = "unpinned" },
        .{ .title = "Tight", .format = "8-bit", .repo = "Org/Tight", .revision = "4444444444444444444444444444444444444444", .download = 61 * gib, .metal = true, .recipe = "tight" },
        .{ .title = "External installer", .format = "4-bit", .repo = "Org/External", .download = 12 * gib, .metal = true, .hf_dir = "External", .recipe = "external" },
        .{ .title = "Too big", .format = "8-bit", .repo = "Org/TooBig", .download = 100 * gib, .metal = true, .recipe = "too-big" },
    };
    var mc: Machine = .{ .backend = .metal, .memory = 96 * gib, .gpu_limit = 64 * gib };
    mc.setChip("Apple M5 Max");
    const picked = (try chooseForStart(a, io, null, missing, &entries, &mc)).?;
    try std.testing.expectEqualStrings("Org/Large", picked.entry.repo);
    try std.testing.expectEqualStrings("Org/Large@2222222222222222222222222222222222222222", picked.model);
    try std.testing.expectEqualStrings("Org/Draft@3333333333333333333333333333333333333333", picked.drafter.?);
    try std.testing.expect(picked.drafter_dir != null);
    try std.testing.expectEqual(@as(u64, 9 * gib), picked.download_bytes);
    try std.testing.expect(!picked.cached);
    mc.setChip("Apple M4 Max");
    const plain = (try chooseForStart(a, io, null, missing, &entries, &mc)).?;
    try std.testing.expectEqualStrings("Org/Large", plain.entry.repo);
    try std.testing.expect(plain.drafter == null and plain.drafter_dir == null);
    mc = .{ .backend = .cuda, .memory = 96 * gib, .gpu_limit = 64 * gib };
    const cuda = (try chooseForStart(a, io, null, missing, &entries, &mc)).?;
    try std.testing.expectEqualStrings("Org/Large", cuda.entry.repo);
    try std.testing.expect(cuda.drafter != null and cuda.drafter_dir != null);
    mc.gpu_limit = gib;
    try std.testing.expect((try chooseForStart(a, io, null, missing, &entries, &mc)) == null);
    mc = .{};
    try std.testing.expect((try chooseForStart(a, io, null, missing, &entries, &mc)) == null);
}

fn has(text: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, text, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return false;
}
