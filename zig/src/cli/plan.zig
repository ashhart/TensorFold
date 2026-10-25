//! The `plan` command: the serve startup's memory arithmetic without loading weights, per RAM class.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");
const engines = @import("native_engines");
const budget = engines.budget;

pub const GiB: u64 = 1 << 30;
const gib: f64 = @floatFromInt(GiB);

fn gibf(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / gib;
}

/// The flags `plan` takes: an explicit budget (as TENSORFOLD_CUDA_MEMORY_LIMIT_GB would set it) and RAM classes.
pub const Options = struct { mem_gb: ?f64 = null, ram: []const u64 = &.{} };

/// MODEL plus the parsed flags, or null when the argv shape is wrong.
pub const Parsed = struct { model: []const u8, mem_gb: ?f64, ram: []const u64 };

/// `tensorfold plan MODEL [--memory-gb GIB] [--ram GB]...`: the model first, then the repeatable flags.
pub fn parseArgs(a: Allocator, args: []const []const u8) ?Parsed {
    if (args.len == 0) return null;
    var mem_gb: ?f64 = null;
    var ram: std.ArrayList(u64) = .empty;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--memory-gb")) {
            i += 1;
            if (i >= args.len) return null;
            mem_gb = std.fmt.parseFloat(f64, args[i]) catch return null;
        } else if (std.mem.eql(u8, args[i], "--ram")) {
            i += 1;
            if (i >= args.len) return null;
            const class = std.fmt.parseInt(u64, args[i], 10) catch return null;
            ram.append(a, class) catch return null;
        } else return null;
    }
    return .{ .model = args[0], .mem_gb = mem_gb, .ram = ram.items };
}

/// The checkpoint and the machine's memory, resolved from `model` and the environment, for `report`.
pub const Inputs = struct {
    title: []const u8,
    model: []const u8,
    weights: u64,
    mem: ?budget.MemInfo,
    reserve_text: ?[]const u8,
    limit_text: ?[]const u8,
    mem_gb: ?f64 = null,
    ram: []const u64 = &.{},
};

/// One budget the admission would grant at startup: its label, its grant with nothing held yet, its reserve.
const Row = struct { label: []const u8, grant: u64, reserve: u64 };

/// The plan's verdicts and hint, on the values `in` carries. Returns the process exit code.
pub fn report(a: Allocator, out: *std.Io.Writer, in: Inputs) !u8 {
    const mem = in.mem orelse {
        try out.print("{s}: plan answers the host memory an integrated device admits from (/proc/meminfo's MemTotal and MemAvailable); a discrete CUDA device's capacity is its own startup report\n", .{in.model});
        return 1;
    };
    const unified = true; // the host-memory admission: an integrated device's counts are procfs's
    const reserve = budget.reserveBytes(in.reserve_text, mem.total, unified) catch {
        try out.print("TENSORFOLD_MEMORY_RESERVE_GIB={s}: a number of GiB from {d} to the memory's size\n", .{ in.reserve_text orelse "", budget.floor(unified).least });
        return 1;
    };
    const limit = budget.limitBytes(in.limit_text) catch {
        try out.print("TENSORFOLD_CUDA_MEMORY_LIMIT_GB={s}: a positive number of GiB whose byte count fits in a 64-bit size\n", .{in.limit_text orelse ""});
        return 1;
    };
    var rows: std.ArrayList(Row) = .empty;
    const pool: budget.Pool = .{ .free = mem.available, .total = mem.total, .reserve = reserve, .limit = limit, .unified = unified };
    try rows.append(a, .{ .label = "this machine's default", .grant = pool.room(0), .reserve = reserve });
    if (in.mem_gb) |asked| {
        const text = try std.fmt.allocPrint(a, "{d}", .{asked});
        const cap = budget.limitBytes(text) catch {
            try out.print("--memory-gb {d}: a positive number of GiB whose byte count fits in a 64-bit size\n", .{asked});
            return 1;
        };
        const capped: budget.Pool = .{ .free = mem.available, .total = mem.total, .reserve = reserve, .limit = cap, .unified = unified };
        try rows.append(a, .{ .label = "--memory-gb", .grant = capped.room(0), .reserve = reserve });
    }
    for (in.ram) |class| {
        const total = std.math.mul(u64, class, GiB) catch {
            try out.print("--ram {d}: a number of GiB whose byte count fits in a 64-bit size\n", .{class});
            return 1;
        };
        const class_reserve = budget.reserveBytes(null, total, unified) catch unreachable; // the default floor always scales
        const class_pool: budget.Pool = .{ .free = total, .total = total, .reserve = class_reserve, .limit = null, .unified = unified };
        try rows.append(a, .{ .label = try std.fmt.allocPrint(a, "--ram {d} GB class", .{class}), .grant = class_pool.room(0), .reserve = class_reserve });
    }
    try out.print("plan for {s} ({s}) on this machine\n", .{ in.title, in.model });
    try out.print("memory {d:.0} GiB, {d:.0} GiB available, {d:.1} GiB reserved for everything else", .{ gibf(mem.total), gibf(mem.available), gibf(reserve) });
    if (limit) |cap| try out.print(", capped to {d:.1} GiB by TENSORFOLD_CUDA_MEMORY_LIMIT_GB", .{gibf(cap)});
    try out.print("\nweights {d:.1} GiB\n", .{gibf(in.weights)});
    var verdict: u8 = 0;
    for (rows.items) |row| {
        if (in.weights > row.grant) {
            verdict = 1;
            try out.print("{s}: {d:.1} GiB budget — does not fit; the model and one prompt chunk need about {d:.1} GiB\n", .{ row.label, gibf(row.grant), gibf(in.weights +| row.reserve) });
        } else {
            try out.print("{s}: {d:.1} GiB budget — fits; about {d:.1} GiB for prompts, streams and caches beside the weights\n", .{ row.label, gibf(row.grant), gibf(row.grant - in.weights) });
        }
    }
    if (verdict != 0) try hint(out, in, mem, reserve, limit);
    return verdict;
}

/// The environment that would fit the weights, or nothing when no single one can (#281's raise_hint).
fn hint(out: *std.Io.Writer, in: Inputs, mem: budget.MemInfo, reserve: u64, limit: ?u64) !void {
    const unified = true;
    const least: u64 = @intFromFloat(budget.floor(unified).least * gib);
    if (mem.available -| least < in.weights) return; // the machine cannot hold them at any reserve
    if (mem.available -| reserve >= in.weights) { // the reserve is fine, so the cap is the only cause
        try out.print("hint: TENSORFOLD_CUDA_MEMORY_LIMIT_GB past {d:.1} would fit the weights (this machine's own grant is {d:.1} GiB)\n", .{ gibf(in.weights +| reserve), gibf(mem.available -| reserve) });
    } else if (limit == null or limit.? >= in.weights) { // a reserve at or under free less the weights fits
        const r = @max((mem.available -| in.weights) -| budget.floor(unified).at_least, least);
        try out.print("hint: TENSORFOLD_MEMORY_RESERVE_GIB={d:.1} would fit the weights (this machine's memory takes up to {d:.1} GiB of them)\n", .{ gibf(r), gibf(mem.available -| least) });
    }
}

/// The `plan` command: the admission's own weights and reserve, before weights load; returns the exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, env: ?*const std.process.Environ.Map, model: []const u8, o: Options) !u8 {
    const dir: []const u8 = blk: {
        if (hub.isDir(io, model)) break :blk model;
        if (hub.isRepoIdLike(model)) {
            const root = try hub.cacheDir(a, env, null);
            if (try hub.cachedSnapshot(a, io, root, model)) |snapshot| break :blk snapshot;
            try out.print("{s} is not in the Hugging Face cache; run: tensorfold pull {s}\n", .{ model, model });
            return 1;
        }
        try out.print("{s} is neither a directory nor a Hugging Face repo id (owner/name)\n", .{model});
        return 1;
    };
    var problem: []const u8 = "";
    const bytes = @import("checkpoint_config").read(a, io, dir, &problem) catch |e| {
        if (e == error.OutOfMemory) return e;
        try out.print("{s}: {s}\n", .{ model, problem });
        return 1;
    };
    if (bytes) |read| a.free(read);
    const model_type = hub.modelType(a, io, dir);
    const family = hub.family(model_type) orelse {
        try out.print("{s}: no registered Zig family serves model_type {s}; the 0.6 line may: tensorfold@0.6\n", .{ model, model_type });
        return 1;
    };
    const weights = budget.weightBytes(a, io, dir) catch |e| {
        try out.print("{s}: cannot count the checkpoint's weight files ({s}); refusing to plan\n", .{ model, @errorName(e) });
        return 1;
    };
    if (weights == 0) {
        try out.print("{s}: no safetensors under {s} to plan from\n", .{ model, dir });
        return 1;
    }
    const text = budget.procText(a, io, "/proc/meminfo");
    defer if (text) |read| a.free(read);
    const mem: ?budget.MemInfo = if (text) |read| budget.meminfo(read) else null;
    const get = struct {
        fn f(e: ?*const std.process.Environ.Map, name: []const u8) ?[]const u8 {
            return if (e) |m| m.get(name) else null;
        }
    }.f;
    return report(a, out, .{
        .title = family.title,
        .model = model,
        .weights = weights,
        .mem = mem,
        .reserve_text = get(env, "TENSORFOLD_MEMORY_RESERVE_GIB"),
        .limit_text = get(env, "TENSORFOLD_CUDA_MEMORY_LIMIT_GB"),
        .mem_gb = o.mem_gb,
        .ram = o.ram,
    });
}

fn inputs(weights: u64, total: u64, available: u64) Inputs {
    return .{ .title = "Test family", .model = "Org/Test", .weights = weights, .mem = .{ .total = total, .available = available }, .reserve_text = null, .limit_text = null };
}

fn written(a: Allocator, in: Inputs) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    _ = try report(a, &out.writer, in);
    return out.written();
}

test "plan: a small checkpoint fits the default budget" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const g: u64 = GiB;
    const out = try written(a, inputs(6 * g, 48 * g, 48 * g));
    try std.testing.expect(std.mem.indexOf(u8, out, "fits") != null and std.mem.indexOf(u8, out, "does not fit") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "weights 6.0 GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "this machine's default: 43.2 GiB budget — fits; about 37.2 GiB") != null);
}

test "plan: a big checkpoint refuses and names the need" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const g: u64 = GiB;
    var in = inputs(21 * g, 48 * g, 48 * g);
    in.mem_gb = 30.0;
    in.ram = &.{24};
    var out: std.Io.Writer.Allocating = .init(a);
    const verdict = try report(a, &out.writer, in);
    try std.testing.expectEqual(@as(u8, 1), verdict);
    const text = out.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "this machine's default: 43.2 GiB budget — fits; about 22.2 GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "--ram 24 GB class: 20.0 GiB budget — does not fit; the model and one prompt chunk need about 25.0 GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "--memory-gb: 30.0 GiB budget — fits; about 9.0 GiB") != null);
}

test "plan: the hint names the reserve that fits the weights" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const g: u64 = GiB;
    const out = try written(a, inputs(21 * g, 24 * g, 24 * g));
    try std.testing.expect(std.mem.indexOf(u8, out, "hint: TENSORFOLD_MEMORY_RESERVE_GIB=2.0 would fit the weights") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "this machine's memory takes up to 22.0 GiB of them") != null);
}

test "plan: the hint names the cap when the limit refuses what the machine would grant" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const g: u64 = GiB;
    var in = inputs(21 * g, 48 * g, 48 * g);
    in.limit_text = "10";
    const out = try written(a, in);
    try std.testing.expect(std.mem.indexOf(u8, out, "hint: TENSORFOLD_CUDA_MEMORY_LIMIT_GB past 25.8 would fit the weights") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "this machine's own grant is 43.2 GiB") != null);
}

test "plan: no hint when even the least reserve cannot hold the weights" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const g: u64 = GiB;
    const out = try written(a, inputs(21 * g, 24 * g, 10 * g));
    try std.testing.expect(std.mem.indexOf(u8, out, "hint:") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "does not fit") != null);
    var both_short = inputs(21 * g, 48 * g, 48 * g);
    both_short.reserve_text = "40";
    both_short.limit_text = "10"; // both the reserve and the cap are short; no single env fits
    const both = try written(a, both_short);
    try std.testing.expect(std.mem.indexOf(u8, both, "hint:") == null);
}

test "plan: an unreadable host memory refuses with the startup report line" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var in = inputs(6 * GiB, 48 * GiB, 48 * GiB);
    in.mem = null;
    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try report(a, &out.writer, in));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "its own startup report") != null);
}

test "plan: a bad reserve variable is refused with the admission's own text" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var in = inputs(6 * GiB, 48 * GiB, 48 * GiB);
    in.reserve_text = "1";
    const out = try written(a, in);
    try std.testing.expect(std.mem.indexOf(u8, out, "TENSORFOLD_MEMORY_RESERVE_GIB=1: a number of GiB from 2 to the memory's size") != null);
}

test "plan: ram classes floor at 4 GiB and share a tenth above it" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const g: u64 = GiB;
    var in = inputs(g, 48 * g, 48 * g);
    in.ram = &.{ 16, 128 };
    const out = try written(a, in);
    try std.testing.expect(std.mem.indexOf(u8, out, "--ram 16 GB class: 12.0 GiB budget") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "--ram 128 GB class: 115.2 GiB budget") != null);
}

test "plan: parseArgs takes the model first and the repeatable flags after it" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const argv = [_][]const u8{ "Org/Test", "--memory-gb", "30.5", "--ram", "32", "--ram", "64" };
    const parsed = parseArgs(a, &argv).?;
    try std.testing.expectEqualStrings("Org/Test", parsed.model);
    try std.testing.expectEqual(@as(?f64, 30.5), parsed.mem_gb);
    try std.testing.expectEqualSlices(u64, &.{ 32, 64 }, parsed.ram);
    try std.testing.expect(parseArgs(a, &.{"Org/Test"}) != null);
    try std.testing.expect(parseArgs(a, &.{}) == null);
    try std.testing.expect(parseArgs(a, &.{ "Org/Test", "--ram" }) == null);
    try std.testing.expect(parseArgs(a, &.{ "Org/Test", "--wat" }) == null);
}
