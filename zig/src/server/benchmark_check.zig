//! The board's receipt rules (tensorfold.dev/benchmarks/protocol.js), plus a save mode that keeps local and rocm runs.
const std = @import("std");
const json = @import("json");
const suite = @import("benchmark_suite.zig");
const Value = json.Value;

/// ``save``: what a local file may hold; ``publish``: exactly what the board accepts.
pub const Mode = enum { save, publish };

pub const local_label = "Local or private model labels cannot be published; benchmark a public checkpoint ID";
pub const over_budget = "A sample's output count exceeds the requested budget; this receipt cannot be published";
pub const rocm_local = "The board accepts mlx and cuda receipts; ROCm receipts stay local until it accepts rocm";

const Bad = error{Bad};

/// True when ``v`` passes; otherwise ``why`` names the field and the rule.
pub fn receipt(v: Value, mode: Mode, now_ms: i64, why: *[]const u8) bool {
    var c: Check = .{ .why = why, .mode = mode, .now_ms = now_ms };
    c.all(v) catch return false;
    return true;
}

const top_keys = [_][]const u8{ "schema_version", "suite", "run_id", "created_at", "runtime", "model", "hardware", "settings", "samples" };
const runtime_keys = [_][]const u8{ "tensorfold_version", "backend", "python_version", "platform", "dependencies" };
const model_keys = [_][]const u8{ "repo_id", "revision", "family", "config_sha256", "tokenizer_sha256", "quantization", "drafter_repo_id", "local" };
const hardware_keys = [_][]const u8{ "cpu_model", "system_memory_bytes", "memory_type", "gpus", "gpu_count", "source" };
const settings_keys = [_][]const u8{ "tokens", "repetitions", "temperatures", "serial", "context_tokens", "managed", "rank_count" };
const sample_keys = [_][]const u8{ "fixture_id", "temperature", "repeat", "seed", "status", "prompt_tokens", "completion_tokens", "cached_tokens", "delivery_seconds", "delivery_tps", "ttft_seconds", "end_to_end_seconds", "server_decode_tps", "server_decode_seconds", "prefill_seconds", "token_sha", "error_code" };

const Num = struct { nullable: bool = false, integer: bool = false, min: f64 = 0, max: f64 = 1e15 };

const Check = struct {
    why: *[]const u8,
    mode: Mode,
    now_ms: i64,

    fn fail(c: *Check, comptime message: []const u8) Bad {
        c.why.* = message;
        return error.Bad;
    }

    fn need(c: *Check, ok: bool, comptime message: []const u8) Bad!void {
        if (!ok) return c.fail(message);
    }

    fn all(c: *Check, v: Value) Bad!void {
        try c.need(exact(v, &top_keys), "receipt must hold exactly the community-v1 fields");
        try c.need(isNumber(v.get("schema_version").?, 1), "schema_version must be 1");
        const s = v.get("suite").?;
        try c.need(exact(s, &.{ "id", "sha256" }) and eqlStr(s.get("id").?, suite.id) and eqlStr(s.get("sha256").?, suite.sha256_hex), "suite must be community-v1 with its fixture digest");
        try c.need(text(v.get("run_id").?, uuid, false), "run_id must be a lowercase UUID v4");
        try c.need(createdAt(v.get("created_at").?, c.now_ms), "created_at must be a real UTC time like 2026-01-01T00:00:00Z, at most 15 minutes ahead");
        try c.runtime(v.get("runtime").?);
        try c.model(v.get("model").?);
        try c.hardware(v.get("hardware").?);
        const st = v.get("settings").?;
        try c.settings(st);
        const samples = v.get("samples").?;
        try c.need(samples == .array and samples.array.len >= 1 and samples.array.len <= 200, "samples must hold 1 to 200 attempts");
        for (samples.array, 0..) |x, i| try c.sample(x, st, samples.array[0..i]);
    }

    fn runtime(c: *Check, r: Value) Bad!void {
        try c.need(exact(r, &runtime_keys), "runtime must hold exactly its fields");
        const b = r.get("backend").?;
        const p = r.get("platform").?;
        const board = (eqlStr(b, "mlx") and eqlStr(p, "macos")) or (eqlStr(b, "cuda") and eqlStr(p, "linux"));
        const rocm = eqlStr(b, "rocm") and eqlStr(p, "linux");
        if (c.mode == .publish and rocm) return c.fail(rocm_local);
        if (c.mode == .publish) try c.need(board, "runtime.backend and platform must be mlx on macos or cuda on linux");
        try c.need(board or rocm, "runtime.backend and platform must be mlx on macos, or cuda or rocm on linux");
        try c.need(text(r.get("tensorfold_version").?, version, false), "runtime.tensorfold_version must be a version string");
        try c.need(text(r.get("python_version").?, version, false), "runtime.python_version must be a version string");
        const deps = r.get("dependencies").?;
        try c.need(deps == .object, "runtime.dependencies must be an object");
        for (deps.object.keys(), deps.object.values()) |k, d| {
            const known = for ([_][]const u8{ "mlx", "mlx_lm", "torch", "triton", "cuda", "driver" }) |n| {
                if (std.mem.eql(u8, n, k)) break true;
            } else false;
            try c.need(known, "runtime.dependencies may name only mlx, mlx_lm, torch, triton, cuda and driver");
            try c.need(text(d, version, true), "runtime.dependencies values must be version strings or null");
        }
    }

    fn model(c: *Check, m: Value) Bad!void {
        try c.need(exact(m, &model_keys), "model must hold exactly its fields");
        const repo = m.get("repo_id").?;
        const local = m.get("local").?;
        if (c.mode == .publish and (repo == .null or local != .bool or local.bool)) return c.fail(local_label);
        try c.need(text(repo, repoId, c.mode == .save), "model.repo_id must be a public owner/name checkpoint ID");
        try c.need(text(m.get("drafter_repo_id").?, repoId, true), "model.drafter_repo_id must be a public owner/name checkpoint ID or null");
        try c.need(local == .bool, "model.local must be true or false");
        try c.need(hex(m.get("revision").?, &.{ 40, 64 }), "model.revision must be 40 or 64 lowercase hex or null");
        try c.need(hex(m.get("config_sha256").?, &.{64}) and hex(m.get("tokenizer_sha256").?, &.{64}), "model checkpoint hashes must be 64 lowercase hex or null");
        try c.need(text(m.get("family").?, name, true), "model.family must be a plain name or null");
        try c.need(text(m.get("quantization").?, name, true), "model.quantization must be a plain name or null");
    }

    fn hardware(c: *Check, h: Value) Bad!void {
        try c.need(exact(h, &hardware_keys), "hardware must hold exactly its fields");
        try c.need(text(h.get("cpu_model").?, name, true), "hardware.cpu_model must be a plain name or null");
        try c.need(num(h.get("system_memory_bytes").?, .{ .nullable = true, .integer = true }), "hardware.system_memory_bytes must be a byte count or null");
        const count = h.get("gpu_count").?;
        try c.need(num(count, .{ .nullable = true, .integer = true, .max = 256 }), "hardware.gpu_count must be 0 to 256 or null");
        try c.need(oneOf(h.get("memory_type").?, &.{ "unified", "dedicated", "unknown" }) and oneOf(h.get("source").?, &.{ "detected", "partial", "unavailable" }), "hardware.memory_type and source must be known values");
        const gpus = h.get("gpus").?;
        try c.need(gpus == .array and gpus.array.len <= 256, "hardware.gpus must be a list of at most 256");
        for (gpus.array) |g| {
            try c.need(exact(g, &.{ "name", "memory_bytes", "cores" }), "hardware.gpus entries must hold exactly name, memory_bytes and cores");
            try c.need(text(g.get("name").?, name, false), "hardware.gpus name must be a plain name");
            try c.need(num(g.get("memory_bytes").?, .{ .nullable = true, .integer = true }), "hardware.gpus memory_bytes must be a byte count or null");
            try c.need(num(g.get("cores").?, .{ .nullable = true, .integer = true, .max = 100000 }), "hardware.gpus cores must be 0 to 100000 or null");
        }
        try c.need(count == .null or toF64(count).? == @as(f64, @floatFromInt(gpus.array.len)), "hardware.gpu_count must equal the number of GPUs");
    }

    fn settings(c: *Check, s: Value) Bad!void {
        try c.need(exact(s, &settings_keys), "settings must hold exactly its fields");
        try c.need(num(s.get("tokens").?, .{ .integer = true, .min = 2, .max = 8192 }), "settings.tokens must be 2 to 8192");
        try c.need(num(s.get("repetitions").?, .{ .integer = true, .min = 1, .max = 20 }), "settings.repetitions must be 1 to 20");
        try c.need(num(s.get("context_tokens").?, .{ .nullable = true, .integer = true, .min = 1, .max = 10000000 }), "settings.context_tokens must be 1 to 10000000 or null");
        try c.need(num(s.get("rank_count").?, .{ .nullable = true, .integer = true, .min = 1, .max = 2 }), "settings.rank_count must be 1, 2 or null");
        try c.need(s.get("serial").? == .bool and s.get("managed").? == .bool, "settings.serial and managed must be true or false");
        const temps = s.get("temperatures").?;
        try c.need(temps == .array and temps.array.len >= 1 and temps.array.len <= 4, "settings.temperatures must hold one to four values");
        for (temps.array, 0..) |t, i| {
            try c.need(num(t, .{ .max = 2 }), "settings.temperatures must be numbers from 0 to 2");
            for (temps.array[0..i]) |u| try c.need(toF64(u).? != toF64(t).?, "settings.temperatures must be distinct");
        }
    }

    fn sample(c: *Check, x: Value, s: Value, before: []const Value) Bad!void {
        try c.need(exact(x, &sample_keys), "samples entries must hold exactly the 17 sample fields");
        const t = toF64(x.get("temperature").?);
        const in_cell = t != null and for (s.get("temperatures").?.array) |u| {
            if (toF64(u).? == t.?) break true;
        } else false;
        try c.need(oneOf(x.get("fixture_id").?, &.{ "code", "chat" }) and in_cell, "samples must belong to a code or chat cell at a requested temperature");
        const reps = toF64(s.get("repetitions").?).?;
        try c.need(num(x.get("repeat").?, .{ .integer = true, .min = -1, .max = reps - 1 }), "samples repeat must be -1 to repetitions - 1");
        try c.need(num(x.get("seed").?, .{ .integer = true, .max = 4294967295 }), "samples seed must be 0 to 2^32 - 1");
        for (before) |y| {
            const same = std.mem.eql(u8, y.get("fixture_id").?.string, x.get("fixture_id").?.string) and toF64(y.get("temperature").?).? == t.? and toF64(y.get("repeat").?).? == toF64(x.get("repeat").?).?;
            try c.need(!same, "samples must not repeat a fixture, temperature and repeat");
        }
        const status = x.get("status").?;
        try c.need(oneOf(status, &.{ "ok", "early_eos", "error", "unmeasured" }), "samples status must be ok, early_eos, error or unmeasured");
        try c.need(toF64(x.get("repeat").?).? != -1 or eqlStr(status, "error"), "only a failed warm-up may use repeat -1");
        inline for (.{ "prompt_tokens", "completion_tokens", "cached_tokens" }) |k| try c.need(num(x.get(k).?, .{ .nullable = true, .integer = true, .max = 10000000 }), "samples " ++ k ++ " must be 0 to 10000000 or null");
        inline for (.{ "delivery_seconds", "delivery_tps", "ttft_seconds", "end_to_end_seconds", "server_decode_tps", "server_decode_seconds", "prefill_seconds" }) |k| try c.need(num(x.get(k).?, .{ .nullable = true, .max = 1e9 }), "samples " ++ k ++ " must be 0 to 1e9 or null");
        try c.need(hex(x.get("token_sha").?, &.{ 12, 64 }), "samples token_sha must be 12 or 64 lowercase hex or null");
        const code = x.get("error_code").?;
        const known = code == .null or code == .string and std.meta.stringToEnum(suite.ErrorCode, code.string) != null;
        try c.need(known, "samples error_code must be null or a known code");
        const count = toF64(x.get("completion_tokens").?);
        const seconds = toF64(x.get("delivery_seconds").?);
        const rate = toF64(x.get("delivery_tps").?);
        if (rate) |r| {
            try c.need(count != null and count.? >= 2 and seconds != null and seconds.? > 0, "samples delivery_tps needs completion_tokens of 2 or more and a positive delivery_seconds");
            const expected = (count.? - 1) / seconds.?;
            try c.need(@abs(r - expected) <= @max(1e-6, 1e-6 * @max(r, expected)), "samples delivery_tps must equal (completion_tokens - 1) / delivery_seconds");
        }
        const tokens = toF64(s.get("tokens").?).?;
        if (eqlStr(status, "ok")) try c.need(count != null and count.? == tokens and rate != null and code == .null, "an ok sample needs its requested output, a delivery rate and no error code");
        const cached = toF64(x.get("cached_tokens").?);
        const prompt = toF64(x.get("prompt_tokens").?);
        try c.need(cached == null or prompt == null or cached.? <= prompt.?, "samples cached_tokens must not exceed prompt_tokens");
        if (count != null and count.? > tokens and c.mode == .publish) return c.fail(over_budget);
    }
};

/// A JSON number as JavaScript reads it; bools and other kinds are not numbers.
fn toF64(v: Value) ?f64 {
    return switch (v) {
        .int => |t| std.fmt.parseFloat(f64, t) catch null,
        .float => |f| f,
        else => null,
    };
}

fn num(v: Value, o: Num) bool {
    if (o.nullable and v == .null) return true;
    const x = toF64(v) orelse return false;
    if (!std.math.isFinite(x)) return false;
    if (o.integer and !(@floor(x) == x and @abs(x) <= 9007199254740991)) return false;
    return x >= o.min and x <= o.max;
}

fn isNumber(v: Value, want: f64) bool {
    const x = toF64(v) orelse return false;
    return x == want;
}

fn exact(v: Value, keys: []const []const u8) bool {
    if (v != .object or v.object.count() != keys.len) return false;
    for (keys) |k| if (!v.object.contains(k)) return false;
    return true;
}

fn eqlStr(v: Value, s: []const u8) bool {
    return v == .string and std.mem.eql(u8, v.string, s);
}

fn oneOf(v: Value, options: []const []const u8) bool {
    for (options) |o| if (eqlStr(v, o)) return true;
    return false;
}

fn text(v: Value, comptime pattern: fn ([]const u8) bool, nullable: bool) bool {
    if (nullable and v == .null) return true;
    return v == .string and pattern(v.string);
}

fn hex(v: Value, lengths: []const usize) bool {
    if (v == .null) return true;
    if (v != .string or std.mem.indexOfScalar(usize, lengths, v.string.len) == null) return false;
    for (v.string) |ch| if (!(std.ascii.isDigit(ch) or (ch >= 'a' and ch <= 'f'))) return false;
    return true;
}

fn charsIn(s: []const u8, extra: []const u8) bool {
    for (s) |ch| if (!std.ascii.isAlphanumeric(ch) and std.mem.indexOfScalar(u8, extra, ch) == null) return false;
    return true;
}

/// The board's VERSION pattern: ``^[A-Za-z0-9][A-Za-z0-9._+ -]{0,79}$``.
pub fn version(s: []const u8) bool {
    return s.len >= 1 and s.len <= 80 and std.ascii.isAlphanumeric(s[0]) and charsIn(s[1..], "._+ -");
}

/// The board's NAME pattern: ``^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,119}$``.
pub fn name(s: []const u8) bool {
    return s.len >= 1 and s.len <= 120 and std.ascii.isAlphanumeric(s[0]) and charsIn(s[1..], " ._()+-");
}

/// The board's REPO pattern: owner of 1 to 96 and name of 1 to 160 ``[A-Za-z0-9._-]``, each starting alphanumeric.
pub fn repoId(s: []const u8) bool {
    const slash = std.mem.indexOfScalar(u8, s, '/') orelse return false;
    const owner = s[0..slash];
    const rest = s[slash + 1 ..];
    for ([_]struct { []const u8, usize }{ .{ owner, 96 }, .{ rest, 160 } }) |part| {
        if (part[0].len == 0 or part[0].len > part[1] or !std.ascii.isAlphanumeric(part[0][0]) or !charsIn(part[0], "._-")) return false;
    }
    return true;
}

fn uuid(s: []const u8) bool {
    if (s.len != 36) return false;
    for (s, 0..) |ch, i| {
        const dash = i == 8 or i == 13 or i == 18 or i == 23;
        if (dash != (ch == '-')) return false;
        if (!dash and !(std.ascii.isDigit(ch) or (ch >= 'a' and ch <= 'f'))) return false;
    }
    return s[14] == '4' and std.mem.indexOfScalar(u8, "89ab", s[19]) != null;
}

/// ``YYYY-MM-DDTHH:MM:SS[.f{1,6}]Z``, a real calendar time, at most 15 minutes past ``now_ms``.
fn createdAt(v: Value, now_ms: i64) bool {
    if (v != .string) return false;
    const s = v.string;
    if (s.len < 20 or s[s.len - 1] != 'Z') return false;
    const shape = "dddd-dd-ddTdd:dd:dd";
    for (shape, 0..) |want, i| if (if (want == 'd') !std.ascii.isDigit(s[i]) else s[i] != want) return false;
    var frac_ms: i64 = 0;
    const rest = s[19 .. s.len - 1];
    if (rest.len > 0) {
        if (rest[0] != '.' or rest.len < 2 or rest.len > 7) return false;
        for (rest[1..]) |ch| if (!std.ascii.isDigit(ch)) return false;
        var padded: [3]u8 = "000".*;
        @memcpy(padded[0..@min(3, rest.len - 1)], rest[1..@min(4, rest.len)]);
        frac_ms = std.fmt.parseInt(i64, &padded, 10) catch return false;
    }
    const field = struct {
        fn f(t: []const u8, at: usize, n: usize) i64 {
            return std.fmt.parseInt(i64, t[at .. at + n], 10) catch unreachable;
        }
    }.f;
    const y = field(s, 0, 4);
    const mo = field(s, 5, 2);
    const d = field(s, 8, 2);
    const hh = field(s, 11, 2);
    const mi = field(s, 14, 2);
    const ss = field(s, 17, 2);
    if (mo < 1 or mo > 12 or d < 1 or d > daysIn(y, mo) or hh > 23 or mi > 59 or ss > 59) return false;
    const ms = ((daysFromCivil(y, mo, d) * 24 + hh) * 60 + mi) * 60_000 + ss * 1000 + frac_ms;
    return ms <= now_ms + 15 * 60 * 1000;
}

fn daysIn(y: i64, m: i64) i64 {
    const leap = (@mod(y, 4) == 0 and @mod(y, 100) != 0) or @mod(y, 400) == 0;
    return switch (m) {
        2 => if (leap) 29 else 28,
        4, 6, 9, 11 => 30,
        else => 31,
    };
}

fn daysFromCivil(y0: i64, m: i64, d: i64) i64 {
    const y = if (m <= 2) y0 - 1 else y0;
    const era = @divFloor(y, 400);
    const yoe = y - era * 400;
    const mp = @mod(m + 9, 12);
    const doy = @divFloor(153 * mp + 2, 5) + d - 1;
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return era * 146097 + doe - 719468;
}

test "the time rule takes a real date, at most six fraction digits, Z only, at most 15 minutes ahead" {
    const now: i64 = 1767225600000;
    try std.testing.expectEqual(@as(i64, 1767225600000), daysFromCivil(2026, 1, 1) * 86400000);
    for ([_][]const u8{ "2026-01-01T00:00:00Z", "2026-01-01T00:14:59.999999Z", "2024-02-29T23:59:59.5Z" }) |ok| try std.testing.expect(createdAt(.{ .string = ok }, now));
    for ([_][]const u8{ "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00.1234567Z", "2026-02-30T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01 00:00:00Z", "2026-01-01Ownership rollback::00:00Z", "2026-01-01T00:00:00" }) |bad|
        try std.testing.expect(!createdAt(.{ .string = bad }, now));
}

test "the identifier patterns follow the board's VERSION, NAME, REPO and run ID patterns" {
    try std.testing.expect(repoId("example-org/test-model") and !repoId("/home/x") and !repoId("a/b/c") and !repoId("-a/b"));
    try std.testing.expect(version("none") and version("unknown") and version("1.0.7-benchmark") and !version("") and !version("n/a") and !version("-"));
    try std.testing.expect(name("Apple M5 Max") and name("mlx-4bit-g64") and !name("a@b") and !name("a,b"));
    try std.testing.expect(uuid("00000000-0000-4000-8000-000000000001") and !uuid("00000000-0000-4000-8000-00000000000A") and !uuid("00000000-0000-3000-8000-000000000001"));
}
