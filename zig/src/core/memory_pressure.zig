//! macOS memory pressure as the lane host acts on it: a rise takes effect at once, a fall only after a calm spell.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");

pub const Level = enum(u8) { normal, warning, critical };

/// The kernel's level for its sysctl value: 4 and up critical, 2 and 3 warning, anything else normal.
pub fn ofKernel(v: u32) Level {
    return if (v >= 4) .critical else if (v >= 2) .warning else .normal;
}

/// Where readings come from: the kernel's sysctl, or a test's fake (null: no reading, taken as normal).
pub const Source = struct {
    ctx: ?*anyopaque = null,
    read: *const fn (ctx: ?*anyopaque) ?Level,

    pub const kernel: Source = .{ .read = readKernel };

    fn readKernel(_: ?*anyopaque) ?Level {
        return ofKernel(platform.memory_policy.pressure() orelse return null);
    }
};

/// A change of the governed level, reported once.
pub const Change = struct { from: Level, to: Level };

/// Where a change's one log line goes; tests record it instead.
pub const Log = struct {
    ctx: ?*anyopaque = null,
    line: *const fn (ctx: ?*anyopaque, change: Change, mark: ?u64) void = write,
};

/// TENSORFOLD_MEMORY_PRESSURE (0 or off: no governor) and TENSORFOLD_PRESSURE_CACHE_PERCENT (the soft watermark).
pub const Settings = struct {
    on: bool = true,
    percent: u8 = default_percent,

    pub const default_percent: u8 = 50;

    /// Settings from their texts: 0 or off turns it off; a whole percent in [0, 100], else the default.
    pub fn fromText(switch_text: ?[]const u8, percent_text: ?[]const u8) Settings {
        const s = std.mem.trim(u8, switch_text orelse "", " \t");
        const off = std.mem.eql(u8, s, "0") or std.ascii.eqlIgnoreCase(s, "off");
        const p = std.fmt.parseInt(u8, std.mem.trim(u8, percent_text orelse "", " \t"), 10) catch default_percent;
        return .{ .on = !off, .percent = if (p <= 100) p else default_percent };
    }
};

pub const Governor = struct {
    source: Source,
    percent: u8 = Settings.default_percent,
    calm_ns: i96 = 5 * std.time.ns_per_s, // a lower level holds this long before the governed level falls to it
    every_ns: i96 = 250 * std.time.ns_per_ms, // at most one reading this often, however often the host asks
    log: Log = .{},
    level: Level = .normal,
    read_at: ?i96 = null,
    calm_since: ?i96 = null, // the first reading below `level` in the current calm spell

    /// The lane host's governor on a Mac where the kernel reads (null: off, a test, or another OS).
    pub fn fromEnv() ?Governor {
        if (builtin.is_test or builtin.os.tag != .macos) return null; // host tests give their own source
        const s = Settings.fromText(platform.getenv("TENSORFOLD_MEMORY_PRESSURE"), platform.getenv("TENSORFOLD_PRESSURE_CACHE_PERCENT"));
        if (!s.on or Source.kernel.read(null) == null) return null;
        return .{ .source = Source.kernel, .percent = s.percent };
    }

    /// The soft watermark the prompt cache sheds toward under pressure, in bytes of its budget.
    pub fn mark(g: *const Governor, budget: u64) u64 {
        return budget / 100 * g.percent + budget % 100 * g.percent / 100;
    }

    /// A reading at `now` unless one was taken within `every_ns`; the governed level's change, once.
    pub fn poll(g: *Governor, now: i96) ?Change {
        if (g.read_at) |t| if (now - t < g.every_ns) return null;
        g.read_at = now;
        return g.observe(g.source.read(g.source.ctx) orelse .normal, now);
    }

    /// The governed level after `raw` at `now`: up at once, down after `calm_ns` of lower readings.
    pub fn observe(g: *Governor, raw: Level, now: i96) ?Change {
        const from = g.level;
        if (@backingInt(raw) >= @backingInt(from)) {
            g.calm_since = null;
            if (raw == from) return null;
        } else {
            const since = g.calm_since orelse now;
            g.calm_since = since;
            if (now - since < g.calm_ns) return null;
            g.calm_since = null;
        }
        g.level = raw;
        return .{ .from = from, .to = raw };
    }
};

/// A change's line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn write(_: ?*anyopaque, c: Change, mark: ?u64) void {
    if (builtin.is_test) return;
    const cache = if (mark) |m| m >> 20 else 0;
    switch (c.to) {
        .normal => std.log.info("memory pressure: normal again: streams are admitted and prompt states kept as before", .{}),
        .warning => if (c.from == .critical)
            std.log.warn("memory pressure: eased to warning: new requests are queued again, one more stream only when none runs", .{})
        else
            std.log.warn("memory pressure: warning: one more stream only when none runs, no new kept prompt states, the prompt cache sheds toward {d} MiB", .{cache}),
        .critical => std.log.warn("memory pressure: critical: new requests are refused with 503 until it eases; the prompt cache sheds toward {d} MiB", .{cache}),
    }
}
