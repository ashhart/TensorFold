//! Explicit diagnostic policy tapes leave ordinary CUDA clocks untouched and bind paired costs to workloads.
const std = @import("std");
const lanes = @import("lanes");
const env = @import("sys").environment;
const pc = lanes.policy_clock;
pub const Receipt = struct { config: lanes.Config, events: []const pc.Event };
pub const Policy = struct {
    tape: pc.Tape,
    parsed: ?std.json.Parsed(Receipt) = null,
    pub fn open(a: std.mem.Allocator, io: std.Io, base: lanes.backend.Clock, cfg: *lanes.Config) !?*Policy {
        if (comptime !@import("profile").round_attribution.enabled) return null;
        const record = try env.read(bool, "TF_CUDA_ROUND_POLICY_RECORD", false);
        const path = try env.path("TF_CUDA_ROUND_POLICY_REPLAY", env.rawText("TF_CUDA_ROUND_POLICY_REPLAY"));
        if (!record and path == null) return null;
        if (record and path != null) return error.ConflictingPolicyClockModes;
        const p = try a.create(Policy);
        errdefer a.destroy(p);
        p.* = .{ .tape = .{ .a = a, .base = base } };
        errdefer if (p.parsed) |*v| v.deinit();
        if (path) |file| {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(64 << 20));
            defer a.free(bytes);
            p.parsed = try std.json.parseFromSlice(Receipt, a, bytes, .{ .allocate = .alloc_always });
            if (p.parsed.?.value.events.len == 0) return error.EmptyPolicyClockTape;
            p.tape.replay = p.parsed.?.value.events;
            var frozen = p.parsed.?.value.config;
            if (!frozen.timed) return error.PolicyClockMustKeepAdaptiveTiming;
            frozen.depth_prior = try a.dupe(f64, frozen.depth_prior);
            errdefer a.free(frozen.depth_prior);
            frozen.family_costs.slots = try a.dupe(?f64, frozen.family_costs.slots);
            errdefer a.free(frozen.family_costs.slots);
            frozen.shared_costs.slots = try a.dupe(?f64, frozen.shared_costs.slots);
            errdefer a.free(frozen.shared_costs.slots);
            frozen.lane_costs.slots = try a.dupe(?f64, frozen.lane_costs.slots);
            cfg.deinit(a);
            cfg.* = frozen;
        }
        const config_json = try std.json.Stringify.valueAlloc(a, cfg.*, .{});
        defer a.free(config_json);
        std.debug.print("TF_POLICY_CONFIG {s}\n", .{config_json});
        return p;
    }
    pub fn close(p: *Policy) void {
        const count = if (p.tape.replay) |v| v.len else p.tape.recorded.items.len;
        std.debug.print("TF_POLICY_END complete={s} events={d}\n", .{ if (p.tape.complete()) "true" else "false", count });
        const a = p.tape.a;
        p.tape.deinit();
        if (p.parsed) |*parsed| parsed.deinit();
        a.destroy(p);
    }
};
