//! Prompt-level pocket firing rates and the strongest off-subject witnesses, never a training verdict.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const probes = @import("slide_audit_probes.zig");
const A = std.mem.Allocator;

pub const activation_floor = 1e-4;
pub const default_threshold = 0.01;

pub fn value(a: A, plan: probes.Plan, ns: []const api.Neuron, peaks: []const api.pocket_audit.Peak, threshold: f64) !json.Value {
    if (peaks.len != plan.probes.len * ns.len or !std.math.isFinite(threshold) or threshold < 0 or threshold > 1)
        return error.BadAudit;
    const root = try json.newObject(a);
    try root.put(a, "version", .{ .string = probes.version });
    try root.put(a, "read_only", .{ .bool = true });
    try root.put(a, "activation_source", .{ .string = "host pocket math on live-weight snapshot and forward bf16 layer inputs" });
    try root.put(a, "activation_floor", .{ .float = activation_floor });
    try root.put(a, "off_subject_threshold", .{ .float = threshold });
    try root.put(a, "measure", .{ .string = "max absolute activation over every rendered prompt row; firing strictly above floor" });
    try root.put(a, "subject_method", .{ .string = "heuristic from fact and questions; empty means unavailable; questions asked cold" });
    var facts: std.ArrayList(json.Value) = .empty;
    var offset: usize = 0;
    for (plan.facts) |f| {
        const fact = try json.newObject(a);
        try fact.put(a, "id", try json.intValue(a, f.source.id));
        try fact.put(a, "subject", .{ .string = f.subject });
        var neurons: std.ArrayList(json.Value) = .empty;
        for (f.source.neurons) |n| {
            if (offset >= ns.len or !std.meta.eql(n, ns[offset])) return error.BadAudit;
            const v = try neuron(a, plan, peaks, ns.len, offset, f.source.id, threshold);
            try neurons.append(a, v);
            offset += 1;
        }
        try fact.put(a, "neurons", .{ .array = neurons.items });
        try facts.append(a, .{ .object = fact });
    }
    if (offset != ns.len) return error.BadAudit;
    try root.put(a, "facts", .{ .array = facts.items });
    return .{ .object = root };
}

fn neuron(a: A, plan: probes.Plan, peaks: []const api.pocket_audit.Peak, width: usize, col: usize, fact: u32, threshold: f64) !json.Value {
    var counts: [4]usize = @splat(0);
    var fires: [4]usize = @splat(0);
    var strongest: [5]?usize = @splat(null);
    for (plan.probes, 0..) |p, i| {
        const x = peaks[i * width + col].activation;
        if (!std.math.isFinite(x)) return error.AuditNonfinite;
        const c = probes.cohort(p, fact) orelse continue;
        const kind = @backingInt(c);
        counts[kind] += 1;
        fires[kind] += @intFromBool(@abs(x) > activation_floor);
        if (c == .own) continue;
        for (0..strongest.len) |rank| {
            if (strongest[rank]) |old| if (@abs(x) <= @abs(peaks[old * width + col].activation)) continue;
            var j = strongest.len - 1;
            while (j > rank) : (j -= 1) strongest[j] = strongest[j - 1];
            strongest[rank] = i;
            break;
        }
    }
    const n = blk: {
        var at: usize = 0;
        for (plan.facts) |f| for (f.source.neurons) |n| {
            if (at == col) break :blk n;
            at += 1;
        };
        return error.BadAudit;
    };
    const o = try json.newObject(a);
    try o.put(a, "layer", try json.intValue(a, n.layer));
    try o.put(a, "neuron", try json.intValue(a, n.neuron));
    try o.put(a, "key", .{ .string = @tagName(n.key) });
    const groups = try json.newObject(a);
    inline for (std.enums.values(probes.Cohort)) |c| {
        const k = @backingInt(c);
        try groups.put(a, @tagName(c), try rate(a, counts[k], fires[k]));
    }
    try o.put(a, "cohorts", .{ .object = groups });
    const count = counts[1] + counts[2] + counts[3];
    const fired = fires[1] + fires[2] + fires[3];
    try o.put(a, "off_subject", try rate(a, count, fired));
    try o.put(a, "overall", try rate(a, count + counts[0], fired + fires[0]));
    try o.put(a, "flagged", .{ .bool = count > 0 and fraction(fired, count) > threshold });
    var top: std.ArrayList(json.Value) = .empty;
    for (strongest) |idx| if (idx) |i| {
        const w = try json.newObject(a);
        try w.put(a, "prompt", .{ .string = plan.probes[i].text });
        try w.put(a, "cohort", .{ .string = @tagName(probes.cohort(plan.probes[i], fact).?) });
        try w.put(a, "activation", .{ .float = peaks[i * width + col].activation });
        try w.put(a, "row", try json.intValue(a, peaks[i * width + col].row));
        try top.append(a, .{ .object = w });
    };
    try o.put(a, "strongest_off_subject", .{ .array = top.items });
    return .{ .object = o };
}

fn fraction(n: usize, d: usize) f64 {
    return @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(@max(d, 1)));
}

fn rate(a: A, count: usize, fired: usize) !json.Value {
    const o = try json.newObject(a);
    try o.put(a, "prompts", try json.intValue(a, count));
    try o.put(a, "fired", try json.intValue(a, fired));
    try o.put(a, "rate", if (count == 0) .null else .{ .float = fraction(fired, count) });
    return .{ .object = o };
}
