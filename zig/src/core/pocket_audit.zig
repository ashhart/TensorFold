//! An explicit, read-only forward probe; outputs belong to the caller until the queue sends done.
const std = @import("std");
const Neuron = @import("learning_api.zig").Neuron;

pub const Probe = struct { ids: []const u32 };
pub const Peak = struct { activation: f32 = 0, row: u32 = 0 };
pub const Request = struct {
    probe_only: bool = false,
    completed: ?*bool = null,
    neurons: []const Neuron,
    probes: []const Probe,
    peaks: []Peak,

    pub fn validate(r: Request) !void {
        if (r.probe_only) {
            if (r.neurons.len != 0 or r.probes.len != 0 or r.peaks.len != 0) return error.BadAudit;
            return;
        }
        if (r.neurons.len == 0 or r.neurons.len > 1024 or r.probes.len == 0 or r.probes.len > 2048)
            return error.AuditTooLarge;
        if (r.peaks.len != r.neurons.len * r.probes.len) return error.BadAudit;
        for (r.probes) |p| if (p.ids.len == 0 or p.ids.len > 4096) return error.BadAuditPrompt;
    }
};
