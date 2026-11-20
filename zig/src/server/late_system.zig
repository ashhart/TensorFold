//! The startup probe for a template's late system role.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;

/// ``template_late_system``: "system" when the template keeps a later system message in place, else "user".
pub fn lateSystem(a: Allocator, text: model_text.Text) ![]const u8 {
    const probe_text = "tensorfold-late-system-probe";
    const probe = (try json.parse(a, "[{\"role\": \"system\", \"content\": \"s\"}, {\"role\": \"user\", \"content\": \"u\"}, {\"role\": \"assistant\", \"content\": \"a\"}, {\"role\": \"system\", \"content\": \"tensorfold-late-system-probe\"}, {\"role\": \"user\", \"content\": \"v\"}]")).ok;
    var problem: []const u8 = "";
    const rendered = text.render(a, probe, .{ .add_generation_prompt = false }, &problem) catch return "user";
    return if (std.mem.indexOf(u8, rendered.text, probe_text) != null) "system" else "user";
}
