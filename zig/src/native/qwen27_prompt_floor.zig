//! Native shared lanes need their full row bank even when an individual prompt pass is smaller.
const std = @import("std");
pub const minimum: u32 = 128;
pub const Source = enum { environment, frame };

pub fn check(rows: usize) !void {
    if (rows < minimum) return error.PromptRowsTooSmall;
}

pub fn describe(buffer: []u8, rows: usize, source: Source) ![]const u8 {
    const origin = switch (source) {
        .environment => "TF_PROMPT_ROWS",
        .frame => "model.frame.capacity",
    };
    const action = switch (source) {
        .environment => "Set TF_PROMPT_ROWS to at least 128 (a multiple of 16), or unset it to use the device default.",
        .frame => "Create the model frame with at least 128 rows before attaching native lanes.",
    };
    return std.fmt.bufPrint(buffer, "native Qwen prompt serving needs at least {d} prompt rows; {s} supplied {d}. {s}", .{ minimum, origin, rows, action });
}

pub fn admit(a: std.mem.Allocator, rows: usize, source: Source, why: *[]const u8) !void {
    check(rows) catch |err| {
        var buffer: [320]u8 = undefined;
        why.* = try a.dupe(u8, try describe(&buffer, rows, source));
        return err;
    };
}

pub fn environment(a: std.mem.Allocator, value: ?[]const u8, why: *[]const u8) !void {
    const text = value orelse return;
    const settings = @import("sys").environment;
    _ = settings.parse("TF_PROMPT_ROWS", text) catch {
        // a whole count below the floor gets the floor's own message; any other bad text the table's
        const rows = std.fmt.parseInt(u32, text, 10) catch minimum;
        if (rows < minimum) return admit(a, rows, .environment, why);
        why.* = try settings.describe(a, "TF_PROMPT_ROWS", text);
        return error.InvalidEnvironment;
    };
}
