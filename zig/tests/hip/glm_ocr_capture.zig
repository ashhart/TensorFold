//! Independent BF16 stage comparison shared by the native image and text graph checks.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");

pub const Capture = struct {
    a: std.mem.Allocator,
    reference: []const u8,
    stream: hip.Stream,
    final_mismatches: ?usize = null,
    compared: usize = 0,
    total_mismatches: usize = 0,
    quiet: bool = false,
    required_names: []const []const u8 = &.{},
    final_name: []const u8 = "",
    output_root: ?[]const u8 = null,
    name_prefix: []const u8 = "",
    pub fn tap(context: *anyopaque, name: []const u8, address: u64, values: usize) !void {
        const s: *Capture = @ptrCast(@alignCast(context));
        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}{s}-bf16.npy", .{ s.reference, s.name_prefix, name });
        const bytes = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, s.a, .limited(256 << 20)) catch |err| switch (err) {
            error.FileNotFound => {
                for (s.required_names) |required| {
                    if (std.mem.eql(u8, name, required)) return error.MissingStageReference;
                }
                return;
            },
            else => return err,
        };
        defer s.a.free(bytes);
        try s.stream.synchronize();
        const out = try s.a.alloc(u16, values);
        defer s.a.free(out);
        const device: hip.DeviceBuffer = .{ .r = s.stream.r, .ptr = @ptrFromInt(address), .len = values * 2 };
        try device.download(0, std.mem.sliceAsBytes(out));
        if (!s.quiet) {
            if (s.output_root) |root| {
                const output_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}{s}.bf16", .{ root, s.name_prefix, name });
                try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = output_path, .data = std.mem.sliceAsBytes(out) });
            }
        }
        const expected = try core.npy.parse(bytes);
        try std.testing.expectEqualStrings("<u2", expected.descr);
        try std.testing.expectEqual(values * 2, expected.data.len);
        var mismatches: usize = 0;
        var max_abs: f32 = 0;
        for (out, std.mem.bytesAsSlice(u16, expected.data)) |x, y| {
            if (x != y) mismatches += 1;
            const xf: f32 = @bitCast(@as(u32, x) << 16);
            const yf: f32 = @bitCast(@as(u32, y) << 16);
            if (!std.math.isFinite(xf) or !std.math.isFinite(yf)) return error.NonFiniteStage;
            max_abs = @max(max_abs, @abs(xf - yf));
        }
        if (!s.quiet or mismatches != 0)
            std.debug.print("{s}: {d}/{d} BF16 differences, max_abs {d}\n", .{ name, mismatches, values, max_abs });
        s.compared += 1;
        s.total_mismatches += mismatches;
        if (std.mem.eql(u8, name, s.final_name)) s.final_mismatches = mismatches;
    }
};
