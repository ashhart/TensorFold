const std = @import("std");

pub const weights_name = "model.safetensors";
pub const Kind = enum {
    mtp,
    dspark,

    pub fn modelType(self: Kind) []const u8 {
        return switch (self) {
            .mtp => "deepseek_v4_mtp",
            .dspark => "deepseek_v4_dspark",
        };
    }
};

pub fn kind(value: std.json.Value) !Kind {
    if (value != .object) return error.InvalidDraftFolder;
    const model_type = value.object.get("model_type") orelse return error.InvalidDraftFolder;
    if (model_type != .string) return error.InvalidDraftFolder;
    for ([_]Kind{ .mtp, .dspark }) |k| if (std.mem.eql(u8, model_type.string, k.modelType())) return k;
    return error.InvalidDraftFolder;
}

pub fn read(a: std.mem.Allocator, io: std.Io, directory: []const u8) !std.json.Parsed(std.json.Value) {
    var path: [4096]u8 = undefined;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&path, "{s}/config.json", .{directory}), a, .limited(16 * 1024 * 1024)) catch |err| return if (err == error.OutOfMemory) err else error.InvalidDraftFolder;
    defer a.free(bytes);
    const parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always }) catch |err| return if (err == error.OutOfMemory) err else error.InvalidDraftFolder;
    errdefer parsed.deinit();
    _ = try kind(parsed.value);
    const stat = std.Io.Dir.cwd().statFile(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ directory, weights_name }), .{}) catch return error.InvalidDraftFolder;
    if (stat.kind != .file) return error.InvalidDraftFolder;
    return parsed;
}
