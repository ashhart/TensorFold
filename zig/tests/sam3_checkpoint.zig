const std = @import("std");
const sam = @import("sam3");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2 and args.len != 3) return error.ExpectedCheckpointPath;
    var store = try sam.checkpoint.Store.open(init.gpa, init.io, args[1]);
    defer store.deinit();
    if (args.len == 3) {
        switch (store.source) {
            .pytorch => |archive| {
                var it = archive.tensors.iterator();
                while (it.next()) |entry| {
                    if (std.mem.indexOf(u8, entry.key_ptr.*, args[2]) == null) continue;
                    std.debug.print("{s} {s} {any}\n", .{ entry.key_ptr.*, @tagName(entry.value_ptr.dtype), entry.value_ptr.shape });
                }
            },
            .safetensors => |file| {
                for (file.names.keys(), file.names.values()) |name, entry| {
                    if (std.mem.indexOf(u8, name, args[2]) == null) continue;
                    std.debug.print("{s} {s} {any}\n", .{ name, @tagName(entry.dtype), entry.shape[0..entry.rank] });
                }
            },
        }
        return;
    }
    const config = try store.config();
    std.debug.print("SAM checkpoint: {s}, {d} tensors, memory width {d}, {d} object slots per bucket\n", .{
        config.version.name(), store.count(), config.memory_width, config.bucket_size,
    });
}
