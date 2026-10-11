//! TENSORFOLD_REQUEST_LOG: each chat and completion body appended as one JSON line, images redacted, for replays.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The body with every image part's URL replaced, the rest kept.
fn redact(a: Allocator, v: Value) Allocator.Error!Value {
    switch (v) {
        .array => |items| {
            const out = try a.alloc(Value, items.len);
            for (items, out) |item, *slot| slot.* = try redact(a, item);
            return .{ .array = out };
        },
        .object => |o| {
            if (v.typeIs("image_url")) {
                const copy = try json.copyObject(a, o);
                const url = try json.newObject(a);
                try url.put(a, "url", .{ .string = "<redacted>" });
                try copy.put(a, "image_url", .{ .object = url });
                return .{ .object = copy };
            }
            const out = try json.newObject(a);
            for (o.keys(), o.values()) |k, item| try out.put(a, k, try redact(a, item));
            return .{ .object = out };
        },
        else => return v,
    }
}

/// Appends ``body`` to ``path`` unless it is background work (batch jobs are not client traffic).
pub fn append(a: Allocator, path: []const u8, body: Value) void {
    if (body.strField("priority")) |p| if (std.mem.eql(u8, p, "background")) return;
    const line = json.stringify(a, redact(a, body) catch return, .{}) catch return;
    const pathz = a.dupeSentinel(u8, path, 0) catch return;
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, pathz, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true }, 0o644) catch return;
    defer _ = std.posix.system.close(fd);
    const framed = std.mem.concat(a, u8, &.{ line, "\n" }) catch return;
    _ = std.posix.system.write(fd, framed.ptr, framed.len);
}
