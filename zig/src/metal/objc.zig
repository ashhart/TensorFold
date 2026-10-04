//! Objective-C runtime: classes, cached selectors, objc_msgSend cast to typed function pointers, pools.
const std = @import("std");

pub const Object = opaque {};
pub const Id = *Object;
pub const Class = *opaque {};
pub const Sel = *opaque {};

extern "c" fn objc_getClass(name: [*:0]const u8) ?Class;
extern "c" fn sel_registerName(name: [*:0]const u8) ?Sel;
extern "c" fn objc_msgSend() void;
extern "c" fn objc_retain(obj: Id) Id;
extern "c" fn objc_release(obj: Id) void;
extern "c" fn objc_autoreleasePoolPush() ?*anyopaque;
extern "c" fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;

pub fn class(name: [*:0]const u8) Class {
    return objc_getClass(name) orelse std.debug.panic("objc: no class {s}", .{name});
}

/// One registered selector per name, looked up once.
pub fn sel(comptime name: [:0]const u8) Sel {
    // the struct must capture `name`, or Zig shares one cache between every selector
    const Cache = struct {
        const key = name;
        var value: ?Sel = null;
    };
    if (Cache.value) |s| return s;
    const s = sel_registerName(name.ptr) orelse std.debug.panic("objc: bad selector {s}", .{name});
    Cache.value = s;
    return s;
}

/// The C function type objc_msgSend has for a receiver, a selector and `Args`, returning `Ret`.
fn MsgFn(comptime Ret: type, comptime Args: type) type {
    const fields = @typeInfo(Args).@"struct".field_types;
    var params: [fields.len + 2]type = undefined;
    params[0] = *anyopaque;
    params[1] = Sel;
    for (fields, 0..) |t, i| params[i + 2] = t;
    const attrs: [fields.len + 2]std.lang.Type.Fn.ParamAttributes = @splat(.{});
    return @Fn(&params, &attrs, Ret, .{ .@"callconv" = .c });
}

/// Send `name` to `target` (an object or class) with `args`; arm64 needs no stret variants.
pub fn msg(comptime Ret: type, target: anytype, comptime name: [:0]const u8, args: anytype) Ret {
    const f: *const MsgFn(Ret, @TypeOf(args)) = @ptrCast(&objc_msgSend);
    return @call(.auto, f, .{ @as(*anyopaque, @ptrCast(target)), sel(name) } ++ args);
}

pub fn retain(obj: Id) Id {
    return objc_retain(obj);
}

pub fn release(obj: Id) void {
    objc_release(obj);
}

/// An autorelease pool scope: objects returned autoreleased (command buffers, encoders) live until pop.
pub const Pool = struct {
    handle: ?*anyopaque,

    pub fn push() Pool {
        return .{ .handle = objc_autoreleasePoolPush() };
    }

    pub fn pop(self: Pool) void {
        objc_autoreleasePoolPop(self.handle);
    }
};

pub const OsVersion = extern struct { major: isize, minor: isize, patch: isize };

/// The running OS version (NSProcessInfo.operatingSystemVersion).
pub fn osVersion() OsVersion {
    return msg(OsVersion, msg(Id, class("NSProcessInfo"), "processInfo", .{}), "operatingSystemVersion", .{});
}

const utf8_encoding: usize = 4;

/// An owned NSString holding a copy of `bytes`.
pub fn string(bytes: []const u8) Id {
    const raw = msg(Id, class("NSString"), "alloc", .{});
    return msg(?Id, raw, "initWithBytes:length:encoding:", .{ bytes.ptr, bytes.len, utf8_encoding }) orelse
        std.debug.panic("objc: string is not UTF-8", .{});
}

/// The NUL-terminated UTF-8 view of an NSString (valid while the string lives).
pub fn utf8(str: Id) [*:0]const u8 {
    return msg([*:0]const u8, str, "UTF8String", .{});
}

/// An NSError's description, for messages.
pub fn errorText(err: ?Id) [*:0]const u8 {
    const e = err orelse return "(no NSError)";
    return utf8(msg(Id, e, "localizedDescription", .{}));
}

test "each selector name keeps its own cache" {
    try std.testing.expect(sel("alloc") != sel("init"));
    try std.testing.expect(sel("alloc") == sel("alloc"));
}

test "NSString round trip through typed objc_msgSend" {
    const pool = Pool.push();
    defer pool.pop();
    const s = string("tensorfold \u{2713}");
    defer release(s);
    try std.testing.expectEqualStrings("tensorfold \u{2713}", std.mem.span(utf8(s)));
    try std.testing.expectEqual(@as(usize, 14), msg(usize, s, "lengthOfBytesUsingEncoding:", .{utf8_encoding}));
    try std.testing.expect(msg(bool, s, "isKindOfClass:", .{class("NSString")}));
}
