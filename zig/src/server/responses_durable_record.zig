//! Private durable Responses records: principal, model identity, committed items and expiry.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

pub const Principal = struct {
    key: ?[32]u8 = null,
    scope: []const u8 = "",
};

pub const Metadata = struct {
    principal: Principal,
    model: []const u8,
    template: []const u8,
    /// Zero has no expiry; otherwise this is a Unix timestamp in seconds.
    expires_at: i64 = 0,
};

pub const Record = struct {
    arena: std.heap.ArenaAllocator,
    response: json.Value,
    messages: []json.Value,
    parent: ?[]const u8,
    metadata: Metadata,

    pub fn deinit(r: *Record) void {
        r.arena.deinit();
        r.* = undefined;
    }
};

pub fn encode(a: Allocator, response: json.Value, messages: []json.Value, parent: ?[]const u8, metadata: Metadata) ![]u8 {
    if (metadata.model.len == 0 or metadata.template.len == 0 or metadata.expires_at < 0) return error.BadConversationMetadata;
    if (response.get("id") == null or response.get("id").? != .string) return error.BadConversationResponse;
    const root = try json.newObject(a);
    try root.put(a, "schema", try json.intValue(a, 1));
    if (metadata.principal.key) |key| {
        try root.put(a, "owner", .{ .string = try a.dupe(u8, &std.fmt.bytesToHex(key, .lower)) });
    } else try root.put(a, "owner", .null);
    try root.put(a, "scope", .{ .string = try a.dupe(u8, metadata.principal.scope) });
    try root.put(a, "model", .{ .string = try a.dupe(u8, metadata.model) });
    try root.put(a, "template", .{ .string = try a.dupe(u8, metadata.template) });
    try root.put(a, "expires_at", try json.intValue(a, metadata.expires_at));
    if (parent) |rid| try root.put(a, "parent", .{ .string = try a.dupe(u8, rid) }) else try root.put(a, "parent", .null);
    try root.put(a, "response", try json.deepCopy(a, response));
    try root.put(a, "messages", try json.deepCopy(a, .{ .array = messages }));
    return json.stringify(a, .{ .object = root }, .{});
}

pub fn decode(gpa: Allocator, bytes: []const u8) !Record {
    var record: Record = .{
        .arena = .init(gpa),
        .response = .null,
        .messages = &.{},
        .metadata = .{ .principal = .{}, .model = "", .template = "" },
        .parent = null,
    };
    errdefer record.arena.deinit();
    const a = record.arena.allocator();
    const root = (try json.parse(a, bytes)).ok;
    if (root != .object) return error.BadConversationRecord;
    const schema = (root.get("schema") orelse return error.BadConversationRecord).int64() orelse return error.BadConversationRecord;
    if (schema != 1) return error.BadConversationRecord;
    const owner = root.get("owner") orelse return error.BadConversationRecord;
    if (owner == .string) {
        if (owner.string.len != 64) return error.BadConversationRecord;
        var key: [32]u8 = undefined;
        _ = std.fmt.hexToBytes(&key, owner.string) catch return error.BadConversationRecord;
        record.metadata.principal.key = key;
    } else if (owner != .null) return error.BadConversationRecord;
    record.metadata.principal.scope = try requiredString(root, "scope", true);
    record.metadata.model = try requiredString(root, "model", false);
    record.metadata.template = try requiredString(root, "template", false);
    record.metadata.expires_at = (root.get("expires_at") orelse return error.BadConversationRecord).int64() orelse return error.BadConversationRecord;
    if (record.metadata.expires_at < 0) return error.BadConversationRecord;
    record.response = root.get("response") orelse return error.BadConversationRecord;
    if (record.response != .object or record.response.strField("id") == null) return error.BadConversationRecord;
    const messages = root.get("messages") orelse return error.BadConversationRecord;
    if (messages != .array) return error.BadConversationRecord;
    record.messages = messages.array;
    const parent = root.get("parent") orelse return error.BadConversationRecord;
    if (parent == .string) {
        if (parent.string.len == 0) return error.BadConversationRecord;
        record.parent = parent.string;
    } else if (parent != .null) return error.BadConversationRecord;
    return record;
}

fn requiredString(root: json.Value, name: []const u8, empty: bool) ![]const u8 {
    const value = root.strField(name) orelse return error.BadConversationRecord;
    if (!empty and value.len == 0) return error.BadConversationRecord;
    return value;
}

pub fn expired(metadata: Metadata, now: i64) bool {
    return metadata.expires_at != 0 and now >= metadata.expires_at;
}

pub fn sameMetadataIgnoringExpiry(a: Metadata, b: Metadata) bool {
    return std.meta.eql(a.principal.key, b.principal.key) and
        std.mem.eql(u8, a.principal.scope, b.principal.scope) and
        std.mem.eql(u8, a.model, b.model) and
        std.mem.eql(u8, a.template, b.template);
}
