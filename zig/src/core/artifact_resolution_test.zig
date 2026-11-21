const std = @import("std");
const resolution = @import("artifact_resolution.zig");

const revision = "0123456789abcdef0123456789abcdef01234567";
const other_revision = "1123456789abcdef0123456789abcdef01234567";

fn artifact(name: []const u8) resolution.Artifact {
    return .{
        .repository = name,
        .revision = revision,
        .family = "target-family",
        .tensor_format = "mlx-q4g64",
        .template = "chat-v4",
        .vocabulary = "tokenizer-v4",
        .checkpoint = @splat(7),
        .context = 262144,
    };
}

fn request() resolution.Request {
    var draft = artifact("example/proposal");
    draft.family = "proposal-family";
    draft.context = 32768;
    return .{
        .base = artifact("example/target"),
        .drafter = draft,
        .capability = .{ .backend = "metal", .device = "M4-Max" },
        .context = 65536,
    };
}

const Receiver = struct {
    observed_context: u32 = 0,
    calls: usize = 0,
    summary_seen: bool = false,

    fn replace(ctx: *anyopaque, plan: resolution.Plan) !void {
        const self: *Receiver = @ptrCast(@alignCast(ctx));
        if (!self.summary_seen) return error.ReplacedBeforeReceipt;
        self.observed_context = plan.effective_context;
        self.calls += 1;
    }
};

test "artifact resolution receipt precedes replacement and carries the common context" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    const plan = try resolution.resolve(request());
    try std.testing.expectEqual(@as(u32, 32768), plan.effective_context);
    try plan.describe(&out.writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const receipt = parsed.value.object;
    try std.testing.expectEqual(@as(i64, 32768), receipt.get("effective_context").?.integer);
    try std.testing.expectEqualStrings("metal", receipt.get("backend").?.string);
    try std.testing.expectEqualStrings("M4-Max", receipt.get("device").?.string);
    try std.testing.expectEqualStrings(revision, receipt.get("base").?.object.get("revision").?.string);
    try std.testing.expectEqualStrings("proposal-family", receipt.get("drafter").?.object.get("family").?.string);

    var receiver: Receiver = .{ .summary_seen = out.written().len != 0 };
    try plan.commit(request(), .{ .ctx = &receiver, .run = Receiver.replace });
    try std.testing.expectEqual(@as(usize, 1), receiver.calls);
    try std.testing.expectEqual(@as(u32, 32768), receiver.observed_context);
}

test "artifact resolution commit refuses every changed tuple member before replacement" {
    const plan = try resolution.resolve(request());
    var receiver: Receiver = .{ .summary_seen = true };
    const replacement: resolution.Replacement = .{ .ctx = &receiver, .run = Receiver.replace };

    var changed = request();
    changed.base.revision = other_revision;
    try std.testing.expectError(error.ArtifactTupleChanged, plan.commit(changed, replacement));
    changed = request();
    changed.base.checkpoint[0] +%= 1;
    try std.testing.expectError(error.ArtifactTupleChanged, plan.commit(changed, replacement));
    changed = request();
    changed.base.tensor_format = "bf16";
    try std.testing.expectError(error.ArtifactTupleChanged, plan.commit(changed, replacement));
    changed = request();
    changed.capability.backend = "cuda";
    try std.testing.expectError(error.ArtifactTupleChanged, plan.commit(changed, replacement));
    changed = request();
    changed.capability.device = "other-device";
    try std.testing.expectError(error.ArtifactTupleChanged, plan.commit(changed, replacement));
    changed = request();
    changed.drafter = null;
    try std.testing.expectError(error.ArtifactTupleChanged, plan.commit(changed, replacement));
    changed = request();
    changed.context = 16384;
    try std.testing.expectError(error.ArtifactTupleChanged, plan.commit(changed, replacement));
    try std.testing.expectEqual(@as(usize, 0), receiver.calls);
}

test "artifact resolution rejects incompatible or unpinned proposals" {
    var requested = request();
    requested.base.revision = "main";
    try std.testing.expectError(error.UnpinnedRevision, resolution.resolve(requested));
    requested = request();
    requested.drafter.?.revision = "refs/pr/1";
    try std.testing.expectError(error.UnpinnedRevision, resolution.resolve(requested));
    requested = request();
    requested.drafter.?.template = "other-template";
    try std.testing.expectError(error.TemplateMismatch, resolution.resolve(requested));
    requested = request();
    requested.drafter.?.vocabulary = "other-vocabulary";
    try std.testing.expectError(error.VocabularyMismatch, resolution.resolve(requested));
    requested = request();
    requested.base.context = 0;
    try std.testing.expectError(error.ContextUnavailable, resolution.resolve(requested));
    requested = request();
    requested.capability.device = "";
    try std.testing.expectError(error.CapabilityMissing, resolution.resolve(requested));
}

test "artifact resolution transaction identity covers the full immutable tuple" {
    const first = try resolution.resolve(request());
    const again = try resolution.resolve(request());
    try std.testing.expectEqual(first.id, again.id);
    var changed = request();
    changed.drafter.?.checkpoint[31] +%= 1;
    const different_draft = try resolution.resolve(changed);
    try std.testing.expect(!std.mem.eql(u8, &first.id, &different_draft.id));
    changed = request();
    changed.capability.backend = "cuda";
    const different_backend = try resolution.resolve(changed);
    try std.testing.expect(!std.mem.eql(u8, &first.id, &different_backend.id));
}
