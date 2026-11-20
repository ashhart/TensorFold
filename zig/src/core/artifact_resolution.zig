//! Resolve a target/proposal tuple before a caller downloads bytes or replaces a loaded model.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Artifact = struct {
    repository: []const u8,
    revision: []const u8,
    family: []const u8,
    tensor_format: []const u8,
    template: []const u8,
    vocabulary: []const u8,
    checkpoint: [32]u8,
    context: u32,
};

pub const Capability = struct {
    backend: []const u8,
    device: []const u8,
};

pub const Request = struct {
    base: Artifact,
    drafter: ?Artifact = null,
    capability: Capability,
    /// Zero leaves the context at the artifacts' common limit.
    context: u32 = 0,
};

pub const Plan = struct {
    request: Request,
    effective_context: u32,
    id: [32]u8,

    /// The receipt is available before the caller performs any download or replacement.
    pub fn describe(p: Plan, w: *std.Io.Writer) !void {
        try std.json.Stringify.value(.{
            .schema = 1,
            .transaction = std.fmt.bytesToHex(p.id, .lower),
            .base = artifactJson(p.request.base),
            .drafter = if (p.request.drafter) |draft| artifactJson(draft) else null,
            .backend = p.request.capability.backend,
            .device = p.request.capability.device,
            .effective_context = p.effective_context,
        }, .{}, w);
        try w.writeByte('\n');
    }

    /// A replacement is permitted only for the exact tuple from which this plan was resolved.
    pub fn commit(p: Plan, current: Request, replacement: Replacement) !void {
        if (!requestEqual(p.request, current)) return error.ArtifactTupleChanged;
        try replacement.run(replacement.ctx, p);
    }
};

pub const Replacement = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, plan: Plan) anyerror!void,
};

const ArtifactJson = struct {
    repository: []const u8,
    revision: []const u8,
    family: []const u8,
    tensor_format: []const u8,
    template: []const u8,
    vocabulary: []const u8,
    checkpoint: [64]u8,
};

fn artifactJson(a: Artifact) ArtifactJson {
    return .{
        .repository = a.repository,
        .revision = a.revision,
        .family = a.family,
        .tensor_format = a.tensor_format,
        .template = a.template,
        .vocabulary = a.vocabulary,
        .checkpoint = std.fmt.bytesToHex(a.checkpoint, .lower),
    };
}

pub fn resolve(request: Request) !Plan {
    try validate(request.base);
    if (request.drafter) |draft| {
        try validate(draft);
        if (!std.mem.eql(u8, request.base.template, draft.template)) return error.TemplateMismatch;
        if (!std.mem.eql(u8, request.base.vocabulary, draft.vocabulary)) return error.VocabularyMismatch;
    }
    if (request.capability.backend.len == 0 or request.capability.device.len == 0) return error.CapabilityMissing;
    var context = request.base.context;
    if (request.drafter) |draft| context = @min(context, draft.context);
    if (request.context != 0) context = @min(context, request.context);
    return .{ .request = request, .effective_context = context, .id = digest(request, context) };
}

fn validate(a: Artifact) !void {
    if (a.repository.len == 0 or a.family.len == 0 or a.tensor_format.len == 0) return error.ArtifactIdentityMissing;
    if (a.template.len == 0 or a.vocabulary.len == 0) return error.ArtifactIdentityMissing;
    if (a.context == 0) return error.ContextUnavailable;
    if (a.revision.len != 40) return error.UnpinnedRevision;
    for (a.revision) |c| if (!std.ascii.isHex(c)) return error.UnpinnedRevision;
    // A symbolic revision must never authorize replacement, even if its repository is unchanged.
}

fn digest(request: Request, context: u32) [32]u8 {
    var h = Sha256.init(.{});
    h.update("tensorfold-artifact-resolution-v1");
    artifactDigest(&h, request.base);
    if (request.drafter) |draft| {
        h.update("draft");
        artifactDigest(&h, draft);
    } else h.update("no-draft");
    field(&h, request.capability.backend);
    field(&h, request.capability.device);
    var encoded: [4]u8 = undefined;
    std.mem.writeInt(u32, &encoded, context, .little);
    h.update(&encoded);
    return h.finalResult();
}

fn artifactDigest(h: *Sha256, a: Artifact) void {
    field(h, a.repository);
    field(h, a.revision);
    field(h, a.family);
    field(h, a.tensor_format);
    field(h, a.template);
    field(h, a.vocabulary);
    h.update(&a.checkpoint);
    var encoded: [4]u8 = undefined;
    std.mem.writeInt(u32, &encoded, a.context, .little);
    h.update(&encoded);
}

fn field(h: *Sha256, value: []const u8) void {
    var encoded: [8]u8 = undefined;
    std.mem.writeInt(u64, &encoded, value.len, .little);
    h.update(&encoded);
    h.update(value);
}

fn requestEqual(a: Request, b: Request) bool {
    if (!artifactEqual(a.base, b.base)) return false;
    if ((a.drafter == null) != (b.drafter == null)) return false;
    if (a.drafter) |draft| if (!artifactEqual(draft, b.drafter.?)) return false;
    return a.context == b.context and
        std.mem.eql(u8, a.capability.backend, b.capability.backend) and
        std.mem.eql(u8, a.capability.device, b.capability.device);
}

fn artifactEqual(a: Artifact, b: Artifact) bool {
    return a.context == b.context and std.mem.eql(u8, &a.checkpoint, &b.checkpoint) and
        std.mem.eql(u8, a.repository, b.repository) and
        std.mem.eql(u8, a.revision, b.revision) and
        std.mem.eql(u8, a.family, b.family) and
        std.mem.eql(u8, a.tensor_format, b.tensor_format) and
        std.mem.eql(u8, a.template, b.template) and
        std.mem.eql(u8, a.vocabulary, b.vocabulary);
}

test {
    _ = @import("artifact_resolution_test.zig");
}
