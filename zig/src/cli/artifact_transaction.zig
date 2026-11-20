//! One immutable target/drafter plan and its recoverable cache publication, before either artifact replaces the visible head.
const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("core");
const resolution = core.artifact_resolution;
const release = core.drafter_release;
const sys = @import("sys");
const hub = @import("hub.zig");
const catalog = @import("catalog.zig");
const models = @import("models.zig");
const snapshot_equal = @import("snapshot_equal.zig");

pub const Refusal = struct {
    kind: Kind,
    why: release.Why,

    pub const Kind = enum {
        artifact_identity_missing,
        context_unavailable,
        unpinned_revision,
        template_mismatch,
        vocabulary_mismatch,
        capability_missing,
        capability_unsupported,
        backend_unsupported,
        capacity_exceeded,
        multi_machine_required,
        raise_limit_required,
        non_hub_artifact,
        catalog_target_mismatch,
        catalog_revision_mismatch,
        catalog_drafter_mismatch,
        drafter_target_mismatch,
        drafter_release_missing,
        target_adapter_missing,
        adapter_mismatch,
        artifact_collision,
        bad_drafter_release,
    };

    pub fn name(r: *const Refusal) []const u8 {
        return @tagName(r.kind);
    }

    pub fn message(r: *const Refusal) []const u8 {
        return r.why.text();
    }
};

pub const Plan = struct {
    resolved: resolution.Plan,

    pub fn describe(p: Plan, out: *std.Io.Writer) !void {
        return p.resolved.describe(out);
    }

    pub fn begin(p: Plan, a: Allocator, io: std.Io, cache_root: []const u8) !Transaction {
        return beginTransaction(a, io, cache_root, p);
    }
};

pub const Outcome = union(enum) { plan: Plan, refusal: Refusal };

pub const Constraints = struct {
    catalog_entry: ?*const catalog.Entry = null,
    machine: ?*const sys.machine.Machine = null,
    drafter_release: ?*const release.Config = null,
    target_adapter: ?release.Adapter = null,
};

pub fn resolve(request: resolution.Request) Outcome {
    return resolveChecked(request, .{});
}

pub fn resolveChecked(request: resolution.Request, constraints: Constraints) Outcome {
    const resolved = resolution.resolve(request) catch |e| {
        const kind: Refusal.Kind = switch (e) {
            error.ArtifactIdentityMissing => .artifact_identity_missing,
            error.ContextUnavailable => .context_unavailable,
            error.UnpinnedRevision => .unpinned_revision,
            error.TemplateMismatch => .template_mismatch,
            error.VocabularyMismatch => .vocabulary_mismatch,
            error.CapabilityMissing => .capability_missing,
        };
        return .{ .refusal = refusal(kind, "{s}", .{@errorName(e)}) };
    };
    if (request.drafter) |draft| if (std.mem.eql(u8, draft.repository, request.base.repository))
        return .{ .refusal = refusal(.artifact_collision, "target and drafter must be distinct repositories in one cache transaction", .{}) };
    if (constraints.drafter_release) |manifest| {
        if (request.drafter == null) return .{ .refusal = refusal(.drafter_release_missing, "the release has no drafter artifact in this request", .{}) };
        if (!std.mem.eql(u8, manifest.target.repo, request.base.repository) or !std.mem.eql(u8, manifest.target.revision, request.base.revision))
            return .{ .refusal = refusal(.drafter_target_mismatch, "the release targets {s}@{s}, not {s}@{s}", .{ manifest.target.repo, manifest.target.revision, request.base.repository, request.base.revision }) };
        const adapter = constraints.target_adapter orelse return .{ .refusal = refusal(.target_adapter_missing, "the loaded target's adapter is required to check this release", .{}) };
        var why: release.Why = .{};
        release.validate(manifest, &why) catch return .{ .refusal = .{ .kind = .bad_drafter_release, .why = why } };
        release.check(manifest, adapter, &why) catch return .{ .refusal = .{ .kind = .adapter_mismatch, .why = why } };
    } else if (constraints.target_adapter != null) {
        return .{ .refusal = refusal(.drafter_release_missing, "a target adapter was supplied without a drafter release", .{}) };
    }
    if (constraints.catalog_entry) |entry| {
        if (entry.hf_dir != null) return .{ .refusal = refusal(.non_hub_artifact, "{s} is installed as a plain folder, not a Hub cache transaction", .{entry.repo}) };
        if (!std.mem.eql(u8, entry.repo, request.base.repository)) return .{ .refusal = refusal(.catalog_target_mismatch, "catalog target {s} does not match {s}", .{ entry.repo, request.base.repository }) };
        if (entry.revision) |revision| if (!std.mem.eql(u8, revision, request.base.revision))
            return .{ .refusal = refusal(.catalog_revision_mismatch, "catalog pins {s}, not the requested {s}", .{ revision, request.base.revision }) };
        if (request.drafter) |draft| if (entry.drafter) |expected| {
            if (!std.mem.eql(u8, expected, draft.repository)) return .{ .refusal = refusal(.catalog_drafter_mismatch, "catalog drafter {s} does not match {s}", .{ expected, draft.repository }) };
            if (entry.drafter_revision) |revision| if (!std.mem.eql(u8, revision, draft.revision))
                return .{ .refusal = refusal(.catalog_revision_mismatch, "catalog pins drafter {s}, not {s}", .{ revision, draft.revision }) };
        };
    }
    if (constraints.machine) |machine| {
        if (machine.backend == .none or !std.mem.eql(u8, request.capability.backend, @tagName(machine.backend)))
            return .{ .refusal = refusal(.backend_unsupported, "the {s} plan does not match this machine's {s} backend", .{ request.capability.backend, @tagName(machine.backend) }) };
        if (constraints.catalog_entry) |entry| switch (models.fitOf(entry, machine)) {
            .other_backend, .unknown, .untested => return .{ .refusal = refusal(.capability_unsupported, "{s} is not served by this machine's capability", .{entry.repo}) },
            .too_big => return .{ .refusal = refusal(.capacity_exceeded, "{s} exceeds this machine's admitted GPU capacity", .{entry.repo}) },
            .pair => return .{ .refusal = refusal(.multi_machine_required, "{s} requires a multi-machine plan", .{entry.repo}) },
            .raise_limit => return .{ .refusal = refusal(.raise_limit_required, "{s} requires the machine's GPU limit to be raised before this plan", .{entry.repo}) },
            .fits, .tight => {},
        };
    } else if (constraints.catalog_entry != null) {
        return .{ .refusal = refusal(.capability_missing, "a catalog capability check needs the actual machine", .{}) };
    }
    return .{ .plan = .{ .resolved = resolved } };
}

pub fn exactCachedSnapshot(a: Allocator, io: std.Io, cache_root: []const u8, artifact: resolution.Artifact) !?[]const u8 {
    if (!hub.isRepoIdLike(artifact.repository) or !hub.hexSha(artifact.revision)) return null;
    const repo_dir = try hub.repoDirName(a, artifact.repository);
    const snapshot = try std.fs.path.join(a, &.{ cache_root, repo_dir, "snapshots", artifact.revision });
    if (!hub.isDir(io, snapshot)) return null;
    std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ snapshot, "config.json" }), .{}) catch return null;
    return snapshot;
}

pub const State = enum { prepared, publishing, committed, rolled_back };
pub const Recovery = enum { committed, rolled_back, already_rolled_back };

const StoredArtifact = struct {
    repository: []const u8,
    revision: []const u8,
    family: []const u8,
    tensor_format: []const u8,
    template: []const u8,
    vocabulary: []const u8,
    checkpoint: []const u8,
    context: u32,
};

const JsonArtifact = struct {
    repository: []const u8,
    revision: []const u8,
    family: []const u8,
    tensor_format: []const u8,
    template: []const u8,
    vocabulary: []const u8,
    checkpoint: [64]u8,
    context: u32,
};

const StoredSlot = struct { artifact: StoredArtifact, previous_head: ?[]const u8, replacement_existed: bool };
const JsonSlot = struct { artifact: JsonArtifact, previous_head: ?[]const u8, replacement_existed: bool };

const StoredReceipt = struct {
    schema: u32,
    transaction: []const u8,
    state: State,
    target: StoredSlot,
    drafter: ?StoredSlot,
    capability: resolution.Capability,
    requested_context: u32,
    effective_context: u32,
};

const JsonReceipt = struct {
    schema: u32,
    transaction: [64]u8,
    state: State,
    target: JsonSlot,
    drafter: ?JsonSlot,
    capability: resolution.Capability,
    requested_context: u32,
    effective_context: u32,
};

const Slot = struct {
    work: []const u8,
    snapshot: []const u8,
    ref: []const u8,
    previous_head: ?[]const u8,
    replacement_existed: bool,
};

pub const Transaction = struct {
    a: Allocator,
    io: std.Io,
    plan: resolution.Plan,
    receipt_path: []const u8,
    work_root: []const u8,
    transactions_root: []const u8,
    target: Slot,
    drafter: ?Slot,
    state: State,
    lease: sys.folder_lock.Lease,

    pub fn targetStaging(t: *const Transaction) []const u8 {
        return t.target.work;
    }

    pub fn drafterStaging(t: *const Transaction) ?[]const u8 {
        return if (t.drafter) |slot| slot.work else null;
    }

    pub fn commit(t: *Transaction, current: resolution.Request) !void {
        if (t.state != .prepared) return error.TransactionState;
        return t.plan.commit(current, .{ .ctx = t, .run = publish });
    }

    pub fn rollback(t: *Transaction) !void {
        if (t.state == .committed) return error.TransactionState;
        if (t.state == .rolled_back) return;
        if (t.state == .publishing) try t.rollbackPublishing();
        std.Io.Dir.cwd().deleteTree(t.io, t.work_root) catch |e| if (e != error.FileNotFound) return e;
        try t.changeState(.rolled_back);
    }

    pub fn deinit(t: *Transaction) void {
        t.lease.release();
        t.* = undefined;
    }

    fn changeState(t: *Transaction, next: State) !void {
        const old = t.state;
        t.state = next;
        t.writeReceipt() catch |e| {
            t.state = old;
            return e;
        };
    }

    fn writeReceipt(t: *const Transaction) !void {
        var out: std.Io.Writer.Allocating = .init(t.a);
        defer out.deinit();
        try std.json.Stringify.value(t.jsonReceipt(), .{}, &out.writer);
        try sys.atomic_file.replace(t.a, t.receipt_path, &.{out.written()});
    }

    fn jsonReceipt(t: *const Transaction) JsonReceipt {
        var json: JsonReceipt = .{
            .schema = 1,
            .transaction = std.fmt.bytesToHex(t.plan.id, .lower),
            .state = t.state,
            .target = jsonSlot(t.plan.request.base, t.target),
            .drafter = null,
            .capability = t.plan.request.capability,
            .requested_context = t.plan.request.context,
            .effective_context = t.plan.effective_context,
        };
        if (t.drafter) |slot| json.drafter = jsonSlot(t.plan.request.drafter.?, slot);
        return json;
    }

    fn publishResolved(t: *Transaction, resolved: resolution.Plan) !void {
        if (!std.mem.eql(u8, &resolved.id, &t.plan.id)) return error.ArtifactTupleChanged;
        try t.preflight();
        try t.changeState(.publishing);
        t.publishAll() catch |e| {
            t.abortPublishing() catch return error.RecoveryRequired;
            return e;
        };
        t.changeState(.committed) catch |e| {
            t.abortPublishing() catch return error.RecoveryRequired;
            return e;
        };
        std.Io.Dir.cwd().deleteTree(t.io, t.work_root) catch {};
    }

    fn preflight(t: *Transaction) !void {
        for (0..t.slotCount()) |i| {
            const slot = t.slotAt(i);
            try checkStaged(t.a, t.io, slot.work);
            if (hub.isDir(t.io, slot.snapshot) and !try snapshot_equal.equal(t.io, slot.work, slot.snapshot)) return error.CachedSnapshotMismatch;
            const current = try readHead(t.a, t.io, slot.ref);
            if (!sameHead(current, slot.previous_head)) return error.WrongHead;
            try std.Io.Dir.cwd().createDirPath(t.io, std.fs.path.dirname(slot.snapshot).?);
            try std.Io.Dir.cwd().createDirPath(t.io, std.fs.path.dirname(slot.ref).?);
        }
    }

    fn publishAll(t: *Transaction) !void {
        for (0..t.slotCount()) |i| {
            const slot = t.slotAt(i);
            if (hub.isDir(t.io, slot.snapshot)) {
                std.Io.Dir.cwd().deleteTree(t.io, slot.work) catch |e| if (e != error.FileNotFound) return e;
            } else {
                try std.Io.Dir.cwd().rename(slot.work, std.Io.Dir.cwd(), slot.snapshot, t.io);
            }
            try writeHead(t.a, slot.ref, t.artifact(i).revision);
        }
    }

    fn abortPublishing(t: *Transaction) !void {
        try t.rollbackPublishing();
        try t.changeState(.rolled_back);
    }

    fn rollbackPublishing(t: *Transaction) !void {
        for (0..t.slotCount()) |i| {
            const slot = t.slotAt(i);
            const current = try readHead(t.a, t.io, slot.ref);
            if (!sameHead(current, slot.previous_head) and !sameHead(current, t.artifact(i).revision)) return error.WrongHead;
        }
        var i = t.slotCount();
        while (i > 0) {
            i -= 1;
            const slot = t.slotAt(i);
            if (slot.previous_head) |old| {
                try writeHead(t.a, slot.ref, old);
            } else {
                std.Io.Dir.cwd().deleteFile(t.io, slot.ref) catch |e| if (e != error.FileNotFound) return e;
            }
        }
        i = t.slotCount();
        while (i > 0) {
            i -= 1;
            const slot = t.slotAt(i);
            if (!slot.replacement_existed and !pathExists(t.io, slot.work) and hub.isDir(t.io, slot.snapshot))
                try std.Io.Dir.cwd().deleteTree(t.io, slot.snapshot);
        }
        std.Io.Dir.cwd().deleteTree(t.io, t.work_root) catch |e| if (e != error.FileNotFound) return e;
    }

    fn slotCount(t: *const Transaction) usize {
        return if (t.drafter == null) 1 else 2;
    }

    fn slotAt(t: *Transaction, i: usize) *Slot {
        return if (i == 0) &t.target else &t.drafter.?;
    }

    fn artifact(t: *const Transaction, i: usize) resolution.Artifact {
        return if (i == 0) t.plan.request.base else t.plan.request.drafter.?;
    }
};

fn publish(ctx: *anyopaque, plan: resolution.Plan) !void {
    const t: *Transaction = @ptrCast(@alignCast(ctx));
    return t.publishResolved(plan);
}

pub fn begin(a: Allocator, io: std.Io, cache_root: []const u8, plan: Plan) !Transaction {
    return beginTransaction(a, io, cache_root, plan);
}

fn beginTransaction(a: Allocator, io: std.Io, cache_root: []const u8, plan: Plan) !Transaction {
    try validateRepositories(plan.resolved.request);
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, cache_root);
    const lease = try sys.folder_lock.hold(a, cache_root, true);
    errdefer lease.release();
    const transactions_root = try std.fs.path.join(a, &.{ cache_root, ".tensorfold-transactions" });
    try cwd.createDirPath(io, transactions_root);
    const id = std.fmt.bytesToHex(plan.resolved.id, .lower);
    const work_root = try std.fs.path.join(a, &.{ transactions_root, &id });
    const receipt_path = try std.fmt.allocPrint(a, "{s}.json", .{work_root});
    if (pathExists(io, receipt_path) or hub.isDir(io, work_root)) return error.TransactionExists;
    const target = try makeSlot(a, io, cache_root, work_root, "target", plan.resolved.request.base);
    const drafter = if (plan.resolved.request.drafter) |draft| try makeSlot(a, io, cache_root, work_root, "drafter", draft) else null;
    var t: Transaction = .{ .a = a, .io = io, .plan = plan.resolved, .receipt_path = receipt_path, .work_root = work_root, .transactions_root = transactions_root, .target = target, .drafter = drafter, .state = .prepared, .lease = lease };
    errdefer {
        cwd.deleteTree(io, work_root) catch {};
        cwd.deleteFile(io, receipt_path) catch {};
    }
    try t.writeReceipt();
    try cwd.createDirPath(io, target.work);
    if (t.drafter) |slot| try cwd.createDirPath(io, slot.work);
    return t;
}

pub fn recover(a: Allocator, io: std.Io, cache_root: []const u8, id: [32]u8) !Recovery {
    var lease: ?sys.folder_lock.Lease = try sys.folder_lock.hold(a, cache_root, true);
    errdefer if (lease) |held| held.release();
    const transactions_root = try std.fs.path.join(a, &.{ cache_root, ".tensorfold-transactions" });
    const hex = std.fmt.bytesToHex(id, .lower);
    const work_root = try std.fs.path.join(a, &.{ transactions_root, &hex });
    const receipt_path = try std.fmt.allocPrint(a, "{s}.json", .{work_root});
    const stored = try readReceipt(a, io, receipt_path, id);
    var t = try transactionFromReceipt(a, io, cache_root, transactions_root, work_root, receipt_path, stored, lease.?);
    lease = null;
    if (stored.state == .committed) {
        t.deinit();
        return .committed;
    }
    if (stored.state == .rolled_back) {
        t.deinit();
        return .already_rolled_back;
    }
    errdefer t.deinit();
    try t.rollback();
    t.deinit();
    return .rolled_back;
}

pub fn recoverHex(a: Allocator, io: std.Io, cache_root: []const u8, transaction: []const u8) !Recovery {
    if (transaction.len != 64) return error.BadTransactionReceipt;
    var id: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&id, transaction) catch return error.BadTransactionReceipt;
    return recover(a, io, cache_root, id);
}

fn makeSlot(a: Allocator, io: std.Io, cache_root: []const u8, work_root: []const u8, name: []const u8, artifact: resolution.Artifact) !Slot {
    const repo_dir = try hub.repoDirName(a, artifact.repository);
    const snapshot = try std.fs.path.join(a, &.{ cache_root, repo_dir, "snapshots", artifact.revision });
    return .{
        .work = try std.fs.path.join(a, &.{ work_root, name }),
        .snapshot = snapshot,
        .ref = try std.fs.path.join(a, &.{ cache_root, repo_dir, "refs", "main" }),
        .previous_head = try readHead(a, io, try std.fs.path.join(a, &.{ cache_root, repo_dir, "refs", "main" })),
        .replacement_existed = hub.isDir(io, snapshot),
    };
}

fn transactionFromReceipt(a: Allocator, io: std.Io, cache_root: []const u8, transactions_root: []const u8, work_root: []const u8, receipt_path: []const u8, stored: StoredReceipt, lease: sys.folder_lock.Lease) !Transaction {
    const draft = if (stored.drafter) |slot| try artifactFromStored(slot.artifact) else null;
    const plan = resolution.resolve(.{ .base = try artifactFromStored(stored.target.artifact), .drafter = draft, .capability = stored.capability, .context = stored.requested_context }) catch return error.BadTransactionReceipt;
    var id: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&id, stored.transaction) catch return error.BadTransactionReceipt;
    if (!std.mem.eql(u8, &id, &plan.id) or stored.effective_context != plan.effective_context) return error.BadTransactionReceipt;
    const target = try slotFromStored(a, io, cache_root, work_root, "target", stored.target);
    const drafter = if (stored.drafter) |slot| try slotFromStored(a, io, cache_root, work_root, "drafter", slot) else null;
    return .{ .a = a, .io = io, .plan = plan, .receipt_path = receipt_path, .work_root = work_root, .transactions_root = transactions_root, .target = target, .drafter = drafter, .state = stored.state, .lease = lease };
}

fn slotFromStored(a: Allocator, io: std.Io, cache_root: []const u8, work_root: []const u8, name: []const u8, stored: StoredSlot) !Slot {
    _ = io;
    const artifact = try artifactFromStored(stored.artifact);
    if (stored.previous_head) |head| if (!hub.hexSha(head)) return error.BadTransactionReceipt;
    const repo_dir = try hub.repoDirName(a, artifact.repository);
    return .{
        .work = try std.fs.path.join(a, &.{ work_root, name }),
        .snapshot = try std.fs.path.join(a, &.{ cache_root, repo_dir, "snapshots", artifact.revision }),
        .ref = try std.fs.path.join(a, &.{ cache_root, repo_dir, "refs", "main" }),
        .previous_head = stored.previous_head,
        .replacement_existed = stored.replacement_existed,
    };
}

fn readReceipt(a: Allocator, io: std.Io, path: []const u8, expected: [32]u8) !StoredReceipt {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch return error.BadTransactionReceipt;
    const stored = std.json.parseFromSliceLeaky(StoredReceipt, a, text, .{ .allocate = .alloc_always }) catch return error.BadTransactionReceipt;
    if (stored.schema != 1 or stored.transaction.len != 64) return error.BadTransactionReceipt;
    var actual: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&actual, stored.transaction) catch return error.BadTransactionReceipt;
    if (!std.mem.eql(u8, &actual, &expected)) return error.BadTransactionReceipt;
    return stored;
}

fn artifactFromStored(stored: StoredArtifact) !resolution.Artifact {
    if (stored.checkpoint.len != 64) return error.BadTransactionReceipt;
    var checkpoint: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&checkpoint, stored.checkpoint) catch return error.BadTransactionReceipt;
    if (!hub.isRepoIdLike(stored.repository)) return error.BadTransactionReceipt;
    return .{ .repository = stored.repository, .revision = stored.revision, .family = stored.family, .tensor_format = stored.tensor_format, .template = stored.template, .vocabulary = stored.vocabulary, .checkpoint = checkpoint, .context = stored.context };
}

fn jsonSlot(artifact: resolution.Artifact, slot: Slot) JsonSlot {
    return .{ .artifact = .{ .repository = artifact.repository, .revision = artifact.revision, .family = artifact.family, .tensor_format = artifact.tensor_format, .template = artifact.template, .vocabulary = artifact.vocabulary, .checkpoint = std.fmt.bytesToHex(artifact.checkpoint, .lower), .context = artifact.context }, .previous_head = slot.previous_head, .replacement_existed = slot.replacement_existed };
}

fn validateRepositories(request: resolution.Request) !void {
    if (!hub.isRepoIdLike(request.base.repository)) return error.BadRepoId;
    if (request.drafter) |draft| {
        if (!hub.isRepoIdLike(draft.repository)) return error.BadRepoId;
        if (std.mem.eql(u8, draft.repository, request.base.repository)) return error.ArtifactCollision;
    }
}

fn checkStaged(a: Allocator, io: std.Io, work: []const u8) !void {
    const path = try std.fs.path.join(a, &.{ work, "config.json" });
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return error.InvalidStagedSnapshot;
    const parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch return error.InvalidStagedSnapshot;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidStagedSnapshot;
}

fn readHead(a: Allocator, io: std.Io, path: []const u8) !?[]const u8 {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(128)) catch |e| if (e == error.FileNotFound) return null else return e;
    const head = std.mem.trim(u8, bytes, " \r\n");
    if (!hub.hexSha(head)) return error.InvalidCacheHead;
    return head;
}

fn sameHead(current: ?[]const u8, expected: ?[]const u8) bool {
    if (current == null or expected == null) return current == null and expected == null;
    return std.mem.eql(u8, current.?, expected.?);
}

fn writeHead(a: Allocator, path: []const u8, head: []const u8) !void {
    try sys.atomic_file.replace(a, path, &.{ head, "\n" });
}

fn pathExists(io: std.Io, path: []const u8) bool {
    _ = std.Io.Dir.cwd().statFile(io, path, .{ .follow_symlinks = false }) catch return false;
    return true;
}

fn refusal(kind: Refusal.Kind, comptime fmt: []const u8, args: anytype) Refusal {
    var why: release.Why = .{};
    why.set(fmt, args);
    return .{ .kind = kind, .why = why };
}

test {
    _ = @import("artifact_transaction_test.zig");
}
