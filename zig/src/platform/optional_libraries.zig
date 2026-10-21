//! Optional runtime dependencies: one diagnostic and discovery description per feature.
const std = @import("std");
const builtin = @import("builtin");
const mac = builtin.os.tag == .macos;
const win = builtin.os.tag == .windows;
pub const video_library = @import("video_library.zig");

pub const Id = enum { cublaslt, cublas, nccl, verbs, video, jpeg, webp, grammar, rccl, mcdma_rpc, mcdma_fabric };
pub const Failure = error{ CublasLtUnavailable, CublasUnavailable, NcclUnavailable, VerbsUnavailable, VideoLibraryUnavailable, JpegLibraryUnavailable, WebpLibraryUnavailable, GrammarLibraryUnavailable, RcclUnavailable, McdmaRpcUnavailable, McdmaFabricUnavailable };
pub const Spec = struct {
    id: Id,
    name: []const u8,
    names: []const []const u8,
    feature: []const u8,
    install: []const u8,
    message: []const u8,
    failure: Failure,
    env: ?[*:0]const u8 = null,
    local: bool = false,
};

fn item(comptime id: Id, comptime name: []const u8, comptime names: []const []const u8, comptime feature: []const u8, comptime install: []const u8, comptime failure: Failure, comptime env: ?[*:0]const u8, comptime local: bool) Spec {
    return .{ .id = id, .name = name, .names = names, .feature = feature, .install = install, .message = names[0] ++ " is required for " ++ feature ++ "; " ++ install, .failure = failure, .env = env, .local = local };
}

const loader_path = if (win) "add its directory to PATH" else "on Linux, add its directory to LD_LIBRARY_PATH";

pub const all = [_]Spec{
    item(.cublaslt, "cuBLASLt", if (win) &.{"cublasLt64_13.dll"} else &.{ "libcublasLt.so.13", "libcublasLt.so" }, "CUDA vision towers (--vision) and CUDA cuBLASLt projections", "install the NVIDIA CUDA 13 cuBLAS runtime; " ++ loader_path ++ "; a driver-only host does not include it", error.CublasLtUnavailable, null, false),
    item(.cublas, "cuBLAS", &.{ "libcublas.so.13", "libcublas.so" }, "Drama CUDA prompt and codec GEMMs", "install the NVIDIA CUDA 13 cuBLAS runtime; " ++ loader_path ++ "", error.CublasUnavailable, null, false),
    item(.nccl, "NCCL", &.{"libnccl.so.2"}, "CUDA ranks (including the GLM pair)", "install the NVIDIA NCCL 2 runtime compatible with CUDA 13; " ++ loader_path ++ "", error.NcclUnavailable, null, false),
    item(.verbs, "RDMA verbs", if (mac) &.{"/usr/lib/librdma.dylib"} else &.{"libibverbs.so.1"}, "RoCE/RDMA gathers", if (mac) "use a macOS installation with its RDMA runtime" else "install rdma-core's libibverbs runtime and the provider for the network adapter", error.VerbsUnavailable, null, false),
    item(.video, "TensorFold video", if (mac) &.{"libtfvideo.dylib"} else if (win) &.{"tfvideo.dll"} else &.{video_library.soname}, "video inputs (--vision)", "build the optional zig build video -Dffmpeg=PREFIX runtime against FFmpeg 9.0.2 and place it in lib beside the server; see packaging/RUNTIME.md", error.VideoLibraryUnavailable, null, true),
    item(.jpeg, "libjpeg-turbo", if (mac) &.{"libturbojpeg.0.dylib"} else if (win) &.{"turbojpeg.dll"} else &.{"libturbojpeg.so.0"}, "JPEG inputs (--vision)", "install libjpeg-turbo 3.1.4.1 and set TENSORFOLD_LIBTURBOJPEG to its file, or put it on the documented loader path", error.JpegLibraryUnavailable, "TENSORFOLD_LIBTURBOJPEG", true),
    item(.webp, "libwebp", if (mac) &.{"libwebpdecoder.3.dylib"} else &.{"libwebpdecoder.so.3"}, "WebP inputs (--vision)", "install libwebp 1.6.0 and set TENSORFOLD_LIBWEBP to its decoder file, or put it on the documented loader path", error.WebpLibraryUnavailable, "TENSORFOLD_LIBWEBP", true),
    item(.grammar, "TensorFold grammar", if (mac) &.{"libtfgrammar.dylib"} else &.{"libtfgrammar.so"}, "structured output and required tool calls", "restore the matching release archive's lib directory or build zig build grammar", error.GrammarLibraryUnavailable, null, true),
    item(.rccl, "RCCL", &.{ "librccl.so.1", "librccl.so", "/opt/rocm/lib/librccl.so.1" }, "HIP tensor-parallel ranks", "install the RCCL runtime matching the ROCm installation; " ++ loader_path ++ "", error.RcclUnavailable, null, false),
    item(.mcdma_rpc, "MCDMA RPC", if (mac) &.{"libmcdma-rpc.dylib"} else &.{"libmcdma-rpc.so"}, "MCDMA mailbox and Metal speed-up communication", "build and install MCDMA's RPC helper and set MCDMA_RPC_LIBRARY to its file", error.McdmaRpcUnavailable, "MCDMA_RPC_LIBRARY", false),
    item(.mcdma_fabric, "MCDMA fabric", if (mac) &.{"libmcdma-fabric.dylib"} else &.{"libmcdma-fabric.so"}, "MCDMA peer fabric", "build MCDMA's fabric helper and name its inspected absolute path in the peer settings", error.McdmaFabricUnavailable, null, false),
};

pub fn spec(id: Id) Spec {
    return all[@backingInt(id)];
}

pub fn explain(err: anyerror) ?[]const u8 {
    for (all) |s| if (err == s.failure) return s.message;
    return null;
}

pub fn unavailable(id: Id) Failure {
    const s = spec(id);
    warn(id);
    return s.failure;
}

threadlocal var boundary_reports = false;

pub const Boundary = struct {
    previous: bool,
    pub fn deinit(b: Boundary) void {
        boundary_reports = b.previous;
    }
};

/// The native caller renders the returned refusal; direct users still get the dependency warning.
pub fn atBoundary() Boundary {
    const previous = boundary_reports;
    boundary_reports = true;
    return .{ .previous = previous };
}

pub fn warn(id: Id) void {
    if (boundary_reports) return;
    std.log.warn("{s}", .{spec(id).message});
}

/// The RPC runtime and discovery share these absolute defaults, after a nonempty MCDMA_RPC_LIBRARY.
pub fn rpcPaths(comptime os: std.Target.Os.Tag) []const [:0]const u8 {
    return if (os == .macos) &.{"/usr/local/lib/libmcdma-rpc.dylib"} else &.{ "/usr/local/lib/libmcdma-rpc.so", "/usr/lib/libmcdma-rpc.so" };
}

pub const Status = struct { name: []const u8, feature: []const u8, soname: []const u8, found: bool, install: []const u8 };

/// Loader discovery only: symbols, pinned codec versions, device support and feature configuration remain unchecked.
pub fn probe(a: std.mem.Allocator, io: std.Io, comptime Sys: type) ![]Status {
    const out = try a.alloc(Status, all.len);
    errdefer a.free(out);
    const exe = std.process.executableDirPathAlloc(io, a) catch null;
    defer if (exe) |p| a.free(p);
    for (all, out) |s, *row| row.* = .{ .name = s.name, .feature = s.feature, .soname = s.names[0], .found = found(s, a, Sys, exe), .install = s.install };
    return out;
}

pub fn found(s: Spec, a: std.mem.Allocator, comptime Sys: type, exe: ?[]const u8) bool {
    if (builtin.os.tag == .linux and s.id == .video) return video_library.open(a, Sys, exe) != null;
    if (s.env) |name| if (Sys.getenv(name)) |value| {
        if (value.len == 0 and s.id != .mcdma_rpc) return false;
        if (value.len > 0) return loads(Sys, value);
    };
    if (s.local) if (exe) |dir| for (s.names) |name| {
        for ([_][]const []const u8{ &.{ dir, "..", "lib", name }, &.{ dir, name } }) |parts| {
            const path = std.fs.path.join(a, parts) catch continue;
            defer a.free(path);
            if (loads(Sys, path)) return true;
        }
    };
    if (mac and (s.id == .jpeg or s.id == .webp)) {
        const package = if (s.id == .jpeg) "jpeg-turbo" else "webp";
        for ([_][]const u8{ "/opt/homebrew/opt", "/usr/local/opt" }) |root| {
            const path = std.fs.path.join(a, &.{ root, package, "lib", s.names[0] }) catch continue;
            defer a.free(path);
            if (loads(Sys, path)) return true;
        }
    }
    if (s.id == .mcdma_rpc) {
        for (rpcPaths(builtin.os.tag)) |path| if (loads(Sys, path)) return true;
        return false;
    }
    for (s.names) |name| if (loads(Sys, name)) return true;
    return false;
}

fn loads(comptime Sys: type, path: []const u8) bool {
    // Constructors may register process or thread exit handlers: keep every found loader reference until exit.
    _ = Sys.DynLib.open(path) catch return false;
    return true;
}
