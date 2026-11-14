# Provenance: Apple M2 Max runs at 1aa60a91

Host: 14-inch M2 Max (Apple M2 Max, G14C B1), Linux side.
Kernel: 7.1.12-2-7-ARCH. Driver: Honeykrisp, Mesa 26.2.2-arch1.1. Zig: 0.17.0.

The v2d packet's PROVENANCE.txt names a 16-inch M1 Max. That line is wrong. Runs were on the 14-inch M2 Max; the runner output and vulkaninfo both read "Apple M2 Max (G14C B1)". PROVENANCE.txt is not edited, because SHA256SUMS covers it.

Run: `tf-vulkan-kernel-contract`, head 1aa60a91, binary sha256 19d640ec553b0f97549670fd931284c26700e8988a721d37572b135aac5cb3d4, under `flock` on the GPU lock.
Result: rc 0, 48.5 s wall. SUMMARY: 4 kernels pass gates 1-4, 0 fail, 0 known findings; 9 controls caught, 0 missed; 0 limit failures.
