# CUDA source/include check

Run `python3 tools/zig/check_cuda_sources.py` with Clang available. This is a semantic name and type check of every
`zig/kernels/cuda/**/*.cu` and its real transitive includes, including standalone sources normally included by another
translation unit. New sources are discovered automatically. Six macro configurations cover sm_80, sm_86, sm_89,
sm_90, sm_120 and sm_121. The default scan reads each unit's literal min_sm from the build catalogs and records
ineligible pairs separately. Explicit source arguments force checks even below their declared floor.

Clang runs with `-fsyntax-only -nocudainc -nocudalib`. Its device semantic pass checks function bodies, explicit template
instantiations, CUDA launches, launch bounds and inline-assembly operands. It emits no PTX, object, assembly or executable and never
invokes nvcc. The macro values select source branches, not a code-generation target. Production source text is not
rewritten. No failed eligible translation unit or unresolved project identifier is excluded. The cluster declarations
exist only from sm_90. Below sm_89, preprocessed inline PTX cvt instructions using e4m3x2 or e5m2x2 are rejected.
Below sm_90, inline PTX barrier.cluster instructions, mapa and ::cluster modifiers are also rejected.
Preprocessing respects include chains and architecture guards; comments and disabled branches cannot trigger that rule.

The headers in this directory supply a fixed, manually reviewed subset of CUDA SDK and standard math declarations.
They contain no project helpers and must never be populated from unknown identifiers in compiler diagnostics. CUDA
runtime builtins are implicitly visible, as in nvcc; other headers are reached through each source's own includes.
`algorithm` supplies only min/max, and `vector` is empty because production defines exclude the Torch-only wrappers.
The remaining standard integer, size and type-trait headers come from the host compiler.

This check catches missing declarations such as EXL3's unpack helpers and these known architecture floors. It is not
an SDK ABI, link, general PTX instruction, register, resource, architecture-support or numerical qualification.
Templates not instantiated by a source and
preprocessor branches outside the production no-Torch configurations remain unchecked. Clang warnings are preserved
in the JSON receipt. A real CUDA image build, the SASS symbol gate and device exactness tests remain required.

Run `python3 tools/zig/test_cuda_sources.py` for include, launch-bound, layout, cluster API, inline PTX and FP8-floor controls.
The layout assertion covers every coordinate of the 16 by 16 mma B fragment using an independent inverse mapping.
