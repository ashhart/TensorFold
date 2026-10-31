# Current-source host checks, original failure and explicit repair

The original broad host graph does NOT pass unchanged:285 steps, tile283 succeeded/3 crashes; attention281 succeeded/4 crashes. The failures are mock target geometry reads, reproduced in untouched main. Original logs retained without suppressing failures. These are separate from passing served-model measurements.

With the common three-file test-only initialization repair bind-mounted, each branch passes285/285 steps, with108 fresh tests reported and remaining tests cached. Do not describe this as4053 fresh tests or an unmodified-source pass. Exact overlay files/hashes and both commands are attached. The MLP's existing extra field initializations are preserved in its overlay. Native binaries remain byte-identical to those used for timing. This does not replace the still-incomplete fullGPU contract SUMMARY. All results remain local and unpublished.
