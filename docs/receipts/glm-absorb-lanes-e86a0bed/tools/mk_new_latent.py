#!/usr/bin/env python3
"""mk_new_latent.py BRANCH_HEADER OUT: the branch's glm_glue_latent.cuh absorb/expand renamed new_*, for latent_eq.cu."""
import sys
s = open(sys.argv[1]).read()
a = s.index("namespace glm_latent_rows {")
b = s.index("namespace glm_pages {")
seg = s[a:b].replace("tf_glm_absorb_q", "new_absorb_q").replace("tf_glm_expand_v", "new_expand_v")
seg = seg.replace("namespace glm_latent_rows", "namespace glm_latent_rows_new").replace("glm_latent_rows::", "glm_latent_rows_new::")
open(sys.argv[2], "w").write('#pragma once\n#include "glm_triton.cuh"\nusing namespace glm_triton;\n' + seg)
