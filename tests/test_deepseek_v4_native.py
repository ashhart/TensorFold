"""Native ABI ownership and fatal-error isolation without loading a model."""
import ctypes as C
import os
import shutil
import struct
import subprocess

import numpy as np
import pytest

from tensorfold.families.deepseek_v4.cuda.build import PIN, verify_sources, VENDOR
from tensorfold.families.deepseek_v4.cuda.native import NativeError, NativeLibrary, NativeSession


def test_pinned_cpu_library_iq2_donor_primitive():
    verify_sources(VENDOR)
    path = os.environ.get('TENSORFOLD_TEST_CPU_LIBRARY')
    if not path:
        pytest.skip('set TENSORFOLD_TEST_CPU_LIBRARY to the separately built CPU ABI library')
    api = NativeLibrary(path)
    assert api.backend() == 2
    activation = (C.c_int8 * 256)(*([1] * 256))
    # Grid zero stores eight magnitudes of 8, scale .125 gives unit weights.
    # Sign code 127 flips all eight signs; top nibble 1 multiplies scale by 3.
    for scale, word, expected in ((1., 0, 256.), (0., 0, 0.),
                                  (1., 0x0fffffff, -256.), (1., 0x10000000, 768.)):
        payload = struct.pack('<e', scale) + (bytes(4) + struct.pack('<I', word)) * 8
        packed, result = C.create_string_buffer(payload), C.c_float()
        assert api.iq2_dot(packed, 66, activation, 256, C.byref(result)) == 0
        assert result.value == expected


def test_native_rpc_ownership_and_fatal_exit(tmp_path):
    cc = shutil.which('cc')
    if not cc:
        pytest.skip('C compiler required for process boundary fixture')
    source = tmp_path / 'stub.c'
    source.write_text(r'''
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>
int tf_ds4_abi(void) { return 1; }
const char *tf_ds4_revision(void) { return "REVISION"; }
int tf_ds4_backend(void) { return 1; }
int tf_ds4_open(const char *p,int n,int t,void **out,char *err,size_t cap) {
    if (strstr(p,"crash")) _exit(7);
    if (strstr(p,"error")) { snprintf(err,cap,"fixture failure"); return 1; }
    int *v=malloc(sizeof(int)); *v=n; *out=v; return 0;
}
void tf_ds4_close(void *p) { free(p); }
int tf_ds4_vocab(void *p) { return 4; }
int tf_ds4_eos(void *p) { return 3; }
int tf_ds4_context(void *p) { return *(int *)p; }
void tf_ds4_reset(void *p) {}
int tf_ds4_sync(void *p,const int *ids,int n,char *err,size_t cap) { return 0; }
int tf_ds4_eval(void *p,int id,char *err,size_t cap) { return 0; }
int tf_ds4_logits(void *p,float *out,int n) {
    for(int i=0;i<n;i++) out[i]=(float)i; return n;
}
int tf_ds4_encode(void *p,const char *s,int rendered,int **out) {
    *out=malloc(sizeof(int)); **out=rendered?2:1; return 1;
}
void tf_ds4_free(void *p) { free(p); }
char *tf_ds4_token_text(void *p,int id,size_t *len) {
    *len=2; return strdup("ok");
}
int tf_ds4_iq2_dot(void *p,int b,void *a,int n,float *out) { return 1; }
'''.replace('REVISION', PIN).replace('#include <stddef.h>', '#include <stddef.h>\n#include <stdio.h>'))
    library = tmp_path / 'stub.so'
    subprocess.run([cc, '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    with NativeSession(library=library, model_path='normal', context=8, timeout=10) as session:
        session.reset()
        session.sync([0, 1])
        np.testing.assert_array_equal(session.logits(), np.arange(4, dtype=np.float32))
        session.eval(2)
        np.testing.assert_array_equal(session.eval_logits(1), np.arange(4, dtype=np.float32))
        assert session.encode('text') == [1]
        assert session.encode('text', rendered=True) == [2]
        assert session.token_text(2) == b'ok'
    assert session._closed
    with pytest.raises(NativeError, match='fixture failure'):
        NativeSession(library=library, model_path='error', context=8, timeout=10)
    with pytest.raises(NativeError, match='exited'):
        NativeSession(library=library, model_path='crash', context=8, timeout=10)
    # The parent remains usable after a C _exit() in a preceding session.
    with NativeSession(library=library, model_path='normal', context=8, timeout=10) as session:
        assert session.vocab_size == 4
    wrong = tmp_path / 'wrong-abi.so'
    source.write_text(source.read_text().replace('tf_ds4_abi(void) { return 1;',
                                                 'tf_ds4_abi(void) { return 0;'))
    subprocess.run([cc, '-shared', '-fPIC', str(source), '-o', str(wrong)], check=True)
    with pytest.raises(NativeError, match='ABI/revision mismatch'):
        NativeLibrary(wrong)
