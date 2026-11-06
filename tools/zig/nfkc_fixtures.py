"""Write a curated NFKC/NFKD golden sample (per tag: spaced, first, last, longest) with the Rust tokenizers library."""
import argparse
import json
import subprocess
import unicodedata
from pathlib import Path

PER_TAG = 11
# Compatibility jamo, half-width jamo, and half-width voiced and semi-voiced sound marks.
REQUIRED = {0x3131, 0x3164, 0x318E, 0xFFA0, 0xFFDC, 0xFF9E, 0xFF9F}
HAND_MADE = ['\uff21\uff22\uff23 \u2460 \ufb01 e\u0301 \u3351 \u1100\u1161\u11a8', '\u212b\u0323',
             '\uff76\uff9e \uff73\uff9e \uff8a\uff9f', '\u0301\u0323a', 'a\u0315\u0300\u05ae\u0300b',
             '\ufb03\u0301 \u2163\u0301', '\uac00\uac01\ud7a3', '\U0001e030\U0001ccd6\U00016d68',
             '<predict_bbox>\uff21\uff22\uff23', '\xb5\xa0\u2460', '']


def tag(text):
    """The compatibility tag of a single tagged code point, else None."""
    if len(text) != 1:
        return None
    decomposition = unicodedata.decomposition(text)
    return decomposition.split('>')[0] + '>' if decomposition.startswith('<') else None


def select(cases):
    """The curated subset of `cases`, in input order; hand-made strings are always kept."""
    groups = {}
    for i, case in enumerate(cases):
        if tag(case['input']) is not None:
            groups.setdefault(tag(case['input']), []).append(i)
    keep = {i for i, case in enumerate(cases) if tag(case['input']) is None or ord(case['input']) in REQUIRED}
    for members in groups.values():
        k = min(PER_TAG, len(members))
        keep.update(members[round(j * (len(members) - 1) / max(k - 1, 1))] for j in range(k))
        keep.add(max(members, key=lambda i: len(cases[i]['nfkd'])))
    return [cases[i] for i in sorted(keep)]


def dump(path, tokenizers_version, cases):
    """Write the manifest with one case per line."""
    lines = ',\n'.join('    ' + json.dumps(case, ensure_ascii=False) for case in cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = f'{{\n  "tokenizers_version": {json.dumps(tokenizers_version)},\n  "normalization_version": "9.0",\n'
    path.write_text(head + '  "cases": [\n' + lines + '\n  ]\n}\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import tokenizers
    from tokenizers import normalizers
    if tokenizers.__version__ != '0.22.2':
        raise RuntimeError('these goldens require tokenizers 0.22.2, whose normalization data is Unicode 9.0')
    perl = 'print join(",", Unicode::UCD::prop_invlist("Present_In=9.0"))'
    raw = subprocess.check_output(['perl', '-MUnicode::UCD', '-e', perl], text=True)
    bounds = [int(x) for x in raw.split(',')]
    bounds += [0x110000] * (len(bounds) % 2)
    assigned = [cp for lo, hi in zip(bounds[::2], bounds[1::2]) for cp in range(lo, hi)]
    samples = [chr(cp) for cp in assigned if unicodedata.decomposition(chr(cp)).startswith('<')] + HAND_MADE
    forms = {k.lower(): getattr(normalizers, k)() for k in ['NFC', 'NFD', 'NFKC', 'NFKD']}
    cases = select([dict(input=text, **{name: form.normalize_str(text) for name, form in forms.items()})
                    for text in samples])
    dump(args.output, tokenizers.__version__, cases)
    print('normalization goldens', len(cases))


if __name__ == '__main__':
    main()
