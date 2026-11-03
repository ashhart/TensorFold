"""Compiler-free controls for the complete five-SM, per-function comparison and source selection."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import cuda_sass_compare as compare
import cuda_sass_identity as sid
import cuda_sass_identity_test as fixtures

IMAGE=('unit',[],False,0)

def dump(word='0x000fc00000000000'):
    return '\n'.join(fixtures.dump('NOP',word).replace('sm_121',f'sm_{sm}') for sm in sid.SMS)

class Parser(unittest.TestCase):
    def test_full_five_sm_coverage(self):
        self.assertEqual(len(compare.parsed(dump(),IMAGE)),10)

    def test_missing_sm_refuses(self):
        with self.assertRaises(ValueError):compare.parsed(fixtures.dump('NOP','0x000fc00000000000'),IMAGE)

    def test_empty_and_duplicate_refuse(self):
        for text in ('',dump()+dump()):
            with self.assertRaises(ValueError):compare.parsed(text,IMAGE)

    def test_image_above_every_served_sm_refuses(self):
        with self.assertRaisesRegex(ValueError,'no emitted'):
            compare.parsed('',('unit',[],False,200))

    def test_missing_encoding_word_refuses(self):
        text=dump().replace('/* 0x000fea0003800000 */','')
        with self.assertRaises(ValueError):compare.parsed(text,IMAGE)

    def test_uppercase_late_address_is_not_skipped(self):
        data=compare.parsed(dump().replace('/*fff0*/','/*FFF0*/'),IMAGE)
        self.assertEqual(len(data['sm_121','late']),6)

class Comparison(unittest.TestCase):
    def run_case(self,build):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rows=[]
            trees={'base':root/'base','head':root/'head'}
            catalogs={tag:([],{'unit':IMAGE}) for tag in trees}
            with mock.patch.object(sid,'build',side_effect=build):
                _,_,equal=compare.compare_image(argparse.Namespace(jobs=1),trees,catalogs,'unit',root,rows,2)
            return equal,rows

    def test_equal_covers_every_function(self):
        equal,rows=self.run_case(lambda *args:dump())
        self.assertTrue(equal)
        self.assertEqual(len(rows),10)
        self.assertEqual({r['status'] for r in rows},{'equal'})

    def test_second_word_change_fails(self):
        equal,rows=self.run_case(lambda tree,*args:dump('0x000fe20000000000') if tree.name=='head' else dump())
        self.assertFalse(equal)
        self.assertEqual(sum(r['status']=='changed' for r in rows),5)
        self.assertTrue(all(r['base_encoding_sha256']!=r['head_encoding_sha256'] for r in rows if r['status']=='changed'))

    def test_unstable_head_fails(self):
        seen=0
        def build(tree,*args):
            nonlocal seen
            if tree.name=='head':seen+=1
            return dump('0x000fe20000000000') if tree.name=='head' and seen==2 else dump()
        equal,rows=self.run_case(build)
        self.assertFalse(equal)
        self.assertEqual(sum(r['status']=='unstable' for r in rows),5)

class Catalog(unittest.TestCase):
    def tree(self,root):
        (root/'zig/build').mkdir(parents=True)
        (root/'zig/kernels/cuda').mkdir(parents=True)
        (root/'zig/kernels/cuda/unit.cu').write_text('extern "C" __global__ void kernel() {}\n')
        (root/'zig/build/cuda_kernels.zig').write_text('''const torch_flags = [_][]const u8{"-O3"};
const local_flags = &[_][]const u8{"-DFIRST=1"};
const images = .{ .{ .name = "unit", .flags = local_flags } };
''')

    def test_owner_flags_are_not_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.tree(root)
            (root/'zig/build/other.zig').write_text('''const local_flags = &[_][]const u8{"-DSECOND=1"};
const images = .{ .{ .name = "other", .src = "unit", .flags = local_flags } };
''')
            torch,images=sid.catalog(root)
            self.assertEqual(torch,['-O3'])
            self.assertEqual(images['unit'][1],['-DFIRST=1'])
            self.assertEqual(images['other'][1],['-DSECOND=1'])

    def test_unknown_flags_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.tree(root)
            p=root/'zig/build/cuda_kernels.zig'
            p.write_text(p.read_text().replace('.flags = local_flags','.flags = missing_flags'))
            with self.assertRaises(ValueError):sid.catalog(root)

    def test_global_flags_change_selects_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=root/'base';b=root/'head';self.tree(a);self.tree(b)
            p=b/'zig/build/cuda_kernels.zig';p.write_text(p.read_text().replace('"-O3"','"-O2"'))
            images,_=compare.select({'base':a,'head':b},{'base':sid.catalog(a),'head':sid.catalog(b)})
            self.assertEqual(images,['unit'])

class Controls(unittest.TestCase):
    def run_control(self,changed=False,unstable=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);head=root/'head';head.mkdir()
            catalogs={'head':([],{ 'unit':IMAGE })}
            calls=0
            def build(*args):
                nonlocal calls
                calls+=1
                word='0x000fe20000000000' if changed or unstable and calls==2 else '0x000fc00000000000'
                return dump(word)
            def mutate(head,dest,*args):
                dest.mkdir();return dest
            with mock.patch.object(sid,'CONTROLS',(('order','common/reduce.cuh','a','b'),)), \
                 mock.patch.object(compare.sources,'mutated',side_effect=mutate), \
                 mock.patch.object(sid,'includes',return_value={'zig/kernels/cuda/common/reduce.cuh'}), \
                 mock.patch.object(sid,'build',side_effect=build):
                result=compare.controls(argparse.Namespace(),{'head':head},catalogs,
                    {'unit':compare.parsed(dump(),IMAGE)},root)
            rows=(root/'control-0/compared-functions.tsv').read_text().splitlines()
            return result,rows

    def test_control_records_all_compared_functions(self):
        result,rows=self.run_control(changed=True)
        self.assertTrue(result[0]['pass'])
        self.assertEqual(len(rows),11)
        self.assertEqual(len(result[0]['changed_functions']),5)

    def test_undetected_control_refuses(self):
        with self.assertRaisesRegex(ValueError,'failed to detect'):self.run_control()

    def test_unstable_control_refuses(self):
        with self.assertRaisesRegex(ValueError,'unstable control'):self.run_control(unstable=True)

if __name__=='__main__':unittest.main()
