import os
import logging
from unittest import TestCase
import unittest

from naruto_skills.voc import Voc


class TestVoc(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)

    def tearDown(self):
        if os.path.isfile('voc_test.pkl'):
            os.remove('voc_test.pkl')

    def test_word_1(self):
        voc = Voc(tokenize_func=Voc.WORD_LV_TOK_FUNC, space_char=Voc.WORD_LV_SPACE_CHR)
        docs = ['hom nay toi di hoc', 'moi met qua di', 'hom nay la 1 ngay dep troi ! :D']
        for doc in docs:
            voc.add_sentence(doc)
        voc.build_from_scratch()

        idxs = voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'])
        self.assertEqual(voc.idx2docs(idxs), ['hom nay toi di hoc __o__ 1 ngay dep troi'])

    def test_word_2(self):
        voc = Voc(tokenize_func=Voc.WORD_LV_TOK_FUNC, space_char=Voc.WORD_LV_SPACE_CHR)
        docs = ['hom nay toi di hoc', 'moi met qua di', 'hom nay la 1 ngay dep troi ! :D']
        for doc in docs:
            voc.add_sentence(doc)
        voc.trim(2)
        voc.build_from_scratch()

        idxs = voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'])
        self.assertEqual(voc.idx2docs(idxs), ['hom nay __o__ di __o__ __o__ __o__ __o__ __o__ __o__'])

    def test_word_dump_and_load(self):
        voc = Voc(tokenize_func=Voc.WORD_LV_TOK_FUNC, space_char=Voc.WORD_LV_SPACE_CHR)
        docs = ['hom nay toi di hoc', 'moi met qua di', 'hom nay la 1 ngay dep troi ! :D']
        for doc in docs:
            voc.add_sentence(doc)
        voc.build_from_scratch()

        idxs1 = voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'])
        voc.dump('voc_test.pkl')
        del voc
        voc_new = Voc.load('voc_test.pkl')
        idxs2 = voc_new.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'])
        self.assertEqual(idxs1, idxs2)

    def test_char_1(self):
        voc = Voc(tokenize_func=Voc.CHR_LV_TOK_FUNC, space_char=Voc.CHR_LV_SPACE_CHR)
        docs = ['hom nay toi di hoc', 'moi met qua di', 'hom nay la 1 ngay dep troi ! :D']
        for doc in docs:
            voc.add_sentence(doc)
        voc.build_from_scratch(special_tokens=['', '¬'])

        idxs = voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'])
        self.assertEqual(voc.idx2docs(idxs), ['hom nay toi di hoc ¬ao 1 ngay dep troi'])

    def test_char_2(self):
        voc = Voc(tokenize_func=Voc.CHR_LV_TOK_FUNC, space_char=Voc.CHR_LV_SPACE_CHR)

        docs = ['hom nay toi di hoc', 'moi met qua di', 'hom nay la 1 ngay dep troi ! :D']
        for doc in docs:
            voc.add_sentence(doc)
        voc.trim(2)
        voc.build_from_scratch(special_tokens=['', '¬'])

        idxs = voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'])
        self.assertEqual(voc.idx2docs(idxs), ['hom nay toi di ho¬ ¬ao ¬ n¬ay de¬ t¬oi'])

    def test_char_dump_and_load(self):
        voc = Voc(tokenize_func=Voc.CHR_LV_TOK_FUNC, space_char=Voc.CHR_LV_SPACE_CHR)

        docs = ['hom nay toi di hoc', 'moi met qua di', 'hom nay la 1 ngay dep troi ! :D']
        for doc in docs:
            voc.add_sentence(doc)
        voc.build_from_scratch(special_tokens=['', '¬'])

        idxs1 = voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'])
        voc.dump('voc_test.pkl')
        del voc
        voc_new = Voc.load('voc_test.pkl')
        idxs2 = voc_new.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'])
        self.assertEqual(idxs1, idxs2)


if __name__ == '__main__':
    unittest.main()