# run the tests using 'PYTHONPATH=src python3 -m unittest discover tests -v'

import unittest
from larp_larp.tokenizer.char import CharTokenizer

text = 'my name is not nan'
tokenizer = CharTokenizer(text)

class TestCharTokenizer(unittest.TestCase):

    def test_1(self):
        '''Make one round trip'''
        self.assertEqual(text, tokenizer.decode(tokenizer.encode(text)))
