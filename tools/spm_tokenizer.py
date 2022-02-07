import sentencepiece as sp
import six
import unicodedata
import os
# import regex as re
from tools.logger_utils import get_logger
# import pdb
from typing import Any, Dict, List, Optional, Tuple
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

logger = get_logger()

__all__ = ['SPMTokenizer', 'DebertaV2Tokenizer']

VOCAB_FILES_NAMES = {"vocab_file": "spm.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/spm.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-v2-xlarge": 512,
    "microsoft/deberta-v2-xxlarge": 512,
    "microsoft/deberta-v2-xlarge-mnli": 512,
    "microsoft/deberta-v2-xxlarge-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-v2-xlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xlarge-mnli": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge-mnli": {"do_lower_case": False},
}


class DebertaV2Tokenizer(PretrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            split_by_punct=False,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # super(DebertaV2Tokenizer, self).__init__(
        #     do_lower_case=do_lower_case,
        #     bos_token=bos_token,
        #     eos_token=eos_token,
        #     unk_token=unk_token,
        #     sep_token=sep_token,
        #     pad_token=pad_token,
        #     cls_token=cls_token,
        #     mask_token=mask_token,
        #     split_by_punct=split_by_punct,
        #     sp_model_kwargs=self.sp_model_kwargs,
        #     **kwargs,
        # )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_lower_case = do_lower_case
        self.split_by_punct = split_by_punct
        self._tokenizer = SPMTokenizer(vocab_file,
                                       split_by_punct=split_by_punct,
                                       sp_model_kwargs=self.sp_model_kwargs)

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self._convert_id_to_token(i))
        return tokens

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    def tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if self.do_lower_case:
            text = text.lower()
        return self._tokenizer.tokenize(text)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if self.do_lower_case:
            text = text.lower()
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self._tokenizer.spm.PieceToId(token)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self._tokenizer.spm.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self._tokenizer.spm.IdToPiece(index) if index < self.vocab_size else self.unk_token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return self._tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return self._tokenizer.save_pretrained(save_directory, filename_prefix=filename_prefix)


class SPMTokenizer:

    def __init__(self, vocab_file, split_by_punct=False, sp_model_kwargs: Optional[Dict[str, Any]] = None):
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        assert os.path.exists(vocab_file)
        spm.load(vocab_file)
        bpe_vocab_size = spm.GetPieceSize()
        # Token map
        # <unk> 0+1
        # <s> 1+1
        # </s> 2+1
        self.vocab = {spm.IdToPiece(i): i for i in range(bpe_vocab_size)}
        self.ids_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
        # self.vocab['[PAD]'] = 0
        # self.vocab['[CLS]'] = 1
        # self.vocab['[SEP]'] = 2
        # self.vocab['[UNK]'] = 3

        self.spm = spm

    def __getstate__(self):
        state = self.__dict__.copy()
        state["spm"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        self.spm.Load(self.vocab_file)

    def tokenize(self, text):
        pieces = self._encode_as_pieces(text)

        def _norm(x):
            if x not in self.vocab or x == "<unk>":
                return "[UNK]"
            else:
                return x

        pieces = [_norm(p) for p in pieces]
        return pieces

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def decode(self, tokens, start=-1, end=-1, raw_text=None):
        if raw_text is None:
            return self.spm.decode_pieces([t for t in tokens])
        else:
            words = self.split_to_words(raw_text)
            word_tokens = [self.tokenize(w) for w in words]
            token2words = [0] * len(tokens)
            tid = 0
            for i, w in enumerate(word_tokens):
                for k, t in enumerate(w):
                    token2words[tid] = i
                    tid += 1
            word_start = token2words[start]
            word_end = token2words[end] if end < len(tokens) else len(words)
            text = "".join(words[word_start:word_end])
            return text

    def add_special_token(self, token):
        if token not in self.special_tokens:
            self.special_tokens.append(token)
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) - 1
                self.ids_to_tokens.append(token)
        return self.id(token)

    def part_of_whole_word(self, token, is_bos=False):
        if is_bos:
            return True
        if (
                len(token) == 1
                and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(list(token)[0]))
        ) or token in self.special_tokens:
            return False

        word_start = b"\xe2\x96\x81".decode("utf-8")
        return not token.startswith(word_start)

    def pad(self):
        return "[PAD]"

    def bos(self):
        return "[CLS]"

    def eos(self):
        return "[SEP]"

    def unk(self):
        return "[UNK]"

    def mask(self):
        return "[MASK]"

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        return self.vocab[sym] if sym in self.vocab else 1

    def _encode_as_pieces(self, text):
        text = convert_to_unicode(text)
        if self.split_by_punct:
            words = self._run_split_on_punc(text)
            pieces = [self.spm.encode(w, out_type=str) for w in words]
            return [p for w in pieces for p in w]
        else:
            return self.spm.encode(text, out_type=str)

    def split_to_words(self, text):
        pieces = self._encode_as_pieces(text)
        word_start = b"\xe2\x96\x81".decode("utf-8")
        words = []
        offset = 0
        prev_end = 0
        for i, p in enumerate(pieces):
            if p.startswith(word_start):
                if offset > prev_end:
                    words.append(text[prev_end:offset])
                prev_end = offset
                w = p.replace(word_start, "")
            else:
                w = p
            try:
                s = text.index(w, offset)
                pn = ""
                k = i + 1
                while k < len(pieces):
                    pn = pieces[k].replace(word_start, "")
                    if len(pn) > 0:
                        break
                    k += 1

                if len(pn) > 0 and pn in text[offset:s]:
                    offset = offset + 1
                else:
                    offset = s + len(w)
            except Exception:
                offset = offset + 1

        if prev_end < offset:
            words.append(text[prev_end:offset])

        return words

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def save_pretrained(self, path: str, filename_prefix: str = None):
        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + "-" + filename
        full_path = os.path.join(path, filename)
        with open(full_path, "wb") as fs:
            fs.write(self.spm.serialized_model_proto())
        return (full_path,)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError(f"Unsupported string type: {type(text)}")
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError(f"Unsupported string type: {type(text)}")
    else:
        raise ValueError("Not running on Python2 or Python 3?")

# class SPMTokenizer:
#     def __init__(self, vocab_file, do_lower_case=False, special_tokens=None, bpe_dropout=0, split_by_punct=False):
#         self.pad_token_type_id = 0
#         self.pad_token_id = 0
#         self.split_by_punct = split_by_punct
#         spm = sp.SentencePieceProcessor()
#         assert os.path.exists(vocab_file)
#         spm.load(vocab_file)
#         bpe_vocab_size = spm.GetPieceSize()
#         # Token map
#         # <unk> 0+1
#         # <s> 1+1
#         # </s> 2+1
#         self.vocab = {spm.IdToPiece(i): i for i in range(bpe_vocab_size)}
#         self.id_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
#         # self.vocab['[PAD]'] = 0
#         # self.vocab['[CLS]'] = 1
#         # self.vocab['[SEP]'] = 2
#         # self.vocab['[UNK]'] = 3
#
#         _special_tokens = ['[MASK]', '[SEP]', '[PAD]', '[UNK]', '[CLS]']
#         self.special_tokens = []
#         if special_tokens is not None:
#             _special_tokens.extend(special_tokens)
#         for t in _special_tokens:
#             self.add_special_token(t)
#
#         self.spm = spm
#         self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
#
#     def tokenize(self, text):
#         pieces = self._encode_as_pieces(text)
#
#         def _norm(x):
#             if x not in self.vocab or x == '<unk>':
#                 return '[UNK]'
#             else:
#                 return x
#
#         pieces = [_norm(p) for p in pieces]
#         return pieces
#
#     def convert_tokens_to_ids(self, tokens):
#         return [self.vocab[t] if t in self.vocab else 1 for t in tokens]
#
#     def convert_ids_to_tokens(self, ids):
#         tokens = []
#         for i in ids:
#             tokens.append(self.ids_to_tokens[i])
#         return tokens
#
#     def decode(self, tokens, start=-1, end=-1, raw_text=None):
#         if raw_text is None:
#             return self.spm.decode_pieces([t for t in tokens if t not in self.special_tokens])
#         else:
#             words = self.split_to_words(raw_text)
#             word_tokens = [self.tokenize(w) for w in words]
#             wt = [w for t in word_tokens for w in t]
#             # assert tokens == wt, f'{tokens} || {wt}'
#             if wt != tokens:
#                 for a, b in zip(wt, tokens):
#                     if a != b:
#                         pdb.set_trace()
#             token2words = [0] * len(tokens)
#             tid = 0
#             for i, w in enumerate(word_tokens):
#                 for k, t in enumerate(w):
#                     token2words[tid] = i
#                     tid += 1
#             word_start = token2words[start]
#             word_end = token2words[end] if end < len(tokens) else len(words)
#             text = ''.join(words[word_start:word_end])
#             return text
#
#     def add_special_token(self, token):
#         if token not in self.special_tokens:
#             self.special_tokens.append(token)
#             if token not in self.vocab:
#                 self.vocab[token] = len(self.vocab)
#                 self.id_to_tokens.append(token)
#         return self.id(token)
#
#     def part_of_whole_word(self, token, is_bos=False):
#         if is_bos:
#             return True
#         if (len(token) == 1 and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(
#                 list(token)[0]))) or token in self.special_tokens:
#             return False
#
#         word_start = b'\xe2\x96\x81'.decode('utf-8')
#         return not token.startswith(word_start)
#
#     def pad(self):
#         return '[PAD]'
#
#     def bos(self):
#         return '[CLS]'
#
#     def eos(self):
#         return '[SEP]'
#
#     def unk(self):
#         return '[UNK]'
#
#     def mask(self):
#         return '[MASK]'
#
#     def sym(self, id):
#         return self.ids_to_tokens[id]
#
#     def id(self, sym):
#         return self.vocab[sym] if sym in self.vocab else 1
#
#     def _encode_as_pieces(self, text):
#         text = convert_to_unicode(text)
#         if self.split_by_punct:
#             words = self._run_split_on_punc(text)
#             pieces = [self.spm.encode_as_pieces(w) for w in words]
#             return [p for w in pieces for p in w]
#         else:
#             return self.spm.encode_as_pieces(text)
#
#     def split_to_words(self, text):
#         pieces = self._encode_as_pieces(text)
#         word_start = b'\xe2\x96\x81'.decode('utf-8')
#         words = []
#         offset = 0
#         prev_end = 0
#         for i, p in enumerate(pieces):
#             if p.startswith(word_start):
#                 if offset > prev_end:
#                     words.append(text[prev_end:offset])
#                 prev_end = offset
#                 w = p.replace(word_start, '')
#             else:
#                 w = p
#             try:
#                 s = text.index(w, offset)
#                 pn = ""
#                 k = i + 1
#                 while k < len(pieces):
#                     pn = pieces[k].replace(word_start, '')
#                     if len(pn) > 0:
#                         break
#                     k += 1
#
#                 if len(pn) > 0 and pn in text[offset:s]:
#                     offset = offset + 1
#                 else:
#                     offset = s + len(w)
#             except:
#                 offset = offset + 1
#
#         if prev_end < offset:
#             words.append(text[prev_end:offset])
#
#         return words
#
#     def _run_strip_accents(self, text):
#         """Strips accents from a piece of text."""
#         text = unicodedata.normalize("NFD", text)
#         output = []
#         for char in text:
#             cat = unicodedata.category(char)
#             if cat == "Mn":
#                 continue
#             output.append(char)
#         return "".join(output)
#
#     def _run_split_on_punc(self, text):
#         """Splits punctuation on a piece of text."""
#         # words = list(re.findall(self.pat, text))
#         chars = list(text)
#         i = 0
#         start_new_word = True
#         output = []
#         while i < len(chars):
#             char = chars[i]
#             if _is_punctuation(char):
#                 output.append([char])
#                 start_new_word = True
#             else:
#                 if start_new_word:
#                     output.append([])
#                 start_new_word = False
#                 output[-1].append(char)
#             i += 1
#
#         return ["".join(x) for x in output]
#
#     def _tokenize_chinese_chars(self, text):
#         """Adds whitespace around any CJK character."""
#         output = []
#         for char in text:
#             cp = ord(char)
#             if self._is_chinese_char(cp):
#                 output.append(" ")
#                 output.append(char)
#                 output.append(" ")
#             else:
#                 output.append(char)
#         return "".join(output)
#
#     def _is_chinese_char(self, cp):
#         """Checks whether CP is the codepoint of a CJK character."""
#         # This defines a "chinese character" as anything in the CJK Unicode block:
#         #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
#         #
#         # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
#         # despite its name. The modern Korean Hangul alphabet is a different block,
#         # as is Japanese Hiragana and Katakana. Those alphabets are used to write
#         # space-separated words, so they are not treated specially and handled
#         # like the all of the other languages.
#         if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
#                 (cp >= 0x3400 and cp <= 0x4DBF) or  #
#                 (cp >= 0x20000 and cp <= 0x2A6DF) or  #
#                 (cp >= 0x2A700 and cp <= 0x2B73F) or  #
#                 (cp >= 0x2B740 and cp <= 0x2B81F) or  #
#                 (cp >= 0x2B820 and cp <= 0x2CEAF) or
#                 (cp >= 0xF900 and cp <= 0xFAFF) or  #
#                 (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
#             return True
#
#         return False
#
#     def _clean_text(self, text):
#         """Performs invalid character removal and whitespace cleanup on text."""
#         output = []
#         for char in text:
#             cp = ord(char)
#             if cp == 0 or cp == 0xfffd or _is_control(char):
#                 continue
#             if _is_whitespace(char):
#                 output.append(" ")
#             else:
#                 output.append(char)
#         return "".join(output)
#
#
# def _is_whitespace(char):
#     """Checks whether `chars` is a whitespace character."""
#     # \t, \n, and \r are technically contorl characters but we treat them
#     # as whitespace since they are generally considered as such.
#     if char == " " or char == "\t" or char == "\n" or char == "\r":
#         return True
#     cat = unicodedata.category(char)
#     if cat == "Zs":
#         return True
#     return False
#
#
# def _is_control(char):
#     """Checks whether `chars` is a control character."""
#     # These are technically control characters but we count them as whitespace
#     # characters.
#     if char == "\t" or char == "\n" or char == "\r":
#         return False
#     cat = unicodedata.category(char)
#     if cat.startswith("C"):
#         return True
#     return False
#
#
# def _is_punctuation(char):
#     """Checks whether `chars` is a punctuation character."""
#     cp = ord(char)
#     # We treat all non-letter/number ASCII as punctuation.
#     # Characters such as "^", "$", and "`" are not in the Unicode
#     # Punctuation class but we treat them as punctuation anyways, for
#     # consistency.
#     if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
#             (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
#         return True
#     cat = unicodedata.category(char)
#     if cat.startswith("P"):
#         return True
#     return False
#
#
# def whitespace_tokenize(text):
#     """Runs basic whitespace cleaning and splitting on a peice of text."""
#     text = text.strip()
#     if not text:
#         return []
#     tokens = text.split()
#     return tokens
#
#
# def convert_to_unicode(text):
#     """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
#     if six.PY3:
#         if isinstance(text, str):
#             return text
#         elif isinstance(text, bytes):
#             return text.decode("utf-8", "ignore")
#         else:
#             raise ValueError("Unsupported string type: %s" % (type(text)))
#     elif six.PY2:
#         if isinstance(text, str):
#             return text.decode("utf-8", "ignore")
#         elif isinstance(text, unicode):
#             return text
#         else:
#             raise ValueError("Unsupported string type: %s" % (type(text)))
#     else:
#         raise ValueError("Not running on Python2 or Python 3?")
