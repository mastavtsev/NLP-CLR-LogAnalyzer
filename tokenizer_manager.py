import json
import os

from transformers import PreTrainedTokenizerFast


class TokenizerManager:
    base_dir = os.path.dirname(__file__)
    _base_dir = os.path.join(base_dir, 'tokenizers')
    _paths = {
        1: "fast_bpe_l_5_v_512",
        2: "fast_bpe_l_10_v_1109",
        3: "fast_bpe_l_20_v_1707",
        4: "fast_bpe_l_25_v_2304",
        5: "fast_bpe_l_30_v_2901",
        6: "fast_bpe_l_40_v_3499",
        7: "fast_bpe_l_60_v_4096",
        8: "fast_bpe_l_NONE_v_5000",
        9: "fast_bpe_l_NONE_v_12500",
        10: "fast_bpe_l_NONE_v_20000",
        11: "fast_bpe_l_100_v_5000",
        12: "fast_bpe_l_200_v_12500",
        13: "fast_bpe_l_300_v_20000"
    }

    _tokenizers = {}

    _mappings = {}

    @staticmethod
    def get_tokenizer(level):
        if level not in TokenizerManager._tokenizers:
            tokenizer_dir = TokenizerManager._paths.get(level)
            tokenizer_dir = os.path.join(TokenizerManager._base_dir, tokenizer_dir)
            if tokenizer_dir:
                TokenizerManager._tokenizers[level] = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
            else:
                raise ValueError(f"No tokenizer defined for level {level}")
        return TokenizerManager._tokenizers[level]

    @staticmethod
    def get_mapping(level):
        if level not in TokenizerManager._mappings:
            tokenizer_dir = TokenizerManager._paths.get(level)
            mapping_path = os.path.join(TokenizerManager._base_dir, tokenizer_dir)
            mapping_path = os.path.join(mapping_path, "new_vocab_mapping_uni_rep.json")
            # mapping_path = TokenizerManager._base_dir + tokenizer_dir + "/new_vocab_mapping_uni_rep.json"
            if mapping_path:
                TokenizerManager._mappings[level] = TokenizerManager.__read_dict_from_json_file(mapping_path)
            else:
                raise ValueError(f"No mapping defined for level {level}")
        return TokenizerManager._mappings[level]

    @staticmethod
    def __read_dict_from_json_file(filepath):
        with open(filepath, 'r') as json_file:
            dictionary = json.load(json_file)
        return dictionary
