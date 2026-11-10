"""Tokenizer loading that survives transformers 5.

transformers 5 maps tokenizer_class "LlamaTokenizerFast" to a rebuilt SentencePiece-style LlamaTokenizer even for
models whose tokenizer.json is byte-level BPE (e.g. deepseek-ai/deepseek-llm-7b-*). That tokenizer drops spaces
and newlines ("Hello world\\n" -> "Helloworld"). load_tokenizer() checks the round trip and, when it fails, loads
the model's own tokenizer.json instead, adding BOS like the original fast tokenizer did.
"""
import json

from huggingface_hub import hf_hub_download
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast

PROBE = "Hello world\n### Response: 4 out of 5"


def load_tokenizer(base):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.decode(tok(PROBE, add_special_tokens=False)["input_ids"]) != PROBE:
        cfg = json.load(open(hf_hub_download(base, "tokenizer_config.json")))
        special = lambda k: cfg[k]["content"] if isinstance(cfg.get(k), dict) else cfg.get(k)
        tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(base, "tokenizer.json"),
                                      bos_token=special("bos_token"), eos_token=special("eos_token"))
        if cfg.get("add_bos_token", True) and tok.bos_token:
            tok.backend_tokenizer.post_processor = TemplateProcessing(
                single=f"{tok.bos_token} $A", pair=f"{tok.bos_token} $A {tok.bos_token} $B",
                special_tokens=[(tok.bos_token, tok.bos_token_id)])
        assert tok.decode(tok(PROBE, add_special_tokens=False)["input_ids"]) == PROBE, base
    tok.pad_token = tok.eos_token
    return tok
