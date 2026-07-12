"""GBNF grammar that constrains NER output to syntactically valid JSON at
the sampler level -- a malformed-JSON answer becomes structurally
impossible instead of something to catch and retry after the fact. Built
lazily (not at import time) so importing this module never requires
llama_cpp to already be installed/working.
"""

_NER_GBNF = r'''
root   ::= "[" ws (entity ("," ws entity)*)? ws "]"
entity ::= "{" ws "\"text\"" ws ":" ws string ws "," ws "\"type\"" ws ":" ws string ws "}"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" .
ws     ::= [ \t\n]*
'''

_grammar = None


def get_ner_grammar():
    global _grammar
    if _grammar is None:
        from llama_cpp import LlamaGrammar
        _grammar = LlamaGrammar.from_string(_NER_GBNF)
    return _grammar
