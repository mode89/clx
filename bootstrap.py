#!/usr/bin/env python3

from collections import namedtuple
import re

import pyrsistent as pr

def main():
    text = slurp("example.clj")
    print(pr_str_(read_string(text)))

#************************************************************
# Reader
#************************************************************

TOKEN_RE = re.compile(
    r"[\s,]*"
    "("
        r"~@" "|"
        r"[\[\]{}()'`~^@]" "|"
        r"\"(?:[\\]." "|" r"[^\\\"])*\"?" "|"
        r";.*" "|"
        r"[^\s\[\]{}()'\"`@,;]+"
    ")")

INT_RE = re.compile(r"-?[0-9]+$")
FLOAT_RE = re.compile(r"-?[0-9]+\.[0-9]+$")
STRING_RE = re.compile(r"\"(?:[\\].|[^\\\"])*\"$")

def read_string(text):
    tokens = tokenize(text)
    form, _rest = read_form(tokens)
    return form

def tokenize(text):
    return list(
        filter(
            lambda t: t[0] != ";",
            map(
                lambda t: t[1],
                re.finditer(TOKEN_RE, text))))

def read_form(tokens):
    token = tokens[0]
    if token == "(":
        return read_list(tokens)
    if token == "[":
        return read_vector(tokens)
    if token == "{":
        return read_hash_map(tokens)

    return read_atom(token), tokens[1:]

def read_atom(token):
    if re.match(INT_RE, token):
        return int(token)
    if re.match(FLOAT_RE, token):
        return float(token)
    if re.match(STRING_RE, token):
        return unescape(token[1:-1])
    if token[0] == "\"":
        raise Exception("Expected end of string")
    if token == "true":
        return True
    if token == "false":
        return False
    if token == "nil":
        return None
    if token[0] == ":":
        return keyword(token[1:])
    return symbol(token)

def unescape(text):
    return text \
        .replace(r"\\", "\\") \
        .replace(r"\"", "\"") \
        .replace(r"\n", "\n")

def read_list(tokens):
    elements, rest = read_collection(tokens, "(", ")")
    return list_(*elements), rest

def read_vector(tokens):
    elements, rest = read_collection(tokens, "[", "]")
    return vector(*elements), rest

def read_hash_map(tokens):
    elements, rest = read_collection(tokens, "{", "}")
    return hash_map(*elements), rest

def read_collection(tokens, start, end):
    assert tokens[0] == start, f"Expected `{start}`"
    tokens = tokens[1:]
    sequence = []
    while tokens[0] != end:
        element, tokens = read_form(tokens)
        sequence.append(element)
        if len(tokens) == 0:
            raise Exception(f"Expected `{end}`")
    return sequence, tokens[1:]

#************************************************************
# Printer
#************************************************************

def pr_str_(obj, readably=False):
    if is_nil(obj):
        return "nil"
    if is_bool(obj):
        return "true" if obj else "false"
    if is_string(obj):
        if readably:
            return "\"" + escape(obj) + "\""
        return obj
    if is_keyword(obj):
        return ":" + obj.name
    if is_symbol(obj):
        return obj.name
    if is_list(obj):
        return "(" + \
            " ".join(map(lambda x: pr_str_(x, readably), obj)) + \
            ")"
    if is_vector(obj):
        return "[" + \
            " ".join(map(lambda x: pr_str_(x, readably), obj)) + \
            "]"
    if is_hash_map(obj):
        ret = []
        for key, value in obj.items():
            ret.append(pr_str_(key, readably))
            ret.append(pr_str_(value, readably))
        return "{" + " ".join(ret) + "}"
    return str(obj)

def escape(text):
    return text \
        .replace("\\", r"\\") \
        .replace("\"", r"\"") \
        .replace("\n", r"\n")

#************************************************************
# Core
#************************************************************

def is_nil(obj):
    return obj is None

def is_bool(obj):
    return isinstance(obj, bool)

def is_string(obj):
    return isinstance(obj, str)

Keyword = namedtuple("Keyword", ["name"])

def keyword(name):
    return Keyword(name)

def is_keyword(obj):
    return isinstance(obj, Keyword)

Symbol = namedtuple("Symbol", ["name"])

def symbol(name):
    return Symbol(name)

def is_symbol(obj):
    return isinstance(obj, Symbol)

def list_(*elements):
    return pr.plist(elements)

def is_list(obj):
    return isinstance(obj, pr.PList)

def vector(*elements):
    return pr.pvector(elements)

def is_vector(obj):
    return isinstance(obj, pr.PVector)

def hash_map(*elements):
    assert len(elements) % 2 == 0, "Expected even number of elements"
    return pr.pmap(dict(zip(elements[::2], elements[1::2])))

def is_hash_map(obj):
    return isinstance(obj, pr.PMap)

def slurp(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

if __name__ == '__main__':
    main()
