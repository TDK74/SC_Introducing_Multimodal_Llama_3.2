import base64
import json
import warnings
import matplotlib.pyplot as plt
import tiktoken
import torch

from pathlib import Path
from IPython.display import HTML, display
from tiktoken.load import load_tiktoken_bpe
from utils import html_tokens, llama31


warnings.filterwarnings('ignore')

## ------------------------------------------------------ ##
tokenizer_path = "./content/tokenizer.model"
num_reserved_special_tokens = 256

mergeable_ranks = load_tiktoken_bpe(tokenizer_path)

num_base_tokens = len(mergeable_ranks)

special_tokens = ["<|begin_of_text|>",
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|finetune_right_pad_id|>",
                "<|step_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eom_id|>",
                "<|eot_id|>",
                "<|python_tag|>", ]

reserved_tokens = [f"<|reserved_special_token_{2 + i}|>"
                    for i in range(num_reserved_special_tokens - len(special_tokens))]
special_tokens = special_tokens + reserved_tokens

# source: https://github.com/meta-llama/llama-models/blob/main/models/llama3/api/tokenizer.py#L53

tokenizer = tiktoken.Encoding(name = Path(tokenizer_path).name,
                            pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+\
                                    |\p{N}{1,3}|?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                            mergeable_ranks = mergeable_ranks,
                            special_tokens = {token: len(mergeable_ranks) + i
                                            for i, token in enumerate(special_tokens)}, )

## ------------------------------------------------------ ##
tokenizer.encode("hello")

## ------------------------------------------------------ ##
tokenizer.decode([15339])

## ------------------------------------------------------ ##
tokenizer.encode("hello Andrew")

## ------------------------------------------------------ ##
tokenizer.encode("hello andrew")

## ------------------------------------------------------ ##
input_text = "hello world"
len(tokenizer.encode(input_text))

## ------------------------------------------------------ ##
question = "Who wrote the book Charlotte's Web?"

prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

encoded_tokens = tokenizer.encode(prompt, allowed_special = "all")
len(encoded_tokens)

## ------------------------------------------------------ ##
decoded_tokens = [tokenizer.decode([token]) for token in encoded_tokens]

for e, d in zip(encoded_tokens, decoded_tokens):
    print(e, d)

## ------------------------------------------------------ ##
display(HTML(html_tokens(decoded_tokens)))

## ------------------------------------------------------ ##
prompt = "Supercalifragilisticexpialidocious"

encoded_tokens = tokenizer.encode(prompt, allowed_special = "all")
decoded_tokens = [tokenizer.decode([token]) for token in encoded_tokens]
display(HTML(html_tokens(decoded_tokens)))

## ------------------------------------------------------ ##
prompt = "Непротивоконституционирайте"

encoded_tokens = tokenizer.encode(prompt, allowed_special = "all")
decoded_tokens = [tokenizer.decode([token]) for token in encoded_tokens]
display(HTML(html_tokens(decoded_tokens)))

## ------------------------------------------------------ ##
question = "How many r's in the word strawberry?"

prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

response = llama31(prompt)
print(response)

## ------------------------------------------------------ ##
encoded_tokens = tokenizer.encode(prompt, allowed_special =  "all")
decoded_tokens = [tokenizer.decode([token]) for token in encoded_tokens]
display(HTML(html_tokens(decoded_tokens)))

## ------------------------------------------------------ ##
question = "How many r's in the word s t r a w b e r r y? "

prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

response = llama31(prompt)
print(response)

## ------------------------------------------------------ ##
encoded_tokens = tokenizer.encode(prompt, allowed_special = "all")
decoded_tokens = [tokenizer.decode([token]) for token in encoded_tokens]
display(HTML(html_tokens(decoded_tokens)))

## ------------------------------------------------------ ##
encoded_tokens = []
decoded_byte_tokens = []
decoded_utf8_tokens = []

with open("./content/tokenizer.model", 'r') as file:
    for i, line in enumerate(file):
        k, v = line.strip().split(' ')
        encoded_tokens.append({k: v})
        decoded_byte_tokens.append({base64.b64decode(k) : v})
        decoded_utf8_tokens.append({base64.b64decode(k).decode('utf-8', errors = "replace") : v})

## ------------------------------------------------------ ##
list(encoded_tokens)[ : 10]

## ------------------------------------------------------ ##
list(decoded_byte_tokens)[ : 10]

## ------------------------------------------------------ ##
list(decoded_utf8_tokens)[ : 10]

## ------------------------------------------------------ ##
base64.b64encode('h'.encode('utf-8'))

## ------------------------------------------------------ ##
base64.b64encode('hello'.encode('utf-8'))

## ------------------------------------------------------ ##
question = "Which number is bigger, 9.11 or 9.9? "

prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

response = llama31(prompt)
print(response)

## ------------------------------------------------------ ##
response = llama31(prompt, 70)
print(response)

## ------------------------------------------------------ ##
response = llama31(prompt, 405)
print(response)

## ------------------------------------------------------ ##
encoded_tokens = tokenizer.encode(prompt, allowed_special = "all")
decoded_tokens = [tokenizer.decode([token]) for token in encoded_tokens]
[x for x in zip(encoded_tokens, decoded_tokens)]

## ------------------------------------------------------ ##
display(HTML(html_tokens(decoded_tokens)))

## ------------------------------------------------------ ##
input = "Reverse the string 'amazing'"

prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

response = llama31(prompt)
print(response)

## ------------------------------------------------------ ##
response = llama31(prompt, 70)
print(response)

## ------------------------------------------------------ ##
response = llama31(prompt, 405)
print(response)

## ------------------------------------------------------ ##
encoded_tokens = tokenizer.encode(prompt, allowed_special = "all")
decoded_tokens = [tokenizer.decode([token]) for token in encoded_tokens]
display(HTML(html_tokens(decoded_tokens)))

## ------------------------------------------------------ ##
input = "Reverse the string 'language'"

prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

response = llama31(prompt)
print(response)

## ------------------------------------------------------ ##
response = llama31(prompt, 70)
print(response)

## ------------------------------------------------------ ##
response = llama31(prompt, 405)
print(response)

## ------------------------------------------------------ ##
encoded_tokens = tokenizer.encode(prompt, allowed_special = "all")
decoded_tokens = [tokenizer.decode([token]) for token in encoded_tokens]
display(HTML(html_tokens(decoded_tokens)))

## ------------------------------------------------------ ##
input = "Reverse the string 'XMLElement'"

prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

response = llama31(prompt)
print(response)

## ------------------------------------------------------ ##
response = llama31(prompt, 70)
print(response)

## ------------------------------------------------------ ##
response = llama31(prompt, 405)
print(response)

## ------------------------------------------------------ ##
encoded_tokens = tokenizer.encode(prompt, allowed_special = "all")
decoded_tokens = [tokenizer.decode([token]) for token in encoded_tokens]
display(HTML(html_tokens(decoded_tokens)))
