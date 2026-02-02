import tools.ai_token as tk
import os


file = "../../data/corpus_tokenizer_15mb.txt"
#with open("../../data/corpus_tokenizer_15mb.txt", "r", encoding="utf-8") as f:
#    data = f.read();

token = tk.BPETokenizer(file)

print(token.text_cleaned[:1000])
print(token.token_char[:100])
