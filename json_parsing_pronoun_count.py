import json
from pprint import pprint
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from contextlib import redirect_stdout

with open('tifu_text_samples.json') as f:
    data = json.load(f)

# pprint(data)

print(data["data"][0]["data"]["selftext"])

with open('tifu_text_samples.txt', 'w') as f:
    with redirect_stdout(f):
        # Create array containing all the values of the "self-text"
        for i in range(975):
            # words = word_tokenize(data["data"][i]["data"]["selftext"])
            # word_breakdown = pos_tag(words)
            # print(word_breakdown)

            print(data["data"][i]["data"]["selftext"].encode("utf-8"))
            print("\n------\n")

f.close()
