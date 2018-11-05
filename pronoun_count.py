from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk import FreqDist
from contextlib import redirect_stdout

with open('text_files/tifu_text_samples.txt') as f:
    data = f.read()

words = word_tokenize(data)
word_breakdown = pos_tag(words)
tag_fd = FreqDist(tag for (word, tag) in word_breakdown)


with open('most_common_pos.txt', 'a') as f:
    with redirect_stdout(f):
        print("Power Ledger")
        print(tag_fd.most_common())
        print('\n')