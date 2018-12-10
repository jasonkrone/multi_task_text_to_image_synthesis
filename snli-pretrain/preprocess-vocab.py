import json
from collections import Counter
import itertools
import re
import os

import config
import data
import utils


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def main():
  split = "train"
  questions = []
  for line in file(os.path.join(config.snli_path, split + ".tsv")).read().split("\n")[1:]:
    if not line: continue
    fields = line.split("\t")
    unique_id = "%s-%012d" % (split, int(fields[0]))
    sentence1 = re.sub("\) ?", "", re.sub("\( ?", "", fields[3])).strip()
    sentence2 = re.sub("\) ?", "", re.sub("\( ?", "", fields[4])).strip()
    questions.append(sentence1 + " " + sentence2)

  tokenized_questions = [q.split(" ") for q in questions]
  question_vocab = extract_vocab(tokenized_questions)
  answer_vocab = {
      "entailment": 0,
      "neutral": 1,
      "contradiction": 2
  }

  vocabs = {
      'question': question_vocab,
      'answer': answer_vocab,
  }
  with open(config.vocabulary_path, 'w') as fd:
    json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
