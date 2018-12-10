from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


if __name__ == "__main__":
  snli_dir = "/usr/local/google/home/chrisalberti/data/glue_data/SNLI"
  snli_output_dir = "/usr/local/google/home/chrisalberti/github/AttnGAN/data/SNLI"

  if not os.path.isdir(snli_output_dir + "/test"):
    os.makedirs(snli_output_dir + "/test")
  if not os.path.isdir(snli_output_dir + "/text"):
    os.makedirs(snli_output_dir + "/text")
  dev_data = file(os.path.join(
      snli_dir, "dev.tsv")).read().split("\n")[1:]
  filenames = []

  for line in dev_data:
    if not line: continue
    fields = line.split("\t")
    unique_id = "%012d" % int(fields[0])
    for i in [1, 2]:
      sentence = re.sub("\) ?", "", re.sub("\( ?", "", fields[2 + i])).strip()
      filename = "dev-%s-%d" % (unique_id, i)
      filenames.append(filename)
      with file(os.path.join(
          snli_output_dir, "text", filename + ".txt"), "w") as fout:
        fout.write(sentence)

  with file(os.path.join(snli_output_dir, "test/filenames.pickle"), "wb") as fout:
    pickle.dump(filenames, fout)
