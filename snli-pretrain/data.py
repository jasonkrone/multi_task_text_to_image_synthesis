import json
import os
import os.path
import re

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import config
import utils


def get_loader(train=False, val=False, test=False):
    """ Returns a data loader for the desired split """
    assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'
    split = SNLI(
        config.snli_path,
        config.flickr_path,
        "train" if train else "dev")
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.batch_size,
        shuffle=train,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class SNLI(data.Dataset):
    """ SNLI dataset."""
    def __init__(self, snli_path, flickr_path, split):
        super(SNLI, self).__init__()
        self.snli_path = snli_path
        self.snli_ids, self.questions, self.answers = self._load_text(split)
        with open(config.vocabulary_path, 'r') as fd:
            vocab_json = json.load(fd)

        # load images
        self.transform = utils.get_transform(config.image_size, config.central_fraction)
        self.images = SnliImages(snli_path, flickr_path, split)

        # vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']

        # q and a
        self.questions = [q.split(" ") for q in self.questions]
        self.questions = [self._encode_question(q) for q in self.questions]
        self.answers = [self._encode_answers(a) for a in self.answers]

    def _load_text(self, split):
        snli_ids = []
        questions = []
        answers = []
        for line in file(os.path.join(self.snli_path, split + ".tsv")).read().split("\n")[1:]:
            if not line: continue
            fields = line.split("\t")
            if fields[1].startswith("vg"): continue
            unique_id = "%s-%012d" % (split, int(fields[0]))
            sentence1 = re.sub("\) ?", "", re.sub("\( ?", "", fields[3])).strip()
            sentence2 = re.sub("\) ?", "", re.sub("\( ?", "", fields[4])).strip()
            snli_ids.append(unique_id)
            questions.append(sentence1 + " " + sentence2)
            answers.append(fields[-1])
            # XXX debug only
            # if len(questions) > 100:
            #     break
        return snli_ids, questions, answers

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions))
        return self._max_length

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def _find_answerable(self):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        for i, answers in enumerate(self.answers):
            answer_has_index = len(answers.nonzero()) > 0
            # store the indices of anything that is answerable
            if answer_has_index:
                answerable.append(i)
        return answerable

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
        return vec, len(question)

    def _encode_answers(self, answer):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(self.answer_to_index))
        index = self.answer_to_index.get(answer)
        if index is not None:
            answer_vec[index] += 1
        return answer_vec

    def _load_image(self, image_id):
        """ Load an image """
        if image_id not in self.images.id_to_filename:
          print("Missing image id: ", image_id)
          path = self.images.id_to_filename.values()[0]
        else:
          path = self.images.id_to_filename[image_id]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return torch.from_numpy(np.array(img).astype('float32'))

    def __getitem__(self, item):
        q, q_length = self.questions[item]
        a = self.answers[item]
        image_id = self.snli_ids[item]
        v = self._load_image(image_id)
        # since batches are re-ordered for PackedSequence's, the original question order is lost
        # we return `item` so that the order of (v, q, a) triples can be restored if desired
        # without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
        return v, q, a, item, q_length

    def __len__(self):
        return len(self.questions)


# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    questions = [q['question'] for q in questions_json['questions']]
    for question in questions:
        question = question.lower()[:-1]
        yield question.split(' ')


def prepare_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    def process_punctuation(s):
        # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
        # this version should be faster since we use re instead of repeated operations on str's
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()

    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))


class SnliImages(data.Dataset):
    """ Dataset for SNLI images located in a folder on the filesystem """
    def __init__(self, snli_path, flickr_path, split, transform=None):
        super(SnliImages, self).__init__()
        self.snli_path = snli_path
        self.flickr_path = flickr_path
        self.id_to_filename = self._find_images(split)
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.snli_path))
        self.transform = transform

    def _find_images(self, split):
        id_to_filename = {}
        for line in file(os.path.join(self.snli_path, split + ".tsv")).read().split("\n")[1:]:
            if config.use_generated_images:
                if not line: continue
                fields = line.split("\t")
                if fields[1].startswith("vg"): continue
                unique_id = "%s-%012d-1" % (split, int(fields[0]))
                img_path = os.path.join(config.generated_images_path, unique_id)
                id_to_filename[unique_id] = img_path
            else:
                if not line: continue
                fields = line.split("\t")
                if fields[1].startswith("vg"): continue
                unique_id = "%s-%012d" % (split, int(fields[0]))
                img_path = os.path.join(self.flickr_path, fields[1])
                img_path = re.sub("#[^#]*$", "", img_path)
                id_to_filename[unique_id] = img_path
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = self.id_to_filename[id]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


class Composite(data.Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError('Index too large for composite dataset')

    def __len__(self):
        return sum(map(len, self.datasets))
