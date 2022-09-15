# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess
import sys
import math

import tensorflow as tf

from lib.evaluation_scripts import bleu
from lib.evaluation_scripts import tokens2wordlevel
from collections import Counter

__all__ = ["evaluate"]


def evaluate(ref_file, trans_file, ref_src_file, metric, subword_token='Ġ', subword_option=None):
  """Pick a metric and evaluate depending on task."""
  global token
  token = subword_token
  
  if len(metric.lower()) > 4 and metric.lower()[0:4]=='bleu':
    max_order = int(metric.lower()[5:])
    evaluation_score = _bleu(ref_file, trans_file, max_order=max_order,
                             subword_option=subword_option)
  elif metric.lower()[0:len('distinct')] == 'distinct':
    max_order = int(metric.lower()[len('distinct')+1:])
    evaluation_score = _distinct(trans_file, max_order, subword_option=subword_option)
  elif metric.lower() == 'entropy':
    dict_files = [ref_src_file]
    evaluation_score = _entropy_nrg(dict_files, trans_file, subword_option=subword_option)
  else:
    raise ValueError("Unknown metric %s" % metric)

  return evaluation_score


def _clean(sentence, subword_option):
  
  sentence = tokens2wordlevel.revert_from_sentence(sentence, subword_option)
  sentence = sentence.replace(token,'')

  return sentence


def _distinct(trans_file, max_order=1, subword_option=None):
  """Compute Distinct Score"""

  translations = []
  with codecs.getreader("utf-8")(tf.io.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=subword_option)
      translations.append(line.split(" "))

  num_tokens = 0
  unique_tokens = set()
  scores = []
  for items in translations:
      local_unique_tokens = set()
      local_count = 0.0
      
      for i in range(0, len(items) - max_order + 1):
        tmp = ' '.join(items[i:i+max_order])
        unique_tokens.add(tmp)
        num_tokens += 1
        local_unique_tokens.add(tmp)
        local_count += 1
      if local_count == 0:
        scores.append(0)
      else:
        scores.append(100*len(local_unique_tokens) / local_count)
  if num_tokens == 0:
    ratio = 0
  else:
    ratio = len(unique_tokens) / num_tokens
  return 100 * ratio, scores

# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file,max_order=4, subword_option=None):
  """Compute BLEU scores and handling BPE."""
  
  smooth = False
  ref_files = [ref_file]
  
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(
        tf.io.gfile.GFile(reference_filename, "rb")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference = _clean(reference, subword_option)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  translations = []
  with codecs.getreader("utf-8")(tf.io.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=subword_option)
      translations.append(line.split(" "))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      per_segment_references, translations, max_order, smooth)

  blue_scores = []
  for ref, trans in zip(per_segment_references, translations):
    tmp_bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      [ref], [trans], max_order, smooth)
    blue_scores.append(tmp_bleu_score * 100)
   
  return 100 * bleu_score, blue_scores

def _entropy_nrg(dict_files, trans_file, subword_option=None):
  """Compute Entropy Score"""
  counter = Counter()
  num_tokens = 0.0
  
  for dict_file in dict_files:
    with open(dict_file, encoding='utf-8') as fh:
      for line in fh:
        src, tgt = line.split('\t')

        line = src.strip() + ' ' + tgt.strip()

        line = _clean(line, subword_option=subword_option)
        tokens = line.split(" ")
        num_tokens += len(tokens)
        counter.update(tokens)

  entropy = 0
  num_infer_tokens = 0

  scores1 = []
  with open(trans_file, encoding='utf-8') as fh:
    for line in fh:
      line = _clean(line, subword_option=subword_option)
      tokens = line.split(" ")
      local_scores = []
      for item in tokens:
        fre = max(1, counter[item])
        p = fre / num_tokens
        tmp = -math.log(p, 2)
        local_scores += [tmp]
        entropy += tmp
        num_infer_tokens += 1.0
      scores1.append(sum(local_scores)/len(local_scores))
  score1 = entropy/num_infer_tokens
  return (score1, scores1)
