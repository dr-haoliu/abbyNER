# -*- coding: utf-8 -*-
"""LitCoin_data_review_HL_v3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n4zd9a3Vz4TJUjoeCctJP3S3gPYoyj7r
"""
# ! pip install transformers
# ! pip install datasets
# ! pip install sentencepiece  # need to restart the kernel after installation
# ! pip install pytorch_lightning
# ! pip install seqeval[gpu]
# ! pip freeze

# !pip install scispacy
# !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import numpy as np
from tqdm.notebook import tqdm
# from tqdm import tqdm
import pandas as pd
import spacy
import codecs
import collections
import pickle
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertForTokenClassification, AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification
import torch
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple
from torch.distributions.bernoulli import Bernoulli
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(200)
training_option = 2

def replace_unicode_whitespaces_with_ascii_whitespace(string):
    return ' '.join(string.split())


spacy_nlp = spacy.load("en_core_sci_lg")  # using scispacy
verbose = True

# output_file = codecs.open('haotest_v2.tsv', 'w', 'utf-8')

def get_start_and_end_offset_of_token_from_spacy(token):
    start = token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp):
    document = spacy_nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            token_dict = {}
            token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                print("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'], 
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')
            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences

def pre_processing_from_df(abs_df, entity_df, output_file):
    for abs_id, text in zip(abs_df.abstract_id, abs_df.text):
      print(abs_id, ' ', text)
      rows_df = entity_df.loc[entity_df['abstract_id']==abs_id]

      entities = []
      for index, arow in rows_df.iterrows():
        # print(arow)
        # parse entity
        entity = {}
        entity['id'] = arow['abstract_id']
        entity['type'] = arow['type']
        entity['start'] = int(arow['offset_start'])
        entity['end'] = int(arow['offset_finish'])
        entity['text'] = arow['mention']
        print("entity: {0}".format(entity))
        # Check compatibility between brat text and anotation
        if replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']]) != \
            replace_unicode_whitespaces_with_ascii_whitespace(entity['text']):
            print("Warning: brat text and annotation do not match.")
            print("\ttext: {0}".format(text[entity['start']:entity['end']]))
            print("\tanno: {0}".format(entity['text']))
        # add to entitys data
        entities.append(entity)

      entities = sorted(entities, key=lambda entity:entity["start"])
      sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)

      for sentence in sentences:
          inside = False
          previous_token_label = 'O'
          for token in sentence:
              token['label'] = 'O'
              for entity in entities:
                  if entity['start'] <= token['start'] < entity['end'] or \
                      entity['start'] < token['end'] <= entity['end'] or \
                      token['start'] < entity['start'] < entity['end'] < token['end']:
                      token['label'] = entity['type'].replace('-', '_') # Because the ANN doesn't support tag with '-' in it
                      break
                  elif token['end'] < entity['start']:
                      break

              if len(entities) == 0:
                  entity={'end':0}
              if token['label'] == 'O':
                  gold_label = 'O'
                  inside = False
              elif inside and token['label'] == previous_token_label:
                  gold_label = 'I-{0}'.format(token['label'])
              else:
                  inside = True
                  gold_label = 'B-{0}'.format(token['label'])
              if token['end'] == entity['end']:
                  inside = False
              previous_token_label = token['label']
              if verbose: print('{0}\t{1}\t{2}\t{3}\n'.format(token['text'], token['start'], token['end'], gold_label))
              output_file.write('{0}\t{1}\t{2}\t{3}\n'.format(token['text'], token['start'], token['end'], gold_label))
          if verbose: print('\n')
          output_file.write('\n')

    output_file.close()


def parse_dataset(dataset_filepath):
  token_count = collections.defaultdict(lambda: 0)
  label_count = collections.defaultdict(lambda: 0)
  character_count = collections.defaultdict(lambda: 0)

  line_count = -1
  tokens = []
  labels = []
  new_token_sequence = []
  new_label_sequence = []
  if dataset_filepath:
      f = codecs.open(dataset_filepath, 'r', 'UTF-8')
      for line in f:
          line_count += 1
          line = line.strip().split('\t')
          if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
              if len(new_token_sequence) > 0:
                  labels.append(new_label_sequence)
                  tokens.append(new_token_sequence)
                  new_token_sequence = []
                  new_label_sequence = []
              continue
          token = str(line[0])
          label = str(line[-1])
          token_count[token] += 1
          label_count[label] += 1

          new_token_sequence.append(token)
          new_label_sequence.append(label)

          for character in token:
              character_count[character] += 1

          # if line_count > 200: break# for debugging purposes

      if len(new_token_sequence) > 0:
          labels.append(new_label_sequence)
          tokens.append(new_token_sequence)
      f.close()
  return labels, tokens, token_count, label_count, character_count

filepath = 'output/pico_conll.tsv'

labels, tokens, token_count, label_count, character_count = parse_dataset(filepath)

print(labels[:10])
print(tokens[:10])

print(len(labels))


labels_set = set()
for x in labels:
  for y in x:
    labels_set.add(y)
print(labels_set)
labels_to_ids = {k: v for v, k in enumerate(labels_set)}
ids_to_labels = {v: k for v, k in enumerate(labels_set)}
label_dict = labels_to_ids
print(label_dict)

#################################################################################
# only dump once:
# delete old ones if training new batch with different labels
# dump label_dict, need use the same trained dict for prediction
if os.path.isfile('output/label_dict.pickle'):
    with open('output/label_dict.pickle', 'rb') as handle:
        label_dict = pickle.load(handle)
else:
    with open('output/label_dict.pickle', 'wb') as handle:
        pickle.dump(label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if os.path.isfile('output/labels_to_ids.pickle'):
    with open('output/labels_to_ids.pickle', 'rb') as handle:
        labels_to_ids = pickle.load(handle)
else:
    with open('output/labels_to_ids.pickle', 'wb') as handle:
        pickle.dump(labels_to_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

if os.path.isfile('output/ids_to_labels.pickle'):
    with open('output/ids_to_labels.pickle', 'rb') as handle:
        ids_to_labels = pickle.load(handle)
else:
    with open('output/ids_to_labels.pickle', 'wb') as handle:
        pickle.dump(ids_to_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# way to load pickle
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)
###########################################################################################

df = pd.DataFrame(zip(tokens, labels), columns=['tokens', 'labels'])

print(df[:10])
print(df.shape)



# let's create a new column called "sentence" which groups the words by sentence 
df['sentence'] = df['tokens'].transform(lambda x: ' '.join(x))
# let's also create a new column called "word_labels" which groups the tags by sentence 
df['word_labels'] = df['labels'].transform(lambda x: ','.join(x))
df.head()

data = df[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
data.head()


class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.sentence[index].strip().split()  
        word_labels = self.data.word_labels[index].split(",") 

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                            #  is_pretokenized=True, 
                            # add_special_tokens=True,  
                            is_split_into_words=True,
                             return_offsets_mapping=True, 
                             padding= 'max_length',  # 'longest'
                             truncation=True, 
                             max_length=self.max_len)
        
        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels]
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
            encoded_labels[idx] = labels[i]
            i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

  def __len__(self):
        return self.len

class LitCoindataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.sentence[index].strip().split()  
        word_labels = self.data.word_labels[index].split(",") 

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                            #  is_pretokenized=True, 
                            is_split_into_words=True,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels] 
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
            encoded_labels[idx] = labels[i]
            i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

  def __len__(self):
        return self.len


############################################
# option 1: split to 80:20
if training_option == 1:
    train_size = 0.8
    validation_size = 0.2
    train_dataset = data.sample(frac=train_size, random_state=200)

    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    validation_dataset = test_dataset.copy()

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("Validation Dataset: {}".format(validation_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    train_dataset.head()
    validation_dataset.head()
    test_dataset.head()
###################################################################
# option 2:
# use all data for training
if training_option == 2:
    train_dataset = data
    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print(train_dataset.head())
##################################################################
# parameters
MAX_LEN = 256  # 128 #
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10

##################################################################


# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
# testing_set = dataset(test_dataset, tokenizer, MAX_LEN)
# validation_set = dataset(validation_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
# testing_loader = DataLoader(testing_set, **test_params)

# model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
model = BertForTokenClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_labels=len(label_dict))



model.to(device)

# print(training_set[2])
#
# inputs = training_set[2]
# input_ids = inputs["input_ids"].unsqueeze(0)
# attention_mask = inputs["attention_mask"].unsqueeze(0)
# labels = inputs["labels"].unsqueeze(0)
#
# input_ids = input_ids.to(device)
# attention_mask = attention_mask.to(device)
# labels = labels.to(device)
#
# outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
# initial_loss = outputs[0]
# print(initial_loss)
#
# outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
# print(outputs[0].item())
#
# tr_logits = outputs[1]
# print(tr_logits.shape)



class ChildTuningAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        reserve_p = 1.0,
        mode = None
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.gradient_mask = None
        self.reserve_p = reserve_p
        self.mode = mode

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # =================== HACK BEGIN =======================         
                if self.mode is not None:
                    if self.mode == 'ChildTuning-D':
                        if p in self.gradient_mask:
                            grad *= self.gradient_mask[p]
                    else: 
                        # ChildTuning-F
                        grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                        grad *= grad_mask.sample() / self.reserve_p
                # =================== HACK END =======================

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
optimizer = ChildTuningAdamW(params=model.parameters(), lr=LEARNING_RATE)


# Defining the training function on the 80% of the dataset for tuning the bert model
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(training_loader):
        
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs[0]
        tr_logits = outputs[1]
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")
    return epoch_loss, tr_accuracy

print('model number of labels: {}'.format(model.num_labels))

for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    epoch_loss, tr_accuracy = train(epoch)
    model.save_pretrained(f'output/{epoch + 1}_acc_{tr_accuracy}/')
    torch.save(model.state_dict(), f'output/{epoch + 1}_acc_{tr_accuracy}.model')
