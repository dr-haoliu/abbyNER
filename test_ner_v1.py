# ! pip install transformers
# ! pip install datasets
# ! pip install sentencepiece  # need to restart the kernel after installation
# ! pip install pytorch_lightning
# !pip install seqeval[gpu]
# ! pip freeze
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import numpy as np
import pandas as pd
import spacy
import pickle
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertForTokenClassification, AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification
import torch
import logging
import uuid
# logging.basicConfig(level=logging.INFO)

pd.set_option('max_colwidth', 400)
spacy_nlp = spacy.load("en_core_sci_lg")  # using scispacy

dir = os.path.dirname(__file__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(200)


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

"""Testing
"""

# way to load pickle
with open(os.path.join(dir, 'output/label_dict.pickle'), 'rb') as handle:
    label_dict = pickle.load(handle)

with open(os.path.join(dir, 'output/labels_to_ids.pickle'), 'rb') as handle:
    labels_to_ids = pickle.load(handle)

with open(os.path.join(dir, 'output/ids_to_labels.pickle'), 'rb') as handle:
    ids_to_labels = pickle.load(handle)


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_labels=len(label_dict))
    # model.load_state_dict(
    #     torch.load(os.path.join(dir, 'trained_model/3_acc_0.9159417462513971.model'), map_location=torch.device('cpu')))
    if str(device).strip() == 'cpu':
        model.load_state_dict(torch.load(os.path.join(dir, 'trained_model/3_acc_0.9159417462513971.model'),
                                         map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(os.path.join(dir, 'trained_model/3_acc_0.9159417462513971.model')))

    # model.load_state_dict(torch.load('trained_model/6_acc_0.9643648329850929.model'))
    # model = AutoModelForTokenClassification.from_pretrained('trained_model/1_acc_0.789362251589862/')
    # model = AutoModelForTokenClassification.from_pretrained('trained_model/3_acc_0.9159417462513971/')
    # model = AutoModelForTokenClassification.from_pretrained('trained_model/6_acc_0.9643648329850929/')
    model.to(device)
    model.eval()

    return model

model = load_model()

class Test_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        abs_id = self.data.abs_id[index]
        sen_offset = self.data.sen_offset[index]
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
        item['abs_id'] = abs_id
        item['sen_offset'] = sen_offset

        return item

    def __len__(self):
        return self.len


def testing_function(text, spacy_nlp):
    my_uid = uuid.uuid1()
    abs_id = str(my_uid.int)[-9:-1]
    abs_test_formated_list = []
    sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)

    for sentence in sentences:
        sen_offset = 0
        tokens = []
        labels = []
        tokens_offsets = []
        for index, token in enumerate(sentence):
            if index == 0:
                sen_offset = token['start']
            tokens.append(token['text'])
            labels.append('O')
            tokens_offsets.append((token['start'], token['end']))
            # print(token['text'], token['start'], token['end'])
        # print('\n')
        # print(sen_offset)
        logging.info(tokens)
        abs_test_formated_list.append([abs_id, sen_offset, tokens, tokens_offsets, labels])

    # logging.info(abs_test_formated_list[:1])

    abs_test_formated_df = pd.DataFrame(abs_test_formated_list,
                                        columns=['abs_id', 'sen_offset', 'tokens', 'tokens_offsets', 'labels'])

    # logging.info(abs_test_formated_df.shape)

    # let's create a new column called "sentence" which groups the words by sentence
    abs_test_formated_df['sentence'] = abs_test_formated_df['tokens'].transform(lambda x: ' '.join(x))
    # let's also create a new column called "word_labels" which groups the tags by sentence
    abs_test_formated_df['word_labels'] = abs_test_formated_df['labels'].transform(lambda x: ','.join(x))
    logging.info(abs_test_formated_df.iloc[0].tokens)
    MAX_LEN = 256
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    VALID_BATCH_SIZE = 1
    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 1
                   }
    testing_set = Test_Dataset(abs_test_formated_df, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set, **test_params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Available device:\t{}'.format(device))
    # print(type(device))
    # model = load_model()

    predicted_labels_list = []
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)
            abs_id = batch['abs_id'][0]
            sen_offset = batch['sen_offset'].item()

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)  #
            eval_logits = outputs[1]

            # compute evaluation accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            labels = [ids_to_labels[id.item()] for id in labels]
            predictions = [ids_to_labels[id.item()] for id in predictions]
            # print(labels)
            # print(predictions)
            # print(abs_id)
            # print(sen_offset)
            predicted_labels_list.append([abs_id, sen_offset, predictions])

    predicted_labels_list_df = pd.DataFrame(predicted_labels_list, columns=['abs_id', 'sen_offset', 'predicted_labels'])
    # logging.info(predicted_labels_list_df.head())

    logging.info(len(predicted_labels_list_df))
    assert len(predicted_labels_list_df) == len(abs_test_formated_df)


    new_df = pd.merge(abs_test_formated_df, predicted_labels_list_df, how='left', on=['abs_id', 'sen_offset'])
    # logging.info(new_df.head())
    # print(new_df.iloc[0].tokens)
    # print(new_df.iloc[0].tokens_offsets)
    # print(new_df.iloc[0].sen_offset)
    # print(new_df.iloc[0].labels)
    # print(new_df.iloc[0].predicted_labels)

    return new_df


def testing_function_with_model(text, spacy_nlp, model):
    my_uid = uuid.uuid1()
    abs_id = str(my_uid.int)[-9:-1]
    abs_test_formated_list = []
    sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)

    for sentence in sentences:
        sen_offset = 0
        tokens = []
        labels = []
        tokens_offsets = []
        for index, token in enumerate(sentence):
            if index == 0:
                sen_offset = token['start']
            tokens.append(token['text'])
            labels.append('O')
            tokens_offsets.append((token['start'], token['end']))
            # print(token['text'], token['start'], token['end'])
        # print('\n')
        # print(sen_offset)
        logging.info(tokens)
        abs_test_formated_list.append([abs_id, sen_offset, tokens, tokens_offsets, labels])

    # logging.info(abs_test_formated_list[:1])

    abs_test_formated_df = pd.DataFrame(abs_test_formated_list,
                                        columns=['abs_id', 'sen_offset', 'tokens', 'tokens_offsets', 'labels'])

    # logging.info(abs_test_formated_df.shape)

    # let's create a new column called "sentence" which groups the words by sentence
    abs_test_formated_df['sentence'] = abs_test_formated_df['tokens'].transform(lambda x: ' '.join(x))
    # let's also create a new column called "word_labels" which groups the tags by sentence
    abs_test_formated_df['word_labels'] = abs_test_formated_df['labels'].transform(lambda x: ','.join(x))
    logging.info(abs_test_formated_df.iloc[0].tokens)
    MAX_LEN = 256
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    VALID_BATCH_SIZE = 1
    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 1
                   }
    testing_set = Test_Dataset(abs_test_formated_df, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set, **test_params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Available device:\t{}'.format(device))
    # print(type(device))
    # model = load_model()

    predicted_labels_list = []
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)
            abs_id = batch['abs_id'][0]
            sen_offset = batch['sen_offset'].item()

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)  #
            eval_logits = outputs[1]

            # compute evaluation accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            labels = [ids_to_labels[id.item()] for id in labels]
            predictions = [ids_to_labels[id.item()] for id in predictions]
            # print(labels)
            # print(predictions)
            # print(abs_id)
            # print(sen_offset)
            predicted_labels_list.append([abs_id, sen_offset, predictions])

    predicted_labels_list_df = pd.DataFrame(predicted_labels_list, columns=['abs_id', 'sen_offset', 'predicted_labels'])
    # logging.info(predicted_labels_list_df.head())

    logging.info(len(predicted_labels_list_df))
    assert len(predicted_labels_list_df) == len(abs_test_formated_df)


    new_df = pd.merge(abs_test_formated_df, predicted_labels_list_df, how='left', on=['abs_id', 'sen_offset'])
    # logging.info(new_df.head())
    # print(new_df.iloc[0].tokens)
    # print(new_df.iloc[0].tokens_offsets)
    # print(new_df.iloc[0].sen_offset)
    # print(new_df.iloc[0].labels)
    # print(new_df.iloc[0].predicted_labels)

    return new_df


def get_result_entity(text, df):
    result_words = []
    for abs_id, tokens_offsets_list, pre_labels_list in zip(df.abs_id, df.tokens_offsets, df.predicted_labels):
        logging.info(abs_id)
        logging.info(pre_labels_list)
        start, end = 0, 1  # 实体开始结束位置标识
        tag_label = "O"  # 实体类型标识
        for i, tag in enumerate(pre_labels_list):
            offsets = tokens_offsets_list[i]
            # print(offsets[0], offsets[1])
            # if label != 'O':
            #   print(f'index {index}\t {label}\t{tokens_offsets_list[index]}')
            if tag.startswith("B"):
                if tag_label != "O":  # 当前实体tag之前有其他实体
                    result_words.append([abs_id, start, end, tag_label])  # 获取实体
                tag_label = tag.split("-")[1]  # 获取当前实体类型
                start, end = offsets[0], offsets[1]  # 开始和结束位置变更
            elif tag.startswith("I"):
                temp_label = tag.split("-")[1]
                if temp_label == tag_label:  # 当前实体tag是之前实体的一部分
                    end = offsets[1]  # 结束位置end扩展
            elif tag == "O":
                if tag_label != "O":  # 当前位置非实体 但是之前有实体
                    result_words.append([abs_id, start, end, tag_label])  # 获取实体
                    tag_label = "O"  # 实体类型置"O"
                start, end = offsets[0], offsets[1]  # 开始和结束位置变更
        if tag_label != "O":  # 最后结尾还有实体
            result_words.append([abs_id, start, end, tag_label])  # 获取结尾的实体

    entity_list = []
    for item in result_words:
        start = int(item[1])
        end = int(item[2])
        tag_label = item[3]
        # print(start)
        # print(end)
        entity_list.append([text[start:end], tag_label])
    return entity_list

"""Evaluation"""


def valid(model, testing_loader):
    # put model in evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs[0]
            eval_logits = outputs[1]
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions


# def evaluation_auto(testing_loader):
#     labels, predictions = valid(model, testing_loader)
#
#     print(labels[:10])
#     print(predictions[:10])
#
#     from seqeval.metrics import classification_report
#
#     print(classification_report([labels], [predictions]))


"""```
2 epochs; PubmedBERT ; 512 ; lr = 1e-4; AdamW

                            precision    recall  f1-score   support

                  CellLine       0.71      0.71      0.71        17
            ChemicalEntity       0.92      0.93      0.93       552
DiseaseOrPhenotypicFeature       0.86      0.91      0.88       727
         GeneOrGeneProduct       0.91      0.94      0.92       855
             OrganismTaxon       0.93      0.96      0.94       295
           SequenceVariant       0.85      0.92      0.88       243

                 micro avg       0.89      0.93      0.91      2689
                 macro avg       0.86      0.89      0.88      2689
              weighted avg       0.89      0.93      0.91      2689
```
"""


def get_PICO(text):
    df = testing_function(text, spacy_nlp)
    results = get_result_entity(text, df)
    return results



class PICO_Class:
    def __init__(self):
        self.model = load_model()
        self.spacy_nlp = spacy.load("en_core_sci_lg")
        with open(os.path.join(dir, 'output/label_dict.pickle'), 'rb') as handle:
            self.label_dict = pickle.load(handle)

        with open(os.path.join(dir, 'output/labels_to_ids.pickle'), 'rb') as handle:
            self.labels_to_ids = pickle.load(handle)

        with open(os.path.join(dir, 'output/ids_to_labels.pickle'), 'rb') as handle:
            self.ids_to_labels = pickle.load(handle)

    def get_pico(self, text):
        df = testing_function_with_model(text, self.spacy_nlp, self.model)
        results = get_result_entity(text, df)
        return results


def main():
    pico_fetcher = PICO_Class()

    # text = 'Rituximab is superior/non-inferior to cyclophosphamide for inducing clinical remission in patients diagnosed with Antineutrophil cytoplasmic antibody (ANCA) vasculitis.'
    # text = 'Proton pump inhibitors (PPI) is superior/non-inferior to histamine-2 receptor blockers for all-cause mortality in ICU patients.'
    # text = 'Among , ICU patients requiring mechanical ventilation(MV) , a strategy of stress ulcer prophylaxis (SUP) with use of proton pump inhibitors (PPI) vs histamine-2 receptor blockers resulted in hospital mortality rates of 18.3% vs 17.5%, respectively, a difference that did not reach the significance threshold.'
    # text = 'Singapore , taiwan and hong kong have brought outbreaks under control.'
    # text = 'The relapse-free and overall survival rates of patients who received adjuvant chemoradiotherapy were significantly higher than those of patients who received adjuvant chemotherapy only ( 68 % vs 56 % at 2 years and 49 % vs 26 % at 5 years , P = 0.013 , and 87 % vs 61 % at 2 years and 59 % vs 33 % at 5 years , P = 0.029 ) .'
    # text = 'Mortality 0-28 days was 42.8 vs 48 % in the prophylaxis vs control group ( p = ns ), due prevalently to intracerebral haemorrhage in both groups.'
    # text = 'The 2.0-mg / kg group , which demonstrated marked reductions in substrate concentrations in the CSF , serum , and urine , was considered to provide the best combination regarding safety and efficacy signals .'
    text = 'Bupivacaine ( 0.5 % ) in combination with Midazolam ( 50 microg x kg-1 ) quickened the onset as well as prolonged the duration of sensory and motor blockade of the brachial plexus for upper limb surgery .'


    results = pico_fetcher.get_pico(text)
    for item in results:
        print('Found entity: {}\t=>\t{}'.format(item[0], item[1]))


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()

    # ['Proton', 'pump', 'inhibitors', '(', 'PPI', ')', 'is',
    # 'superior', 'or', 'non-inferior', 'to', 'histamine-2', '(', 'H2', ')', 'receptor', 'blockers', 'for', 'all-cause', 'mortality', 'in', 'ICU', 'patients', '.']
    # ['B-Intervention', 'I-Intervention', 'I-Intervention', 'O', 'I-Intervention', 'O', 'O',
    # 'B-Observation', 'O', 'B-Observation', 'O', 'B-Intervention', 'O', 'I-Intervention', 'O', 'I-Intervention', 'I-Intervention', 'O', 'B-Outcome', 'I-Outcome', 'O', 'B-Participant', 'O', 'O']