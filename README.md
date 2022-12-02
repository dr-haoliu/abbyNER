# NER

* test with Python 3.8

## set up python env and install required modules
```bash
pip install pandas
pip install spacy
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz
pip install transformers==4.13.0
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pytorch-lightning==1.7.7
pip install sentencepiece  # need to restart the kernel after installation
```


[//]: # (pip install seqeval[gpu])
[//]: # (pip install freeze)
[//]: # (pip install tqdm)

## Train and Test
* Run brat2conll.py to convert brat annotations to conll 
  * Input: put .txt and .ann files under 'data' folder
  * Output: set output file name (e.g., pico_conll.tsv)
* Run train_ner_v1.py to train NER model
  * update filepath to generate outpu file (e.g., 'output/pico_conll.tsv')
  * set tokenizer or model with pretrained models (other models available: https://huggingface.co/models )
    * e.g., "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", or "bert-base-uncased"
  * change parameters to control training
  * run train_ner_v1.py 
  * trained models saved as output/{epoch + 1}_acc_{tr_accuracy}
* Run test_ner_v1.py to test NER model
  * update load_model() function with the path to the trained model (normally the one with the highest accuracy score)
    * e.g., load the trained model -- trained_model/3_acc_0.9159417462513971.model
  * Input: text 
  * Output: recognized entities from the text