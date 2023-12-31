import transformers



MAX_LEN = 128
DEVICE = "cuda"
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "../input/bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_PATH = "../input/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH,do_lower_case = True)
TRAINING_FILE = "../input/imdb.csv"