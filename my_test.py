## model url : https://huggingface.co/fnlp/bart-base-chinese
import torch
import time
import glob
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import T5PegasusTokenizer, load_chinese_base_vocab
from bert_seq2seq import T5Model
from bert_seq2seq.bart_chinese import BartGenerationModel
from bert_seq2seq import Tokenizer
from tqdm import tqdm
from bert_seq2seq.extend_model_method import ExtendModel
from transformers import BertTokenizer
from modeling_bart import BartForConditionalGeneration

vocab_path = "./state_dict/bart-base-chinese" ## 字典
model_path = "./state_dict/bart-base-chinese" ## 预训练参数
tokenizer = BertTokenizer.from_pretrained(vocab_path)
word2idx = tokenizer.vocab
model = BartForConditionalGeneration.from_pretrained(model_path)
