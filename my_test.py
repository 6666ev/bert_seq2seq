import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import T5PegasusTokenizer, load_chinese_base_vocab
from bert_seq2seq import T5Model
from bert_seq2seq.bart_chinese import BartGenerationModel
from bert_seq2seq import Tokenizer
from tqdm import tqdm
from bert_seq2seq.extend_model_method import ExtendModel

from transformers import BertTokenizer, Text2TextGenerationPipeline

from modeling_bart import BartForConditionalGeneration

src_dir = 'data/laic2021/train.src'
zm_tgt_dir = 'data/laic2021/train_zm.tgt'
xq_tgt_dir = 'data/laic2021/train_xq.tgt'

vocab_path = "./state_dict/bart-base-chinese"  # 字典
model_path = "./state_dict/bart-base-chinese"  # 预训练参数

model_save_path = "./state_dict/bart_autotile.bin"  # 训练完模型 保存在哪里
batch_size = 8
lr = 1e-5

tokenizer = BertTokenizer.from_pretrained(vocab_path)
word2idx = tokenizer.vocab
model = BartForConditionalGeneration.from_pretrained(model_path)


def init_weight(model):
    print("=== init weight ===")
    TMP_PATH = "state_dict/bart_tmp.bin"

    # modules = model.state_dict()
    # for name in modules.keys():
    #     print(name)
    #     if "decoder2" in name or "lm_head2" in name:
    #         modules[name] = torch.zeros(modules[name].shape)
    # torch.save(modules, PATH)
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint)

    modules = model.state_dict()
    for name in modules.keys():
        if "decoder2" in name or "lm_head2" in name:
            origin_name = name.replace(
                "decoder2", "decoder").replace("lm_head2", "lm_head")
            modules[origin_name] = modules[name]

    torch.save(modules, TMP_PATH)
    checkpoint = torch.load(TMP_PATH)
    model.load_state_dict(checkpoint)
    return model


init_weight(model)
