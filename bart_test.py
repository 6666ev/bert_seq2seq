# model url : https://huggingface.co/fnlp/bart-base-chinese
import torch
import torch.nn as nn
import time
import glob
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import T5PegasusTokenizer, load_chinese_base_vocab
from bert_seq2seq import T5Model
from bert_seq2seq.bart_chinese import BartGenerationModel
from bert_seq2seq import Tokenizer
from tqdm import tqdm
from bert_seq2seq.extend_model_method import ExtendModel

from transformers import BertTokenizer, Text2TextGenerationPipeline

from modeling_bart import BartForConditionalGeneration

train_src_dir = "data/laic2021/train/fact.src"
train_zm_tgt_dir = "data/laic2021/train/zm.tgt"
train_xq_tgt_dir = "data/laic2021/train/xq.tgt"

valid_src_dir = "data/laic2021/valid/fact.src"
valid_zm_tgt_dir = "data/laic2021/valid/zm.tgt"
valid_xq_tgt_dir = "data/laic2021/valid/xq.tgt"

test_src_dir = "data/laic2021/test/fact.src"
test_zm_tgt_dir = "data/laic2021/test/zm.tgt"
test_xq_tgt_dir = "data/laic2021/test/xq.tgt"

# src_dir = 'corpus/csl/train.src'
# zm_tgt_dir = 'corpus/csl/train.tgt'
# xq_tgt_dir = 'corpus/csl/train.tgt'

vocab_path = "./state_dict/bart-base-chinese"  # 字典
model_path = "./state_dict/bart-base-chinese"  # 预训练参数


batch_size = 8
lr = 1e-5

tokenizer = BertTokenizer.from_pretrained(vocab_path)
word2idx = tokenizer.vocab
model = BartForConditionalGeneration.from_pretrained(model_path)


def init_weight(model):
    print("=== init weight ===")
    TMP_PATH = "state_dict/bart_tmp.bin"

    modules = model.state_dict()
    for name in modules.keys():
        if "decoder2" in name or "lm_head2" in name:
            origin_name = name.replace(
                "decoder2", "decoder").replace("lm_head2", "lm_head")
            modules[name] = modules[origin_name]

    torch.save(modules, TMP_PATH)
    checkpoint = torch.load(TMP_PATH)
    model.load_state_dict(checkpoint)
    return model


model = init_weight(model)


def read_file(src_dir, zm_tgt_dir, xq_tgt_dir, data_num=10000000000):
    src, tgt_zm, tgt_xq = [], [], []

    with open(src_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            src.append(line)

    with open(zm_tgt_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            tgt_zm.append(line)

    with open(xq_tgt_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            tgt_xq.append(line)

    return src[:data_num], tgt_zm[:data_num], tgt_xq[:data_num]


class SeqDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt_zm, sents_tgt_xq):
        # 一般init函数是加载所有数据
        super(SeqDataset, self).__init__()
        # 读原始数据
        self.sents_src = sents_src
        self.sents_tgt_zm = sents_tgt_zm
        self.sents_tgt_xq = sents_tgt_xq

        self.idx2word = {k: v for v, k in word2idx.items()}

    def __getitem__(self, i):
        # 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt_zm = self.sents_tgt_zm[i]
        tgt_xq = self.sents_tgt_xq[i]
        token_ids_src = tokenizer.encode(src, max_length=300)
        token_ids_tgt_zm = tokenizer.encode(tgt_zm, max_length=200)
        token_ids_tgt_xq = tokenizer.encode(tgt_xq, max_length=200)

        output = {
            "token_ids_src": token_ids_src,
            "token_ids_tgt_zm": token_ids_tgt_zm,
            "token_ids_tgt_xq": token_ids_tgt_xq,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] *
                      max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids_src = [data["token_ids_src"] for data in batch]
    max_length_src = max([len(t) for t in token_ids_src])
    token_ids_tgt_zm = [data["token_ids_tgt_zm"] for data in batch]
    max_length_tgt_zm = max([len(t) for t in token_ids_tgt_zm])
    token_ids_tgt_xq = [data["token_ids_tgt_xq"] for data in batch]
    max_length_tgt_xq = max([len(t) for t in token_ids_tgt_xq])

    fact_ids_padded = padding(token_ids_src, max_length_src)
    zm_ids_padded = padding(token_ids_tgt_zm, max_length_tgt_zm)
    xq_ids_padded = padding(token_ids_tgt_xq, max_length_tgt_xq)

    zm_labels_ids = zm_ids_padded.clone()
    zm_ids_padded = zm_ids_padded[:, :-1].contiguous()
    zm_labels_ids = zm_labels_ids[:, 1:].contiguous()

    xq_labels_ids = xq_ids_padded.clone()
    xq_ids_padded = xq_ids_padded[:, :-1].contiguous()
    xq_labels_ids = xq_labels_ids[:, 1:].contiguous()

    return fact_ids_padded, zm_ids_padded, zm_labels_ids, xq_ids_padded, xq_labels_ids


class Trainer:
    def __init__(self):
        # 加载数据
        self.train_src, self.train_tgt_zm, self.train_tgt_xq = read_file(
            train_src_dir, train_zm_tgt_dir, train_xq_tgt_dir)
        self.test_src, self.test_tgt_zm, self.test_tgt_xq = read_file(
            test_src_dir, test_zm_tgt_dir, test_xq_tgt_dir, data_num=1000000000000)
        # 判断是否有可用GPU
        self.device = torch.device(
            "cuda:3" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.model = ExtendModel(
            model, tokenizer=tokenizer, bos_id=word2idx["[CLS]"], eos_id=word2idx["[SEP]"], device=self.device)

        # 将模型发送到计算设备(GPU或CPU)
        self.model.to(self.device)
        # self.model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(
            self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = SeqDataset(
            self.train_src, self.train_tgt_zm, self.train_tgt_xq)
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)

    def test(self, model_path):
        self.model.eval()
        self.model = torch.load(model_path)
        zm_gen, xq_gen = [], []
        for i in tqdm(range(len(self.test_src))):
            fact, zm, xq = self.test_src[i], self.test_tgt_zm[i], self.test_tgt_xq[i]
            gen_text = self.model.sample_generate_encoder_decoder(fact, add_eos=True)
            zm_gen.append(gen_text[0])
            xq_gen.append(gen_text[1])

        with open("gen_text/zm_gen.txt", "w") as f:
            for line in zm_gen:
                f.write(line+"\n")

        with open("gen_text/xq_gen.txt", "w") as f:
            for line in xq_gen:
                f.write(line+"\n")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.test("save_model/bart_6.bin")
