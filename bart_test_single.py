# model url : https://huggingface.co/fnlp/bart-base-chinese
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

from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline


data_name="judgment_gen"
gen_type="result"
cur_device="cuda:2"
batch_size = 32
input_max_seq_len = 300
output_max_seq_len = 50

input_data_name = "plea_fact"
# input_data_name = "plea"
# input_data_name = "fact"

result_path = "res/{}/judgment_gen.txt".format(input_data_name)
load_path = "logs/judgments_gen/{}/bart_9.bin".format(input_data_name)

src_dir = 'data/{}/train/{}.src'.format(data_name, input_data_name)
tgt_dir = 'data/{}/train/judgment.tgt'.format(data_name)

test_src_dir = 'data/{}/test/{}.src'.format(data_name, input_data_name)
test_tgt_dir = 'data/{}/test/judgment.tgt'.format(data_name)

vocab_path = "./ptm/bart-base-chinese"  # 字典
model_path = "./ptm/bart-base-chinese"  # 预训练参数



tokenizer = BertTokenizer.from_pretrained(vocab_path)
word2idx = tokenizer.vocab
model = BartForConditionalGeneration.from_pretrained(model_path)


def read_file(src_dir, tgt_dir, data_num=10000000000):
    src, tgt = [], []

    with open(src_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            src.append(line)

    with open(tgt_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            tgt.append(line)

    return src[:data_num], tgt[:data_num]


class SeqDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt):
        # 一般init函数是加载所有数据
        super(SeqDataset, self).__init__()
        # 读原始数据
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        self.idx2word = {k: v for v, k in word2idx.items()}

    def __getitem__(self, i):
        # 得到单个数据
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        output = {
            "fact": src,
            "rat_12": tgt,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


class Trainer:
    def __init__(self):
        # 加载数据
        self.sents_src, self.sents_tgt = read_file(src_dir, tgt_dir)

        self.test_src, self.test_tgt = read_file(
            test_src_dir, test_tgt_dir, data_num=1000000000000)

        # 判断是否有可用GPU
        self.device = torch.device(
            cur_device if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.model = ExtendModel(
            model, tokenizer=tokenizer, bos_id=word2idx["[CLS]"], eos_id=word2idx["[SEP]"], device=self.device)

        # 将模型发送到计算设备(GPU或CPU)
        self.model.to(self.device)


    def test(self, load_path, result_path):
        self.model.eval()
        device_str="{}:{}".format(self.device.type, self.device.index)
        self.model = torch.load(load_path, map_location={'cuda:1':device_str})
        self.model.to(self.device)
        self.model.device=self.device

        tgt_gen = []
        test_dataset = SeqDataset(self.test_src, self.test_tgt)
        test_dataloader = DataLoader(
            test_dataset, drop_last=False, batch_size=batch_size)
        for data in tqdm(test_dataloader):
            fact, rat = data["fact"], data["rat_12"]
            gen_text = self.model.my_generate_text_beam(
                fact,
                gen_type="single",
                input_max_length=input_max_seq_len,
                output_max_length=output_max_seq_len,
                add_eos=True)

            tgt_gen += gen_text

        with open(result_path, "w") as f:
            for line in tgt_gen:
                f.write(line+"\n")


if __name__ == '__main__':

    trainer = Trainer()
    trainer.test(load_path, result_path)
