# model url : https://huggingface.co/fnlp/bart-base-chinese
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from bert_seq2seq.extend_model_method import ExtendModel

from transformers import BertTokenizer, Text2TextGenerationPipeline

from modeling_bart import BartForConditionalGeneration

data_name="laic2021_filter"
gen_type = "xq"
cur_device="cuda:3"
batch_size = 32
input_max_seq_len = 512
output_max_seq_len = 256

load_path = "logs/enc1dec2/bart_5.bin"
result_path = "res/dec2/{}_gen.txt".format(gen_type)


train_src_dir = "data/{}/train/fact.src".format(data_name)
train_zm_tgt_dir = "data/{}/train/zm.tgt".format(data_name)
train_xq_tgt_dir = "data/{}/train/xq.tgt".format(data_name)

valid_src_dir = "data/{}/valid/fact.src".format(data_name)
valid_zm_tgt_dir = "data/{}/valid/zm.tgt".format(data_name)
valid_xq_tgt_dir = "data/{}/valid/xq.tgt".format(data_name)

test_src_dir = "data/{}/test/fact.src".format(data_name)
test_zm_tgt_dir = "data/{}/test/zm.tgt".format(data_name)
test_xq_tgt_dir = "data/{}/test/xq.tgt".format(data_name)

vocab_path = "./state_dict/bart-base-chinese"  # 字典
model_path = "./state_dict/bart-base-chinese"  # 预训练参数


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
        src = self.sents_src[i]
        tgt_zm = self.sents_tgt_zm[i]
        tgt_xq = self.sents_tgt_xq[i]

        output = {
            "fact": src,
            "zm": tgt_zm,
            "xq": tgt_xq,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


class Trainer:
    def __init__(self):
        # 加载数据
        self.train_fact, self.train_zm, self.train_xq = read_file(
            train_src_dir, train_zm_tgt_dir, train_xq_tgt_dir)
        self.valid_fact, self.valid_zm, self.valid_xq = read_file(
            valid_src_dir, valid_zm_tgt_dir, valid_xq_tgt_dir)
        self.test_fact, self.test_zm, self.test_xq = read_file(
            test_src_dir, test_zm_tgt_dir, test_xq_tgt_dir)
        # 判断是否有可用GPU
        self.device = torch.device(cur_device if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.model = ExtendModel(
            model, tokenizer=tokenizer, bos_id=word2idx["[CLS]"], eos_id=word2idx["[SEP]"], device=self.device)

        # 将模型发送到计算设备(GPU或CPU)
        self.model.to(self.device)

    def test(self, load_path, result_path, gen_type="zm"):
        self.model.eval()
        device_str="{}:{}".format(self.device.type, self.device.index)
        self.model = torch.load(load_path, map_location={'cuda:1':device_str})
        self.model.to(self.device)
        self.model.device=self.device
        gen_res = []

        
        test_dataset = SeqDataset(self.test_fact, self.test_zm, self.test_xq)
        test_dataloader = DataLoader(
            test_dataset, drop_last=False, batch_size=batch_size)
        for data in tqdm(test_dataloader):
            fact, zm, xq = data["fact"], data["zm"], data["xq"]
            gen_text = self.model.my_generate_text_beam(
                fact,
                gen_type=gen_type,
                input_max_length=input_max_seq_len,
                output_max_length=output_max_seq_len,
                add_eos=True
            )
            gen_res += gen_text

        with open(result_path, "w") as f:
            for line in gen_res:
                f.write(line+"\n")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.test(load_path, result_path, gen_type=gen_type)
    # trainer.test("save_model/bart_6.bin")
