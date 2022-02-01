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
            modules[name] = modules[origin_name]
    # 'model.decoder.layers.0.self_attn.k_proj.weight'
    # 1: 0.0422,  0.0159, -0.0557
    # 2: -0.0014,  0.0127, -0.0223
    # 'lm_head.weight'
    # 1: 3.1082e-02, 2.4719e-02, -1.7227e-02
    # 2: -0.0322,  0.0351,  0.0018
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
            test_src_dir, test_zm_tgt_dir, test_xq_tgt_dir, data_num=10)
        # 判断是否有可用GPU
        self.device = torch.device(
            "cuda:1" if torch.cuda.is_available() else "cpu")
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

    def save(self, save_path):
        """
        保存模型
        """
        self.model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        report_loss = 0
        start_time = time.time()  # 得到当前时间
        step = 0
        for fact_ids, zm_ids, zm_labels_ids, xq_ids, xq_labels_ids in tqdm(dataloader, total=len(dataloader)):
            step += 1
            fact_ids = fact_ids.to(self.device)
            zm_ids = zm_ids.to(self.device)
            zm_labels_ids = zm_labels_ids.to(self.device)
            xq_ids = xq_ids.to(self.device)
            xq_labels_ids = xq_labels_ids.to(self.device)

            if step % 100 == 0:
                # self.save(model_save_path)
                self.model.eval()
                test_data = [
                    "本文总结了十个可穿戴产品的设计原则，而这些原则同样也是笔者认为是这个行业最吸引人的地方：1为人们解决重复性问题，2从人开始而不是从机器开始，3要引起注意但不要刻意，4提升用户能力而不是取代人",
                    "2007年乔布斯向人们展示iPhone并宣称它将会改变世界，还有人认为他在夸大其词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落，未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献",
                    "雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东截止发稿前雅虎股价上涨了大约7％至5145美元"
                ]

                for i in range(len(self.test_src)):
                    fact, zm, xq = self.test_src[i], self.test_tgt_zm[i], self.test_tgt_xq[i]

                    def show(zm, xq):
                        print("zm rationale:")
                        print(zm)
                        print("xq rationale:")
                        print(xq)

                    print("=== true ===")
                    show(zm, xq)
                    gen_text = self.model.sample_generate_encoder_decoder(
                        fact, add_eos=True, top_k=20)
                    print("=== gen ===")
                    show(gen_text[0], gen_text[1])
                    print("="*30)

                self.model.train()
                print("report loss is " + str(report_loss))
                report_loss = 0

            # 因为传入了target标签，因此会计算loss并且返回
            labels = {"zm": zm_labels_ids, "xq": xq_labels_ids}
            decoder_input_ids = {"zm": zm_ids, "xq": xq_ids}
            outputs = self.model(fact_ids, labels=(zm_labels_ids, xq_labels_ids),
                                 decoder_input_ids=(zm_ids, xq_ids))
            loss = outputs[0]
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()
            report_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch) + ". loss is " +
              str(total_loss) + ". spend time is " + str(spend_time))
        # 保存模型
        # self.save(model_save_path)


if __name__ == '__main__':

    trainer = Trainer()
    train_epoches = 10
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
