# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 下午 11:00
# @Author  : AlexJean
import torch
import torch.nn as nn
from torchtext import data, datasets
import spacy
from ModelRun import make_model, LabelSmoothing, NoamOpt, run_epoch
from LoadData import MyIterator, rebatch
from MultiGPU import MultiGPULossCompute

global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


if __name__ == "__main__":
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)
    MAX_LEN = 100
    train, val, test = \
        datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                              filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    devices = [torch.device('cuda:0')]
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 1000
    print("4")
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    print("5")
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    print("6")
    model_par = nn.DataParallel(model, device_ids=devices)
    print("7")

    if True:
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(10):
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                      MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                             MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))

            print(loss)
    else:
        model = torch.load("iwslt.pt")