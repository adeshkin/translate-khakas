import os

import torch
import torch.nn as nn
from typing import List
from nltk.tokenize import WordPunctTokenizer

from inference.custom_bpe import init_bpe
from inference.utils import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX
from model import Seq2SeqTransformer
from utils import generate_square_subsequent_mask


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder, positional_encoding, src_tok_emb):
        super(CustomTransformerEncoder, self).__init__()
        self.encoder = encoder
        self.positional_encoding = positional_encoding
        self.src_tok_emb = src_tok_emb

    def forward(self, src, src_mask):
        return self.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder, positional_encoding, tgt_tok_emb):
        super(CustomTransformerDecoder, self).__init__()
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.tgt_tok_emb = tgt_tok_emb

    def forward(self, tgt, memory, tgt_mask):
        return self.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


def tokenize_text(text):
    return text.split()


def custom_greedy_decode(model, onnx_dir, src_sentence, device, text_transform, SRC_LANGUAGE):
    os.makedirs(onnx_dir, exist_ok=True)

    model_encoder = CustomTransformerEncoder(model.transformer.encoder, model.positional_encoding, model.src_tok_emb)
    model_decoder = CustomTransformerDecoder(model.transformer.decoder, model.positional_encoding, model.tgt_tok_emb)
    model_generator = model.generator

    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    src = src.to(device)
    src_mask = src_mask.to(device)

    max_len = num_tokens + 5
    start_symbol = BOS_IDX

    memory = model_encoder(src, src_mask)
    torch.onnx.export(model_encoder,
                      (src, src_mask),
                      f'{onnx_dir}/model_encoder.onnx',
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['src', 'src_mask'],
                      output_names=['memory'],
                      dynamic_axes={'src': {0: 'src0'},
                                    'src_mask': {0: 'src_mask0', 1: 'src_mask1'},
                                    'memory': {0: 'memory0'}})

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    memory = memory.to(device)

    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)

        out = model_decoder(ys, memory, tgt_mask)
        torch.onnx.export(model_decoder,
                          (ys, memory, tgt_mask),
                          f'{onnx_dir}/model_decoder.onnx',
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['ys', 'memory', 'tgt_mask'],
                          output_names=['out'],
                          dynamic_axes={'ys': {0: 'ys0'},
                                        'memory': {0: 'memory0'},
                                        'tgt_mask': {0: 'tgt_mask0', 1: 'tgt_mask1'},
                                        'out': {0: 'out0'}})

        out = out.transpose(0, 1)
        prob = model_generator(out[:, -1])

        torch.onnx.export(model_generator,
                          out[:, -1],
                          f'{onnx_dir}/model_generator.onnx',
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['out[:, -1]'],
                          output_names=['prob'])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break

    return ys


def custom_translate(model, onnx_dir, src_sentence, device, vocab_transform, text_transform, SRC_LANGUAGE, TGT_LANGUAGE):
    tgt_tokens = custom_greedy_decode(model, onnx_dir, src_sentence, device, text_transform, SRC_LANGUAGE).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>",
                                                                                                         "").replace(
        "<eos>", "")


def main():
    bpe_dir = 'inference/data'
    voc_dir = 'inference/data'
    onnx_dir = 'inference/onnx_models'

    model_path = 'inference/models/ru_kjh_best.pt'
    vocabulary_threshold = 50
    DEVICE = torch.device('cuda:0')
    SRC_LANGUAGE = 'ru'
    TGT_LANGUAGE = 'kjh'
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512

    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    MAXLEN = 350

    bpe = init_bpe(bpe_dir, 'ru', vocabulary_threshold)

    language_pair_comb = (SRC_LANGUAGE, TGT_LANGUAGE)

    vocab_transform = dict()
    for lang in language_pair_comb:
        vocab_transform[lang] = torch.load(f'{voc_dir}/vocab_{lang}.pth')

    token_transform = dict()
    for ln in language_pair_comb:
        token_transform[ln] = tokenize_text

    text_transform = {}
    for ln in language_pair_comb:
        text_transform[ln] = sequential_transforms(token_transform[ln],
                                                   vocab_transform[ln],
                                                   tensor_transform)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                               NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, MAXLEN)
    model = model.to(DEVICE)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_sent = 'Новый тренер хотя бы освоится, с игроками нормально пообщается, на стандартах по-своему расставит и закрепит тренировками.'
    input_sent = ' '.join(WordPunctTokenizer().tokenize(input_sent.strip()))
    src_sent = bpe.process_line(input_sent)

    prd_sent_ = custom_translate(model, onnx_dir, src_sent, DEVICE, vocab_transform, text_transform, SRC_LANGUAGE, TGT_LANGUAGE)
    prd_sent = prd_sent_ + ' '
    print(input_sent)
    print(prd_sent.replace('@@ ', '').strip())


if __name__ == '__main__':
    main()
