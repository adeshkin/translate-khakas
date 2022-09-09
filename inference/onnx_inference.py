import onnxruntime
import numpy as np


def generate_square_subsequent_mask(sz):
    mask = (np.triu(np.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def custom_greedy_decode(src_sentence, device):
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    src = src.to(device)
    src_mask = src_mask.to(device)

    max_len = num_tokens + 5
    start_symbol = BOS_IDX


def main():
    onnx_dir = ''
    encoder_sess = onnxruntime.InferenceSession(f'{onnx_dir}/model_encoder.onnx')
    decoder_sess = onnxruntime.InferenceSession(f'{onnx_dir}/model_decoder.onnx')
    generator_sess = onnxruntime.InferenceSession(f'{onnx_dir}/model_generator.onnx')

    memory1 = encoder_sess.run(None, {'src': src.cpu().numpy(), 'src_mask': src_mask.cpu().numpy()})
    memory = memory1[0]

    ys = np.ones(1, 1).fill_(start_symbol).type(np.long)

    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(np.bool))

        out1 = decoder_sess.run(None, {'ys': ys,
                                       'memory': memory,
                                       'tgt_mask': tgt_mask})
        out = out1[0]

        out = out.transpose(0, 1)

        prob1 = generator_sess.run(None, {'out[:, -1]': out[:, -1]})
        prob = prob1[0]

        next_word = np.argmax(prob, dim=1)
        next_word = next_word.item()

        ys = np.cat([ys,
                        np.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break




