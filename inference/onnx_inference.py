import onnxruntime


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
    memory = torch.from_numpy(memory1[0])


