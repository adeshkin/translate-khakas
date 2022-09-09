import onnxruntime


def main():
    onnx_dir = ''
    encoder_sess = onnxruntime.InferenceSession(f'{onnx_dir}/model_encoder.onnx')
    decoder_sess = onnxruntime.InferenceSession(f'{onnx_dir}/model_decoder.onnx')
    generator_sess = onnxruntime.InferenceSession(f'{onnx_dir}/model_generator.onnx')

    memory1 = encoder_sess.run(None, {'src': src.cpu().numpy(), 'src_mask': src_mask.cpu().numpy()})
    memory = torch.from_numpy(memory1[0])


