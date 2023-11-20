# GFPGAN

GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior

## Dependencies

- CUDA-11.8
- cuDNN-8.9.5.29
- TensorRT-8.6.1.6
- PyTorch-2.0.1
- cuda-python-12.3.0
- onnx-1.14.0

## Depoly GFPGANv1.4

- step 1: generate onnx model
```bash
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.4 -s 2 --bg_upsampler xyz
```
Set bg_upsampler as xyz to disable background upsampler. 

- step 2: generate TensorRT engine

```bash
python onnx_to_trt/build_engine.py --onnx GFPGANv1.4.onnx --engine GFPGANv1.4_fp16.engine -p fp16
```