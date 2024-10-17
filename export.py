from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/runs/detect/yolov8l_all_100epoch/weights/best.pt')  # load a custom trained model

# Export the model
model.export(format='engine', half=False)





# try yolov9 export
# import tensorrt as trt
# import torch
# import json
# # from 

# half = True
# model = ''
# model.eval()
# model.float()
# model = model.fuse()
# for m in model.modules():
#     if isinstance(m, (Detect, RTDETRDecoder)):  # Segment and Pose use Detect base class
#         m.export = True
# im = torch.zeros(1,3,640,640)
# path = 'weights/yolov9c.onnx'
# torch.onnx.export(model, im, path, verbose=False, opset_version=10, do_constant_folding=True, input_names=["images"], output_names=['output0'], dynamic_axes=False)


# print(f"starting export with TensorRT {trt.__version__}...")
# f = 'yolov9c.engine'  # TensorRT engine file
# logger = trt.Logger(trt.Logger.INFO)

# builder = trt.Builder(logger)
# config = builder.create_builder_config()
# config.max_workspace_size = 4 * 1 << 30
# # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

# flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# network = builder.create_network(flag)
# parser = trt.OnnxParser(network, logger)
# if not parser.parse_from_file('/data/vjuicefs_sz_ocr_001/72179367/codes/TensorRT-YOLO/yolov9c.onnx'):
#     raise RuntimeError(f"failed to load ONNX file: {'/data/vjuicefs_sz_ocr_001/72179367/codes/TensorRT-YOLO/yolov9c.onnx'}")

# inputs = [network.get_input(i) for i in range(network.num_inputs)]
# outputs = [network.get_output(i) for i in range(network.num_outputs)]
# for inp in inputs:
#     print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
# for out in outputs:
#     print(f'output "{out.name}" with shape{out.shape} {out.dtype}')


# print(f"building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}")
# if builder.platform_has_fast_fp16 and half:
#     config.set_flag(trt.BuilderFlag.FP16)

# # del model
# torch.cuda.empty_cache()

# # Write file
# with builder.build_engine(network, config) as engine, open(f, "wb") as t:
#     # Metadata
#     with open('metadata.json', 'r') as file:
#         metadata = json.load(file)
#     meta = json.dumps(metadata)
#     t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
#     t.write(meta.encode())
#     # Model
#     t.write(engine.serialize())