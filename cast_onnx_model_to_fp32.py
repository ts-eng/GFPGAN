import onnx
import numpy as np

model = onnx.load('GFPGANv1.4.onnx')

node = model.graph.node

for o in model.graph.output:
    s = o.type.tensor_type.shape
    t = onnx.TypeProto(
        tensor_type=onnx.TypeProto.Tensor(
            elem_type=onnx.TensorProto.FLOAT,
            shape=s
        ))
    o.type.CopyFrom(t)

for n in node:
    if n.op_type == 'Cast':
        for attr in n.attribute:
            old_i = attr.i
            if attr.i == onnx.TensorProto.DOUBLE:
                attr.i = onnx.TensorProto.FLOAT
                print(f'{n.name}:{attr.name} {old_i} -> {attr.i}')
    elif n.op_type == 'Constant':
        for attr in n.attribute:
            old_type = attr.t.data_type
            if attr.t.data_type == onnx.TensorProto.DOUBLE:
                data = onnx.numpy_helper.to_array(attr.t).astype(np.float32)
                attr.t.CopyFrom(onnx.numpy_helper.from_array(data, attr.t.name))
                print(f'{n.name}:{attr.name} {old_type} -> {attr.t.data_type}')

weights = model.graph.initializer
for w in weights:
    old_type = w.data_type
    if w.data_type != onnx.TensorProto.FLOAT:
        data = onnx.numpy_helper.to_array(w).astype(np.float32)
        w.CopyFrom(onnx.numpy_helper.from_array(data, w.name))
        print(f'{w.name}: {old_type} -> {w.data_type}')

onnx.save(model, 'GFPGANv1.4_.onnx')