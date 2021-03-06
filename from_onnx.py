
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient
from ir_module_traverser import construct_op_graph, profile_memory
from tvm.relay.visualize import visualize
# python -m memory_profiler tvm-analyzer/relay_profiler/from_onnx.py


model_url = "".join(
    [
        "https://gist.github.com/zhreshold/",
        "bcda4716699ac97ea44f791c24310193/raw/",
        "93672b029103648953c4e5ad3ac3aadf346a4cdc/",
        "super_resolution_0.2.onnx",
    ]
)

#model_url = "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx"
#model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
model_path = download_testdata(model_url, "super_resolution.onnx", module="onnx")
#model_path = "/root/models/vision/classification/resnet/model/resnet18-v2-7.onnx"
onnx_model = onnx.load(model_path)
#onnx_model = onnx.load('resnet18.onnx')

'''
for input in onnx_model.graph.input:
    print (input.name, end=": ")
    # get type of input tensor
    tensor_type = input.type.tensor_type
    # check if it has a shape:
    if (tensor_type.HasField("shape")):
        # iterate through dimensions of the shape:
        for d in tensor_type.shape.dim:
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if (d.HasField("dim_value")):
                print (d.dim_value, end=", ")  # known dimension
            elif (d.HasField("dim_param")):
                print (d.dim_param, end=", ")  # unknown dimension with symbolic name
            else:
                print ("?", end=", ")  # unknown dimension with no name
    else:
        print ("unknown rank", end="")
'''

from PIL import Image


img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

'''
img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# Resize it to 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")
# ONNX expects NCHW input, so convert the array
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to ImageNet
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_stddev = np.array([0.229, 0.224, 0.225])
norm_img_data = np.zeros(img_data.shape).astype("float32")
for i in range(img_data.shape[0]):
    norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

# Add batch dimension
x = np.expand_dims(norm_img_data, axis=0)
'''

target = "cuda"

input_name = "1"
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

#visualize(mod["main"])
construct_op_graph(mod)
print('-----')
print(mod)
print('-----')
profile_memory(params, x)

# ???????????????module?????????????????????????????????????????????
# print(mod)
# print(dir(mod))
# # print(mod.type_definitions)
# print(mod.get_global_vars()[0])
# print(len(mod.get_global_type_vars()))
# ????????????function?????????,????????????????????????function????????????,???????????????????????????

# ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# print(main_function.params)
# # ??????function?????????,?????????????????????
# print(type(main_function.body))
# print(main_function.body)
# # ???????????????call Node?????????????????????
# print("elements in call Node:")
# # ???????????????????????????????????????????????????
# print(main_function.body.args)
# # ??????????????????????????????
# print(main_function.body.op)
# # ??????????????????
# print(main_function.body.attrs['newshape'])
# for k,v in mod.functions.items():
#     print(type(k))
#     print(v)

# ?????????????????????????????????????????????????????????????????????????????????????????????
# temp_x = np.random.rand(672,672,1,1)
# temp_params_map = {}
# temp_tvm_var = tvm.relay.var("1",shape=temp_x.shape)
# # temp_params_map['1'] = temp_tvm_var
# temp_params_list = []
# temp_params_list.append(temp_tvm_var)
# temp_body = tvm.relay.Call(main_function.body.op,temp_params_list,attrs=main_function.body.attrs)
# temp_function = tvm.relay.Function(temp_params_list, temp_body)
# # print(temp_function)
# temp_functions = {"GlobalVar": None, "main": temp_function}
# temp_ir_module = tvm.ir.IRModule(functions=temp_functions)
# print(temp_ir_module)

# print(temp_params_map)
######################################################################
# Execute on TVM
# ---------------------------------------------

# def start_profile():
#     return

# print(memory_usage(proc=start_profile))
# print("zero ir")

# dtype = "float32"
# print(type(intrp))
# def myfunc():
#     # may consume too many memory.
#     tvm_output = intrp.evaluate()(tvm.nd.array(temp_x.astype(dtype)), **temp_params_map).asnumpy()
#     print(tvm_output)

# # myfunc()
# print(memory_usage(proc=myfunc))
# print("first ir")

# determine base:
# dshape = (1,)
# base_input_x = np.random.rand(1)
# base_x = tvm.relay.var("x", shape = dshape)
# base_x = relay.add(base_x, relay.const(1, "float32"))
# base_function = tvm.relay.Function(relay.analysis.free_vars(base_x), base_x)
# base_functions = {"GlobalVar": None, "main": base_function}
# base_ir_module = tvm.ir.IRModule(functions=base_functions)
# print(base_ir_module)

# with tvm.transform.PassContext(opt_level=1):
#     base_intrp = relay.build_module.create_executor("graph", base_ir_module, tvm.cpu(0), target)

# def myfunc_base():
#     # may consume too many memory.
#     tvm_output_base = base_intrp.evaluate()(tvm.nd.array(base_input_x.astype(dtype)), **temp_params_map).asnumpy()
#     print(tvm_output_base)

# # myfunc()
# print(memory_usage(proc=myfunc_base))
# print("second ir")

# get gradient:
# dshape = (2,2)
# sigmoid_input_x = np.random.rand(2,2).astype("float32")
# sigmoid_x = tvm.relay.var("input", shape = dshape)
# sigmoid_output = tvm.relay.sigmoid(sigmoid_x)
# sigmoid_function = tvm.relay.Function([sigmoid_x],sigmoid_output)
# sigmoid_function = run_infer_type(sigmoid_function)
# bwd_func = run_infer_type(gradient(sigmoid_function))

# print("bwd_func)")
# print(bwd_func)

# tvm_output_grad = tvm.relay.create_executor(
#     mod=tvm.IRModule.from_expr(sigmoid_function),device = tvm.cpu(0), target = target
# ).evaluate()(tvm.nd.array(sigmoid_input_x.astype(dtype)), **temp_params_map).asnumpy()

# print("output_grad")
# print(tvm_output_grad)

# intrp = relay.create_executor(device = tvm.cpu(0), target = target)
# op_res, (op_grad, ) = intrp.evaluate(bwd_func)(sigmoid_input_x)

# print("output_grad2")
# print(op_res)
# print(op_grad)
# sigmoid_functions = {"GlobalVar": None, "main": sigmoid_function}
# sigmoid_ir_module = tvm.ir.IRModule(functions=sigmoid_functions)
# sigmoid_ir_module = tvm.relay.transform.InferType()(sigmoid_ir_module)

# sigmoid_ir_module['main'] = relay.transform.gradient(sigmoid_ir_module['main'],  mod=sigmoid_ir_module, mode='higher_order')
# sigmoid_ir_module = relay.transform.InferType()(sigmoid_ir_module)

# e = tvm.relay.create_executor("graph", sigmoid_ir_module, tvm.cpu(0), target).evaluate()
# with tvm.transform.PassContext(opt_level=1):
#     grad_intrp = relay.build_module.create_executor("graph", sigmoid_ir_module, tvm.cpu(0), target)

######################################################################
# Display results
# ---------------------------------------------
# We put input and output image neck to neck. The luminance channel, `Y` is the output
# from the model. The chroma channels `Cb` and `Cr` are resized to match with a simple
# bicubic algorithm. The image is then recombined and converted back to `RGB`.
# from matplotlib import pyplot as plt

# out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode="L")
# out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
# out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
# result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
# canvas = np.full((672, 672 * 2, 3), 255)
# canvas[0:224, 0:224, :] = np.asarray(img)
# canvas[:, 672:, :] = np.asarray(result)
# plt.imshow(canvas.astype(np.uint8))
# plt.show()

######################################################################
# Notes
# ---------------------------------------------
# By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
# retains that dynamism upon import, and the compiler attemps to convert the model
# into a static shapes at compile time. If this fails, there may still be dynamic
# operations in the model. Not all TVM kernels currently support dynamic shapes,
# please file an issue on discuss.tvm.apache.org if you hit an error with dynamic kernels.
#
# This particular model was build using an older version of ONNX. During the import
# phase ONNX importer will run the ONNX verifier, which may throw a `Mismatched attribute type`
# warning. Because TVM supports a number of different ONNX versions, the Relay model
# will still be valid.
