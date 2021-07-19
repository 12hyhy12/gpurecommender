import torch
import numpy as np
import torchvision
from tvm.contrib.download import download_testdata
# 加载torchvision中的ResNet18模型
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
# 使用推理模型
model = model.eval()

from PIL import Image
img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# 处理图像并转成Tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

# 使用Pytorch进行预测结果
with torch.no_grad():
    torch_img = torch.from_numpy(img)
    output = model(torch_img)

    top1_torch = np.argmax(output.numpy())

print(top1_torch)

# export onnx

torch_out = torch.onnx.export(model, torch_img, 'resnet18.onnx', verbose=True, export_params=True)
