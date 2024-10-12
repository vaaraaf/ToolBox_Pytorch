## Downloading The model
import torch
import torchvision
#############################   EFFICIENTNET   ###################################
def download_effnet_b0(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.efficientnet_b0(weights = "DEFAULT")
  my_transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Dropout(p=0.2, inplace = True),
  #     torch.nn.Linear(in_features=1280, out_features = num_classes, bias = True))
  return my_model, my_transform

def download_effnet_b1(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.efficientnet_b1(weights = "DEFAULT")
  my_transform = torchvision.models.EfficientNet_B1_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Dropout(p=0.2, inplace = True),
  #     torch.nn.Linear(in_features=1280, out_features = num_classes, bias = True))
  return my_model, my_transform

def download_effnet_b2(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.efficientnet_b2(weights = "DEFAULT")
  my_transform = torchvision.models.EfficientNet_B2_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Dropout(p=0.3, inplace = True),
  #     torch.nn.Linear(in_features=1408, out_features = num_classes, bias = True))
  return my_model, my_transform

def download_effnet_b3(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.efficientnet_b3(weights = "DEFAULT")
  my_transform = torchvision.models.EfficientNet_B3_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Dropout(p=0.3, inplace = True),
  #     torch.nn.Linear(in_features=1536, out_features = num_classes, bias = True))
  return my_model, my_transform

def download_effnet_b4(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.efficientnet_b4(weights = "DEFAULT")
  my_transform = torchvision.models.EfficientNet_B4_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Dropout(p=0.4, inplace = True),
  #     torch.nn.Linear(in_features=1792, out_features = num_classes, bias = True))
  return my_model, my_transform


def download_effnet_b5(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.efficientnet_b5(weights = "DEFAULT")
  my_transform = torchvision.models.EfficientNet_B5_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Dropout(p=0.4, inplace = True),
  #     torch.nn.Linear(in_features=2048, out_features = num_classes, bias = True))
  return my_model, my_transform

def download_effnet_b6(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.efficientnet_b6(weights = "DEFAULT")
  my_transform = torchvision.models.EfficientNet_B6_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Dropout(p=0.5, inplace = True),
  #     torch.nn.Linear(in_features=2304, out_features = num_classes, bias = True))
  return my_model, my_transform

def download_effnet_b7(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.efficientnet_b7(weights = "DEFAULT")
  my_transform = torchvision.models.EfficientNet_B7_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Dropout(p=0.5, inplace = True),
  #     torch.nn.Linear(in_features=2560, out_features = num_classes, bias = True))
  return my_model, my_transform


#############################   MOBILENET   ###################################
def download_mobilenet_v2(disable_grad= True,
                          num_classes=1000):
  my_model = torchvision.models.mobilenet_v2(weights = "DEFAULT")
  my_transform =  torchvision.models.MobileNet_V2_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Dropout(p=0.2, inplace = False),
  #     torch.nn.Linear(in_features=1280, out_features = num_classes, bias = True))
  return my_model, my_transform


def download_mobilenet_v3_small(disable_grad= True,
                                num_classes=1000):
  my_model = torchvision.models.mobilenet_v3_small(weights = "DEFAULT")
  my_transform =  torchvision.models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Linear(in_features=576, out_features = 1024, bias = True),
  #     torch.nn.Hardswish(),
  #     torch.nn.Dropout(p=0.2, inplace = True),
  #     torch.nn.Linear(in_features = 1024, out_features = num_classes, bias = True))
  return my_model, my_transform


def download_mobilenet_v3_large(disable_grad= True,
                                num_classes=1000):
  my_model = torchvision.models.mobilenet_v3_large(weights = "DEFAULT")
  my_transform =  torchvision.models.MobileNet_V3_Large_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.classifier = torch.nn.Sequential(
  #     torch.nn.Linear(in_features=960, out_features = 1280, bias = True),
  #     torch.nn.Hardswish(),
  #     torch.nn.Dropout(p=0.2, inplace = True),
  #     torch.nn.Linear(in_features = 1280, out_features = num_classes, bias = True))
  return my_model, my_transform


#############################   RESNET   ###################################
def download_resnet18(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.resnet18(weights = "DEFAULT")
  my_transform = torchvision.models.ResNet18_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.fc = torch.nn.Linear(in_features=512, out_features = num_classes, bias = True)
  return my_model, my_transform

def download_resnet34(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.resnet34(weights = "DEFAULT")
  my_transform = torchvision.models.ResNet34_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.fc = torch.nn.Linear(in_features=512, out_features = num_classes, bias = True)
  return my_model, my_transform

def download_resnet50(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.resnet50(weights = "DEFAULT")
  my_transform = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.fc = torch.nn.Linear(in_features=2048, out_features = num_classes, bias = True)
  return my_model, my_transform

def download_resnet101(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.resnet101(weights = "DEFAULT")
  my_transform = torchvision.models.ResNet101_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.fc = torch.nn.Linear(in_features=2048, out_features = num_classes, bias = True)
  return my_model, my_transform


def download_resnet152(disable_grad= True,
                       num_classes=1000):
  my_model = torchvision.models.resnet152(weights = "DEFAULT")
  my_transform = torchvision.models.ResNet152_Weights.DEFAULT.transforms()
  if disable_grad:
    for param in my_model.parameters():
      param.requires_grad = False
  # my_model.fc = torch.nn.Linear(in_features=2048, out_features = num_classes, bias = True)
  return my_model, my_transform
