
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.vision_transformer import vit_base_patch16_224_in21k
from segment_anything import sam_model_registry
import torch

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, nclasses)

    def forward(self, x):
        return self.base_model(x)

class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.base_model = models.resnext50_32x4d(pretrained=True)
        
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, nclasses)

    def forward(self, x):
        return self.base_model(x)
    

class DinoTransformer(nn.Module):
    def __init__(self):
        super(DinoTransformer, self).__init__()
        self.base_model = vit_base_patch16_224_in21k(pretrained=True)
        
        num_features = self.base_model.head.in_features
        self.base_model.head = nn.Linear(num_features, nclasses)

    def forward(self, x):
        return self.base_model(x)
    

class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()
        self.feature_extractor = torch.hub.load(
            'google-research/vision_transformer', 
            'vit_base_patch16_224_in21k', 
            pretrained=True
        )
        self.feature_extractor.head = nn.Identity()  
        hidden_size = 768  
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, nclasses)

    def forward(self, x):
        with torch.no_grad():  
            features = self.feature_extractor(x)
        
        x = F.relu(self.fc1(features))
        x = self.fc2(x)
        return x
    
    
class Dino(nn.Module):
    def __init__(self):
        super(Dino, self).__init__()
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', "dinov2_vitg14_lc")
        self.feature_extractor.eval()
        self.feature_extractor.head = nn.Identity()  
        
        
        self.fc1 = nn.Linear(1000, 512)  
        self.fc2 = nn.Linear(512, nclasses)  

    def forward(self, x):
        with torch.no_grad():  
            features = self.feature_extractor(x)
        x = F.relu(self.fc1(features))  
        x = self.fc2(x)  
        return x


class SAMClassifier(nn.Module):
    def __init__(self, nclasses, sam_checkpoint="sam_vit_b_01ec64.pth"):
        super(SAMClassifier, self).__init__()
        self.feature_extractor = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.feature_extractor.eval() 
        self.feature_extractor.prompt_encoder = nn.Identity()  
        self.feature_extractor.mask_decoder = nn.Identity()    

        hidden_size = 768  
        self.fc1 = nn.Linear(hidden_size, 512) 
        self.fc2 = nn.Linear(512, nclasses)     

    def forward(self, x):
        with torch.no_grad():  
            features = self.feature_extractor.image_encoder(x)  
            cls_token_features = features[:, 0, :]  

        x = F.relu(self.fc1(cls_token_features))
        x = self.fc2(x)
        return x