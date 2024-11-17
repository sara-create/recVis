"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms , data_transforms_plus
from model import Net ,ResNet , ResNeXt , DinoTransformer



class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet" :
            return ResNet()
        elif self.model_name == "resnext" :
            return ResNeXt()
        elif self.model_name == "dino" :
            return DinoTransformer()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        elif self.model_name == "resnet" :
            return data_transforms
        elif self.model_name == "resnext" :
            return data_transforms_plus
        elif self.model_name == "dino" :
            return data_transforms_plus
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
