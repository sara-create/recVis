import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet
data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_transforms_plus = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # Randomly resize and crop to 224x224
        transforms.RandomHorizontalFlip(),   # Random horizontal flip
        transforms.ToTensor(),               # Convert image to tensor
        transforms.Normalize(                # Normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406],      # Mean of ImageNet RGB channels
            std=[0.229, 0.224, 0.225]        # Std dev of ImageNet RGB channels
        ),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),              # Resize image to 256x256
        transforms.CenterCrop(224),          # Crop to center 224x224
        transforms.ToTensor(),               # Convert image to tensor
        transforms.Normalize(                # Normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406],      # Mean of ImageNet RGB channels
            std=[0.229, 0.224, 0.225]        # Std dev of ImageNet RGB channels
        ),
    ]),
}