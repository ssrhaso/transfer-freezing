import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights


# A. RESNET-50 IMPLEMENTATION
class AudioResNet50(nn.Module):
    def __init__(self, freeze_mode='freeze_none', num_classes=10):
        super().__init__()
        
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self._freeze_layers(freeze_mode)
        self.model.fc = nn.Linear(2048, num_classes)

    def _freeze_layers(self, mode):
        # Reset: Everything trainable
        for param in self.model.parameters():
            param.requires_grad = True

        if mode == 'freeze_none':
            pass 
            
        elif mode == 'freeze_0':
            for p in self.model.conv1.parameters(): p.requires_grad = False
            for p in self.model.bn1.parameters(): p.requires_grad = False
            
        elif mode == 'freeze_0_1':
            # Freeze conv1 + bn1
            for p in self.model.conv1.parameters(): p.requires_grad = False
            for p in self.model.bn1.parameters(): p.requires_grad = False
            # Freeze layer1
            for p in self.model.layer1.parameters(): p.requires_grad = False
            
        elif mode == 'freeze_0_1_2':
            # Freeze conv1 + bn1
            for p in self.model.conv1.parameters(): p.requires_grad = False
            for p in self.model.bn1.parameters(): p.requires_grad = False
            # Freeze layers 1-2
            for p in self.model.layer1.parameters(): p.requires_grad = False
            for p in self.model.layer2.parameters(): p.requires_grad = False
            
        elif mode == 'freeze_0_1_2_3':
            # Freeze conv1 + bn1
            for p in self.model.conv1.parameters(): p.requires_grad = False
            for p in self.model.bn1.parameters(): p.requires_grad = False
            # Freeze layers 1-3
            for p in self.model.layer1.parameters(): p.requires_grad = False
            for p in self.model.layer2.parameters(): p.requires_grad = False
            for p in self.model.layer3.parameters(): p.requires_grad = False
            
        elif mode == 'freeze_0_1_2_3_4':
            # Freeze everything except head
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown freeze mode: {mode}")

    def forward(self, x):
        return self.model(x)


# B. VISION TRANSFORMER (ViT-B/16) IMPLEMENTATION
class AudioViT(nn.Module):
    def __init__(self, freeze_mode='freeze_none', num_classes=10):
        super().__init__()
        
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=weights)
        self._freeze_layers(freeze_mode)
        self.model.heads = nn.Linear(768, num_classes)

    def _freeze_layers(self, mode):
        # Reset: Everything trainable
        for param in self.model.parameters():
            param.requires_grad = True
            
        if mode == 'freeze_none':
            pass
            
        elif mode == 'freeze_patch':
            for p in self.model.conv_proj.parameters(): 
                p.requires_grad = False
            self.model.encoder.pos_embedding.requires_grad = False
            
        elif mode == 'freeze_patch_0_2':
            for p in self.model.conv_proj.parameters(): 
                p.requires_grad = False
            self.model.encoder.pos_embedding.requires_grad = False
            for i in range(3):
                for p in self.model.encoder.layers[i].parameters(): 
                    p.requires_grad = False
                
        elif mode == 'freeze_patch_0_5':
            for p in self.model.conv_proj.parameters(): 
                p.requires_grad = False
            self.model.encoder.pos_embedding.requires_grad = False
            for i in range(6): 
                for p in self.model.encoder.layers[i].parameters(): 
                    p.requires_grad = False
                
        elif mode == 'freeze_patch_0_8':
            for p in self.model.conv_proj.parameters(): 
                p.requires_grad = False
            self.model.encoder.pos_embedding.requires_grad = False
            for i in range(9): 
                for p in self.model.encoder.layers[i].parameters(): 
                    p.requires_grad = False

        elif mode == 'freeze_patch_0_11':
            for param in self.model.parameters():
                param.requires_grad = False
            # Un-freeze head
            for param in self.model.heads.parameters():
                param.requires_grad = True
                
        else:
            raise ValueError(f"Unknown freeze mode: {mode}")

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    print("Checking ResNet Freezing...")
    model = AudioResNet50(freeze_mode='freeze_0_1')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet (freeze_0_1) Trainable Params: {trainable:,}")
    
    print("\nChecking ViT Freezing...")
    model = AudioViT(freeze_mode='freeze_patch_0_5')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT (freeze_patch_0_5) Trainable Params: {trainable:,}")
