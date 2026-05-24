import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear
import torch.nn as nn

debug = False

def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

def apply_lora_to_attention_blocks(encoder, lora_config):
    for i, block in enumerate(encoder.blocks):
        attn = block.attn
        if isinstance(attn.qkv, nn.Linear):
            old_qkv = attn.qkv
            lora_qkv = LoraLinear(
                old_qkv,                   # pass the original layer
                adapter_name="default",    # pass a string adapter name
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                fan_in_fan_out=False,
                bias="none",
                config=lora_config
            )
            attn.qkv = lora_qkv

        if isinstance(attn.proj, nn.Linear):
            old_proj = attn.proj
            lora_proj = LoraLinear(
                old_proj,                  # pass the original layer
                adapter_name="default",    # adapter name string
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                fan_in_fan_out=False,
                bias="none",
                config=lora_config
            )
            attn.proj = lora_proj

def freeze_except_lora(model):
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class BED_SAM2(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(BED_SAM2, self).__init__()    
        model_cfg = ".\\configs\\sam2.1\\sam2.1_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=["qkv", "proj"],  # Only names like "qkv" or "proj", not full paths
        )
        
        #### RGB + Edges ####
        if hasattr(self.encoder, "patch_embed") and hasattr(self.encoder.patch_embed, "proj"):
            old_conv = self.encoder.patch_embed.proj
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight  # Copy RGB weights
                new_conv.weight[:, 3:] = (old_conv.weight.mean(dim=1, keepdim=True) * 3)
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            self.encoder.patch_embed.proj = new_conv

        apply_lora_to_attention_blocks(self.encoder, lora_config)
        freeze_except_lora(self.encoder)

        rfb_size = 256

        self.rfb1 = RFB_modified(144, rfb_size)
        self.rfb2 = RFB_modified(288, rfb_size)
        self.rfb3 = RFB_modified(576, rfb_size)
        self.rfb4 = RFB_modified(1152, rfb_size)

        self.up = (Up(rfb_size*2, rfb_size))

        self.head = nn.Conv2d(rfb_size, 1, kernel_size=1)

    def forward(self, x):
        global debug
        
        if debug:
            print("Input shape:", x.shape)
        x1, x2, x3, x4 = self.encoder(x)
        if debug:
            print("Encoder outputs shapes:", x1.shape, x2.shape, x3.shape, x4.shape)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        if debug:
            print("RFB outputs shapes:", x1.shape, x2.shape, x3.shape, x4.shape)

        x = self.up(x4, x3)

        out1 = F.interpolate(self.head(x), scale_factor=16, mode='bilinear')
        x = self.up(x, x2)

        out2 = F.interpolate(self.head(x), scale_factor=8, mode='bilinear')
        x = self.up(x, x1)

        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2
