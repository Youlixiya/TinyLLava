import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# from tokenize_anything import model_registry
from .tap_image_encoder import tap_vit_b_encoder, tap_vit_l_encoder
class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

class LlavaTAPLlamaModel(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class TAPImageProcessor(LlavaTAPLlamaModel):
    def __init__(self, image_size=1024, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        self.crop_size = {'height':image_size,
                          'width': image_size}
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, images, return_tensors='pt'):
        return self.preprocess(images, return_tensors)
    
    def preprocess(self, images, return_tensors='pt'):
        images_tensor = []
        if type(images) == list:
            for image in images:
                images_tensor.append(self.transform(image))
            images_tensor = torch.stack(images_tensor)
        else:
            images_tensor = self.transform(images)[None]
        return {'pixel_values': images_tensor}
        

class TAPVisionTower(nn.Module):
    hidden_size=256
    num_patches=4096
    image_size=1024
    # tap_model_type='tap_vit_b'
    tap_image_encoder_checkpoint='./ckpts/tap/tap_b_image_encoder.pt'
    # tap_checkpoint='./ckpts/tap/tap_vit_b_b45cbf.pkl'
    # concept_weights='./ckpts/tap/merged_2560.pkl'
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.args = args
        # self.tap = model_registry[self.tap_model_type](checkpoint=self.tap_checkpoint).eval()
        # self.tap.concept_projector.reset_weights(self.concept_weights)
        # self.tap.text_decoder.reset_cache(max_batch_size=8)
        # self.vision_tower = self.tap.image_encoder

        if not delay_load:
            self.load_model()

    def load_model(self):
        self.image_processor = TAPImageProcessor(image_size=self.image_size)
        # self.tap = model_registry[self.tap_model_type](checkpoint=self.tap_checkpoint)
        # self.tap.concept_projector.reset_weights(self.concept_weights)
        # self.tap.text_decoder.reset_cache(max_batch_size=8)
        self.vision_tower = tap_vit_b_encoder()
        self.vision_tower.load_state_dict(torch.load(self.tap_image_encoder_checkpoint))
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == 'patch':
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == 'cls_patch':
    #         image_features = image_features
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))[0]
                # image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))[0]
            # image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.patch_embed.proj.weight.data.dtype

    @property
    def device(self):
        return self.vision_tower.patch_embed.proj.weight.data.device

    @property
    def config(self):
        return self.args
        # if self.is_loaded:
        #     return self.vision_tower.config
        # else:
        #     return self.cfg_only

    # @property
    # def hidden_size(self):
    #     return self.config.hidden_size

    # @property
    # def num_patches(self):
    #     return (self.config.image_size // self.config.patch_size) ** 2
