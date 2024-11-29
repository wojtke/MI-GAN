import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class lrelu_agc(nn.Module):
    """
    Leaky ReLU layer with alpha, gain, and clamp as configurable parameters.
    """

    def __init__(self, alpha=0.2, gain=1, clamp=None):
        super(lrelu_agc, self).__init__()
        self.alpha = alpha
        self.gain = np.sqrt(2).item() if gain == 'sqrt_2' else gain
        self.clamp = clamp * self.gain if clamp is not None else None

    def forward(self, x):
        x = F.leaky_relu(x, negative_slope=self.alpha, inplace=True)
        x = x * self.gain
        if self.clamp is not None:
            x = x.clamp(-self.clamp, self.clamp)
        return x


def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    """
    Creates a gaussian-like downsampling filter.
    """
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[None]

    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f


class Downsample2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=4,
            padding=1,
            bias=False,
            stride=2
        )
        with torch.no_grad():
            f = setup_filter([1, 3, 3, 1], gain=1)
            self.conv.weight.copy_(f)
            

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(-1, 1, h, w)
        x = self.conv(x)
        x = x.view(b, c, h//2, w//2)
        return x

class Upsample2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        with torch.no_grad():
            f = setup_filter([1, 3, 3, 1], gain=4)
            self.conv.weight.copy_(f)


    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(-1, 1, h, w)
        x = self.conv(x)
        x = x.view(b, c, h*2, w*2)
        return x

class SeparableConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            bias=True,
            resolution=None,
            use_noise=False,
            down=1,
            up=1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
            groups=in_channels
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            groups=1
        )

        self.downsample = None
        if down > 1:
            self.downsample = Downsample2d()

        self.upsample = None
        if up > 1:
            self.upsample = Upsample2d()

        self.use_noise = use_noise
        if use_noise:
            assert resolution is not None
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.activation = lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        if self.downsample is not None:
            x = self.downsample(x)
        x = self.conv2(x)
        if self.upsample is not None:
            x = self.upsample(x)

        if self.use_noise:
            noise = self.noise_const * self.noise_strength
            x = x.add_(noise)
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        ic_n,
        oc_n,
        rgb_n=None,
        down=2
    ):
        super().__init__()

        self.fromrgb = None
        if rgb_n is not None:
            self.fromrgb = nn.Conv2d(rgb_n, ic_n, 1)

        self.conv1 = SeparableConv2d(ic_n, ic_n, 3)
        self.conv2 = SeparableConv2d(ic_n, oc_n, 3, down=down)
        self.activation = lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)

    def forward(self, x, img):
        if self.fromrgb is not None:
            y = self.fromrgb(img)
            y = self.activation(y)
            x = x + y if x is not None else y

        feat = self.conv1(x)
        x = self.conv2(feat)
        return x, feat


class Encoder(nn.Module):
    def __init__(
        self,
        resolution=256,
        ic_n=4,
        ch_base=32768,
        ch_max=512,
    ):
        super().__init__()

        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError(f"Resolution {resolution} is not a power of 2")

        self.encode_res = [2 ** i for i in range(log2res, 1, -1)]
        self.ic_n = ic_n
        self.blocks = nn.ModuleDict()

        for idx, (res_in, res_out) in enumerate(zip(self.encode_res[:-1], self.encode_res[1:])):
            hidden_ch_in = min(ch_base // res_in, ch_max)
            hidden_ch_out = min(ch_base // res_out, ch_max)
            
            if idx == 0:
                block = EncoderBlock(hidden_ch_in, hidden_ch_out, rgb_n=ic_n)
            else:
                block = EncoderBlock(hidden_ch_in, hidden_ch_out)
                
            self.blocks[f'b{res_in}'] = block

        hidden_ch_last = min(ch_base // self.encode_res[-1], ch_max)
        self.blocks['b4'] = EncoderBlock(hidden_ch_last, hidden_ch_last, down=1)

    def forward(self, img):
        x = None
        feats = {}

        for res in self.encode_res[:-1]:
            block = self.blocks[f'b{res}']
            x, feat = block(x, img)
            feats[res] = feat

        x, feat = self.blocks['b4'](x, img)
        feats[4] = feat

        return x, feats


class SynthesisBlockFirst(nn.Module):
    def __init__(
        self,
        oc_n,
        resolution,
        rgb_n=None
    ):
        """
        Args:
            oc_n: output channel number
        """
        super().__init__()
        self.resolution = resolution

        self.conv1 = SeparableConv2d(oc_n, oc_n, 3)
        self.conv2 = SeparableConv2d(oc_n, oc_n, 3, resolution=4)

        if rgb_n is not None:
            self.torgb = nn.Conv2d(oc_n, rgb_n, 1)

    def forward(self, x, enc_feat):
        x = self.conv1(x)
        x = x + enc_feat
        x = self.conv2(x)

        img = None
        if self.torgb is not None:
            img = self.torgb(x)

        return x, img


class SynthesisBlock(nn.Module):
    def __init__(
        self,
        ic_n,
        oc_n,
        resolution,
        rgb_n
    ):
        super().__init__()

        self.resolution = resolution

        self.conv1 = SeparableConv2d(ic_n, oc_n, 3, resolution=resolution, up=2, use_noise=True)
        self.conv2 = SeparableConv2d(oc_n, oc_n, 3, resolution=resolution, up=1, use_noise=True)

        self.torgb = None
        if rgb_n is not None:
            self.torgb = nn.Conv2d(oc_n, rgb_n, 1)
        self.upsample = Upsample2d()

    def forward(self, x, enc_feat, img):
        x = self.conv1(x)
        x = x + enc_feat
        x = self.conv2(x)

        if img is not None:
            img = self.upsample(img)

        if self.torgb is not None:
            y = self.torgb(x)
            img = img.add_(y) if img is not None else y

        return x, img


class Synthesis(nn.Module):
    def __init__(
        self,
        resolution=256,
        rgb_n=3,
        ch_base=32768,
        ch_max=512,
    ):
        super().__init__()

        self.resolution = resolution
        self.rgb_n = rgb_n
        self.block_res = self._calculate_block_resolutions(resolution)
        self.blocks = nn.ModuleDict()

        # Initialize the first synthesis block (smallest resolution).
        initial_ch = min(ch_base // self.block_res[0], ch_max)
        self.blocks['b4'] = SynthesisBlockFirst(initial_ch, resolution=4, rgb_n=rgb_n)

        # Initialize the remaining synthesis blocks.
        for res_in, res_out in zip(self.block_res[:-1], self.block_res[1:]):
            self.blocks[f'b{res_out}'] = self._create_synthesis_block(res_in, res_out, ch_base, ch_max, rgb_n)

    def _calculate_block_resolutions(self, resolution):
        """Calculate block resolutions for power-of-2 sizes up to the target resolution."""
        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError(f"Resolution {resolution} is not a power of 2")
        return [2 ** i for i in range(2, log2res + 1)]

    def _create_synthesis_block(self, res_in, res_out, ch_base, ch_max, rgb_n):
        """Helper function to create a SynthesisBlock with computed channels for input and output resolutions."""
        ch_in = min(ch_base // res_in, ch_max)
        ch_out = min(ch_base // res_out, ch_max)
        return SynthesisBlock(ch_in, ch_out, resolution=res_out, rgb_n=rgb_n)

    def forward(self, x, enc_feats):
        """Forward pass through synthesis blocks, updating x and img for each resolution."""
        x, img = self.blocks['b4'](x, enc_feats[4])  # First block with fixed resolution 4

        # Process through all remaining blocks by resolution
        for res in self.block_res[1:]:
            x, img = self.blocks[f'b{res}'](x, enc_feats[res], img)
        
        return img


class Generator(nn.Module):
    def __init__(self, resolution=512):
        super().__init__()

        self.synthesis = Synthesis(resolution=resolution)
        self.encoder = Encoder(resolution=resolution)

    def forward(self, x):
        x, feats = self.encoder(x)
        img = self.synthesis(x, feats)
        return img


def gaussian_kernel(size, sigma):
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


def first_and_last_nonzero_indices(tensor):
    indices = torch.arange(tensor.size(0), device=tensor.device)

    mask = (tensor > 0) * 1.0
    masked_indices = indices * mask
    last_index = masked_indices.max() + 1
    masked_indices = indices * mask + (1 - mask) * tensor.size(0)
    first_index = masked_indices.min()
    
    return first_index, last_index

class Cropper(nn.Module):
    def __init__(self, crop_size=(512, 512), downsample_size=(12, 18), interpolation='bilinear'):
        super().__init__()
        self.crop_size = crop_size
        self.dilate = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.dilate.weight.fill_(1)

        self.downsample_size = downsample_size
        self.max_stretch = 0.8
        self.interpolation = interpolation

    def crop_out(self, img, mask):
        img_h, img_w = img.size(-2), img.size(-1)
        mask_downsampled = F.interpolate(1-mask, size=self.downsample_size, mode=self.interpolation, align_corners=False)
        mask_downsampled = self.dilate(mask_downsampled).clamp(0, 1)

        mask_vertical = mask_downsampled.sum(dim=-1).view(-1)

        min_h, max_h = first_and_last_nonzero_indices(mask_vertical)
        min_h = (img_h * min_h / self.downsample_size[0]).floor()
        max_h = (img_h * max_h / (self.downsample_size[0])).ceil()
        height = torch.tensor([max_h - min_h, self.crop_size[0]]).max()

        mask_horizontal = mask_downsampled.sum(dim=-2).view(-1)
        min_w, max_w = first_and_last_nonzero_indices(mask_horizontal)
        min_w = (img_w * min_w / self.downsample_size[1]).floor()
        max_w = (img_w * max_w / (self.downsample_size[1])).ceil()
        width = torch.tensor([(max_w - min_w), self.crop_size[1]]).max()

        aspect_ratio = self.crop_size[1] / self.crop_size[0]
        height = torch.max(height, width / aspect_ratio * self.max_stretch)
        width = torch.max(width, height * aspect_ratio * self.max_stretch)

        center_h = (min_h + max_h) / 2
        min_h = torch.tensor([center_h - height / 2, 0]).max()
        max_h = torch.tensor([min_h + height, img_h]).min()
        min_h = torch.tensor([max_h - height, 0]).max()

        center_w = (min_w + max_w) / 2
        min_w = torch.tensor([center_w - width / 2, 0]).max()
        max_w = torch.tensor([min_w + width, img_w]).min()
        min_w = torch.tensor([max_w - width, 0]).max()

        pts = torch.tensor([min_h, max_h, min_w, max_w])
        
        x = torch.cat([mask, img], dim=1)
        x = self.pts_to_crop(x, pts, self.crop_size)
        mask_crop, img_crop = torch.split(x, [1, 3], dim=1)
        return img_crop, mask_crop, pts


    def pts_to_crop(self, x, pts, crop_size):
        N, C, H, W = x.size()
        top, bottom, left, right = torch.unbind(pts, dim=-1)

        center_y = (top + bottom) / 2 / H * 2 - 1
        center_x = (left + right) / 2 / W * 2 - 1
        scale_y = (bottom - top) / H
        scale_x = (right - left) / W

        affine_matrix = torch.tensor([
            [scale_x, 0, center_x],
            [0, scale_y, center_y]
        ]).unsqueeze(0).repeat(N, 1, 1)

        grid = F.affine_grid(affine_matrix, size=(N, C, crop_size[0], crop_size[1]), align_corners=False)
        cropped = F.grid_sample(x, grid, mode=self.interpolation, align_corners=False)

        return cropped
    
    def reverse_pts(self, pts, img_size):
        top, bottom, left, right = torch.unbind(pts, dim=-1)
        crop_scale_h, crop_scale_w = self.crop_size[0] / (bottom - top), self.crop_size[1] / (right - left)

        new_top = -top * crop_scale_h
        new_bottom = new_top + img_size[0] * crop_scale_h 
        new_left = -left * crop_scale_w
        new_right = new_left + img_size[1] * crop_scale_w 

        return torch.tensor([new_top, new_bottom, new_left, new_right])

    def paste_crop(self, img, crop, pts):
        #reversed_pts = self.reverse_pts(pts, img.size()[-2:])

        #crop = torch.cat([crop, torch.ones_like(crop[:, :1])], dim=1)
        #reverse_crop = self.pts_to_crop(crop, reversed_pts, img.size()[-2:])
        #crop, mask = torch.split(reverse_crop, [3, 1], dim=1)
        # mask = (mask > 0.5) * 1.0
        # mask = self.dilate(1-mask).clamp(0, 1)
        # out = img * mask + crop * (1 - mask)
        return F.interpolate(crop, size=(img.size(2), img.size(3)), mode=self.interpolation, align_corners=False)
        return reverse_crop
    


class GeneratorWrapper(nn.Module):
    def __init__(self, resolution=512, weights=None, invert_mask=False, dilate_mask=3, matte_mask=3, interpolation='bilinear', use_crop=True):
        super().__init__()

        self.model_res = resolution
        self.model = Generator(resolution=resolution)

        #assert weights is not None, "Weights path must be provided"
        self._load_weights(weights)

        self.invert_mask = invert_mask
        self.interpolation = interpolation

        if dilate_mask > 0:
            dilate = nn.Conv2d(1, 1, dilate_mask, padding=dilate_mask//2, bias=False)
            with torch.no_grad():
                dilate.weight.copy_(gaussian_kernel(dilate_mask, dilate_mask))
            self.dilate_mask = dilate
        else:
            self.dilate_mask = None

        if matte_mask > 0:
            matte = nn.Conv2d(1, 1, matte_mask, padding=matte_mask//2, bias=False)
            with torch.no_grad():
                matte.weight.copy_(gaussian_kernel(matte_mask, matte_mask/2))
            self.matte_mask = matte
        else:
            self.matte_mask = None

        if use_crop:
            self.cropper = Cropper()
        self.use_crop = use_crop


    def _load_weights(self, weights="migan_512_places2.pt"):
        state_dict = torch.load(weights, map_location='cpu', weights_only=True)

        for key in list(state_dict.keys()):
            if key.startswith('synthesis.b'):
                state_dict[key.replace('synthesis.', 'synthesis.blocks.')] = state_dict.pop(key)
            if key.startswith('encoder.b'):
                state_dict[key.replace('encoder.', 'encoder.blocks.')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


    def forward(self, img, mask):
        if self.invert_mask:
            mask = 1 - mask

        if self.use_crop:
            img_crop, mask_crop, pts = self.cropper.crop_out(img, mask)
        else:
            img_crop = F.interpolate(img, size=(self.model_res, self.model_res), mode=self.interpolation, align_corners=False).clamp(0, 1)
            mask_crop = F.interpolate(mask, size=(self.model_res, self.model_res), mode=self.interpolation, align_corners=False).clamp(0, 1)

    
        if self.dilate_mask is not None:
            mask_crop = 1 - (self.dilate_mask(1 - mask_crop) > 0.01) * 1.0

        img_crop = 2 * img_crop - 1
        mask_crop = (mask_crop > 0.5) * 1.0
        img_crop = img_crop * mask_crop
        mask_crop = mask_crop - 0.5

        x = torch.cat([mask_crop, img_crop], dim=1)
        out_img_crop = self.model(x)
        out_img_crop = (out_img_crop + 1) / 2

        if self.use_crop:
            out_img = self.cropper.paste_crop(img, out_img_crop, pts)
        else:
            out_img = F.interpolate(out_img_crop, size=(img.size(2), img.size(3)), mode=self.interpolation, align_corners=False)

        if self.matte_mask is not None:
            mask = 1 - (self.matte_mask(1 - mask) > 0.01) * 1.0
            mask = self.matte_mask(mask - 0.5) + 0.5
            mask = torch.clamp(mask, 0, 1)

        #out_img = out_img * (1 - mask) + img * mask
        
        return out_img

    