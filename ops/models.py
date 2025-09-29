from torch import nn
from torch.nn.init import normal_, constant_

from ops.basic_ops import ConsensusModule
from ops.transforms import *


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, base_model='mobilenetv3_deptheca_mega',
                 dropout=0.5, partial_bn=True, is_shift=False, shift_div=8):
        super(TSN, self).__init__()
        self.num_segments = num_segments
        self.base_model_name = base_model
        self.dropout = dropout
        self.is_shift = is_shift
        self.shift_div = shift_div

        print(("""
                TSN Configurations:
                    base model:         {}
                    num_segments:       {}
                    dropout_ratio:      {}
                    shift_div:          {}
                """.format(base_model, self.num_segments, self.dropout, self.shift_div)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

        self.consensus = ConsensusModule()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = self.base_model.last_channel
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        self.new_fc = nn.Linear(feature_dim, num_class)
        if hasattr(self.new_fc, 'weight'):
            normal_(self.new_fc.weight, 0, 0.001)
            constant_(self.new_fc.bias, 0)

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))
        if base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.is_shift:
                from ops.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
        elif base_model == 'mobilenetv3_deptheca':
            from archs.mobilenet_v3_deptheca import mobilenet_v3_deptheca
            # import the V3 inverted‐residual class so isinstance checks work
            from torchvision.models.mobilenetv3 import InvertedResidual as MV3Block
            from ops.temporal_shift import TemporalShift
            from ops.dtemporal_shift import DiscriminativeTemporalShift

            self.base_model = mobilenet_v3_deptheca(pretrained=True)
            self.base_model.last_layer_name = 'classifier'
            self.input_size   = 224
            self.input_mean   = [0.485, 0.456, 0.406]
            self.input_std    = [0.229, 0.224, 0.225]

            if self.is_shift:
                for m in self.base_model.modules():
                    # only patch those blocks that use a residual connection
                    if isinstance(m, MV3Block) and m.use_res_connect:
                        # m.conv is an nn.Sequential; we replace its very first conv
                        m.block[0] = DiscriminativeTemporalShift(
                            m.block[0],
                            n_segment=self.num_segments,
                            n_div=self.shift_div
                        )
                        print(f'Adding temporal shift to MV3Block at {m}')
            else:
                print(f'not using any temporal shift module ')


        elif base_model == 'mobilenetv3_deptheca_mega':
            from archs.mobilenet_v3_deptheca_mega import mobilenet_v3_deptheca_mega
            from torchvision.models.mobilenetv3 import InvertedResidual as MV3Block
            from ops.temporal_shift import TemporalShift
            from ops.dtemporal_shift import DiscriminativeTemporalShift
            from ops.attention_shift import discshift
            from ops.gated_dtsm import GatedDTSM

            self.base_model = mobilenet_v3_deptheca_mega(
                pretrained=True,
                use_eca=True,
                use_se=True,
                use_ema=True
            )
            self.base_model.last_layer_name = 'classifier'
            self.input_size   = 224
            self.input_mean   = [0.485, 0.456, 0.406]
            self.input_std    = [0.229, 0.224, 0.225]

            if self.is_shift:
                for m in self.base_model.modules():
                    if isinstance(m, MV3Block) and m.use_res_connect:
                        old_block = m.block[0]  # this is a Conv2dNormActivation
                
                        # get the input‐channels of the wrapped Conv2d
                        if hasattr(old_block, 'conv'):
                            C = old_block.conv.in_channels
                        else:
                            # fallback if it's a Sequential
                            C = old_block[0].in_channels
                
                        # replace with your GatedDTSM, passing C *positionally*:
                        m.block[0] = GatedDTSM(
                            old_block,   # the 2D module
                            C,           # in_channels
                            self.num_segments,
                            self.shift_div,
                            reduction=4
                        )
                        print(f'Patched GatedDTSM(in_channels={C}) into {m}')
            else:
                print(f'not using any temporal shift module ')

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"}
        ]

    def forward(self, input):
        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        base_out = self.new_fc(base_out)
        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = self.consensus(base_out)

        return output.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 10, 'NO FLIP!!!', '#' * 10)
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
