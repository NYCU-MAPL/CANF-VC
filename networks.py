import torch
from torch import nn

from context_model import ContextModel
from entropy_models import __CONDITIONS__, EntropyBottleneck
from generalizedivisivenorm import GeneralizedDivisiveNorm
from modules import AugmentedNormalizedFlow, Conv2d, ConvTranspose2d


class CompressesModel(nn.Module):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()
        self.divisor = None
        self.num_bitstreams = 1

    def named_main_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                yield (name, param)

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def _cal_base_cdf(self):
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                m._cal_base_cdf()

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.aux_loss())

        return torch.stack(aux_loss).sum() if len(aux_loss) else torch.zeros(1, device=next(self.parameters()).device)


class FactorizedCoder(CompressesModel):
    """FactorizedCoder"""

    def __init__(self, num_priors, quant_mode='noise'):
        super(FactorizedCoder, self).__init__()
        self.analysis = nn.Sequential()
        self.synthesis = nn.Sequential()

        self.entropy_bottleneck = EntropyBottleneck(
            num_priors, quant_mode=quant_mode)

        self.divisor = 16


class HyperPriorCoder(FactorizedCoder):
    """HyperPrior Coder"""

    def __init__(self, num_condition, num_priors, use_mean=False, use_abs=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(HyperPriorCoder, self).__init__(
            num_priors, quant_mode=quant_mode)
        self.use_mean = use_mean
        self.use_abs = not self.use_mean or use_abs
        self.conditional_bottleneck = __CONDITIONS__[condition](
            use_mean=use_mean, quant_mode=quant_mode)
        if use_context:
            self.conditional_bottleneck = ContextModel(
                num_condition, num_condition*2, self.conditional_bottleneck)
        self.hyper_analysis = nn.Sequential()
        self.hyper_synthesis = nn.Sequential()

        self.divisor = 64
        self.num_bitstreams = 2

    def compress(self, input, return_hat=False):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, y_hat = ret
            x_hat = self.synthesis(y_hat)
            return x_hat, [stream, side_stream], [hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [hyperpriors.size()]

    def decompress(self, strings, shape):
        stream, side_stream = strings
        z_shape = shape

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, condition.size(), condition=condition)

        reconstructed = self.synthesis(y_hat)

        return reconstructed


class GoogleHyperAnalysisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=5, stride=2)
        )


class GoogleHyperSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters * 3 // 2,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2, num_features,
                            kernel_size=3, stride=1)
        )


class AugmentedNormalizedAnalysisTransform(AugmentedNormalizedFlow):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_affine, distribution):
        super(AugmentedNormalizedAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_features *
                   (2 if use_affine else 1), kernel_size, stride=2),
            nn.Identity(),
            use_affine=use_affine, transpose=False, distribution=distribution
        )


class AugmentedNormalizedSynthesisTransform(AugmentedNormalizedFlow):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, use_affine, distribution):
        super(AugmentedNormalizedSynthesisTransform, self).__init__(
            nn.Identity(),
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_affine else 1), kernel_size, stride=2),
            use_affine=use_affine, transpose=True, distribution=distribution
        )


class DQ_ResBlock(nn.Sequential):
    def __init__(self, num_filters):
        super().__init__(
            Conv2d(num_filters, num_filters, 3),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(num_filters, num_filters, 3)
        )

    def forward(self, input):
        return super().forward(input) + input


class DeQuantizationModule(nn.Module):

    def __init__(self, in_channels, out_channels, num_filters, num_layers):
        super(DeQuantizationModule, self).__init__()
        self.conv1 = Conv2d(in_channels, num_filters, 3)
        self.resblock = nn.Sequential(
            *[DQ_ResBlock(num_filters) for _ in range(num_layers)])
        self.conv2 = Conv2d(num_filters, num_filters, 3)
        self.conv3 = Conv2d(num_filters, out_channels, 3)

    def forward(self, input):
        conv1 = self.conv1(input)
        x = self.resblock(conv1)
        conv2 = self.conv2(x) + conv1
        conv3 = self.conv3(conv2) + input

        return conv3


class AugmentedNormalizedFlowHyperPriorCoder(HyperPriorCoder):
    """AugmentedNormalizedFlowHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_QE=False, use_affine=True,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        for i in range(num_layers):
            self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[i], kernel_size, use_affine=use_affine and init_code != 'zeros', distribution=init_code))
            self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[i], kernel_size, use_affine=use_affine and i != num_layers-1, distribution=init_code))

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            pass

        if use_QE:
            self.QE = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.QE = None

    def __getitem__(self, key):
        return self.__getattr__(key)

    def encode(self, input, code=None, jac=None):
        for i in range(self.num_layers):
            _, code, jac = self['analysis'+str(i)](input, code, jac)

            if i < self.num_layers-1:
                input, _, jac = self['synthesis'+str(i)](input, code, jac)

        return input, code, jac

    def decode(self, input, code=None, jac=None):
        for i in range(self.num_layers-1, -1, -1):
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1)

            if i or jac is not None:
                _, code, jac = self['analysis' +
                                    str(i)](input, code, jac, rev=True)

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def compress(self, input, code=None, return_hat=False):
        input, code, _ = self.encode(input, code, jac=None)

        hyperpriors = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        side_stream, h_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(h_hat)

        ret = self.conditional_bottleneck.compress(
            code, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, z_hat = ret

            x_hat = self.decode(None, z_hat, jac=None)[0]

            if self.QE is not None:
                x_hat = self.QE(x_hat)

            return x_hat, [stream, side_stream], [z_hat.size(), h_hat.size()]
        else:
            stream = ret
            return [stream, side_stream], [code.size(), h_hat.size()]

    def decompress(self, strings, shapes):
        stream, side_stream = strings
        z_shape, h_shape = shapes

        h_hat = self.entropy_bottleneck.decompress(side_stream, h_shape)

        condition = self.hyper_synthesis(h_hat)

        z_hat = self.conditional_bottleneck.decompress(
            stream, z_shape, condition=condition)

        reconstructed = self.decode(None, z_hat, jac=None)[0]

        if self.QE is not None:
            reconstructed = self.QE(reconstructed)

        return reconstructed

    def forward(self, input, code=None, jac=None):
        # Encode

        ori_input = input
        jac = [] if jac else None

        input, code, jac = self.encode(input, code, jac)

        # Enrtopy coding

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True)

        input, code, hyper_code = None, y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        # input = Y_error

        # Decode
        input, code, jac = self.decode(input, code, jac)

        if self.QE is not None:
            input = self.QE(input)

        return input, (y_likelihood, z_likelihood), Y_error


class CondAugmentedNormalizedFlowHyperPriorCoder(HyperPriorCoder):
    """CondAugmentedNormalizedFlowHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1, # Note: out_channels is useless
                 init_code='gaussian', use_QE=False, use_affine=True,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise',
                 output_nought=True, # Set False when setting upper-left corner(x_2) as MC frame
                 cond_coupling=False, #Set True when applying conditional affine transform
                 num_cond_frames:int =1 # Set 1 when only MC frame is for condition ; >1 whwn multi-refertence frames as conditions
                 ):
        super(CondAugmentedNormalizedFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        self.output_nought=output_nought
        self.cond_coupling = cond_coupling
        assert num_cond_frames > 0, 'number of conditioning frames must >=1'

        print('self.output_nought = ',self.output_nought)
        print('self.cond_coupling = ',self.cond_coupling)

        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        for i in range(num_layers):
            if self.cond_coupling:
                # Make encoding transform conditional
                self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                    in_channels*(1+num_cond_frames), num_features, num_filters[i], kernel_size, 
                    in_channels, num_features, num_filters[i], kernel_size, 
                    use_affine=use_affine and init_code != 'zeros', distribution=init_code))
                # Keep decoding transform unconditional
                self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                    in_channels, num_features, num_filters[i], kernel_size, 
                    use_affine=use_affine and init_code != 'zeros', distribution=init_code))
            else: 
                raise NotImplementedError

        self.hyper_analysis = GoogleHyperAnalysisTransform(num_features, hyper_filters, num_hyperpriors) 

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(num_features*2, hyper_filters, num_hyperpriors) 
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors) 
        else:
            pass

        if use_QE:
            self.QE = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.QE = None

    def __getitem__(self, key):
        return self.__getattr__(key)

    def encode(self, input, code=None, jac=None, cond_coupling_input=None):
        for i in range(self.num_layers):
            # Concat input with condition (MC frame)
            if self.cond_coupling:
                cond = cond_coupling_input
                cond_input = torch.cat([input, cond], dim=1)
                _, code, jac = self['analysis'+str(i)](cond_input, code, jac)
            else:
                _, code, jac = self['analysis'+str(i)](input, code, jac)

            if i < self.num_layers-1:
                input, _, jac = self['synthesis'+str(i)](input, code, jac)

        return input, code, jac

    def decode(self, input, code=None, jac=None):
        for i in range(self.num_layers-1, -1, -1):
            input, _, jac = self['synthesis'+str(i)](input, code, jac, rev=True, last_layer=i == self.num_layers-1)

            if i or jac is not None:
                # Concat input with condition (MC frame)
                if self.cond_coupling:
                    cond = cond_coupling_input
                    cond_input = torch.cat([input, cond], dim=1)
                    _, code, jac = self['analysis'+str(i)](cond_input, code, jac, layer=i, rev=True)
                else:
                    _, code, jac = self['analysis'+str(i)](input, code, jac, layer=i, rev=True)

        return input, code, jac

    def entropy_model(self, input, code):
        # Enrtopy coding
        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)
        y_tilde, y_likelihood = self.conditional_bottleneck(code, condition=condition)
        # y_tilde = code # No quantize on z2
        
        return y_tilde, z_tilde, y_likelihood, z_likelihood

    def compress(self, input, cond_coupling_input=None, reverse_input=None, return_hat=False):
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        
        code = None
        jac = None
        input, features, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            jac = None
            stream, y_hat = ret

            # Decode
            if not self.output_nought:
                assert not (reverse_input is None), "reverse_input should be specified"
                input = reverse_input
            else:
                input = torch.zeros_like(input)
                
            x_hat, code, jac = self.decode(
                input, y_hat, jac, cond_coupling_input=cond_coupling_input)
            if self.use_QE:
                x_hat = self.QE(x_hat)

            return x_hat, [stream, side_stream], [hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [hyperpriors.size()]

    def decompress(self, strings, shapes, cond_coupling_input=None, reverse_input=None):
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        
        jac = None

        stream, side_stream = strings
        y_shape, z_shape = shapes

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, condition=condition)
        
        # Decode
        if not self.output_nought:
            assert not (reverse_input is None), "reverse_input should be specified"
            input = reverse_input
        else:
            input = torch.zeros_like(input)

        x_hat, code, jac = self.decode(
            input, y_hat, jac, cond_coupling_input=cond_coupling_input)

        if self.use_QE:
            reconstructed = self.QE(x_hat)

        return reconstructed

    def forward(self, input, code=None, jac=None 
                output=None, # Should assign value when self.output_nought==False
                cond_coupling_input=None # Should assign value when self.cond_coupling==True
                                         # When using ANFIC on residual coding, output & cond_coupling_input should both be MC frame
               ):
        if not self.output_nought:
            assert not (output is None), "output should be specified"
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"

        # Encode
        jac = [] if jac else None
        input, code, jac = self.encode(input, code, jac, cond_coupling_input=cond_coupling_input)

        # Entropy model
        y_tilde, z_tilde, y_likelihood, z_likelihood = entropy_model(input, code)

        # Encode distortion (last synthesis transform)
        x_2, _, jac = self['synthesis'+str(self.num_layers-1)](input, y_tilde, jac, last_layer=True, layer=self.num_layers-1)

        #input, code, hyper_code = x_2, y_tilde, z_tilde # x_2 directly backward
        input, code, hyper_code = output, y_tilde, z_tilde # Correct setting ; MC frame as x_2 when decoding

        # Decode
        input, code, jac = self.decode(input, code, jac, cond_coupling_input=cond_coupling_input)

        if self.use_QE:       
            BDQ = input
            input = self.QE(input)

        return input, (y_likelihood, z_likelihood), x_2, jac, code, BDQ


class CondAugmentedNormalizedFlowHyperPriorCoderPredPrior(CondAugmentedNormalizedFlowHyperPriorCoder):
    def __init__(self, in_channels_predprior=3, num_predprior_filters=None, **kwargs):
        super(CondAugmentedNormalizedFlowHyperPriorCoderPredPrior, self).__init__(**kwargs)

        if num_predprior_filters is None:  # When not specifying, it will align to num_filters
            num_predprior_filters = kwargs['num_filters']

        if self.use_mean or "Mixture" in kwargs["condition"]:
            self.pred_prior = GoogleAnalysisTransform(in_channels_predprior,
                                                      kwargs['num_features'] * self.conditional_bottleneck.condition_size,
                                                      num_predprior_filters,  # num_filters=64,
                                                      kwargs['kernel_size'],  # kernel_size=3,
                                                     )
            self.PA = nn.Sequential(
                nn.Conv2d((kwargs['num_features'] * self.conditional_bottleneck.condition_size) * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, kwargs['num_features'] * self.conditional_bottleneck.condition_size, 1)
            )
        else:
            self.pred_prior = GoogleAnalysisTransform(in_channels_predprior,
                                                      kwargs['num_features'],
                                                      num_predprior_filters,  # num_filters=64,
                                                      kwargs['kernel_size'],  # kernel_size=3,
                                                     )
            self.PA = nn.Sequential(
                nn.Conv2d(kwargs['num_features'] * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, kwargs['num_features'], 1)
            )
    
    def entropy_model(self, input, code, pred_prior_input):
        # Enrtopy coding
        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        hp_feat = self.hyper_synthesis(z_tilde)
        pred_feat = self.pred_prior(pred_prior_input)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        y_tilde, y_likelihood = self.conditional_bottleneck(code, condition=condition)

        # y_tilde = code # No quantize on z2

        return y_tilde, z_tilde, y_likelihood, z_likelihood

    def compress(self, input, cond_coupling_input=None, reverse_input=None, pred_prior_input=None, return_hat=False):
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        if pred_prior_input is None:
            pred_prior_input = cond_coupling_input
        
        code = None
        jac = None
        input, features, jac = self.encode(
            input, code, jac, visual=visual, figname=figname, cond_coupling_input=cond_coupling_input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        hp_feat = self.hyper_synthesis(z_hat)
        pred_feat = self.pred_prior(pred_prior_input)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            jac = None
            stream, y_hat = ret

            # Decode
            if not self.output_nought:
                assert not (reverse_input is None), "reverse_input should be specified"
                input = reverse_input
            else:
                input = torch.zeros_like(input)
                
            x_hat, code, jac = self.decode(
                input, y_hat, jac, cond_coupling_input=cond_coupling_input)
            if self.DQ is not None:
                x_hat = self.QE(x_hat)

            return x_hat, [stream, side_stream], [hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [hyperpriors.size()]

    def decompress(self, strings, shapes, cond_coupling_input=None, reverse_input=None, pred_prior_input=None):
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        if pred_prior_input is None:
            pred_prior_input = cond_coupling_input
        
        jac = None

        stream, side_stream = strings
        y_shape, z_shape = shapes

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        hp_feat = self.hyper_synthesis(z_hat)
        pred_feat = self.pred_prior(pred_prior_input)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, condition=condition)
        
        # Decode
        if not self.output_nought:
            assert not (reverse_input is None), "reverse_input should be specified"
            input = reverse_input
        else:
            input = torch.zeros_like(input)

        x_hat, code, jac = self.decode(
            input, y_hat, jac, cond_coupling_input=cond_coupling_input)

        if self.use_QE:
            reconstructed = self.QE(x_hat)

        return reconstructed

    def forward(self, input, code=None, jac=None 
                output=None, # Should assign value when self.output_nought==False
                cond_coupling_input=None # Should assign value when self.cond_coupling==True
                                         # When using ANFIC on residual coding, output & cond_coupling_input should both be MC frame
                pred_prior_input=None  # cond_coupling_input will replace this when None
               ):
        if not self.output_nought:
            assert not (output is None), "output should be specified"
        if self.cond_coupling:
            assert not (cond_coupling_input is None), "cond_coupling_input should be specified"
        if pred_prior_input is None:
            pred_prior_input = cond_coupling_input

        # Encode
        jac = [] if jac else None
        input, code, jac = self.encode(
            input, code, jac, visual=visual, cond_coupling_input=cond_coupling_input)

        # Enrtopy coding
        y_tilde, z_tilde, y_likelihood, z_likelihood = entropy_model(input, code, pred_prior_input)
        
        # Encode distortion
        x_2, _, jac = self['synthesis' + str(self.num_layers - 1)](input, y_tilde, jac, last_layer=True, layer=self.num_layers - 1)

        input, code, hyper_code = output, y_tilde, z_tilde  # MC frame as x_2 when decoding

        # Decode
        input, code, jac = self.decode(input, code, jac, rec_code=rec_code, cond_coupling_input=cond_coupling_input)

        if self.use_QE:
            BDQ = input
            input = self.QE(input)

        return input, (y_likelihood, z_likelihood), x_2, jac, code, BDQ

