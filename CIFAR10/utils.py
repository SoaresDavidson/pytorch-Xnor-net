import torch
import torch.nn as nn
import torch.nn.functional as F

class BinFunction(torch.autograd.Function): #função de ativação
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        input, = ctx.saved_tensors
        # print(grad_output)
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class Binarize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return BinFunction.apply(input)
    
class BinWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        dim = input.size()
        with torch.no_grad():
            negMean = input.mean(1, keepdim=True).mul(-1).expand_as(input)
            input.add_(negMean)
            
            input.copy_(input.clamp(-1.0, 1.0))

            if len(dim) == 4:
                alpha = input.abs()\
                        .mean(dim=(1,2,3), keepdim=True)\
                        .expand(dim).clone()
            else:
                alpha = input.abs().mean(1, keepdim=True).expand(dim).clone()

        ctx.save_for_backward(input, alpha)

        return input.sign().mul(alpha)  
       
    @staticmethod
    def backward(ctx, *grad_output):
        input, alpha, = ctx.saved_tensors
        dim = input.size()

        alpha[grad_output[0].lt(-1.0)] = 0 #type: ignore
        alpha[grad_output[0].gt(1.0)] = 0 #type: ignore

        d = torch.ones(dim).mul(input.sign())
        return alpha + d
   
class Conv2dBinary(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
    
    def forward(self, input):
        BinWeight(self.weight)
        output = F.conv2d(
            input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups 
        ) 
        return output

class LinearBinary(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)


    def forward(self, input):
        BinWeight(self.weight)
        output = F.linear(input, self.weight, self.bias) 
        return output
    