import torch
from model_18_34 import Resnet18_34_Unet
from model_50_101 import Resnet50_101_Unet

def init_model(model_type='Resnet50'):
    if model_type == 'Resnet18':
        model = Resnet18_34_Unet(model_type)
    elif model_type == 'Resnet34':
        model = Resnet18_34_Unet(model_type)
    elif model_type == 'Resnet50':
        model = Resnet50_101_Unet(model_type)
    elif model_type == 'Resnet101':
        model = Resnet50_101_Unet(model_type)

    return model

if __name__=='__main__':
    model = init_model('Resnet101').cuda()
    num = sum([param.nelement() for param in model.parameters()])
    num_require_grad = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    #print(model)
    print("Number of parameter: %.5fM" % (num / 1e6))
    print("Number of require grad parameter: %.5fM" % (num_require_grad / 1e6))
    
    inp = torch.rand((2, 3, 512, 512)).cuda()
    out,_ = model(inp)
    print(out.shape)
    