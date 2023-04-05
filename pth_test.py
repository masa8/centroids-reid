import torch

# load the model checkpoint file
checkpoint = torch.load('./models/resnet50-19c8e357.pth')

# check if the 'state_dict' key is present in the checkpoint dictionary
if 'state_dict' in checkpoint:
    print('The checkpoint file contains a state dictionary.')
else:
    print('The checkpoint file does not contain a state dictionary.')

