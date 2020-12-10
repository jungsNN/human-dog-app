import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import OrderedDict
import os

from human_dog_app import app as application
import human_dog_api

path = os.getcwd()
SERVE_PATH = os.path.join(path, 'mysite/static/serve')
UPLOAD_PATH = os.path.join(path, 'mysite/static/uploads')
application.config['SERVE_PATH'] = SERVE_PATH
application.config['UPLOAD_PATH'] = UPLOAD_PATH

''' [DOG LABELS] '''
# MIGHT HAVE TO CHECK IF THESE LOADS AS A WHOLE LIST ***
id_to_true_tuple = sorted([tup[0] for tup in np.load(os.path.join(application.config['SERVE_PATH'],
                                                                    'id_to_true_key.npy'))])
ID_TO_BREED = {id_to_true_tuple[i]:
                sorted(np.load(os.path.join(application.config['SERVE_PATH'], 'labels.npy')))
                for i in range(len(id_to_true_tuple))}

def ready_model(device):
    state_dict_path = os.path.join(application.config['SERVE_PATH'], 'resnet_state_dict.pt')

    resnet = torchvision.models.resnet152(pretrained=True, progress=False)
    classifier, old_classifier = resnet._modules.popitem()

    for param in resnet.parameters():
        param.requires_grad = False

    input_size = old_classifier.in_features
    #output_size = old_classifier.out_features

    new_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, 133)),
        ('output', nn.LogSoftmax(dim=1)),
    ]))
    resnet.add_module(classifier, new_classifier)
    resnet.load_state_dict(torch.load(state_dict_path, map_location=device))
    resnet.to(device)

    return resnet

def run_model():
    device = torch.device('cpu')
    model = ready_model(device)
    model.eval()
    result_path = os.path.join(application.config['SERVE_PATH'], 'results.npy')

    if len(os.listdir(application.config['UPLOAD_PATH'])) >= 1:
        img_paths = []
        for i, item in enumerate(os.listdir(application.config['UPLOAD_PATH'])):
            if not item.startswith('.') or not item.startswith('_'):
                img_paths.append(os.path.join(application.config['UPLOAD_PATH'], item))

        np.save(result_path,
                np.array([val for val in human_dog_api.serialize_results(model,
                                                                        img_paths,
                                                                        ID_TO_BREED,
                                                                        device).values()]))


if __name__ == "__main__":
    run_model()
