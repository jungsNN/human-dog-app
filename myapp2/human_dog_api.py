import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms


def face_detector(img_path):
    """
    Takes in prepped face-cascade model and an image path and returns whether or
    not a face was detected.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def detection_rate(detector, model, img_files, device):
    """
    Do not need this function for current app.
    """
    class_idx = [0] * len(img_files)
    for i, img in enumerate(img_files):
        if detector(model, img, device):
            class_idx[i] += 1
    total_correct = np.sum(class_idx) / len(img_files)
    return total_correct

def class_predict(model, img_path, device):
    """
    Takes in the classifier model, an image path and torch device.
    Transforms the image input and returns a predicted trained id reference.
    """
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    prob = torch.exp(model(image))
    top_p, top_k = prob.topk(1)
    return int(top_k.squeeze().detach().cpu().numpy())

def dog_detector(model, img_path, label_dict, device):
    #dog_idx = range(151, 269)
    if class_predict(model, img_path, device) in label_dict.keys():
        return True
    else:
        return False

def serialize_results(model, img_path_list, label_dict, device):
    """
    Takes in single to many image paths to feed into the model
    """
    dog = "Definitely a dog!"
    human = "Hello human!"
    neither = "Must be something I can't recognize. Try other images!"
    breed = 'You look like a {}.'
    result = {}

    for i, img_path in enumerate(img_path_list):
        result[i] = ''
        dog_true = dog_detector(model, img_path, label_dict, device)
        human_true = face_detector(img_path)
        class_pred = class_predict(model, img_path, device)

        if dog_true or human_true:
            breed_name = ' '.join(label_dict[class_pred].split('_'))
            if dog_true and not human_true:
                result[i] += dog + ' ' + breed.format(breed_name)
            elif human_true and not dog_true:
                result[i] += human + ' ' + breed.format(breed_name)
            elif dog_true and human_true:
                result[i] += "Must be either human or dog! " + breed.format(breed_name)
        else:
            result[i] += neither

    return result
