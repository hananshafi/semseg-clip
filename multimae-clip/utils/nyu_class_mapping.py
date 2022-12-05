import torch
import numpy



class_mapping = {0: 'wall',
 1: 'floor',
 2: 'cabinet',
 3: 'bed',
 4: 'chair',
 5: 'sofa',
 6: 'table',
 7: 'door',
 8: 'window',
 9: 'bookshelf',
 10: 'picture',
 11: 'counter',
 12: 'blinds',
 13: 'desk',
 14: 'shelves',
 15: 'curtain',
 16: 'dresser',
 17: 'pillow',
 18: 'mirror',
 19: 'floor mat',
 20: 'clothes',
 21: 'ceiling',
 22: 'books',
 23: 'fridge',
 24: 'TV',
 25: 'paper',
 26: 'towel',
 27: 'shower curtain',
 28: 'box',
 29: 'white board',
 30: 'person',
 31: 'night-stand',
 32: 'toilet',
 33: 'sink',
 34: 'lamp',
 35: 'bathtub',
 36: 'bag',
 37: 'other structure',
 38: 'other furniture',
 39: 'other prop'}


def get_class_names(image_tensor):
    n = image_tensor.shape[0]
    class_text_list = []
    for i in range(n):
    	unique_classes = image_tensor[i].cpu().unique().numpy().tolist()
    	unique_classes = list(map(int, unique_classes))
    	unique_classes = sorted([class_ for class_ in unique_classes if class_ not in [-1,255] ])
    	class_text = ", ".join([class_mapping[k] for k in unique_classes])
    	class_text_list.append(class_text)
    return class_text_list
    