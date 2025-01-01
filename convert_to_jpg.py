import os
from PIL import Image

dataset_path = 'dataset'

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            if img_name.endswith('.jpeg'):
                img_path = os.path.join(folder_path, img_name)
                img = Image.open(img_path)
                
                new_img = os.path.splitext(img_path)[0] + '.jpg'
                img.convert('RGB').save(new_img, 'JPEG')
                os.remove(img_path)