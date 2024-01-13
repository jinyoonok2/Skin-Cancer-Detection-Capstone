from ultralytics import YOLO

import os
import numpy as np
import torch
import gc
import glob
import heapq
import re
import shutil
class ImageDetails:
    def __init__(self, result):
        self.conf = torch.max(result.boxes.conf).item()  # Get the maximum confidence score
        self.img_file = re.search(r'/([^/]+)\.jpg', result.path).group(1)  # Extract file name from path


def __lt__(self, other):
        return self.conf > other.conf  # Changed to select the best (highest) confidence

class ModelHandler:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.heap_size = {}

    def train(self, epoch, exp_name, imgsz=256):
        # train with current model_path, then change the model path to EXP_NAME model.
        model = YOLO(self.model_path)
        model.train(data=self.data_path, epochs=epoch, name=exp_name, imgsz=imgsz, device=0)

        # Update model_path to point to the newly trained model
        model_path = f"runs/segment/{exp_name}/weights/best.pt"
        self.model_path = os.path.join(os.getcwd(), model_path)

        torch.cuda.empty_cache()
        gc.collect()

    def infer(self, img_path, output_path, label_map, tr, split_rate=0.8, batch_size=100):
        plants = list(label_map.values())
        model = YOLO(self.model_path)
        heaps = {ptype: [] for ptype in plants}

        # Create train and valid directories
        train_path = os.path.join(output_path, 'train')
        valid_path = os.path.join(output_path, 'valid')
        os.makedirs(os.path.join(train_path, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(train_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(valid_path, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(valid_path, 'images'), exist_ok=True)

        for ptype in plants:
            plant_path = os.path.join(img_path, ptype)
            img_files = glob.glob(os.path.join(plant_path, "*.jpg"))

            if ptype not in self.heap_size:
                self.heap_size[ptype] = int(len(img_files) * (tr / 100))

            for i in range(0, len(img_files), batch_size):
                batch_files = img_files[i:i + batch_size]
                results = model.predict(batch_files, device=0)

                for result in results:
                    if result.masks is not None:
                        detail = ImageDetails(result)
                        heaps[ptype].append(detail)
                        if len(heaps[ptype]) > self.heap_size[ptype]:
                            heapq.heappop(heaps[ptype])

        # Splitting into training and validation sets and saving labels and images
        for ptype, heap in heaps.items():
            train_count = int(split_rate * len(heap))
            valid_count = len(heap) - train_count

            for i, detail in enumerate(heap):
                result = model.predict([detail.img_file])[0]  # Predict again to get the result object
                dataset_type = 'train' if i < train_count else 'valid'
                label_path = os.path.join(output_path, dataset_type, 'labels', detail.img_file + '.txt')
                img_dest = os.path.join(output_path, dataset_type, 'images', detail.img_file + '.jpg')

                result.save_txt(txt_file=label_path)  # Save label file
                shutil.move(result.path, img_dest)    # Move image file

        torch.cuda.empty_cache()
        gc.collect()

    def val(self, TEST_DATA_PATH):
        model = YOLO(self.model_path)
        metrics = model.val(data=TEST_DATA_PATH, device=0)
        torch.cuda.empty_cache()
        gc.collect()
        return metrics

# Now, instead of passing data to model directly, we are going to use predict method instead. predict method also returns results object like above.
# If we pass list of image path to predict, it will return a list of result objects.
# now using this iterate, we will adjust our existing code.
# 1. result.save_txt(txt_file=) allows us to select the label file path we want to save.
# we dont need to call masks and cls any longer since we can just save the label file using this function.
# if you input the string path you want, it will save according to that path.
#
# 2. we also do not need to manipulate label files name to move the image file in the active learning folders. result.path gives you the path of the image used for the inference.
#
# 3. we still need conf, since this is the standard we use to select the inferences results in the heap.
#
# tensor([0.5092], device='cuda:0')
# datasets/HAM10000/active_learning/BKL/ISIC_0024324-BKL.jpg
#
# this is example it prints, save_txt does not return anything, since it is just saving process
# but boxes.conf return the result of the confidence socre in tensor form
# and path just return the path of the image for inference as string.
#
# 4. Now we will modify the code in this way:
# we are doing the same inference on img_path and same selection algorithm using confidence score, but instead we will be using the best confidence instead of the least confidence.
# perform inference every image in every class in img_path, put them in heap for the top best confidence results.
#
# I think you also need to modify image details class for it, we dont need mask and cls in it any longer. we will just save the result object itself.
# we only additionally need img_file to save the name of the image.
# if image path was img path + class name + ISIC_0024324-BKL.jpg,
# we only take ISIC_0024324-BKL and use it for save_txt later on to save it as ISIC_0024324-BKL.txt in the wanted path.
# you might be able to do so by getting img_file variable using regex something like like */{img_file}.jpg from the result.path and save it for the object
#
# Now after you collected heap size of imagedetails objects for each class, we should start saving the label and image pairs.
# I will modify the parameters of infer function now on. add train_dir, valid_dir and split_rate
# first, split them in train and valid with given split_rate (default is 0.8 for train).
# now we will save masks of results to train/labels folder, save valid results for valid/labels folder using save_txt function of the result object.
# their names will be like train_dir + labels + img_file + .txt
# for valid list, it will be like valid_dir + labels + img_file + .txt
# after saving them, now move images, using result.path, move it from there to train_dir + images.
# repeat this process for every class.
#
# Now can you implement the modified infer function and imagedetails class for me?

