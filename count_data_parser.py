import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import scipy.misc
import glob
import pickle
import os.path
import json
from ast import literal_eval as make_tuple
import scipy.ndimage as nimg

rnd_state = np.random.RandomState(123)


class CountCell(object):
    def __init__(self, jsondata):
        self.original_x = jsondata['position']['x']
        self.original_y = jsondata['position']['y']
        self.positive = jsondata['positive']
        self.positiveProbability = jsondata['positiveProbability']
        self.removed = jsondata['removed']
        self.modified = jsondata['modified']

    def renormalize(self, offset, scaling):
        self.x = (self.original_x*scaling) - offset[0]
        self.y = (self.original_y*scaling) - offset[1]
    
    @staticmethod
    def deserialize(list_of_json):
        return [CountCell(d) for d in list_of_json]


class CountData(object):
    def __init__(self, slide_info, count_info, stiched_image):
        self.slide_width = slide_info['baseLevel']['size']['width']
        self.slide_height = slide_info['baseLevel']['size']['height']
        self.cells = CountCell.deserialize(count_info['cells'])
        self.stiched_dimensions = stiched_image[0]
        self.image_data = stiched_image[1]
        # Compute the cell positions in relative coordinates
        # to the stiched image.
        tx = self.stiched_dimensions[0] / self.stiched_dimensions[2]
        ty = self.stiched_dimensions[1] / self.stiched_dimensions[2]
        scaling = self.slide_width / self.stiched_dimensions[2]
        for c in self.cells:
            c.renormalize((tx, ty), scaling)

    def get_annotated(self, annotation_image = None):
        if annotation_image is None:
            annotation_image = np.copy(self.image_data)
        scaling = annotation_image.shape[1]
        for c in self.cells:
            value = [1, 0, 0] if c.removed else [0, 1, 0]
            annotation_image[int(c.y*scaling), int(c.x*scaling)] = value
        return annotation_image

    def scaled_image(self, scale):
        raw_scaled = nimg.zoom(self.image_data, (scale, scale, 1))
        # Rescale so that the largest channel value is <= 1
        raw_scaled /= np.max([1.0, np.max(raw_scaled)])
        return raw_scaled
        
    def create_patches(self, patchSize, scale, balanced = False):
                
        max_per_class = len(self.cells)
        if balanced:
            removed_count = 0
            for c in self.cells:
                if c.removed:
                    removed_count += 1

            max_per_class = min(removed_count, len(self.cells) - removed_count)

        scaled_image_data = self.scaled_image(scale)
        norm_factor = scaled_image_data.shape[1]
        patches = []
        labels = []                
        
        current_active_count = 0
        current_removed_count = 0

        for c in self.cells:
            # Track the number of examples per class.
            add = True
            if c.removed:
                add = current_removed_count <= max_per_class
                current_removed_count += 1
            else:
                add = current_active_count <= max_per_class
                current_active_count += 1
            if add:
                norm_x = int(c.x*norm_factor)
                norm_y = int(c.y*norm_factor)
                start_x = int(norm_x - patchSize/2.0)
                start_y = int(norm_y - patchSize/2.0)
                end_x = int(start_x + patchSize)
                end_y = int(start_y + patchSize)
                if start_x >= 0 and start_y >= 0 and end_x < scaled_image_data.shape[1] and end_y < scaled_image_data.shape[0]:
                    patch = scaled_image_data[start_y:end_y, start_x:end_x]
                    patches.append(patch)
                    label = 0 if c.removed else 1
                    labels.append(label)
        return patches, labels


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_slide_paths(base):
    return get_immediate_subdirectories(base)

def get_source_info(dir):
    source_filename = "{0}/sourceinfo.json".format(dir)
    with open(source_filename, "r") as f:
        return json.load(f)

def get_result_directories(slidePath):
    return get_immediate_subdirectories(slidePath)

def get_patch_image(filename):
    # we expect the filename format for be patch_(x, y, w, h).png"
    a = "af"    
    dim_str = filename[filename.rindex('('):filename.rindex(')') + 1]
    dim  = make_tuple(dim_str)
    data = matplotlib.image.imread(filename)[:, :, :3] # Use the last 3
    return dim , data

def get_stiched_image(patches):
    # Each patch has the format (dims, data)
    # where dims = (x, y, w, h)
    x_s = [p[0][0] for p in patches]
    y_s = [p[0][1] for p in patches]
    x_e = [p[0][0] + p[0][2] for p in patches]
    y_e = [p[0][1] + p[0][3] for p in patches]

    min_x = np.min(x_s)
    min_y = np.min(y_s)
    max_x = np.max(x_e)
    max_y = np.max(y_e)

    result_shape =(max_y - min_y, max_x - min_x, 3)
    result = np.zeros((result_shape), dtype='float32')
    for p in patches:
        x_idx = p[0][0] - min_x
        y_idx = p[0][1] - min_y
        width = p[1].shape[1]
        height = p[1].shape[0]
        result[y_idx:y_idx + height, x_idx:x_idx + width] = p[1]

    out_patch_dim = (min_x, min_y, result.shape[1], result.shape[0])
    return (out_patch_dim, result)

def get_count_result(result_dir):
    content_filename = "{0}/content.json".format(result_dir)
    content = None
    with open(content_filename, "r") as f:
        content = json.load(f)
    
    data_pattern = "{0}/patch_*.png".format(result_dir)
    data_patches = [get_patch_image(patch_path) for patch_path in glob.glob(data_pattern)]
    stitched = get_stiched_image(data_patches)
    return content, stitched


def process_slide(slidePath, patch_size = 32, scale = 1.0, balanced = True, test_split = 0.0):
    info = get_source_info(slidePath)
    result_paths = get_result_directories(slidePath)
    slide_patches = []
    slide_labels = []
    num_test = int(test_split * len(result_paths))
    test_paths = []
    if num_test > 0:
        test_paths = result_paths[-num_test:]
        result_paths = result_paths[:-num_test]

    for result in result_paths:
        count_info, stiched_image = get_count_result(result)
        count_data = CountData(info, count_info, stiched_image)
        partial_patches, partial_labels = count_data.create_patches(patch_size, scale, balanced)
        slide_patches.extend(partial_patches)
        slide_labels.extend(partial_labels)
        
    if len(slide_patches) > 0:
        # Convert to numpy
        patch_data = np.asarray(slide_patches, dtype='float32')
        patch_labels = np.asarray(slide_labels, dtype='float32')
        patch_filename = "{0}/patches_{1}_{2:.2f}.pkl".format(slidePath, patch_size, scale)    
        label_filename = "{0}/labels_{1}_{2:.2f}.pkl".format(slidePath, patch_size, scale)
        with open(patch_filename, "wb") as f:
            pickle.dump(patch_data, f)
        with open(label_filename, "wb") as f:
            pickle.dump(patch_labels, f)

    test_image_data = []
    if len(test_paths) > 0:
        for test in test_paths:
            count_info, stiched_image = get_count_result(result)
            count_data = CountData(info, count_info, stiched_image)
            test_image = count_data.scaled_image(scale)
            test_image_data.append(test_image)
        test_filename = "{0}/test_{1:.2f}.pkl".format(slidePath, scale)
        with open(test_filename, "wb") as f:
            pickle.dump(test_image_data, f)

def process_slides(basePath, patch_size = 32, scale = 1.0, balanced = True, test_split=0.0):
    slide_paths = get_slide_paths(basePath)
    for slide in slide_paths:
        process_slide(slide, patch_size, scale, balanced, test_split)


def load_from_extract_path(basePath, pattern="*"):
    return load_slides(get_slide_paths(basePath), pattern=pattern)

def test_from_extract_path(basePath, pattern="*"):
    return load_test_data(get_slide_paths(basePath), pattern=pattern)

# Load slide data from all listed directories
def load_slides(list_of_paths, pattern="*"):    
    data = []
    labels = []
    for current_path in list_of_paths:
        patch_pattern = "{0}/patches_{1}.pkl".format(current_path, pattern)
        label_pattern = "{0}/labels_{1}.pkl".format(current_path, pattern)
        for file in glob.glob(patch_pattern):
            with open(file, "rb") as f:
                data.append(pickle.load(f))
        for file in glob.glob(label_pattern):
            with open(file, "rb") as f:
                labels.append(pickle.load(f))
    if len(data) > 1:
        data = np.concatenate(data)
        labels = np.concatenate(labels)
    elif len(data) == 1:
        data = data[0]
        labels = labels[0]

    return data, labels             

def load_test_data(list_of_paths, pattern="*"):
    data = []
    labels = []
    for current_path in list_of_paths:
        test_pattern = "{0}/test_{1}.pkl".format(current_path, pattern)        
        for file in glob.glob(test_pattern):
            with open(file, "rb") as f:
                data.extend(pickle.load(f))

    return data

def test():
    data_path = r"C:\work\PathologyCore\Tools\CellCountingEvaluator\bin\Debug\extract_test"    
    patch_size = 32
    scale = 0.5
    process_slides(data_path, patch_size, scale,balanced = True, test_split=0.35)

    #data, labels = load_from_extract_path(data_path)
    #print(data.shape)


if __name__ == "__main__":
    test()