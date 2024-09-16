import numpy as np

LABEL_MAP = np.asarray(
    [0, 0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8, 9, 10]
    + [11, 12, 13, 14, 15]
    + [0] * 6
    + [1, 16, 0, 17]
    + [0] * 12
    + [18, 19, 20, 21, 0, 22, 23]
    + [0, 24, 25, 26, 27, 28, 29, 0, 0, 18, 30, 0, 31]
    + [0] * 75
    + [3, 4]
    + [0] * 25
    + [20, 21]
    + [0] * 366,
    dtype="int",
).astype(np.uint8)

def preprocess_label(lab, label_map=LABEL_MAP):
    return label_map[lab].astype(np.uint8)

def preprocess_image_min_max(img: np.ndarray):
    "Min max scaling preprocessing for the range 0..1"
    img = (img - img.min()) / (img.max() - img.min())
    return img

def preprocessing_pipe(data):
    """ Set up your preprocessing options here, ignore if none are needed """
    img, lab = data
    print("I'm in preprocessing pipe")
    img = preprocess_image_min_max(img) * 255
    print("Img is preprocessed")
    img = img.astype(np.uint8)
    print("img is being converted to uint8")
    lab = preprocess_label(lab)
    print("lab is being preprocessed")
    lab = lab.astype(np.uint8)
    print("lab being converted to uint8")
    return (img, lab)
