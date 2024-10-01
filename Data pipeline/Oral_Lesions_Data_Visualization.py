from PIL import Image, ImageDraw
import glob
import os

label_path = './teeth_dataset_labels/training/labels/Image_2021-12-13 11_00_54_003.txt'

image_path = "./Dataset -teeth/clean/train/images/Image_2021-12-13 11_00_54_003.JPG"

# image_path_list = [
#     'Image_2021-12-13 09_54_56_464.JPG',
#     '',
#     '',
#     '',
#     '',
#     '',
#     '',
#     '',
#     '',
#     '',
#     '',
#     '',
#     '',
#     '',
#     ''
# ]

image_path_list = glob.glob(r'./untitled folder/test/*.JPG')
print(image_path_list)

label_path_list = ['./untitled folder/test/Image_2021-12-13 11_14_12_766.txt',
                   './untitled folder/test/Image_2021-12-13 11_14_30_958.txt',
                   './untitled folder/test/Image_2021-12-13 12_16_44_199.txt',
                   './untitled folder/test/Image_2021-12-13 11_12_48_676.txt',
                   './untitled folder/test/Image_2021-12-13 11_13_58_156.txt',
                   './untitled folder/test/Image_2021-12-13 11_14_00_505.txt',
                   './untitled folder/test/Image_2021-12-13 12_20_59_249.txt',
                   './untitled folder/test/Image_2021-12-13 11_13_26_367.txt',
                   './untitled folder/test/Image_2021-12-13 11_13_54_088.txt',
                   './untitled folder/test/Image_2021-12-13 09_54_56_464.txt',
                   './untitled folder/test/Image_2021-12-13 12_18_21_412.txt',
                   './untitled folder/test/Image_2021-12-13 11_14_34_245.txt',
                   './untitled folder/test/Image_2021-12-13 11_09_08_100.txt',
                   './untitled folder/test/Image_2021-12-13 11_13_20_878.txt',
                   './untitled folder/test/Image_2021-12-13 12_02_16_103.txt',
                   './untitled folder/test/Image_2021-12-13 12_02_20_071.txt']
# Get the labels list whose shape is (n, 5)
# n - the number of labels in a label file
# 5 - class, x, y, length, width


def get_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        num = line.split(" ")
        num[-1] = num[-1][0:-1]
        labels.append(num)
    return labels


# labels = get_labels(label_path)

class_color = {
    '0': "green",
    '1': "red",
    '2': "blue"
}


# Get the boxes whose shape is (n, 5)
# n - the number of box in an image
# 5 - x0, y0, x1, y1, class
def get_boxes(labels):
    boxes = []
    for label in labels:
        oral_class = label[0]
        x = float(label[1])
        y = float(label[2])
        l = float(label[3])
        w = float(label[4])
        box = [(x-l/2) * 480, (y-w/2) * 480, (x+l/2) * 480, (y+w/2) * 480, oral_class]
        boxes.append(box)
    return boxes


# boxes = get_boxes(labels)


def draw_bnd(boxes, image_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box[0:4], width=2, outline=class_color[box[-1]])
    image.show()


# draw_bnd(boxes, image_path)

for index in range(len(image_path_list)):
    labels = get_labels(label_path_list[index])
    boxes = get_boxes(labels)
    draw_bnd(boxes, image_path_list[index])


# images_paths = glob.glob(r'./Dataset -teeth/clean/train/images/*.JPG')
# images_paths = sorted(images_paths, key=os.path.getctime)
# print(images_paths)
#
# labels_paths = glob.glob(r'./teeth_dataset_labels/training/labels/*.txt')
# labels_paths = sorted(labels_paths, key=os.path.getctime)
# print(labels_paths)
