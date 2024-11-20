import __init__
import vsrl_utils as vu
import numpy as np
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

## TODO
"""YOU change path to your all images here"""
base="/data1/tangjq/COCO_data/all_data/"

def draw_bbox(plt, ax, rois, fill=False, linewidth=2, edgecolor=[1.0, 0.0, 0.0], **kwargs):
    for i in range(rois.shape[0]):
        roi = rois[i,:].astype(np.int32)
        ax.add_patch(plt.Rectangle((roi[0], roi[1]),
            roi[2] - roi[0], roi[3] - roi[1],
            fill=False, linewidth=linewidth, edgecolor=edgecolor, **kwargs))

def subplot(plt, Y, X, sz_y=10, sz_x=10):
    plt.rcParams['figure.figsize'] = (X * sz_x, Y * sz_y)
    fig, axes = plt.subplots(Y, X)
    return fig, axes

# Load COCO annotations for V-COCO images
coco = vu.load_coco()

# Load the VCOCO annotations for vcoco_train image set
vcoco_all = vu.load_vcoco('vcoco_train')
for x in vcoco_all:
    x = vu.attach_gt_boxes(x, coco)

# Action classes and roles in V-COCO
classes = [x['action_name'] for x in vcoco_all]
for i, x in enumerate(vcoco_all):
    print ('{:>20s}'.format(x['action_name']), x['role_name'])

cls_id = classes.index('hit')
vcoco = vcoco_all[cls_id]

np.random.seed(1)
positive_index = np.where(vcoco['label'] == 1)[0]
positive_index = np.random.permutation(positive_index)

# Set the path where you want to save the images
save_dir = "./"  # Replace with the actual path where you want to save images

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cc = plt.get_cmap('hsv', lut=4)

for i in range(5):
    id = positive_index[i]

    # load image
    coco_image = coco.loadImgs(ids=[vcoco['image_id'][id][0]])[0]
    im = np.asarray(Image.open(os.path.join(base, coco_image['file_name'])))
    
    sy = 4.
    sx = float(im.shape[1]) / float(im.shape[0]) * sy
    fig, ax = subplot(plt, 1, 1, sy, sx)
    ax.set_axis_off()
    ax.imshow(im)

    # Draw bounding box for agent
    draw_bbox(plt, ax, vcoco['bbox'][[id], :], edgecolor=cc(0)[:3])
    role_bbox = vcoco['role_bbox'][id, :] * 1.
    role_bbox = role_bbox.reshape((-1, 4))
    for j in range(1, len(vcoco['role_name'])):
        if not np.isnan(role_bbox[j, 0]):
            draw_bbox(plt, ax, role_bbox[[j], :], edgecolor=cc(j)[:3])

    # Save the image instead of displaying it
    save_path = os.path.join(save_dir, f"image_{i}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free up memory