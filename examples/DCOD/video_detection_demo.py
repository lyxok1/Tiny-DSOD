import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import os
import cv2
import argparse
import time
# %matplotlib inline

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'

# Make sure that caffe is on the python path:
import os
import sys
sys.path.append('./python')

import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def parse_args():
    parser = argparse.ArgumentParser("video detector")
    parser.add_argument('-model',dest='model',help='path to model prototxt file',type=str)
    parser.add_argument('-weights',dest='weights', help='path to weight file',type=str)
    parser.add_argument('-video', dest='video', help='path to input video',type=str)
    parser.add_argument('-gpu', dest='gpu', help='specifiy using GPU or not', action='store_true')
    parser.add_argument('-threshold', dest='threshold', help='threshold to filter bbox with low confidence', type=float, default=0.3)
    return parser.parse_args()

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def get_color_map(num_classes):

    color_unit = int(255/num_classes)

    colors = []
    for i in range(num_classes):
        if colors == []:
            colors.append((color_unit, 0, 0))
        else:
            last = colors[-1]
            colors.append((last[2],last[0]+2*color_unit,last[1]+color_unit))

    return colors

if __name__ == '__main__':

    num_classes = 21

    args = parse_args()
    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()
    #Load the net in the test phase for inference, and configure input preprocessing.
    model_def = args.model
    model_weights = args.weights

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': ((1,)+net.blobs['data'].data.shape[1:])})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    #transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # DSOD detection

    # set net to batch size of 1
    image_resize_h = 300
    image_resize_w = 300
    batch_size = 8
    net.blobs['data'].reshape(batch_size, 3, image_resize_h, image_resize_w)

    # set colors
    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    colors = get_color_map(num_classes)

    #Load image.
    video_dir = args.video
    
    total_time = 0.0

    batch_list = []

    cap = cv2.VideoCapture(args.video)

    end = False
    while cap.isOpened():

        images = []
        for i in range(batch_size):
            ret, img = cap.read()
            if not ret:
                # drop the last batch
                end = True
                break
            else:
                images.append(img)

        if end:
            break
        else:
            batch = np.stack(images, axis=0) # stack images in a batch to form a 4-dimensional tensor
            batch_list.append(batch)

    print('finish reading videos...')

    output = []

    for idx, batch in enumerate(batch_list):
        
        #Run the net and examine the top_k results

        for l in range(batch.shape[0]):
            net.blobs['data'].data[l,:,:,:] = transformer.preprocess('data', batch[l], copy=True)

        # Forward pass.
        t1 = time.time()
        detections = net.forward()['detection_out']
        time_increment = time.time() - t1
        fps = batch_size/time_increment

        # Parse the outputs.
        det_item = detections[0,0,:,0]
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [j for j, conf in enumerate(det_conf) if conf >= args.threshold]

        top_item = det_item[top_indices]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        origin = np.array(batch, dtype=np.uint8)

        for k in range(top_conf.shape[0]):
            item = int(top_item[k])
            xmin = int(round(top_xmin[k] * batch.shape[2]))
            ymin = int(round(top_ymin[k] * batch.shape[1]))
            xmax = int(round(top_xmax[k] * batch.shape[2]))
            ymax = int(round(top_ymax[k] * batch.shape[1]))
            score = top_conf[k]
            label = int(top_label_indices[k])
            label_name = top_labels[k]
            display_txt = '%s: %.2f'%(label_name, score)
            p1 = (xmin, ymin)
            p2 = (xmax, ymax)
            org = (xmin, ymin - 10)
            color = colors[label]

            cv2.rectangle(origin[item], p1, p2, color=color, thickness=4)
            cv2.putText(origin[item], display_txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        fps_str = 'fps: %.1f' % fps
        for i in range(batch_size):
            cv2.putText(origin[i], fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 5)
            output.append(origin[i])

        print('finish processing batch %d' % idx)
        
    fps = (batch_size*len(batch_list))/total_time

    output_dir, name = os.path.split(args.video)
    output_path = os.path.join(output_dir, 'out_'+name)

    output_size = (output[0].shape[0], output[0].shape[1])
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    for image in output:
        writer.write(image)
        time.sleep(1/fps)

