#we are using tensorflow for detecting the objects
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import cv2
sys.path.append("..") #appending the path outside object_detection 
from utils import label_map_util
from utils import visualization_utils as vis_util
cap = cv2.VideoCapture(0) 



tar_fil = tarfile.open('E:/Tensorflow/ssd_mobilenet_v1_coco_11_06_2017.tar.gz')
for file in tar_fil.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_fil.extract(file, os.getcwd())
 
    
    
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile('ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph, name='')
 
    
    
label_map = label_map_util.load_labelmap(os.path.join('data', 'mscoco_label_map.pbtxt'))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 


with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    while True:
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object detection', cv2.resize(image_np, (640,480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
          cap.release()
          cv2.destroyAllWindows()
          break