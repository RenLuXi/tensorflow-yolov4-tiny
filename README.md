# tensorflow-yolov4-tiny

Adapted from https://github.com/mystic123/tensorflow-yolo-v3

Tested on Python 3.6, tensorflow 1.14.0 on Ubuntu 18.04

## Todo list:
- [x] Weights converter to pb
- [x] Pb converter to IR

## How to work:
1. Download COCO class names file: 
`wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download v4-Tiny weights:    
`wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights` 
3. Run `python convert_weights_pb.py`    
4. Pb converter to IR
`cp ./yolo_v4_tiny.json  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf`
`cd /opt/intel/openvino/deployment_tools/model_optimizer`
`python mo_tf.py --input_model yolov4-tiny.pb --tensorflow_use_custom_operations_config extensions/front/tf/yolo_v4_tiny.json --batch 1`
5. Openvino-Object Detection YOLO\*  Python Demo
`python sync_detection_yolo.py`

####Optional Flags
1. convert_weights_pb.py:
    1. `--class_names`
        1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file    
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--tiny`
        1. Use yolov4-tiny
    6. `--output_graph`
        1. Location to write the output .pb graph to
2. sync_detection_yolo.py:
    1. `-m`
        1. Path to an .xml file with a trained model.
    2. `-labels`
        1. Path to the coco.names
 
####  issue
When I run `python sync_detection_yolo.py`

Got an error :
`out_blob = out_blob.reshape(self.net.layers[self.net.layers[layer_name].parents[0]].out_data[0].shape)
KeyError: 'detector/yolo-v4-tiny/split.0'`

If I ignore these three layers(split.0,split_1.0,split_2.0), there is no detection object.