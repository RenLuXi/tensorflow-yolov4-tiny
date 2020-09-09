#!/usr/bin/env python
"""
    openvino sync
"""
from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import time
from time import perf_counter

import cv2
from openvino.inference_engine import IENetwork, IECore

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-m", "--model", default='yolov4-tiny.xml',
                      help="Required. Path to an .xml file with a trained model.",
                      type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default='coco.names', type=str)

    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.3, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.3, type=float)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    return parser


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
#         self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
#                         198.0,
#                         373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]
        self.anchors = [10.0,14.0,  23.0,27.0,  37.0,58.0,  81.0,82.0,  135.0,169.0,  
                        344.0,319.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


class ObjectDetection(object):
    def __init__(self):
        self.args = build_argparser().parse_args()

        model_xml = self.args.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
        log.info("Creating Inference Engine...")
        ie = IECore()

        # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        self.net = ie.read_network(model=model_xml, weights=model_bin)

        # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------
        if "CPU" in self.args.device:
            supported_layers = ie.query_network(self.net, "CPU")
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(self.args.device, ', '.join(not_supported_layers)))
                sys.exit(1)

        assert len(self.net.input_info.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

        # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
        log.info("Preparing inputs")
        self.input_blob = next(iter(self.net.input_info))

        #  Defaulf batch_size is 1
        self.net.batch_size = 1

        if self.args.labels:
            with open(self.args.labels, 'r') as f:
                self.labels_map = [x.strip() for x in f]
        else:
            self.labels_map = None

        # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(network=self.net, num_requests=2, device_name=self.args.device)

    def inference(self, frame):
        '''

        :param frame:
        :return:
        '''
        cur_request_id = 0
        parsing_time = 0
        # ----------------------------------------------- 6. Doing inference -----------------------------------------------
        is_async_mode = False
        while cap.isOpened():
            # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
            # in the regular mode, we capture frame to the CURRENT infer request
            if not ret:
                break

            # Read and pre-process input images
            n, c, h, w = self.net.input_info[self.input_blob].input_data.shape

            request_id = cur_request_id
            in_frame = cv2.resize(frame, (w, h))

            # resize input_frame to network size
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            # Start inference
            infer_time = time()
            self.exec_net.start_async(request_id=request_id, inputs={self.input_blob: in_frame})
            # exec_net.infer(inputs={self.input_blob: in_frame})
            det_time = time() - infer_time

            # Collecting object detection results
            objects = list()
            if self.exec_net.requests[cur_request_id].wait(-1) == 0:
                output = self.exec_net.requests[cur_request_id].outputs
                # for layer_name, out_blob in output.items():
                    # print("-----------The layer name of collecting object detection results:----------")
                    # print(layer_name)
                start_time = time()
                for layer_name, out_blob in output.items():
                    # if layer_name == 'detector/yolo-v4-tiny/strided_slice/Split.0' or layer_name == 'detector/yolo-v4-tiny/strided_slice_1/Split.0' \
                    #         or layer_name == 'detector/yolo-v4-tiny/strided_slice_2/Split.0':
                    #     pass
                    # else:
                    out_blob = out_blob.reshape(self.net.layers[self.net.layers[layer_name].parents[0]].out_data[0].shape)
                    layer_params = YoloParams(self.net.layers[layer_name].params, out_blob.shape[2])
                    objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                                 frame.shape[:-1], layer_params,
                                                 self.args.prob_threshold)
                parsing_time = time() - start_time

            # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
            objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
            for i in range(len(objects)):
                if objects[i]['confidence'] == 0:
                    continue
                for j in range(i + 1, len(objects)):
                    if intersection_over_union(objects[i], objects[j]) > self.args.iou_threshold:
                        objects[j]['confidence'] = 0

            # Drawing objects with respect to the --prob_threshold CLI parameter
            objects = [obj for obj in objects if obj['confidence'] >= self.args.prob_threshold]

            if len(objects) and self.args.raw_output_message:
                log.info("\nDetected boxes for batch {}:".format(1))
                log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

            origin_im_size = frame.shape[:-1]
            for obj in objects:
                # Validation bbox of detected object
                if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                    continue
                color = (int(min(obj['class_id'] * 12.5, 255)),
                         min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
                det_label = self.labels_map[obj['class_id']] if self.labels_map and len(self.labels_map) >= obj['class_id'] else \
                    str(obj['class_id'])

                if self.args.raw_output_message:
                    log.info(
                        "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                                  obj['ymin'], obj['xmax'], obj['ymax'],
                                                                                  color))

                cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
                cv2.putText(frame,
                            "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                            (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

            # Draw performance stats over frame
            inf_time_message = "" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time * 1e3)
            async_mode_message = "sync mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                ''
            parsing_message = "parsing time is {:.3f}".format(parsing_time * 1e3)

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, async_mode_message, (10, int(origin_im_size[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)
            cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            return frame


if __name__ == '__main__':
    yolo = ObjectDetection()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    last_start_time = perf_counter()
    count_frame = 0
    fps_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        start_time = perf_counter()
        out_frame = yolo.inference(frame)
        all_time = perf_counter() - start_time
        print('The processing time of one frame is', all_time)
        cv2.imwrite("result.jpg", frame)
        count_frame = count_frame + 1
        print("FPS is", count_frame / (perf_counter() - last_start_time))
        cv2.imshow("frame", frame)
        key = cv2.waitKey(3)
        if key == 27:
            break

    cv2.destroyAllWindows()
