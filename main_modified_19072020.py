"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import logging as log

import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 120


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    
    ### Connecting to the MQTT client ###
    
    client = None
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    ### Returning the MQTT Client ###
       
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    request_id = 0
    current_person_duration = 0
    previous_person_duration = 0
    total_persons_detected = 0
    
    current_detection_counter = 0
    previous_detection_counter = 0 
    
    frame_time = 0
    frame_counter = 0
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Loading the model through infer_network's load_model method ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    
    ### Getting network input shape ###
    net_input_shape = infer_network.get_input_shape()
    
    ### Check if the input stream is webcam or an image file or a video file ### 
    ### Notifying Users if the given input is not supported ###
    
    if args.input == "CAM":
        input_type = 0
        source_mode = "WebCam"                
    elif args.input.endswith('.bmp') or args.input.endswith('.jpg') or args.input.endswith('.png'): 
        input_type = args.input
        source_mode = "Image"                
    elif args.input.endswith('.mp4'):
        input_type = args.input
        source_mode = "Video"
    else: 
        exit()
        
       
    ### Handling the input stream ###
    vid_cap = cv2.VideoCapture(args.input)
    
    vid_cap.open(args.input)
    
    input_width = int(vid_cap.get(3))
    input_height = int(vid_cap.get(4))
    
    ### Looping until video stream is completely over ###
    while vid_cap.isOpened():
        
        ### Reading frame from the video capture ###
        flag, vid_frame = vid_cap.read()
        
        if not flag: 
            break
        key_pressed = cv2.waitKey(60)
        
        frame_counter = frame_counter + 1
        ftime = time.time()
        
        ### Pre-processing the image as needed by the model ###
               
        ### Resizing the picture frame ###
        picture_frame = cv2.resize(vid_frame, (net_input_shape[3], net_input_shape[2]))
        
        ### Transposing the  picture frame ###
        picture_frame = picture_frame.transpose(2,0,1)
        
        ### Reshaping the picture frame ###
        picture_frame = picture_frame.reshape(1, *picture_frame.shape)
            
        ### Starting asynchronous inference for specified request number ###
        inference_start_time = time.time()
        infer_network.exec_net(picture_frame, request_id)    
        
        ### Waiting for the inference result ###
        if infer_network.wait() == 0:
        
            ### Getting the results of the inference request ###
            inference_end_time = time.time()
            person_detection_time = inference_end_time - inference_start_time
            
            inf_time_message = "Inference Time: {:.3f}ms".format(person_detection_time * 1000) 
            
            #inf_time_message = total_persons_detected 
            cv2.putText(vid_frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            
            infer_result = infer_network.get_output()
                            
            ### TODO: Extract any desired stats from the results ###
            conf = infer_result[0, 0, :, 2]
            
            current_detection_counter = 0
            
            for i, c in enumerate(conf):
                
                if c > prob_threshold: 
                    rect_box = infer_result[0, 0, i, 3:]
                    min_x = int(rect_box[0] * input_width)
                    min_y = int(rect_box[1] * input_height)
                    max_x = int(rect_box[2] * input_width)
                    max_y = int(rect_box[3] * input_height)
                                       
                    vid_frame = cv2.rectangle(vid_frame, (min_x, min_y), (max_x, max_y), (255,0, 0), 1)
                                       
                    current_detection_counter = current_detection_counter + 1
                   
            if current_detection_counter > previous_detection_counter:
                
                current_person_start_time = time.time()
                total_persons_detected = total_persons_detected + current_detection_counter - previous_detection_counter
                client.publish('person', payload=json.dumps({'total': total_persons_detected}))

                
                
            if current_detection_counter < previous_detection_counter:
                current_person_duration = time.time() - current_person_start_time
                client.publish('person/duration', payload=json.dumps({'duration': current_person_duration}))
                
            
            frame_time = frame_time - time.time()
            frames_per_second = frame_counter / float(frame_time)
            fps_label = "FPS : {:.2f}".format(frames_per_second)
            cv2.putText(vid_frame, fps_label,(25,100),cv2.FONT_HERSHEY_COMPLEX, 0.5,(255, 0, 0), 1)
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            
            previous_detection_counter = current_detection_counter
            client.publish('person', payload=json.dumps({'count': current_detection_counter}))
                       
                   
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(vid_frame)
        sys.stdout.flush()
        
        
    vid_cap.release()
        
    ### Destroying all CV2 Windows
        
    cv2.destroyAllWindows()
       
        
            
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    
    ### Disconnecting MQTT Client ###
    client.disconnect()

if __name__ == '__main__':
    main()
