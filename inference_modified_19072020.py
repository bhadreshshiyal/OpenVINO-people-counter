#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        
        ### TODO: Initialize any class variables desired ###
        
        self.plugin = None
        self.network = None 
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
                
    def load_model(self, model, device="CPU", cpu_extension="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so "):
        
        ### Loading the model ###
        
        model_xml = model
        model_bin = os.path.splittext(model_xml)[0] + ".bin"
        self.plugin = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)
                
        ### Checking for supported layers ###
        
        if not all_supported_layers(self.plugin, self.network):
            self.plugin.add_extension(cpu_extension, device)
        
        self.exec_network = self.plugin.load_network(self.network, device)
            
        ### Addingextensions ###
        
        if cpu_extension and "CPU" in device: 
            self.plugin.add_extension(cpu_extension, device)
                    
        ### Returning the loaded inference plugin ###
        
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob  = next(iter(self.network.outputs))
               
        return

    def get_input_shape(self):
        
        ### Returning the shape of the input layer ###
        
        return self.network.inputs[self.input_blob].shape
                
    def exec_net(self, net_input, request_id):
        
        ### Starting an asynchronous request ###
        
        self.exec_network.start_async(request_id, inputs={self.input_blob: net_input})
               
        ### Returning information ###
        
        return

    def wait(self):
        
        ### Waiting for the request to be completed. ###
        
        request_status = self.exec_network.request[0].wait(-1)
                
        ### Returning request status ###
        
        return request_status

    def get_output(self):
        
        ### Extracting and returning the output results ###
        
        return self.exec_network.request[0].outputs[self.output_blob]
    
    def all_supported_layers(inference_engine, network):
        
        ### Checking for Supported and unsupported network layers ###
        
        supported_layers = inference_engine.query_network(network, device_name="CPU")
        all_layers = network.layers.keys()
        
        supported_layers_all = True
        
        ### Iterating through all layers to check which are supported and which are not supported ###
        
        for layer in layers: 
            
            if layer not in supported_layers:
                supported_layers_all = False
        
        ### Returning all supported layers ###
        
        return supported_layers_all
            
            
        
        