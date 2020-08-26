# coding: utf-8

import sys
sys.path.insert(0, "../../")

from convert import convert

input_dict = {
        "input_1": [1,1,50],
        "State_c0": [1,80],
        "State_h0": [1,80],
        "State_c1": [1,30],
        "State_h1": [1,30]
        }
outputs = ["State_c0_out:0","State_h0_out:0","State_c1_out:0","State_h1_out:0","dense:0"]

in_model = "model.h5"
out_model = "model.pb"

convert(input_dict, outputs, in_model, out_model)

