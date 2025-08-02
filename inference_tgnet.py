import sys
import os
sys.path.append(os.getcwd())
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
from glob import glob
from predict_utils import ScanSegmentation


def inference_tgnet(lower_scan, upper_scan, output_dir):

    dir_path = os.path.dirname(os.path.realpath(__file__))

    model_name = "tgnet"
    checkpoint_path = os.path.join(dir_path, "ckpts\\tgnet_fps")
    checkpoint_path_bdl = os.path.join(dir_path, "ckpts\\tgnet_bdl")
    
    pred_obj = ScanSegmentation(make_inference_pipeline(model_name, [checkpoint_path+".h5", checkpoint_path_bdl+".h5"]))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing Lower Scan: {lower_scan}")
    pred_obj.process(lower_scan, os.path.join(output_dir, os.path.basename(lower_scan).replace(".stl", ".json")), "lower")

    print(f"Processing Upper Scan: {upper_scan}")
    pred_obj.process(upper_scan, os.path.join(output_dir, os.path.basename(upper_scan).replace(".stl", ".json")), "upper")