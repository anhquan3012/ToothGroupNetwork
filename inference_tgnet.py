import sys
import os
sys.path.append(os.getcwd())
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
from glob import glob
from predict_utils import ScanSegmentation


def inference_tgnet(input_dir, output_dir):

    dir_path = os.path.dirname(os.path.realpath(__file__))

    model_name = "tgnet"
    checkpoint_path = os.path.join(dir_path, "ckpts\\tgnet_fps")
    checkpoint_path_bdl = os.path.join(dir_path, "ckpts\\tgnet_bdl")

    stl_path_ls = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.stl')]
    
    pred_obj = ScanSegmentation(make_inference_pipeline(model_name, [checkpoint_path+".h5", checkpoint_path_bdl+".h5"]))
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(stl_path_ls)):
        print(f"Processing: ", i,":",stl_path_ls[i])
        base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
        pred_obj.process(stl_path_ls[i], os.path.join(output_dir, os.path.basename(stl_path_ls[i]).replace(".stl", ".json")))