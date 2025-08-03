import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.getcwd())
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
from glob import glob
from predict_utils import ScanSegmentation

def process_scan(pred_obj, scan_path, output_path, scan_type):
    """Process a single scan"""
    try:
        print(f"Processing {scan_type} Scan: {scan_path}")
        pred_obj.process(scan_path, output_path, scan_type)
        print(f"Completed {scan_type} Scan: {scan_path}")
        return True
    except Exception as e:
        print(f"Error processing {scan_type} scan {scan_path}: {str(e)}")
        return False

def inference_tgnet(lower_scan, upper_scan, output_dir):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    model_name = "tgnet"
    checkpoint_path = os.path.join(dir_path, "ckpts\\tgnet_fps")
    checkpoint_path_bdl = os.path.join(dir_path, "ckpts\\tgnet_bdl")
    
    # Create prediction object
    pred_obj = ScanSegmentation(make_inference_pipeline(model_name, [checkpoint_path+".h5", checkpoint_path_bdl+".h5"]))
    os.makedirs(output_dir, exist_ok=True)

    # Prepare scan processing tasks
    tasks = []
    
    if lower_scan != 'null':
        lower_output = os.path.join(output_dir, os.path.basename(lower_scan).replace(".stl", ".json"))
        tasks.append((pred_obj, lower_scan, lower_output, "lower"))
    
    if upper_scan != 'null':
        upper_output = os.path.join(output_dir, os.path.basename(upper_scan).replace(".stl", ".json"))
        tasks.append((pred_obj, upper_scan, upper_output, "upper"))
    
    # Process scans concurrently
    if tasks:
        print(f"Starting processing of {len(tasks)} scan(s) using multi-threading...")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit all tasks
            future_to_scan = {
                executor.submit(process_scan, *task): task[3] for task in tasks
            }
            
            # Wait for completion and collect results
            results = {}
            for future in as_completed(future_to_scan):
                scan_type = future_to_scan[future]
                try:
                    results[scan_type] = future.result()
                except Exception as e:
                    print(f"Exception occurred for {scan_type} scan: {str(e)}")
                    results[scan_type] = False
        
        # Print summary
        print("\nProcessing Summary:")
        for scan_type, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {scan_type.upper()} scan: {status}")
    else:
        print("No valid scans to process.")
