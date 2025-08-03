import sys
import os
sys.path.append(os.getcwd())
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
from glob import glob
from predict_utils import ScanSegmentation
import multiprocessing as mp
import torch

def process_scan_worker(args):
    """Worker function for processing a single scan"""
    scan_path, output_path, scan_type, model_name, checkpoint_paths, gpu_id = args
    
    try:
        # Set GPU device in the worker process (if available)
        if torch.cuda.is_available() and gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            print(f"Worker for {scan_type} using GPU {gpu_id}")
        
        # Create prediction object in worker process
        pred_obj = ScanSegmentation(make_inference_pipeline(model_name, checkpoint_paths))
        
        print(f"Processing {scan_type.title()} Scan: {scan_path}")
        pred_obj.process(scan_path, output_path, scan_type)
        
        # Clear GPU cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Completed {scan_type.title()} Scan processing")
        return True
        
    except Exception as e:
        print(f"Error processing {scan_type} scan: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

def validate_inputs(lower_scan, upper_scan):
    """Validate input scans"""
    if not lower_scan and not upper_scan:
        raise ValueError("At least one scan (lower or upper) must be provided")
    
    if lower_scan and lower_scan.strip() == "null":
        lower_scan = None
    if upper_scan and upper_scan.strip() == "null":
        upper_scan = None
        
    if not lower_scan and not upper_scan:
        raise ValueError("At least one scan (lower or upper) must be provided")
    
    # Check file existence for non-null scans
    if lower_scan and not os.path.exists(lower_scan):
        raise FileNotFoundError(f"Lower scan file not found: {lower_scan}")
    if upper_scan and not os.path.exists(upper_scan):
        raise FileNotFoundError(f"Upper scan file not found: {upper_scan}")
    
    return lower_scan, upper_scan

def inference_tgnet(lower_scan, upper_scan, output_dir):
    """
    Process dental scans using TGNet model with multiprocessing support
    
    Args:
        lower_scan (str or None): Path to lower scan STL file
        upper_scan (str or None): Path to upper scan STL file  
        output_dir (str): Output directory for results
    """
    # Validate inputs
    lower_scan, upper_scan = validate_inputs(lower_scan, upper_scan)
    
    # Setup paths
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_name = "tgnet"
    checkpoint_path = os.path.join(dir_path, "ckpts", "tgnet_fps")
    checkpoint_path_bdl = os.path.join(dir_path, "ckpts", "tgnet_bdl")
    checkpoint_paths = [checkpoint_path + ".h5", checkpoint_path_bdl + ".h5"]
    
    # Verify checkpoint files exist
    for ckpt in checkpoint_paths:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Determine GPU allocation
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Available GPUs: {num_gpus}")
    
    # Prepare tasks only for non-null scans
    tasks = []
    
    if lower_scan:
        print(f"Preparing lower scan: {lower_scan}")
        lower_output = os.path.join(output_dir, os.path.basename(lower_scan).replace(".stl", ".json"))
        lower_args = (
            lower_scan,
            lower_output,
            "lower",
            model_name,
            checkpoint_paths,
            0 if num_gpus > 0 else None
        )
        tasks.append(("lower", lower_args))
    
    if upper_scan:
        print(f"Preparing upper scan: {upper_scan}")
        upper_output = os.path.join(output_dir, os.path.basename(upper_scan).replace(".stl", ".json"))
        upper_args = (
            upper_scan,
            upper_output,
            "upper",
            model_name,
            checkpoint_paths,
            (1 if num_gpus > 1 else 0) if num_gpus > 0 else None
        )
        tasks.append(("upper", upper_args))
    
    # Process based on number of tasks
    if len(tasks) == 1:
        # Process single scan (no multiprocessing needed)
        scan_type, args = tasks[0]
        print(f"Processing single {scan_type} scan...")
        success = process_scan_worker(args)
        if success:
            print(f"{scan_type.title()} scan processed successfully!")
        else:
            raise RuntimeError(f"Failed to process {scan_type} scan")
            
    elif len(tasks) == 2:
        # Process both scans concurrently
        print("Starting concurrent processing...")
        
        try:
            # Create processes
            lower_process = mp.Process(target=process_scan_worker, args=(tasks[0][1],))
            upper_process = mp.Process(target=process_scan_worker, args=(tasks[1][1],))
            
            # Start processes
            lower_process.start()
            upper_process.start()
            
            # Wait for completion
            lower_process.join()
            upper_process.join()
            
            # Check exit codes
            success = True
            if lower_process.exitcode != 0:
                print(f"Lower scan processing failed with exit code: {lower_process.exitcode}")
                success = False
            if upper_process.exitcode != 0:
                print(f"Upper scan processing failed with exit code: {upper_process.exitcode}")
                success = False
            
            if success:
                print("Both scans processed successfully!")
            else:
                raise RuntimeError("One or more scans failed to process")
                
        except Exception as e:
            print(f"Error during multiprocessing: {e}")
            # Clean up processes if they're still running
            if 'lower_process' in locals() and lower_process.is_alive():
                lower_process.terminate()
                lower_process.join()
            if 'upper_process' in locals() and upper_process.is_alive():
                upper_process.terminate()
                upper_process.join()
            raise e

def inference_tgnet_sequential(lower_scan, upper_scan, output_dir):
    """
    Sequential version (no multiprocessing) for comparison/debugging
    """
    # Validate inputs
    lower_scan, upper_scan = validate_inputs(lower_scan, upper_scan)
    
    # Setup (same as multiprocessing version)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_name = "tgnet"
    checkpoint_path = os.path.join(dir_path, "ckpts", "tgnet_fps")
    checkpoint_path_bdl = os.path.join(dir_path, "ckpts", "tgnet_bdl")
    checkpoint_paths = [checkpoint_path + ".h5", checkpoint_path_bdl + ".h5"]
    
    pred_obj = ScanSegmentation(make_inference_pipeline(model_name, checkpoint_paths))
    os.makedirs(output_dir, exist_ok=True)

    # Process scans sequentially
    if lower_scan:
        print(f"Processing Lower Scan: {lower_scan}")
        lower_output = os.path.join(output_dir, os.path.basename(lower_scan).replace(".stl", ".json"))
        pred_obj.process(lower_scan, lower_output, "lower")
        print("Lower scan completed")

    if upper_scan:
        print(f"Processing Upper Scan: {upper_scan}")
        upper_output = os.path.join(output_dir, os.path.basename(upper_scan).replace(".stl", ".json"))
        pred_obj.process(upper_scan, upper_output, "upper")
        print("Upper scan completed")
    
    print("All scans processed successfully!")

