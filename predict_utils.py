import glob
import json
import os
import numpy as np
import traceback
import open3d as o3d
from gen_utils import read_txt_obj_ls

def get_colored_mesh(mesh, label_arr):
    palte = {
        0: [255, 255, 255],  # White
        11: [255, 153, 153],  # Light Red
        12: [153, 76, 0],     # Brown
        13: [153, 153, 0],    # Olive
        14: [76, 153, 0],     # Dark Green
        15: [0, 153, 153],    # Teal
        16: [0, 0, 153],      # Navy Blue
        17: [153, 0, 153],    # Purple
        18: [153, 0, 76],     # Dark Pink
        21: [64, 64, 0],      # Olive Drab
        22: [255, 128, 0],    # Orange
        23: [255, 0, 0],      # Red
        24: [0, 255, 0],      # Green
        25: [0, 0, 255],      # Blue
        26: [255, 255, 0],    # Yellow
        27: [255, 0, 255],    # Magenta
        28: [0, 255, 255],    # Cyan
        31: [255, 153, 153],  # Light Red
        32: [153, 76, 0],     # Brown
        33: [153, 153, 0],    # Olive
        34: [76, 153, 0],     # Dark Green
        35: [0, 153, 153],    # Teal
        36: [0, 0, 153],      # Navy Blue
        37: [153, 0, 153],    # Purple
        38: [153, 0, 76],     # Dark Pink
        41: [64, 64, 0],      # Olive Drab
        42: [255, 128, 0],    # Orange
        43: [255, 0, 0],      # Red
        44: [0, 255, 0],      # Green
        45: [0, 0, 255],      # Blue
        46: [255, 255, 0],    # Yellow
        47: [255, 0, 255],    # Magenta
        48: [0, 255, 255],    # Cyan
    }

    label_arr = label_arr.copy()
    label_colors = np.zeros((label_arr.shape[0], 3))
    for lbl in np.sort(np.unique(label_arr)):
        label_colors[label_arr==lbl] = np.array(palte[lbl])/255
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)
    return mesh

def get_mesh_of_each_tooth(mesh, label_arr, label):
    # Filter vertices
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_indices = np.where(label_arr == label)[0]
    
    # Create a mask for faces that are composed entirely of the filtered vertices
    face_mask = np.all(np.isin(faces, vertex_indices), axis=1)
    filtered_faces = faces[face_mask]
    
    
    # Map the vertex indices to the new mesh
    unique_vertex_indices, new_faces = np.unique(filtered_faces, return_inverse=True)
    new_vertices = vertices[unique_vertex_indices]
    new_faces = new_faces.reshape(filtered_faces.shape)
    new_vertex_normals = np.asarray(mesh.vertex_normals)[unique_vertex_indices]
    
    # Create a new mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
    new_mesh.vertex_normals = o3d.utility.Vector3dVector(new_vertex_normals) 
    
    return new_mesh


def get_brace_location(mesh, label_arr):
    brace_locations = {}
    for lbl in np.unique(label_arr):
        tooth_mesh = get_mesh_of_each_tooth(mesh, label_arr, lbl)
        if lbl in [11, 12, 21, 22, 31, 32, 41, 42]:
            outer_mesh = tooth_mesh.select_by_index(np.where(np.array(tooth_mesh.vertex_normals)[:,1]<=0)[0])
            center = np.mean(np.array(outer_mesh.vertices), axis=0)
            # find the vertex that is closest to the center
            closest_vertex = np.argmin(np.linalg.norm(np.array(outer_mesh.vertices)-center, axis=1))
            closest_vertex_normal = np.array(outer_mesh.vertex_normals)[closest_vertex]
            brace_locations[int(lbl)] = {"center_location": np.array(outer_mesh.vertices)[closest_vertex].tolist(), 
                                    "normal_vector": closest_vertex_normal.tolist()}
        if lbl in [13, 14, 15, 16, 17, 18, 43, 44, 45, 46, 47, 48]:
            outer_mesh = tooth_mesh.select_by_index(np.where(np.array(tooth_mesh.vertex_normals)[:,0]<=0)[0])
            # get the half most <=0 x value vertex
            if lbl in [15, 16, 17, 18, 45, 46, 47, 48]:
                outer_mesh = outer_mesh.select_by_index(np.argsort(np.array(outer_mesh.vertices)[:,0])[:len(outer_mesh.vertices)//3])

            center = np.mean(np.array(outer_mesh.vertices), axis=0)
            # find the vertex that is closest to the center
            closest_vertex = np.argmin(np.linalg.norm(np.array(outer_mesh.vertices)-center, axis=1))
            closest_vertex_normal = np.array(outer_mesh.vertex_normals)[closest_vertex]
            brace_locations[int(lbl)] = {"center_location": np.array(outer_mesh.vertices)[closest_vertex].tolist(), 
                                    "normal_vector": closest_vertex_normal.tolist()}
        if lbl in [23, 24, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38]:
            outer_mesh = tooth_mesh.select_by_index(np.where(np.array(tooth_mesh.vertex_normals)[:,0]>=0)[0])
            if lbl in [25, 26, 27, 28, 35, 36, 37, 38]:
                outer_mesh = outer_mesh.select_by_index(np.argsort(np.array(outer_mesh.vertices)[:,0])[-len(outer_mesh.vertices)//3:])
            center = np.mean(np.array(outer_mesh.vertices), axis=0)
            # find the vertex that is closest to the center
            closest_vertex = np.argmin(np.linalg.norm(np.array(outer_mesh.vertices)-center, axis=1))
            closest_vertex_normal = np.array(outer_mesh.vertex_normals)[closest_vertex]
            brace_locations[int(lbl)] = {"center_location": np.array(outer_mesh.vertices)[closest_vertex].tolist(), 
                                    "normal_vector": closest_vertex_normal.tolist()}
    return brace_locations


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ScanSegmentation():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self, model):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.chl_pipeline = model

        #self.model = load_model()
        #sef.device = "cuda"

        pass

    @staticmethod
    def load_input(input_dir):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """

        # iterate over files in input_dir, assuming only 1 file available
        inputs = glob.glob(f'{input_dir}/*.obj')
        print("scan to process:", inputs)
        return inputs

    @staticmethod
    def write_output(labels, instances, jaw, output_path):
        """
        Write to /output/dental-labels.json your predicted labels and instances
        Check https://grand-challenge.org/components/interfaces/outputs/
        """
        pred_output = {'id_patient': "",
                       'jaw': jaw,
                       'labels': labels,
                       'instances': instances
                       }

        # just for testing
        #with open('./test/test_local/expected_output.json', 'w') as fp:
        with open(output_path, 'w') as fp:
            json.dump(pred_output, fp, cls=NpEncoder)



        return

    @staticmethod
    def get_jaw(scan_path):
        try:
            # read jaw from filename
            _, jaw = os.path.basename(scan_path).split('.')[0].split('_')
        except:
            # read from first line in obj file
            try:
                with open(scan_path, 'r') as f:
                    jaw = f.readline()[2:-1]
                if jaw not in ["upper", "lower"]:
                    return None
            except Exception as e:
                print(str(e))
                print(traceback.format_exc())
                return None

        return jaw

    def predict(self, inputs):
        """
        Your algorithm goes here
        """

        try:
            assert len(inputs) == 1, f"Expected only one path in inputs, got {len(inputs)}"
        except AssertionError as e:
            raise Exception(e.args)
        scan_path = inputs[0]
        #print(f"loading scan : {scan_path}")
        # read input 3D scan .obj
        try:
            # you can use trimesh or other any loader we keep the same order
            #mesh = trimesh.load(scan_path, process=False)
            pred_result = self.chl_pipeline(scan_path)
            jaw = self.get_jaw(scan_path)
            if jaw == "lower":
                pred_result["sem"][pred_result["sem"]>0] += 20
            elif jaw=="upper":
                pass
            else:
                raise "jaw name error"
            print("jaw processed is:", jaw)
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            raise
        # preprocessing if needed
        # prep_data = preprocess_function(mesh)
        # inference data here
        # labels, instances = self.model(mesh, jaw=None)

        # extract number of vertices from mesh
        nb_vertices = pred_result["sem"].shape[0]

        # just for testing : generate dummy output instances and labels
        instances = pred_result["ins"].astype(int).tolist()
        labels = pred_result["sem"].astype(int).tolist()

        try:
            assert (len(labels) == len(instances) and len(labels) == nb_vertices),\
                "length of output labels and output instances should be equal"
        except AssertionError as e:
            raise Exception(e.args)

        return labels, instances, jaw

    def process(self, input_path, output_path):
        """
        Read input from /input, process with your algorithm and write to /output
        assumption /input contains only 1 file
        """
        #input = self.load_input(input_dir='./test/test_local')
        labels, instances, jaw = self.predict([input_path])

        # read mesh from obj file
        _, mesh = read_txt_obj_ls(input_path, ret_mesh=True, use_tri_mesh=True, creating_color_mesh=True)
        mesh = mesh.remove_duplicated_vertices()
        mesh = get_colored_mesh(mesh, np.array(labels))
        o3d.io.write_triangle_mesh(output_path.replace(".json", ".obj"), mesh)

        braces_location = get_brace_location(mesh, np.array(labels))
        # write output
        with open(output_path.replace(".json", "_braces_location.json"), 'w') as fp:
            json.dump(braces_location, fp, indent=4)

        # color the closet vertex to the center of the tooth with black
        # for lbl in braces_location.keys():
        #     center = np.array(braces_location[lbl]["center_location"])
        #     closest_vertex = np.argmin(np.linalg.norm(np.array(mesh.vertices)-center, axis=1))
        #     mesh.vertex_colors[closest_vertex] = [0, 0, 0]
        # o3d.io.write_triangle_mesh(output_path.replace(".json", "_withcenter.obj"), mesh)

        self.write_output(labels=labels, instances=instances, jaw=jaw, output_path=output_path)