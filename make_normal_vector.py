import numpy as np
from scipy.spatial.transform import Rotation as Rot
import hydra
import os
from loguru import logger
import subprocess
from tqdm import tqdm

def read_images_txt(images_path):
    if not os.path.exists(images_path):
        raise Exception(f"No such file : {images_path}")

    with open(images_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {images_path}")

    comments = lines[:4]
    contents = lines[4:]

    img_ids = []
    cam_ids = []
    img_names = []
    poses = []
    cam_position = []
    for img_idx, content in enumerate(contents[::2]):
        content_items = content.split(' ')
        img_id = content_items[0]
        q_xyzw = np.array(content_items[2:5] + content_items[1:2], dtype=np.float32) # colmap uses wxyz
        t_xyz = np.array(content_items[5:8], dtype=np.float32)
        cam_id = content_items[8]
        img_name = content_items[9]

        R = Rot.from_quat(q_xyzw).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = t_xyz

        img_ids.append(img_id)
        cam_ids.append(cam_id)
        img_names.append(img_name)
        poses.append(T)
        cam_position.append(-R.T@ t_xyz)

    return img_ids, cam_ids, img_names, poses, cam_position

def read_point3d_txt(point3d_path):
    if not os.path.exists(point3d_path):
        raise Exception(f"No such file : {point3d_path}")

    with open(point3d_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {point3d_path}")

    comments = lines[:3]
    contents = lines[3:]

    XYZs = []
    RGBs = []
    candidate_ids = {}

    for pt_idx, content in enumerate(contents):
        content_items = content.split(' ')
        pt_id = content_items[0]
        XYZ = content_items[1:4]
        RGB = content_items[4:7]
        error = content_items[7],
        candidate_id = content_items[8::2]
        XYZs.append(np.array(XYZ, dtype=np.float32).reshape(1,3))
        RGBs.append(np.array(RGB, dtype=np.float32).reshape(1, 3) / 255.0)
        candidate_ids[pt_id] = candidate_id
    XYZs = np.concatenate(XYZs, axis=0)
    RGBs = np.concatenate(RGBs, axis=0)

    return XYZs, RGBs, candidate_ids
    
def inverse_relation(candidate_img_ids):
    candidate_point_ids = {}
    pt_ids = list(candidate_img_ids.keys())

    for pt_id in pt_ids:
        candidate_img_id = candidate_img_ids[pt_id]
        for img_id in candidate_img_id:
            if img_id in list(candidate_point_ids.keys()):
                candidate_point_ids[img_id].append(pt_id)
            else:
                candidate_point_ids[img_id] = [pt_id]
    return candidate_point_ids

# make normal vector for each 3d points by averaging its camera poses
# output type is [n, 3] numpy txt by np.savetxt
# output path is /sfm_output/outputs_softmax_loftr_loftr/{names}/tkl_model/normal_vector.txt
def make_normal_vector(cfg):
    # /sfm_output/outputs_softmax_loftr_loftr
    sfm_dir = cfg.sfm_dir
    names = cfg.names

    all_data_names = os.listdir(sfm_dir)
    id2datafullname = {
        data_name[:4]: data_name for data_name in all_data_names if "-" in data_name
    }

    for name in tqdm(names):
        if len(name) == 4:
            if name in id2datafullname:
                name = id2datafullname[name]
            else:
                logger.warning(f"id {name} not exist in sfm directory")
        
        tkl_model_dir = os.path.join(sfm_dir, name, 'sfm_ws', 'model_filted_track')
        cmd = ["colmap", "model_converter", 
               "--input_path", tkl_model_dir, 
               "--output_path", tkl_model_dir, 
               "--output_type", "TXT"]
        colmap_res = subprocess.call(cmd)

        assert colmap_res==0

        img_ids, cam_ids, img_names, poses, cam_position = read_images_txt(os.path.join(tkl_model_dir, 'images.txt'))
        XYZs, _, candidate_ids = read_point3d_txt(os.path.join(tkl_model_dir, 'points3D.txt'))

        normal_vectors = []
        n_points = len(XYZs)
        pt_to_img = list(candidate_ids.values())

        for i in range(n_points):
            tmp_point = XYZs[i]
            tmp_img_ids = pt_to_img[i]
            
            sum_n = np.zeros((3))

            for img_idx in tmp_img_ids:
                if img_idx in cam_ids:
                    tmp_cam_position = cam_position[cam_ids.index(img_idx)] - tmp_point
                    sum_n = sum_n + tmp_cam_position
            
            normal_vectors.append(sum_n / len(tmp_img_ids))

        normal_vectors = np.array(normal_vectors)
        np.savetxt(os.path.join(tkl_model_dir, 'normal_vector.txt'), normal_vectors)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    make_normal_vector(cfg)


if __name__ == "__main__":
    main()