import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction


class PointNet2_model(nn.Module):
    def __init__(self,normal_channel=True):
        super(PointNet2_model, self).__init__()
        in_channel = 128 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=5000, radius=0.2, nsample=256, in_channel=in_channel+3, mlp=[128, 128, 256], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=256, mlp=[256, 256, 256], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=128, in_channel=256+3, mlp=[256, 256, 256], group_all=False)



    def forward(self, data):
        """
        Update:
            data (dict): {
                keypoints3d: [N, n1, 3]
                descriptors3d_db: [N, dim, n1]
                scores3d_db: [N, n1, 1]

                query_image: (N, 1, H, W)
                query_image_scale: (N, 2)
                query_image_mask(optional): (N, H, W)
            }
        """
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = data['keypoints3d'].permute(0, 2, 1)
        norm = data['descriptors3d_db']

        B, _, _ = xyz.shape

        l1_xyz, l1_points, l1_idx = self.sa1(xyz, norm)
        l2_xyz, l2_points, l2_idx = self.sa2(l1_xyz, l1_points)
        idx = []
        for b in range(B):
            idx.append(l1_idx[b, l2_idx[b, :]])
        
        data['fps_idx'] = torch.stack(idx)

        new_desc3d_coarse_db = []
        for b in range(B):
            new_desc3d_coarse_db.append(data['descriptors3d_coarse_db'][b, :, data['fps_idx'][b, :]])
        new_desc3d_coarse_db = torch.stack(new_desc3d_coarse_db)

        if 'scores3d_db' in data.keys():
            new_scores3d_db = []
            for b in range(B):
                new_scores3d_db.append(data['scores3d_db'][b, data['fps_idx'][b, :]])
            new_scores3d_db = torch.stack(new_scores3d_db)

        if 'conf_matrix_gt' in data.keys():
            new_conf_matrix_gt = []
            for b in range(B):
                new_conf_matrix_gt.append(data['conf_matrix_gt'][b, data['fps_idx'][b, :], :])
            new_conf_matrix_gt = torch.stack(new_conf_matrix_gt)

        if 'fine_location_matrix_gt' in data.keys():
            new_fine_loc_matrix_gt = []
            for b in range(B):
                new_fine_loc_matrix_gt.append(
                    data['fine_location_matrix_gt'][b, data['fps_idx'][b,:], :, :]
                )
            new_fine_loc_matrix_gt = torch.stack(new_fine_loc_matrix_gt)




        data.update(
            {
                "keypoints3d": l2_xyz.permute(0, 2,1),  # [n_point_cloud, n_query_coarse_grid] Used for coarse GT
                "descriptors3d_db": l2_points,  # [n_point_cloud, n_query_coarse_grid, 2] (x,y)

                "descriptors3d_coarse_db": new_desc3d_coarse_db, # [batch, channel, n_point_cloud]
            }
        )

        if 'scores3d_db' in data.keys():
            data.update(
                {
                    "scores3d_db": new_scores3d_db, # [batch, n_pc]
                }
            )

        if 'conf_matrix_gt' in data.keys() and 'fine_location_matrix_gt' in data.keys():
            data.update(
                {
                    "conf_matrix_gt": new_conf_matrix_gt, # [batch, n_pc, coarsegrid*coarsegrid(64*64=4096)]
                    "fine_location_matrix_gt": new_fine_loc_matrix_gt, # [batch, n_pc, coarsegrid*coarsegrid, 2]
                }
            )

                        
