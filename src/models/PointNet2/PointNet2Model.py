import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction


class PointNet2_model(nn.Module):
    def __init__(self):
        super(PointNet2_model, self).__init__()
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=256+3, mlp=[256, 256, 512], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=512+3, mlp=[512, 512, 1024], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024+3, mlp=[1024, 2048, 2048], group_all=True)

        
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.4)

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
        norm = data['descriptors3d_coarse_db']

        B, _, N = xyz.shape

        l1_xyz, l1_points, l1_idx = self.sa1(xyz, norm)
        l2_xyz, l2_points, l2_idx = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 2048)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        x = x.expand(N, B, 256).permute(1,2,0)
        new_descriptors3d_coarse_db = torch.cat((data['descriptors3d_coarse_db'], x), 1)
        x = new_descriptors3d_coarse_db.permute(0,2,1).reshape(B*N, 512)
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        new_descriptors3d_coarse_db = x.reshape(B, N, -1).permute(0,2,1)


        data.update(
            {

                "descriptors3d_coarse_db": new_descriptors3d_coarse_db, # [batch, channel, n_point_cloud]
            }
        )
