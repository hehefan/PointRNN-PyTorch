"""
Example usage
"""

import torch

from functions import ChamferDistance, EarthMoversDistance, VideoChamferDistance, VideoEarthMoversDistance


# Create two random pointclouds
# (Batchsize x Number of points x Number of dims)
source_cloud = torch.randn(4, 64, 3).cuda()
target_cloud = torch.randn(4, 64, 3).cuda()
source_cloud.requires_grad = True

# Initialize Chamfer distance module
chamfer_distance = ChamferDistance()
earth_movers_distance = EarthMoversDistance()

# Compute  distance
cd = chamfer_distance(source_cloud, target_cloud)
emd = earth_movers_distance(source_cloud, target_cloud)
print(cd)
print(emd)


# Create two random pointclouds
# (Batchsize x Number of points x Number of dims)
source_video = torch.randn(4, 8, 64, 3).cuda()
target_video = torch.randn(4, 8, 64, 3).cuda()
source_video.requires_grad = True

# Initialize Chamfer distance module
chamfer_distance = VideoChamferDistance()
earth_movers_distance = VideoEarthMoversDistance()

# Compute  distance
cd = chamfer_distance(source_video, target_video)
emd = earth_movers_distance(source_video, target_video)
print(cd)
print(emd)


