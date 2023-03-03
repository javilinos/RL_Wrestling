# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observation state."""

from utils.camera import Camera
from utils.image_processing import ImageProcessing

class Observation(Camera):
    def __init__(self, robot, camera_name='CameraTop'):
        super().__init__(robot, camera_name)
    
    def print_observations(self):
        print(ImageProcessing.locate_opponent(super().get_image()))