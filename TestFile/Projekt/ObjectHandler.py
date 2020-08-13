
# Typing
from typing import List

import sys

from obstacle_detect import *


class ObjectHandler:
    
    """This Class is used to retrieve the object states"""

    def __init__(self) -> None:

        self.ObjectStates: List = []


    def getObjectStates(self) -> List:

        self.updateObjectStates()

        return self.ObjectStates

    def updateObjectStates(self):

        detector = ObstacleDetector()
        img, folder = None, '.'
        if len(sys.argv) > 1 and type(sys.argv[1]) is str:
            if '.jpg' in sys.argv[1]:
                try: 
                    img = cv2.imread(sys.argv[1])
                except: 
                    print('no valid image path', sys.argv[1])

                if img is not None:
                    # load image and analyze
                    cv2.imshow('orig', img)
                    horizon_lines, votes, seps = detector.detect_horizon(img)
                    obstacles = detector.find_obstacles(img, horizon=horizon_lines[0])
                    # print(horizon_lines[0].angle * 180/np.pi, horizon_lines[0].height)

                    States = []
                    for obs in obstacles:
                        # print(obs.x,',',obs.y)
                        States.append([obs.x, obs.y])
                    self.ObjectStates = States

                    detector.plot_img(img, obstacles=obstacles, horizon_lines=horizon_lines, plot_method='matplot')
                    plt.show()

            else:
                folder = sys.argv[1]

        if img is None:
            import os
            image_list = [f for f in os.listdir(folder) if '.jpg' in f or '.JPG' in f]
            # loop over imagelist and analyze each image
            for image_file in image_list:
                img = cv2.imread(folder + '/' + image_file)
                assert img is not None, folder + '/' + image_file
                cv2.imshow('orig', img)
                horizon_lines, votes, seps = detector.detect_horizon(img)
                if horizon_lines:
                    obstacles = detector.find_obstacles(img, horizon=horizon_lines[0])
                else:
                    obstacles = None
                detector.plot_img(img, obstacles=obstacles, 
                                  horizon_lines=horizon_lines, 
                                  plot_method='cv', wait_time=1)

        




