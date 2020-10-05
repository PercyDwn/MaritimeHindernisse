
# Typing
from typing import List

# pathlib module
from pathlib import Path

import sys
import os

from numpy import ndarray

from obstacle_detect import *
from CustomErrors import *


class ObjectHandler:
    
    """This Class is used to retrieve the object states"""

    def __init__(self) -> None:
        
        self.DebugLevel = 0
        self.ObjectStates: List = []
        self.ImageFolder: str = None
        self.ImageBaseName: str = None
        self.ImageFileType: str = '.jpg'
        self.Img: ndarray 
        self.image_height: int
        self.image_width: int
        self.PlotOnUpdate: bool = True

    def setDebugLevel(self, debugLevel: int = 0) -> None:
        self.DebugLevel = debugLevel

    # check, if set debug level is greater equal to given level
    def printDebug(self, level: int) -> bool:
        if(level <= self.DebugLevel):
            return True
        else:
            return False

    def setPlotOnUpdate(self, plot: bool) -> None:
        self.PlotOnUpdate = plot

    def setImageFolder(self, folder: str, addCWD: bool = True) -> bool:
        if addCWD:
            path = os.getcwd()
            path = path.replace(os.sep,'/')
        else:
            path = ''
        self.ImageFolder = path + folder
        return True

    def setImageBaseName(self, baseName: str) -> bool:
        self.ImageBaseName = baseName
        return True

    def setImageFileType(self, fileType: str) -> bool:
        self.ImageFileType = fileType
        return True

    def getTimeStepCount(self) -> int:
        return len(self.ObjectStates)

    def getImg(self) -> ndarray:
        return self.Img

    def getImgHeight(self) -> int:
        return self.image_height

    def getImgWidth(self) -> int:
        return self.image_width

    # return list with object states for all time stemps
    def getObjectStatesList(self) -> List:

        return self.ObjectStates

    # return object states for a given time step t
    def getObjectStates(self, t: int, position: str = 'cc') -> List:
        if(t > self.getTimeStepCount()):
            # try and update object state data
            for i in range(0,t - self.getTimeStepCount()):
                # try to update
                updated = self.updateObjectStates()
                if updated == False:  raise InvalidTimeStepError('time step is out of bound!')
        
        # return data for requested time step
        formattedCoordinates = []
        for obj in self.ObjectStates[t-1]:
            formattedCoordinates.append(self.returnCoordinates(obj, position))
        return formattedCoordinates
                

    # format coordinates
    def returnCoordinates(self, state: Tuple, position: str = 'cc') -> List:

        # print(state)

        x = state["x"]
        y = state["y"]
        w = state["width"]
        h = state["height"]

        if position[0] == 'l':
            return_x = x
        elif position[0] == 'c':
            return_x = x + w//2
        elif position[0] == 'r':
            return_x = x + w
        else:
            return_x = x

        if position[1] == 't':
            return_y = y
        elif position[1] == 'c':
            return_y = y + h //w
        elif position[1] == 'b':
            return_y = y + h
        else:
            return_y = y

        return [return_x, return_y]

    # return last item in object states list
    def getLastObjectStates(self, position: str = 'cc') -> List:

        return self.returnCoordinates(self.ObjectStates[-1], position)

    # update the object states for the next time step
    def updateObjectStates(self) -> bool:

        detector = ObstacleDetector()
        img, folder = None, '.'

        self.image_height = detector._h
        self.image_width = detector._w

        # get current max time step and add 1
        currentTimeStep = self.getTimeStepCount() + 1
        if self.printDebug(2): print('current time step: ' + str(currentTimeStep))

        # check if folder and image base is set
        if type(self.ImageFolder) is str and type(self.ImageBaseName) is str:
            # if so, concat with current time step to next image name
            filepath = self.ImageFolder  + '/' + self.ImageBaseName + str(currentTimeStep) + self.ImageFileType
        else:
            # else, return false
            if self.printDebug(0): print('image folder or base name is not set')
            return False

        nextImage = Path(filepath)
        if nextImage.is_file():
            # file exists, run obstacle detect
            if self.printDebug(2): print('file ' + filepath + ' is valid')
            # read image
            img = cv2.imread(filepath)
            self.Img = img
            # check if image is valid
            assert img is not None, 'file ' + filepath + ' could not be read'
            # plot if plot setting is true
            #if self.PlotOnUpdate == True: cv2.imshow('orig', img)
            # detect horizon
            horizon_lines, votes, seps = detector.detect_horizon(img)
            # if horizon lines found
            if horizon_lines:
                # find obstacles
                obstacles = detector.find_obstacles(img, horizon=horizon_lines[0])
                # write found obstacles in list
                ObstacleStates = []
                for obs in obstacles:
                    # ObstacleStates.append([obs.x + obs.width // 2, obs.y + obs.height])
                    ObstacleStates.append({
                            "x": obs.x,
                            "y": obs.y,
                            "width": obs.width,
                            "height": obs.height
                        })
                # add list to list with obstacles over all time steps
                self.ObjectStates.append(ObstacleStates)
            else:
                obstacles = None
            # plot image with found obstacles
            #print('obstacles:')
            #for obs in obstacles:
            #    print('[' + str(obs.x) + ', ' + str(obs.y) + ']')
            if self.PlotOnUpdate == True: detector.plot_img(img, obstacles=obstacles,horizon_lines=horizon_lines,plot_method='cv', wait_time=1)

            return True

        else:
            if self.printDebug(0): print(filepath + ' is not a valid file')
            return False








