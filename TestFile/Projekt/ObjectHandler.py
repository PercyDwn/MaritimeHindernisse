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
        self.ImageLeadingZeros: int = None
        self.ImageFileType: str = '.jpg'
        self.Img: ndarray 
        self.image_height: int
        self.image_width: int
        self.HorizonData: List = []
        self.PlotOnUpdate: bool = False

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

    def setImageFolder(self, folder: str) -> bool:
        self.ImageFolder = folder
        return True

    def setImageBaseName(self, baseName: str) -> bool:
        self.ImageBaseName = baseName
        return True

    def setImageLeadingZeros(self, zeros: int) -> bool:
        self.ImageLeadingZeros = zeros
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

    def getHorizonData(self, t: int) -> Tuple:
        
        try:
            return self.HorizonData[t-1]
        except IndexError:
            raise InvalidTimeStepError('time step is out of bound!')


    # return list with object states for all time stemps
    def getObjectStatesList(self) -> List:

        return self.ObjectStates

    # return object states for a given time step t
    def getObjectStates(self, t: int) -> List:
        if(t <= self.getTimeStepCount()):
            # return data for requested time step
            return self.ObjectStates[t-1]
        else:
            # try and update object state data
            for i in range(0,t - self.getTimeStepCount()):
                # try to update
                updated = self.updateObjectStates()
                if updated == False:  raise InvalidTimeStepError('time step is out of bound!')
            print(self.getTimeStepCount())
            return self.ObjectStates[t-1]
                

    # return last item in object states list
    def getLastObjectStates(self) -> List:

        return self.ObjectStates[-1]

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
            if self.ImageLeadingZeros:
                # number of leading zeros given
                formatStr = "{:0"+str(self.ImageLeadingZeros)+"d}"
                num = formatStr.format(currentTimeStep)
                filepath = self.ImageFolder  + '/' + self.ImageBaseName + str(num) + self.ImageFileType
            else:
                # number of leading zeros not given, try 1 to 8
                for ic in range(1,8):
                    formatStr = "{:0"+str(ic)+"d}"
                    num = formatStr.format(currentTimeStep)
                    filepath = self.ImageFolder  + '/' + self.ImageBaseName + str(num) + self.ImageFileType
                    nextImage = Path(filepath)
                    if nextImage.is_file():
                        break
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
            # if self.PlotOnUpdate == True: cv2.imshow('orig', img)
            # detect horizon
            horizon_lines, votes, seps = detector.detect_horizon(img)
            # set horizon data
            self.HorizonData.append( (horizon_lines, votes,seps) )
            # if horizon lines found
            if horizon_lines:
                # find obstacles
                obstacles = detector.find_obstacles(img, horizon=horizon_lines[0])
                # write found obstacles in list
                ObstacleStates = []
                for obs in obstacles:
                    ObstacleStates.append([obs.x + obs.width // 2, obs.y + obs.height])
                # add list to list with obstacles over all time steps
                self.ObjectStates.append(ObstacleStates)
            else:
                print('horizont lines were not found')
                obstacles = None
                self.ObjectStates.append([])

            if self.PlotOnUpdate == True: detector.plot_img(img, obstacles=obstacles,horizon_lines=horizon_lines,plot_method='cv', wait_time=1)

            return True
        else:
            if self.printDebug(0): print(filepath + ' is not a valid file')
            return False