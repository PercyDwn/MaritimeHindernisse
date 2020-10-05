import numpy as np
from cv2 import cv2
from math import ceil, cos, sin, sqrt, atan2
import matplotlib.pyplot as plt
from typing import List, Tuple



class ImageObstacle(object):
    def __init__(self, rectangle: List[int], offset: int=0, angle: float=0) -> None:
        """Represents a object bounding box in image plane

        Args:
            rectangle (List[int]): x y position, width and height of rectangle
            offset (int, optional): position offset (e.g. due to subimage used for detection). Defaults to 0.
            angle (float, optional): rotation offset (due to rotate detection). Defaults to 0.
        """
        
        self.x_rel, self.y_rel, self.width, self.height = rectangle
        self.offset = offset
        self.x, self.y = self.x_rel + offset[0], self.y_rel + offset[1]
        self.angle = angle

    def rectangle_2_corner(self, rotate: bool=False, offset: bool=True) -> Tuple[Tuple[int]]:
        """returns bounding box corners for open-cv drawing function

        Args:
            rotate (bool, optional): respect rotation. Defaults to False.
            offset (bool, optional): respect offset. Defaults to True.

        Returns:
            Tuple[Tuple[int]]: coordinates of diagonal corner points
        """
        if not rotate:
            
            if offset:
                x, y = self.x, self.y
            else:
                x, y = self.x_rel, self.y_rel
            return (x, y), (x + self.width, y + self.height)
        assert False, 'not implemented'

    @property
    def bottom_center(self) -> Tuple[int]:
        """return bottom center of obstacle bounding box

        Returns:
            Tuple[int]: x, y of bottom center pixel
        """
        return self.x + self.width // 2, self.y + self.height

    
class ImageLine(object):
    def __init__(self, img_shape: Tuple[int], points: Tuple[Tuple[int]]=None, polar=None):
        """Represent line in the image

        Args:
            img_shape (Tuple[int]): resolution of the image (widht, height)
            points (Tuple[Tuple[int]], optional): end points of line. Defaults to None.
            polar ([type], optional): polar representation (e.g. output from cv2.HoughLines). Defaults to None.
        """
        self._w, self._h = img_shape
        assert (points is None) != (polar is None)
        if points is not None:
            self._init_from_points(points)
        else:
            self._init_from_polar(*polar)

    def _init_from_points(self, points: Tuple[Tuple[int]]):
        assert len(points) == 2
        assert len(points[0]) == 2
        self.x = [p[0] for p in points]
        self.y = [p[1] for p in points]

        self.height = sum(self.y) / 2
        self.height_rel = self.height - self._h // 2
        self.angle = atan2(np.diff(self.y), np.diff(self.x))

    def _init_from_polar(self, r: float, alpha: float) -> None:
        self._r, self._alpha = r, alpha
        a, b = cos(alpha), sin(alpha)
        # center pixel
        self._center = a * r, b * r
        # line_end pixels
        self.y = np.array([r / b, (r - a * self._w) / b]).astype(int)
        self.x = np.array([0, self._w], int)
        # pixels of a line
        self.xs = np.arange(0, self._w)
        self.ys = np.round((r - a * self.xs) / b).astype(int)

        self.angle = self._alpha - np.pi / 2 # of horizon
        d_2, alpha_0 = sqrt(self._h**2 + self._w**2) / 2, np.arctan(self._h / self._w)
        self.height_rel = r - d_2 * cos(alpha - alpha_0) # over image center
        self.height = self.height_rel + self._h // 2

    @property 
    def indices(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.xs, self.ys

    @property
    def end_points(self) -> Tuple[Tuple[int]]:
        return ((self.x[0], self.y[0]), (self.x[1], self.y[1]))

class ObstacleDetector(object):

    def __init__(self, image_height: int=480, image_width: int=640):
        """Canny based horizon extraction and obstacle detection class

        Args:
            image_height (int, optional): height of images in pixel. Defaults to 480.
            image_width (int, optional): width of images in pixel. Defaults to 640.
        """
        
        # TODO(Moritz) image smoothing params depending on image resolution

        # soften used image shape values
        self._shape = image_width, image_height
        self._w, self._h = self._shape

        self._center = image_width // 2, image_height // 2

        self._min_ang = np.arctan(self._w / self._h)


    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """preprocess image, reduce resolution and smooth

        Args:
            img (np.ndarray): image to process

        Returns:
            np.ndarray: processed image
        """

        img = cv2.resize(img, (self._w, self._h), interpolation=cv2.INTER_AREA)
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        blur_size = self._h // 90
        img = cv2.blur(img, (blur_size, blur_size))
        img = cv2.medianBlur(img, 3)
        # img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def _rotate(self, img: np.ndarray, angle: float, crop: bool=True) -> np.ndarray:
        """rotate image about a given angle 

        Args:
            img (np.ndarray): image to rotate
            angle (float): angle to rotate, (in degree!)
            crop (bool, optional): cut image to rectangle (no black areas). Defaults to True.

        Returns:
            (np.ndarray): rotated image
        """
        # get rotation matrix
        rot_mat = cv2.getRotationMatrix2D(self._center, angle, 1)
        # apply transform
        img_rot = cv2.warpAffine(img, rot_mat, dsize=self._shape)
        # crop image
        if crop:
            sin_angle = np.sin(angle * np.pi / 180)
            crop_h = int(ceil(abs(sin_angle * self._w / 2)))
            crop_w = int(ceil(abs(sin_angle * self._h / 2)))
            if crop_h and crop_w:
                img_rot = img_rot[crop_h: -crop_h, crop_w: -crop_w]
            return img_rot, crop_h, crop_w
        return img_rot

    def detect_horizon(self, 
                       img: np.ndarray, 
                       N_max: int=3, 
                       min_visibility: float=.2, 
                       canny_threshold: int=35
                       ) -> Tuple[List[ImageLine], List[int], List[float]]:
        """detect possible horizon line(s)

        Args:
            img (np.ndarray): image to analyze
            N_max (int, optional): max number of lines to extract. Defaults to 3.
            min_visibility (float, optional): minimum length of horizon line to be considered. Defaults to .2.
            canny_threshold (int, optional): threshold for edge detection. Defaults to 35.

        Returns:
            List[ImageLine]: Horizon lines
            List[int]: Line lengths
            List[float]: separation factor of lines
        """
        img = self._preprocess(img)
        return self._detect_lines(img, min_visibility, N_max, canny_threshold)

    def _detect_lines(self, img: np.ndarray, 
                            min_line_length: float=.2, 
                            N_max: int=3,
                            canny_threshold: int=35,
                            angle_resolution: float=.2
        ) -> Tuple[List[ImageLine], List[int], List[float]]:
        # extract lines in image
        edges = cv2.Canny(img, canny_threshold, canny_threshold * 2)
        lines = cv2.HoughLines(edges, 1, np.pi / 180 * angle_resolution, 
                               int(self._w * min_line_length),
                               min_theta=self._min_ang,
                               max_theta=np.pi - self._min_ang
                               )
        if lines is None:
            return [], [], []
        lines = [ImageLine(self._shape, polar=l[0]) for l in lines[: N_max]]
        line_lengths, separations = self._evaluate_lines(img, edges, lines)

        return lines, line_lengths, separations
        
    def _evaluate_lines(self, img: np.ndarray,
                              edges: np.ndarray,
                              lines: List[ImageLine],
        ) -> Tuple[List[int], List[float]]:
        
        # number of points on line
        votes = [np.sum(edges[l.ys, l.xs]>0) for l in lines]
        # dissimilarity of regions separated by the line
        separations = [self._segmentation(img, l.indices) for l in lines]
        return votes, separations
    
    def _segmentation(self, img: np.ndarray, line_inds: Tuple[np.ndarray], N_neigbor: int=10):
        xs, ys = line_inds

        add_ = np.arange(1, N_neigbor+1)
        
        x_ = (xs[:, None] + np.zeros(N_neigbor, int)[None, :]).reshape(-1)
        y_a = (ys[:, None] + add_[None, :]).reshape(-1)
        y_b = (ys[:, None] - add_[None, :]).reshape(-1)
        img_a = img[y_a, x_]
        img_b = img[y_b, x_]
        
        img_both = np.concatenate((img_a, img_b), axis=0)
        
        sep = np.linalg.norm((np.mean(img_a, axis=0) - np.mean(img_b, axis=0)) \
            / np.std(img_both, axis=0))        
        return sep

    def _reduce_glare(self, img: np.ndarray):
        # following https://medium.com/@rcvaram/glare-removal-with-inpainting-opencv-python-95355aa2aa52
        
        # img = cv2.blur(img, (blur_size, blur_size))
        _, thresh_img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        thresh_img = cv2.erode(thresh_img, None, iterations=2)
        thresh_img = cv2.dilate(thresh_img, None, iterations=4)
        thresh_img = thresh_img.max(axis=2)
        return cv2.inpaint(img, thresh_img, 5, cv2.INPAINT_TELEA)
        
        
    def _detect_obstacles(self,
                          img: np.ndarray,
                          canny_threshold: int=45,
                          glare_reduction=False,
                          offset: Tuple[int]=(0, 0),
                          angle: float=0,
                          plot_subresults: bool=False
        ) -> List[ImageObstacle]:

        # blur
        blur_size = self._h // 90
        img = cv2.blur(img, (blur_size, blur_size))

        # glare reduction
        if glare_reduction:
            img = self._reduce_glare(img)

        # extract edges
        edges = cv2.Canny(img, canny_threshold, canny_threshold * 2) 

        # add edges at image boundaries (besides bottom)
        edges[0, :] = 255
        edges[:, 0] = 255
        edges[:, -1] = 255
        
        # close gapes in lines
        n1 = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (2 * n1 + 1, 2 * n1 +1), 
                                           (n1, n1))
        dilation = cv2.dilate(edges, kernel, iterations=1)
        binary = cv2.erode(dilation, kernel, iterations=1)
        
        # find contours to fill
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        # fill contours
        fill = np.zeros_like(edges)
        for cnt in contours: 
            cv2.fillPoly(fill, pts=[cnt], color=(255, 255, 255))
        
        if plot_subresults:
            self.plot_img(fill)
            self.plot_img(binary)
            self.plot_img(edges)
        # delete single lines
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (2, 2))
        erosion = cv2.erode(fill, kernel2, iterations=1)
        binary = cv2.dilate(erosion, kernel2, iterations=1)
        # extract obstacles
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        obstacles = []
        for c in contours:
            contours_poly = cv2.approxPolyDP(c, 3, True)
            bound_rect = cv2.boundingRect(contours_poly)
            obstacles.append(ImageObstacle(bound_rect, offset, angle=angle))
        return obstacles

    def find_obstacles(self, 
                       img: np.ndarray,
                       horizon: ImageLine=None,
                       threshold: int=40,
                       glare_reduction: bool=False
        ) -> List[ImageObstacle]:
        """find obstacles in the image
        therefore the image is separated according to horizon and 
        only the subhorizon part is searched for possible obstacles
        Further the angle of horizon determines the orientation of bounding boxes

        Args:
            img (np.ndarray): image
            horizon (ImageLine, optional): Horizon estimate. Defaults to None.
            threshold (int, optional): threshold for edge detection. Defaults to 40.
            glare_reduction (bool, optional): if glare reduction is used. Defaults to False.

        Returns:
            List[ImageObstacle]: [description]
        """
        # initial image smoothing
        img = self._preprocess(img)

        # horizon extraction
        if horizon is None:
            horizon = self._detect_lines(img)[0][0]

        # rotate image
        rot_image, crop_h, crop_w = self._rotate(img, horizon.angle * 180 / np.pi)
        height = int(horizon.height - crop_h)
        # subimage below horizon
        offset = 0
        sub_image = rot_image[height - offset:]
        return self._detect_obstacles(sub_image, 
                                      offset=(int(crop_w), int(horizon.height - offset)),
                                      angle=horizon.angle,
                                      canny_threshold=threshold,
                                      glare_reduction=glare_reduction)

    def plot_img(self, img, 
                       horizon_lines=None, 
                       obstacles=None, 
                       rotate: bool=True,
                       plot_method: str='matplot',
                       wait_time: int=0,
        ) -> None:
        
        if horizon_lines:

            if not img.shape[:2] == (self._h, self._w):
                img = cv2.resize(img, (self._w, self._h), interpolation=cv2.INTER_AREA)
            for h in horizon_lines:
                color = (255, 0, 0)
                cv2.line(img, *h.end_points, color, thickness=2)
        if obstacles:
            #pass
            if rotate and obstacles[0].angle:
                img = self._rotate(img, obstacles[0].angle * 180/np.pi, crop=False)
            for obst in obstacles:
                # color = np.random.rand(3) * 255
                color = (0, 0, 255)
                cv2.rectangle(img, *obst.rectangle_2_corner(not rotate, True), color, thickness=2)
                cv2.drawMarker(img, obst.bottom_center, (0, 255, 0), cv2.MARKER_CROSS, 10, thickness=2)

        if 'matplot' in plot_method:
            fig, ax = plt.subplots()
            if len(img.shape) > 2 and img.shape[2] > 1:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
        elif 'cv' in plot_method:
            cv2.imshow('image', img)
            cv2.waitKey(wait_time)


def main():
    # run with first argument as imagefile or folder with images
    print('running main()')

    import sys

    detector = ObstacleDetector()
    #################################
    #Dateipfad individuell anpassen!!
    #################################
    img, folder = None, 'C:/Users/lukas/source/repos/PercyDwn/MaritimeHindernisse/TestFile/Projekt/Bilder/list1'
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
                print(horizon_lines[0].angle * 180/np.pi, horizon_lines[0].height)
                for obs in obstacles:
                    print(obs.x,',',obs.y)

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
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

