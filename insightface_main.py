from matplotlib import transforms

import os
import cv2
from itertools import product as product
import glob
import os.path as osp

import numpy as np
from post_processing import RetinaFace

import sys
from utils.face import Face

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self):
        self.det_model = RetinaFace()

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        detetedFace = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            detetedFace.append(face)
        return detetedFace

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
        return dimg

if __name__ == "__main__":
    app = FaceAnalysis()
    img = cv2.imread('C:/images/city.jpg')

    faces = app.get(img)
    rimg = app.draw_on(img, faces)
    while True:
        cv2.imshow("t1_output", rimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        cv2.imshow("t1_output", rimg)