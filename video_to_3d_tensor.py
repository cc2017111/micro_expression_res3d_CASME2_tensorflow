import numpy as np
import cv2


class Video_to_3dtensor:

    def __init__(self, img_size, depth):
        self.img_size = img_size
        self.depth = depth

    def Video_3D(self, filename, color=False, skip=True):
        cap = cv2.VideoCapture(filename)
        n_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if skip:
            frames = [x * n_frame / self.depth for x in range(self.depth)]
        else:
            frames = [x for x in range(self.depth)]
        frame_array = []

        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            if frame is None:
                return
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            if color:
                frame_array.append(frame)
            else:
                frame_array.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()
        return np.array(frame_array)

