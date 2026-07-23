
class ROI(object):
    def __init__(self, x, y, w, h):
        """
        Region of interest bounding box.
        :param x: Horizontal position of upperleft pixel
        :param y: Vertical position of upperleft pixel
        :param w: Width of ROI
        :param h: Height of ROI
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.self_check()

    def __iter__(self):
        return iter((self.x, self.y, self.x + self.w, self.y + self.h))

    def self_check(self):
        if self.x < 0 or self.y < 0:
            raise ValueError("For RoI, x and y must be positive")
        if self.w < 0:
            raise ValueError("For RoI, width must be positive")
        if self.h < 0:
            raise ValueError("For RoI, height must be positive")

    def check(self, dims):
        self.self_check()
        if len(dims) != 2:
            raise ValueError("image must have 2 dimensions")
        elif self.x + self.w > dims[0] or self.y + self.h > dims[1]:
            raise ValueError("RoI exceeds image dimensions")
