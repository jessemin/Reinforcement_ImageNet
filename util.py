class Util:
    def computeIntersection(self, rec1, rec2):
        # rec is given as a tuple (xmin, ymin, xmax, ymax)
        dx = min(rec1[2], rec2[2]) - max(rec1[0], rec2[0])
        dy = min(rec1[3], rec2[3]) - max(rec1[1], rec2[1])
        if (dx>=0) and (dy>=0):
            return dx*dy
        return 0

    def computeArea(self, rec):
        return (rec[2]-rec[0]) * (rec[3]-rec[1])

    def computeIOU(self, rec1, rec2):
        intersection = self.computeIntersection(rec1, rec2)
        total = self.computeArea(rec1) + self.computeArea(rec2)
        return intersection * 1.0 / (total - intersection)


# unittests
if __name__=='__main__':
    util = Util()
    # 1) unittest for computing intersection of two rectangles
    assert(util.computeIntersection((1, 1, 5, 5), (1, 1, 4, 4)) == 9)
    assert(util.computeIntersection((1, 1, 5, 5), (6, 6, 10, 10)) == 0)
    assert(util.computeIntersection((1, 1, 5, 5), (2, 2, 10, 10)) == 9)
    # 2) unittest for computing area of single rectangle
    assert(util.computeArea((1,1,5,5)) == 16)
    assert(util.computeArea((6, 6, 10, 10)) == 16)
    assert(util.computeArea((2, 2, 10, 10)) == 64)
    # 3) unittest for computing union
    assert(util.computeIOU((1, 1, 5, 5), (1, 1, 4, 4)) ==  9.0/16)
    assert(util.computeIOU((1, 1, 5, 5), (6, 6, 10, 10)) == 0)
    assert(util.computeIOU((1, 1, 5, 5), (2, 2, 10, 10)) == 9.0/71)
