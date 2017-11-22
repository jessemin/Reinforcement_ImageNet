import os
import xml.etree.ElementTree as ET

# class that handles the entire imagenet data
# wnid: WordNet Id that starts with 'n' and followed by a unique numerical id
# image_id: unique image id within given wnid directory
class BoundingBoxData:
    def __init__(self, wnid, image_id):
        self.wnid = wnid
        self.image_id = image_id
        self.bounding_boxes = self.get_bounding_box(wnid, image_id)

    def get_bounding_box(self, wnid, image_id):
        bounding_boxes = []
        annotation_file = os.path.join(os.getcwd(), "Annotation", str(wnid), str(wnid)+"_"+str(image_id)+".xml")
        tree = ET.parse(annotation_file)
        objects = tree.findall("object")
        for obj in objects:
            bounding_box = obj.find("bndbox")
            # fetch [xmin, ymin, xmax, ymax]
            bounding_boxes.append([int(vertex.text) for vertex in bounding_box])
        return bounding_boxes

    def __str__(self):
        return "wnid: {}, image_id: {}, bounding boxes: {}".format(self.wnid, self.image_id, str(self.bounding_boxes))


# Collect all bounding box data for a given wnid
class BBCollector:
    def __init__(self, wnid):
        self.allBBs = {}
        self.getAllBoundingBoxesForWnid(wnid)

    def getAllBoundingBoxesForWnid(self, wnid):
        files = os.listdir(os.path.join(os.getcwd(), "Annotation", str(wnid)))
        for f in files:
            image_id = f[len(str(wnid))+1:-len('.xml')]
            self.allBBs[wnid+"_"+image_id] = BoundingBoxData(str(wnid), image_id)

    def __str__(self):
        return str([(str(bb), str(self.allBBs[bb])) for bb in self.allBBs])


if __name__=='__main__':
    print BBCollector("n00007846")
