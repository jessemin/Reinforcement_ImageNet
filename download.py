import requests
import os


class ImageDownloader:
    def __init__(self):
        pass

    def download(self, wnid):
        if os.path.isdir(os.path.join("Images/",str(wnid))):
            print "Images/"+str(wnid)+" already exists..."
            return
        r = requests.get("http://www.image-net.org/download/synset?wnid={}&username=jesikmin&accesskey=d485058e55f504cd59347e22fe024928a171a64e&release=latest&src=stanford".format(wnid))
        with open("Images/"+str(wnid), "wb") as f:
            f.write(r.content)


# script for downloading all images
if __name__=="__main__":
    downloader = ImageDownloader()
    with open('annotation_list.dat', 'r') as f:
        for wnid in f:
            downloader.download(wnid.rstrip())
