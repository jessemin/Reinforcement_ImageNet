import os


class WnidFetcher:
    def __init__(self):
        self.wnids = []
        self.fetchWnids()

    def fetchWnids(self):
        for f in os.listdir(os.path.join(os.getcwd(), 'Annotation')):
            self.wnids.append(f[:f.find('.')])


# script that gets all the wnids in Annotation directory
if __name__=="__main__":
    wnids = WnidFetcher().wnids
    print len(wnids)
    with open('annotation_list.dat', 'wb') as f:
        for wnid in wnids:
            if len(wnid) == 0:
                continue
            f.write(wnid+"\n")
    f.close()
