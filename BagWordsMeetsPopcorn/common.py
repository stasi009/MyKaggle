
import cPickle

def simple_dump(filename, *objects):
    with open(filename, 'wb') as outfile:
        for obj in objects:
            cPickle.dump(obj, outfile)

def simple_load(filename,n_objs=1):
    objects = []
    with open(filename, "rb") as infile:
        for index in xrange(n_objs):
            objects.append(cPickle.load(infile))
        return objects