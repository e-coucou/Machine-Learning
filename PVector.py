from msilib.schema import Class


class PVector():
    def __init__(self,x,y) -> None:
        self.vector = np.array([x,y])

    def setHeading(v,a,mag=1):
        return (np.array([mag*cos(a),-mag*sin(a)]) + v).astype(int)

    def heading(v):
        return (np.arctan2(-v[1],v[0]))

    def setMag(v,mag):
        coef = np.linalg.norm(v)
        return (v/coef*mag)

    def norm(v):
        coef = np.linalg.norm(v)
        return (v/coef)

    def dist(v1,v2):
        d = np.linalg.norm(v2-v1)
        return d

    def limit(v,lim):
        return norm(v)*lim

    def findProjection(pos,a ,b):
        v1 = pos -a
        v2 = b - a
        v2 = norm(v2)
        sp = np.dot(v1,v2)
        return (a+v2*sp).astype(int),sp
    