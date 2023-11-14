# from perception.msg import Object, ObjectBoundingBox

class ObjBBX(object):
    def __init__(self, class_id, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_id = class_id

class Obj(object):
    def __init__(self, class_id, x_bbx = None, y_bbx = None, w_bbx = None, h_bbx = None, x = None, y = None, z = None):
        self.class_id = class_id
        self.x = x
        self.y = y
        self.z = z
        self.bbox = ObjBBX(class_id, x_bbx, y_bbx, w_bbx, h_bbx)

    # def to_msg(self):
    #     msg = Object()
    #     msg.class_id = self.class_id
    #     msg.x = self.x
    #     msg.y = self.y
    #     msg.z = self.z
    #     msg.bbox = ObjectBoundingBox()
    #     msg.bbox.x = self.bbox.x
    #     msg.bbox.y = self.bbox.y
    #     msg.bbox.w = self.bbox.w
    #     msg.bbox.h = self.bbox.h
    #     return msg