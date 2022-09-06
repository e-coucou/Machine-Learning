import numpy as np

class Point:
	def __init__(self,x,y,parent=None):
		self.x =x
		self.y = y
		self.parent = parent

	def dist2centre(self,centre):
		return np.hypot(self.x - centre[0], self.y - centre[1])


class Rectangle:
	def __init__(self,x,y,w,h):
		self.x = x
		self.y = y
		self.w = w
		self.h = h 
		self.divided = False

	def contains(self,point):
		return (point.x >= self.x-self.w and point.x <= self.x + self.w and point.y >= self.y -self.h and point.y <= self.y +self.h)
	
	def intersect(self,range):
		return not(range.x- range.w > self.x + self.w or range.x + range.w < self.x - self.w or range.y- range.h > self.y + self.h or range.y + range.h < self.y - self.h)


class Quadtree:
	def __init__(self,boundary,n):
		self.boundary = boundary
		self.capacity = n
		self.divided = False
		self.points = []

	def insert(self,point):
		if not self.boundary.contains(point):
			return False

		if len(self.points) < self.capacity:
			self.points.append(point)
			return True
		else:
			if not self.divided:
				self.subdivide()
			return ( self.NE.insert(point) or self.NW.insert(point) or self.SE.insert(point) or self.SW.insert(point) )

	def subdivide(self):
		x = self.boundary.x
		y = self.boundary.y
		w = self.boundary.w
		h = self.boundary.h
		NW =  Rectangle(x-w/2,y-h/2,w/2,h/2)
		self.NW = Quadtree(NW,self.capacity)
		NE = Rectangle(x+w/2,y-h/2,w/2,h/2)
		self.NE =  Quadtree(NE,self.capacity)
		SW = Rectangle(x-w/2,y+h/2,w/2,h/2)
		self.SW = Quadtree(SW,self.capacity)
		SE = Rectangle(x+w/2,y+h/2,w/2,h/2)
		self.SE = Quadtree(SE,self.capacity)
		self.divided = True


	def query(self,range,found):
		if not self.boundary.intersect(range):
			return False

		for p in self.points:
			if (range.contains(p)):
				found.append(p)  # modifier pour n'avoir que les data client
		if (self.divided):
			self.NE.query(range,found)
			self.NW.query(range,found)
			self.SE.query(range,found)
			self.SW.query(range,found)
		return found

	def queryCircle(self,range,centre,radius,found):
		if not self.boundary.intersect(range):
			return False

		for p in self.points:
			if (range.contains(p) and p.dist2centre(centre)<=radius):
				found.append(p)  # modifier pour n'avoir que les data client
		if (self.divided):
			self.NE.queryCircle(range,centre,radius,found)
			self.NW.queryCircle(range,centre,radius,found)
			self.SE.queryCircle(range,centre,radius,found)
			self.SW.queryCircle(range,centre,radius,found)
		return found

	def queryRadius(self,centre,radius,found):
		range = Rectangle(centre[0],centre[1],radius,radius)
		return self.queryCircle(range,centre,radius,found)

	def show(self):
		for p in self.points:
			print('points',p.x,p.y)
		if (self.divided):
			self.NE.show()
			self.NW.show()
			self.SE.show()
			self.SW.show()

	def getPoint(self,found):
		for p in self.points:
			found.append([p.x,p.y])
		if (self.divided):
			self.NE.getPoint(found)
			self.NW.getPoint(found)
			self.SE.getPoint(found)
			self.SW.getPoint(found)
		return found

	def getRect(self,found):
		found.append([self.boundary.x-self.boundary.w,self.boundary.y-self.boundary.h,self.boundary.x+self.boundary.w,self.boundary.y+self.boundary.h])
		if (self.divided):
			self.NE.getRect(found)
			self.NW.getRect(found)
			self.SE.getRect(found)
			self.SW.getRect(found)
		return found
