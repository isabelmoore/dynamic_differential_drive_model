import numpy as np
import random
import math
import matplotlib.pyplot as plt
from differential_drive_model import DifferentialDriveModel


class purePursuitController:

	def __init__(self,numPoints=200):

		self.numPoints = numPoints
		self.idxClosest = 0
		self.posesPPC = []
		#self.generateWayPoints2(self.numPoints)





	def generateWayPoints2(self,numPoints):

		r = 0.2
		thetaT = 0.0
		x = []
		y = []
		x.append(0)
		y.append(0)
		for i in range(0,numPoints):
			thetaT = random.uniform(-math.pi/8 + thetaT,math.pi/8 + thetaT)
			x.append(r*np.cos(thetaT) + x[-1])
			y.append(r*np.sin(thetaT) + y[-1])
		
		print(x)
		print(y)
		plt.plot(x,y)
		#plt.show()
		self.purePursuitController(x,y)


	def purePursuitController(self,x,y):

		ddm = DifferentialDriveModel(wheel_radius=0.1, track_width=0.3765,
				plot=False,trajX=x,trajY=y)
		pose = np.zeros(3)
		#deltaTheta = getSteeringWheelAngle(x,y,lookahead=0.3,pose = pose)
		#action = [0.1, -0.7]
		#for i in range(100):
		#print("x[-1],y[-1] : ",x[-1],y[-1])
		while(self.getDistanceBwPoints(pose[0],pose[1],x[-1],y[-1])>0.5):
			deltaTheta = self.getSteeringWheelAngle(x,y,lookahead=0.4,pose = pose)
			action = [1,deltaTheta]
			pose = ddm.step(pose, action,timestep=0.1)
			self.posesPPC.append(pose)
			#print("pose : ",pose)
			#plt.plot(x[self.idxClosest],y[self.idxClosest],'*')
		
		print("Goal reached!!!! ", "(",pose[0],pose[1],"), (",x[-1],y[-1],")")
		return self.posesPPC

			

	def getSteeringWheelAngle(self,x,y,lookahead = 10,pose = np.zeros(3)):

		lookAhead = lookahead


		dClosest = abs(np.sqrt((x[:] - pose[0])**2 + (y[:] - pose[1])**2) - lookAhead)
		idxClosest = np.argmin(dClosest)

		#print("dclosest : ",dClosest)
		if(idxClosest < self.idxClosest):
			self.idxClosest = self.idxClosest 
		elif(idxClosest == self.idxClosest):
			self.idxClosest = self.idxClosest
		else:
			self.idxClosest = idxClosest


		"""
		Get the small angle difference in heading for the current pose and expected pose
		"""
		if(len(x) -1 > 2):
			#headingLPPoint = np.arctan2((lpObj.ly_lin[idxClosest + 1] - lpObj.ly_lin[idxClosest]),( lpObj.lx_lin[idxClosest + 1] - lpObj.lx_lin[idxClosest] ))
			headingLPPoint = np.arctan2((y[self.idxClosest] - pose[1] ),( x[self.idxClosest] - pose[0] ))
			#headingLPPoint = np.arctan2(( lpObj.ly_lin[idxClosest]),( lpObj.lx_lin[idxClosest] ))
			#print(lpObj.ly_lin,lpObj.lx_lin)
		else:
			headingLPPoint = 0.0
		
		deltaTheta = self.angleDiff(headingLPPoint,pose[2])

		print("deltaTheta : ",deltaTheta)
		return deltaTheta

		
	def angleDiff(self,a,b):
		#### SEE IMPLEMENTATION: https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
		diff = a - b
		ahat = [np.cos(a), np.sin(a)]
		bhat = [np.cos(b), np.sin(b)]
		sinAminusB = np.cross(bhat,ahat)
		cosAminuB = np.dot(ahat,bhat)
		return np.arctan2(sinAminusB,cosAminuB)
	#	    return np.fmod((diff+np.pi),2*np.pi)-np.pi



	def getDistanceBwPoints(self,x1,y1,x2,y2):

		return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

if __name__=="__main__":
	x = purePursuitController(200)