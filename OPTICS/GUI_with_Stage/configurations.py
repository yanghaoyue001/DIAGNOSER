import numpy as np

update_period = 0.5
# ArduinoSerial = None

# robot arm DH system info (constant values)
d1=0.3
a1=0
alpha1=90

d2=0
a2=0.3
alpha2=0

d3=0
a3=0
alpha3=90

d4=0.3
a4=0
alpha4=-90

d5=0
a5=0
alpha5=90

d6=0.1
a6=0
alpha6=0

# robot arm DH system info (varying values)
theta1=90+0 # first 90 is fixed
theta2=120
theta3=90-80 # first 90 is fixed
theta4=0
theta5=-30
theta6=0

kp=500
ki=60
kd=0

# camera ball detection parameters
# define range of blue color in HSV green
lower_blue = np.array([40,100,100])
upper_blue = np.array([80,255,255])
# # blue magenta
# lower_blue = np.array([170,0,0])
# upper_blue = np.array([200,255,255])


width_sensor=3.888 # width of camera sensor is 3.888 mm
height_sensor=2.43 #  height of camera sensor is 2.43 mm
f=6 # focal ~ 6 mm
b=153 # baseline between two camera ~ 153 mm

# z_compensation=np.array( [ 1.60823947e-04,  4.31671817e-03, -3.71142418e-04, -4.10922802e-03,
#   1.05834956e-06,  1.69009901e-06, -9.92447566e-07, -6.72921679e-07,
#   3.70622906e-03])

z_compensation=np.array([ 0.00317947, -0.00811515, -0.00319329,  0.00834952,  0.00373402, 0.5720310795394032])
x_compensation=np.array([ 0.1892927, -1.60641958, -0.05108869,  1.59155587,  0.46557747,  0.08902695, -45.96252601733042])
y_compensation=np.array( [ -0.12910153,  17.08524952,  -0.01836012, -17.36378956,   0.69519795,
  -0.07100605, 53.36571236842167])
z_compensation_para=np.array([1,1,1,1,1,1,1,1,1])
x_compensation_para=np.array([1,1,1,1,1,1,1,1,1,1])
y_compensation_para=np.array([1,1,1,1,1,1,1,1,1,1])
# z=np.dot(z_compensation,compensation_para)
# print(z)
x_better=10000
y_better=10000
z_better=10000
