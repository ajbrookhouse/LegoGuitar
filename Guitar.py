from matplotlib import pyplot as plt
import numpy as np
import math
import random
import matplotlib.ticker as plticker
from PIL import Image
from PIL import ImageColor
from PIL import ImageOps
import pickle
import pandas

with open('guitarShape.pkl','rb') as inFile:
    guitarShape = pickle.load(inFile)

def getClosestColor(colors, r,g,b):
    """
    Given a list of colors in tuple form (r,g,b) and three values for r, g, and b respectively
    This function returns the closest color in the list to the r,g, and b values

    I use this in this program to pass in the list of lego tile colors and then pass the raw rgb values of an image
    to get the closest lego color to that part of an image.

    This is very helpful when making a lego mosaic
    """
    closestColor = 'ldsfjk'
    closestNum = 20000000

    for color in colors:
        colorr = color[0]
        colorg = color[1]
        colorb = color[2]

        distance = ( (r - colorr)**2  +(g - colorg)**2  +(b - colorb)**2) **.5
        #print(distance)
        if distance < closestNum:
            closestColor = color
            closestNum = distance
    return closestColor
        
def plotGuitarShape():
    """
    Plots the Les Paul guitar shape with matplotlib
    """
    with open('guitarShape.pkl','rb') as inFile:
        guitarShape = pickle.load(inFile)

    for index, row in enumerate(guitarShape):
        print(list(row).count(1))
        if index == 10:
            print('=======================================')

    plt.imshow(guitarShape, cmap='binary')
    plt.show()


def plotImageOnGuitar(imagePath):
    """
    Takes imagePath as input
    Plots the Les Paul Guitar shape made out of legos
    Colors the guitar to a pixellated version of the imagePath image only using real lego colors
    """
    colors = pandas.read_csv('legoColors.csv',header=None)
    colors = np.array(colors)
    colorList = []
    colorNameList = []
    for index, line in enumerate(colors):
        if line[3] == 'f':
            color = line[2]
            colorList.append(ImageColor.getcolor('#' + str(color),'RGB'))
            colorNameList.append(line[1])
    filename = imagePath
    img = Image.open(filename)

    imgSmall = img.resize(guitarShape.shape,resample=Image.BILINEAR)
    imgSmall = imgSmall.rotate(90, Image.NEAREST, expand = 1)
    imgSmall = ImageOps.flip(imgSmall)
    imgSmall = np.array(imgSmall)

    # plt.imshow(imgSmall)
    # plt.show()
    for i in range(len(imgSmall)):
        for j in range(len(imgSmall[i])):
            currentColor = imgSmall[i][j]
            curR = currentColor[0]
            curG = currentColor[1]
            curB = currentColor[2]
            closestColor = getClosestColor(colorList,curR,curG,curB)
            imgSmall[i][j][0] = closestColor[0]
            imgSmall[i][j][1] = closestColor[1]
            imgSmall[i][j][2] = closestColor[2]
            try:
                imgSmall[i][j][3] = 255
            except:
                pass

    # plt.imshow(imgSmall)
    # plt.show()
    imgSmall[guitarShape == 0] = 0
    plt.imshow(imgSmall)
    plt.show()

def getDataFromRows(rows):
    toReturn = []
    for row in rows:
        toReturn.append(len(row))
    return toReturn

def getRowsFromData(data):
    toReturn = []
    for dataVal in data:
        toReturn.append([1]*dataVal)
    return toReturn

def getArrayFromRows(rows):
    numRows = len(rows)
    longestRow = max(getDataFromRows(rows))
    array = list(np.zeros((numRows,longestRow)))
    for y in range(numRows):
        padding = int((longestRow-len(rows[y]))/2)
        #print(padding)
        for j in range(padding,padding+len(rows[y])):
            array[y][j] = 1
    picture = []
    for row in reversed(array):
        picture.append(row)
    return picture

def getArrayFromData(data):
    rows = getRowsFromData(data)
    return getArrayFromRows(rows)

def monteCarloBottomShape():
    """
    this function and the four functions above are from an earlier attempt I made randomly generating guitar shapes. I do not recoomentd using these
    """
    numRows = 22

    for iteration in range(25):
        successfulGuitar = False
        while not successfulGuitar:
            rows = []
            longestRow = -1
            initialWidth = 41
            rows.append([1] * initialWidth)
            previousIncrease = 999
            previousWidth = initialWidth
            minIncrease = -10
            for i in range(numRows - 1):
                satisfied = False
                while not satisfied:
                    tryWidth = random.randrange(25,43,2)
                    increase = tryWidth - previousWidth
                    if increase <= 0 and increase <= previousIncrease and increase >= minIncrease:
                        rows.append([1] * tryWidth)
                        previousIncrease = increase
                        previousWidth = tryWidth
                        satisfied = True
            if longestRow == 25:
                successfulGuitar = True
        print('iteration:',getDataFromRows(rows))
        array = getArrayFromRows(rows)
        plt.subplot(iteration + 1)
        plt.title(str(iteration + 1))
        plt.imshow(array)

def sin(x):
    return np.sin(x)

def arctan(x):
    return np.arctan(x)

#The following 3 functions use equations found here http://www.mnealon.eosc.edu/The_Nealon_Equation.html
def guitarCurve(x,A=-.06393446,B=-.7410887,C=1.180973,D=-1.24886,E=.3187446,F=-.8305975,G=2.352912,H=-.1870003,I=3.40192,J=-.01303915,K=1.349344,L=4.32767,M=5.228206,N=-.4099881,O=-.000250234,P=.0007021002,Q=0,R=18.26765,S=18.1965,BL=19.35):
    return (A * np.sin(B*x + C) + D * np.sin(E * x + F) + G * np.sin(H * x + I) + J * np.sin(K * x + L)) * (M * x**4 + N * x**3 + O * x**2 + P * x + Q) * (np.arctan(R * x)) * (np.arctan(S * (BL - x)))

def lesPaul(x,A=.3661125,B=-.6542102,C=-3.428853,D=.04257652,E=-1.426924,F=.9901923,G=4.458389,H=.07430593,I=1.892568,J=1.918956,K=.3106034,L=-2.663953,M=.3029042,N=.2545602,O=-.0215465,P=-.0007298772,Q=8.45777e-5,R=24.62258,S=23.2647,BL=17.275):
    return (A * sin(B*x + C) + D * sin(E*x + F) + G * sin(H*x + I) + J * sin(K*x + L)) * (M + N*x + O*(x**2) + P * (x**3) + Q * (x**4)) * (arctan(R*x)) * (arctan(S*(BL-x)))

def lesCutaway(x,A=.002098982,B=-6.793013,C=1.691257,D=3.204554,E=-.8213864,F=1.118254,G=2.366427,H=-.2342140,I=.29424849,J=4.349017,K=-.7233598,L=4.076135,M=5.072948,N=-11.91285,O=12.03647,P=-5.864441,Q=1.146872,R=16.64988,S = 98.20209,BL=3.59):
    return (A * sin(B*x + C) + D * sin(E*x + F) + G * sin(H*x + I) + J * sin(K*x + L)) * (M + N*x + O*(x**2) + P * (x**3) + Q * (x**4)) * (3-arctan(R*x)) * (arctan(S*(BL-x)))

def createVlineBetween(x,yBottom,yTop):
    ys = np.arange(yBottom,yTop,.0001)
    xs = np.array([x] * len(list(ys)))
    return xs,ys

def plotHalfRows(halfRows):
    ys = halfRows
    ys = ys * 2
    widest = 42
    ys = ys * scale
    for index, value in enumerate(ys):
        ys[index] = round(value)
    numRows = len(list(ys))
    longestRow = int(max(list(ys)))
    array = list(np.zeros((numRows,longestRow)))
    rows = ys
    for w in range(numRows):
        padding = int((longestRow-ys[w])/2)
        for z in range(padding,int(padding+ys[w])):
            array[w][z] = 1
    return array


#plotImageOnGuitar('imagePath.jpg') #uncomment to plot an image on the guitar shape



scale = 3.175 #Lego Dot per inch
xOrig = np.arange(0,17.275,.0001)
yOrig = lesPaul(xOrig)

xs = np.arange(0,round(17.275 *scale),1)
ys = lesPaul(xs/scale)

xCutaway = np.arange(0,3.59,.001)
yCutaway = lesCutaway(xCutaway)

for index, value in enumerate(ys):
    ys[index] = round(value * scale)/scale
    
##plt.plot(xOrig,yOrig,label='real')
##plt.plot(xs/scale,ys,label='lego')
##plt.legend()
##plt.show()

guitarpic = plotHalfRows(ys)
#print(len(guitarpic),len(guitarpic[0]))
plt.imshow(guitarpic,cmap='binary')
##plt.show()
##
##plt.plot(xCutaway,yCutaway)
##plt.show()


last = -5000
lastIndex = 0
for index, value in enumerate(-xOrig):
    xVal = yOrig[index]
    if xVal > last:
        last = xVal
        lastIndex = index
    else:
        break

xDelt = last - xCutaway[-1]
yDelt = -xOrig[lastIndex] - yCutaway[-1]
xCutaway = xCutaway + xDelt
yCutaway = yCutaway + yDelt
xVert,yVert = createVlineBetween(xCutaway[0],-1,0)

xOrig = xOrig * scale
yOrig = yOrig * scale
xVert *= scale
yVert *= scale
xCutaway *= scale
yCutaway *= scale

yVert *= -1
yCutaway *= -1
xOrig *= -1

xShift = 20.5

plt.plot(xVert + xShift,yVert,color='orange')
plt.plot(yOrig + xShift,-xOrig,color='blue')
plt.plot(xCutaway + xShift,yCutaway,color='orange')
plt.plot(-yOrig + xShift,-xOrig,color='blue')
plt.show()

#Remove cutout
for row in range(1,5):
    for col in range(25,35):
        guitarpic[row][col] = 0

for col in range(25,34):
    guitarpic[5][col] = 0

for col in range(25,33):
    guitarpic[6][col] = 0

for col in range(26,32):
    guitarpic[7][col] = 0

for col in range(27,31):
    guitarpic[8][col] = 0

for i in range(1,5):
    guitarpic[i][24] = 0
    
listToRemove = []
#listToRemove = [(1,24),(1,25),(4,25),(4,26),(4,27),(4,28),(4,29),(4,30),(5,25),(5,26),(5,27),(5,28),(5,29),(5,30),(5,31),(5,32),(5,33)]
for t in listToRemove:
    guitarpic[t[0]][t[1]] = 0
plt.imshow(guitarpic,cmap='binary')
plt.plot(xVert + xShift,yVert,color='orange')
plt.plot(yOrig + xShift,-xOrig,color='blue')
plt.plot(xCutaway + xShift,yCutaway,color='orange')
plt.plot(-yOrig + xShift,-xOrig,color='blue')

ax = plt.gca()
ax.set_xticks(np.arange(-.5, 42, 1))
ax.set_yticks(np.arange(-.5, 55, 1))

for label in ax.get_xticklabels():
    label.set_visible(False)
for label in ax.get_xticklabels()[::10]:
    label.set_visible(True)

for label in ax.get_yticklabels():
    label.set_visible(False)
for label in ax.get_yticklabels()[::10]:
    label.set_visible(True)

plt.grid()
plt.axvline(20.5)
plt.show()

plt.imshow(guitarpic,cmap='binary')
plt.axvline(20.5)
plt.show()

print(np.array(guitarpic).shape)

with open('guitarShape.pkl','wb') as outFile:
    pickle.dump(np.array(guitarpic),outFile)

print(type(guitarpic))
guitarpic[0][0] = 1

print(len(guitarpic), len(guitarpic[0]))
for row in guitarpic:
    print(list(row).count(1))

plt.imshow(guitarpic,cmap='binary')
ax = plt.gca()
unique, counts = np.unique(guitarpic, return_counts=True)
countDic = dict(zip(unique, counts))
print(countDic)
ax.set_xticks(np.arange(-.5, 42, 1))
ax.set_yticks(np.arange(-.5, 55, 1))

for label in ax.get_xticklabels():
    label.set_visible(False)
for label in ax.get_xticklabels()[::10]:
    label.set_visible(True)

for label in ax.get_yticklabels():
    label.set_visible(False)
for label in ax.get_yticklabels()[::10]:
    label.set_visible(True)

plt.grid()
plt.axvline(20.5)
plt.show()

'''
xs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
xs = np.arange(1,15,.125)
for y in np.arange(15,20):
    t = (math.log(40/y))/(math.log(14))
    ys = []
    for index, value in enumerate(xs):
        ys.append(3 + y * (value-1)**t)
    plt.plot(xs,ys,label=str(y))

plt.legend()
plt.show()
exit()
'''

'''
rows = []
longestRow = -1
longestRowIndex = -1

numRows = int(input('Number of rows to plot: '))
for i in range(numRows):
    num = int(input('Number of dots in row ' + str(i + 1) + ": "))
    toAdd = [1]*num
    rows.append(toAdd)
    lenRow = len(toAdd)
    if lenRow > longestRow:
        longestRow = lenRow
        longestRowIndex = i

array = list(np.zeros((numRows,longestRow)))
for w in range(numRows):
    padding = int((longestRow-len(rows[w]))/2)
    print(padding)
    for y in range(padding,padding+len(rows[w])):
        array[w][y] = 1

picture = []
for row in reversed(array):
    picture.append(row)
    
plt.imshow(picture,cmap='binary')
plt.show()
'''

'''
numRows = 15
maxIncrease = 10

for iteration in range(25):
    successfulGuitar = False
    while not successfulGuitar:
        rows = []
        longestRow = -1
        initialWidth = 1
        rows.append([1] * initialWidth)
        previousIncrease = 99999
        previousWidth = initialWidth
        for i in range(numRows - 1):
            #print(iteration,i)
            #print('--------------------------------------------------------------------------------------------')
            satisfied = False
            while not satisfied:
                tryWidth = random.randrange(3,43,2)
                increase = tryWidth - previousWidth
                #print(tryWidth,increase,previousWidth,previousIncrease)
                if increase >= 0 and increase <= previousIncrease and increase <= maxIncrease:
                    rows.append([1] * tryWidth)
                    if tryWidth > longestRow:
                        longestRow = tryWidth
                    previousIncrease = increase
                    previousWidth = tryWidth
                    satisfied = True
        #print('---------------------------------------------------------------------------------------')
        #print(longestRow)
        #print(rows)
        if longestRow == 43 or longestRow == 41 and not len(rows[-4]) == 41:
            successfulGuitar = True
    print(str(iteration + 1) + " :", getDataFromRows(rows))
    array = list(np.zeros((numRows,longestRow)))
    for y in range(numRows):
        padding = int((longestRow-len(rows[y]))/2)
        #print(padding)
        for j in range(padding,padding+len(rows[y])):
            array[y][j] = 1

    picture = []
    for row in reversed(array):
        picture.append(row)
    plt.subplot(5,5,iteration + 1)
    plt.title(str(iteration + 1))
    plt.axis('off')
    plt.imshow(picture,cmap='binary')
plt.show()
    #exit()
'''

'''
with open('guitarShapes.txt','r') as inFile:
    counter = 1
    for line in inFile:
        plt.subplot(5,4,counter)
        data = eval(line.strip())
        array = getArrayFromData(data)
        plt.title(str(counter))
        plt.axis('off')
        plt.imshow(array,cmap='binary')
        counter += 1
plt.show()
'''
