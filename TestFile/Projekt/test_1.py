from ObjectHandler import *

ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)

for i in range(1,20):
    print('---------------------------------------')
    success = ObjectHandler.updateObjectStates()
    if success == True:
        print('updated states for time step ' + str(i))
    else:
        print('could not update states for time step ' + str(i))
    #img = ObjectHandler.getImg()
    #plt.imshow(img)
    #plt.plot(400,400,'r+')
    #plt.show()
    #a=1
    #print('last object states:')
    #print(ObjectHandler.getLastObjectStates())
    #cv2.waitKey(1000)

print('---------------------------------------')
#print('current time step: ' + str(ObjectHandler.getTimeStepCount()))
#print(ObjectHandler.getLastObjectStates())
#print('---------------------------------------')

i = 1
try:
    print('get data for timestep ' + str(i) + ':')
    
    print(ObjectHandler.getObjectStates(i))
except InvalidTimeStepError as e:
    print(e.args[0])
print('---------------------------------------')


