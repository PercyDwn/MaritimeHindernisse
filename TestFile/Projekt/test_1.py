from ObjectHandler import *

ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(0)

for i in range(1,21):
    success = ObjectHandler.updateObjectStates()
    if success == True:
        print('updated states for time step ' + str(i))
    else:
        print('could not update states for time step ' + str(i))
    #print('last object states:')
    #print(ObjectHandler.getLastObjectStates())
    #cv2.waitKey(1000)

print('current time step: ' + str(ObjectHandler.getTimeStepCount()))
print(ObjectHandler.getLastObjectStates())

for i in range(21,22):
    try:
        print('get data for timestep ' + str(i) + ':')
        print(ObjectHandler.getObjectStates(i))
    except InvalidTimeStepError as e:
        print(e.args[0])

