from ObjectHandler import *

ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('Bilder/list1')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')

for i in range(1,21):
    ObjectHandler.updateObjectStates()
    print('last object states:')
    print(ObjectHandler.getLastObjectStates())
    cv2.waitKey(1000)