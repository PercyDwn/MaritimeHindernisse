from ObjectHandler import *

# ObjectHandler init
ObjectHandler = ObjectHandler()
ObjectHandler.setImageFolder('/TestFile/Projekt/Bilder/list1_0')
ObjectHandler.setImageBaseName('')
ObjectHandler.setImageFileType('.jpg')
ObjectHandler.setDebugLevel(2)
ObjectHandler.setPlotOnUpdate(True)

states = ObjectHandler.getObjectStates(t = 1)
print(states)