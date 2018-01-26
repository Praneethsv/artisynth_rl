# from artisynth_helper import libraries
from artisynth_helper.libraries import *

# def loadModel(artisynth_main, name, *args):
#     classname = artisynth_main.getDemoClassName(name)
#     print('classname: ', classname)
#     if classname == None:
#         print("No class found for model " + name)
#         return False
#         if len(args) == 0:
#             artisynth_main.loadModel (classname, name, None)
#     else:
#         artisynth_main.loadModel (classname, name, args)



def main():
    #artisynth.core.driver.Main() #??
    Main = artisynth.core.driver.Main.getMain()
    if Main is not None:
        Main.quit()
    artisynth.core.driver.Main.setMain(None)
    artisynth.core.driver.Main.main([])
    Main = artisynth.core.driver.Main.getMain()
    jythonInit.Main = Main

    print(Main)
    loadModel('artisynth.demos.fem.HexFrame')
    play(2.5)

if __name__ == '__main__':
    main()