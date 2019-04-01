# Generate indexmap file on the Processed folder

import os


def IndexMap25to30_offset(captureNasName, processNasName, category, dataName):
    captureFolder = "/media/posefs{0}/Captures/{1}/{2}".format(captureNasName, category, dataName)
    processFolder = "/media/posefs{0}/Processed/{1}/{2}".format(processNasName, category, dataName)

    targetFileName = '{0}/IndexMap25to30_offset.txt'.format(processFolder)
    if os.path.exists(targetFileName):
        print ("Already exists: {0}".format(targetFileName))
    else:
        # Run process here
        print ("Processing: {0}".format(dataName))
        datasetPath = captureFolder
        syncPath = "{0}/sync/syncTables".format(datasetPath)
        outputPath = processFolder
        cmd = "./ImageExtractorVHK INDEXMAP_25_30 {0} {1} {2} 1 1000 1 t f".format(datasetPath, syncPath, outputPath)
        print (cmd)
        os.chdir('./panopticstudio/Application/syncTableGeneratorVHK/')
        os.system(cmd)
        os.chdir('../../../')

    # copy to mpm if exists
    mpmFolderPath = '{0}/body_mpm'.format(processFolder)
    targetFileName = '{0}/body_mpm/IndexMap25to30_offset.txt'.format(processFolder)

    # print "check: {0}".format(targetFileName)
    if os.path.exists(mpmFolderPath):
        # print "Check: {0}".format(targetFileName)
        if not os.path.exists(targetFileName):
            cmd = "cp -v {0}/IndexMap* {0}/body_mpm/".format(processFolder)
            print (cmd)
            os.system(cmd)
        else:
            print ("Already exist: {0}".format(targetFileName))
