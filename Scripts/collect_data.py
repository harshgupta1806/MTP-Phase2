import struct
import time
import cv2
import numpy as np
from typing import Dict
from Scripts.Data_Collection.SynexensPythonSDK import *
from .utils import load_configs

# Dictionaries to manage stream type, PCL saving status, and depth/raw saving status
g_mapStreamType: Dict[int, SYStreamTypeEnum] = {}
g_mapSavePCL: Dict[int, bool] = {}
g_mapSaveDepthOrRaw: Dict[int, bool] = {}

# Load configuration settings from the YAML file
configs = load_configs('config/lidar.yaml')

# Extract configuration values for directories, frame numbers, and stream settings
frame_dir = configs['frame_dir'] 
pcd_dir = configs['pcd_dir']
init_frame_number = configs['init_frame_number']
streamtype = configs['streamtype']  # SYStreamTypeEnum.SYSTREAMTYPE_DEPTH
resolution = configs['resolution']  # SYResolutionEnum.SYRESOLUTION_640_480
save_pcd = configs['save_pcd']
save_frame = configs['save_frame']

# Create directories for frames and point cloud data if they don't already exist
os.makedirs(frame_dir, exist_ok=True)  # Create frame directory
os.makedirs(pcd_dir, exist_ok=True)  # Create point cloud directory

# Initialize the frame counter with the value from configuration
image_counter = init_frame_number

# Function to create OpenCV window based on stream type
def CreateOpencvWindow(nDeviceID: int, streamType: SYStreamTypeEnum, bDestoryOld: bool = False) -> None:
    name: str = ""
    
    # Check stream type and set up the corresponding window name
    if streamType == SYStreamTypeEnum.SYSTREAMTYPE_NULL:
        pass
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RAW:
        if bDestoryOld:
            name = "RGBD_depth_{}".format(nDeviceID)
            cv2.destroyWindow(name)
            name = "RGBD_RGB_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "raw_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTH:
        if bDestoryOld:
            name = "raw_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGB:
        if bDestoryOld:
            name = "depth_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "RGB_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR:
        if bDestoryOld:
            name = "RGB_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "ir_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHRGB:
        if bDestoryOld:
            name = "ir_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "RGB_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIRRGB:
        name = "depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "ir_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "RGB_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGBD:
        if bDestoryOld:
            name = "depth_{}".format(nDeviceID)
            cv2.destroyWindow(name)
            name = "ir_{}".format(nDeviceID)
            cv2.destroyWindow(name)
            name = "RGB_{}".format(nDeviceID)
            cv2.destroyWindow(name)
        name = "RGBD_depth_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = "RGBD_RGB_{}".format(nDeviceID)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

# Function to process frame data and display or save the frames
def ProcessFrameData(nDeviceID, pFrameData: POINTER(SYFrameData)):
    if g_mapStreamType.get(nDeviceID) is not None:
        print("**DeviceID ={} StreamType={}".format(nDeviceID, g_mapStreamType[nDeviceID]))
        
        # Handle RGBD stream type
        if streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGBD:
            objFrameData = pFrameData.contents
            mapIndex: Dict[SYFrameTypeEnum, int] = {}
            mapPos: Dict[SYFrameTypeEnum, int] = {}
            nPos: int = 0
            
            # Process each frame in the RGBD stream and store data positions
            for nFrameIndex in range(objFrameData.m_nFrameCount):
                mapIndex[objFrameData.m_pFrameInfo[nFrameIndex].m_frameType] = nFrameIndex
                mapPos[objFrameData.m_pFrameInfo[nFrameIndex].m_frameType] = nPos
                nPos += objFrameData.m_pFrameInfo[nFrameIndex].m_nFrameHeight * objFrameData.m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(c_short)
            
            # Extract depth and RGB frame indices
            itDepthIndex = mapIndex.get(SYFrameTypeEnum.SYFRAMETYPE_DEPTH)
            itRGBIndex = mapIndex.get(SYFrameTypeEnum.SYFRAMETYPE_RGB)
            nRGBDWidth = objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameWidth
            nRGBDHeight = objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameHeight
            
            # Allocate buffers for depth and RGB data
            pRGBDDepth = (c_ushort * (nRGBDWidth * nRGBDHeight))()
            pRGBDRGB = (c_ubyte * (nRGBDWidth * nRGBDHeight * 3))()
            
            # Check if both depth and RGB data are available
            if itDepthIndex is not None and itRGBIndex is not None:
                data_pointer_depth = cast(objFrameData.m_pData + mapPos[SYFrameTypeEnum.SYFRAMETYPE_DEPTH], POINTER(c_ushort))
                data_pointer_rgb = cast(objFrameData.m_pData + mapPos[SYFrameTypeEnum.SYFRAMETYPE_RGB], POINTER(c_ubyte))
                
                # Retrieve RGBD data
                errorCodeGetRGBD = GetRGBD(nDeviceID,
                                           objFrameData.m_pFrameInfo[itDepthIndex].m_nFrameWidth,
                                           objFrameData.m_pFrameInfo[itDepthIndex].m_nFrameHeight, data_pointer_depth,
                                           objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameWidth,
                                           objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameHeight, data_pointer_rgb,
                                           nRGBDWidth, nRGBDHeight, pRGBDDepth, pRGBDRGB)
                
                # Process and display RGBD depth and RGB frames if retrieval was successful
                if errorCodeGetRGBD == SYErrorCode.SYERRORCODE_SUCCESS:
                    # Depth frame processing
                    nCount = nRGBDHeight * nRGBDWidth
                    ArrayType = c_ubyte * (nCount * 3)
                    pColor = ArrayType()

                    rgbd_depth_bgr = np.zeros((nRGBDHeight, nRGBDWidth, 3), dtype=np.uint8)
                    gray16 = np.zeros((nRGBDHeight, nRGBDWidth), dtype=np.uint16)
                    memmove(gray16.ctypes.data, pRGBDDepth, nCount * 2)

                    # Convert depth data to color
                    if GetDepthColor(nDeviceID, nCount, pRGBDDepth, pColor) == SYErrorCode.SYERRORCODE_SUCCESS:
                        memmove(rgbd_depth_bgr.ctypes.data, pColor, nRGBDHeight * nRGBDWidth * 3)
                        rgbd_depth_rgb = cv2.cvtColor(rgbd_depth_bgr, cv2.COLOR_BGR2RGB)
                        cv2.imshow("RGBD_depth_{}".format(nDeviceID), rgbd_depth_rgb)
                    else:
                        tmp = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX)
                        gray8 = cv2.convertScaleAbs(tmp)
                        rgbimg = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)
                        cv2.imshow("RGBD_depth_{}".format(nDeviceID), rgbimg)
                    del pColor

                    # RGB frame processing
                    rgbd_bgr = np.zeros((nRGBDHeight, nRGBDWidth, 3), dtype=np.uint8)
                    memmove(rgbd_bgr.ctypes.data, pRGBDRGB, nRGBDHeight * nRGBDWidth * 3)
                    rgbd_rgb = cv2.cvtColor(rgbd_bgr, cv2.COLOR_BGR2RGB)
                    cv2.imshow("RGBD_RGB_{}".format(nDeviceID), rgbd_rgb)
                else:
                    PrintErrorCode("GetRGBD", errorCodeGetRGBD)

                del pRGBDDepth
                del pRGBDRGB
        else:
            # Handling for non-RGBD stream types (e.g., raw, depth, IR)
            objFrameData = pFrameData.contents
            nPos: int = 0
            for i in range(objFrameData.m_nFrameCount):
                nFrameHeight = objFrameData.m_pFrameInfo[i].m_nFrameHeight
                nFrameWidth = objFrameData.m_pFrameInfo[i].m_nFrameWidth
                nCount = nFrameHeight * nFrameWidth

                # Process raw frame data
                if objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_RAW:
                    pRaw = (c_ushort * nCount)()
                    ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
                    ptr_new = cast(ptr_void_new, POINTER(c_ushort))
                    memmove(pRaw, ptr_new, sizeof(c_ushort) * nCount)
                    gray16 = np.frombuffer(pRaw, dtype=np.uint16).reshape(nFrameHeight, nFrameWidth)
                    gray8 = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                    # Display raw frame
                    cv2.imshow("raw_{}".format(nDeviceID), gray8)

                    # Save raw data if required
                    if g_mapSaveDepthOrRaw.get(nDeviceID) is not None and g_mapSaveDepthOrRaw[nDeviceID]:
                        pre = f"{nDeviceID}_{objFrameData.m_pFrameInfo[i].m_nFrameWidth}x{objFrameData.m_pFrameInfo[i].m_nFrameHeight}-{int(time.time())}"
                        depth_name = f"{pre}.raw"
                        with open(depth_name, "wb") as fp:
                            fp.write(objFrameData.m_pData[nPos:].tobytes())
                        g_mapSaveDepthOrRaw[nDeviceID] = False
                    nPos += nCount * sizeof(c_short)

                # Process depth frames
                elif objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_DEPTH:
                    global image_counter
                    pDepth = (c_ushort * nCount)()
                    ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
                    ptr_new = cast(ptr_void_new, POINTER(c_ushort))
                    memmove(pDepth, ptr_new, sizeof(c_ushort) * nCount)
                    ArrayType = c_ubyte * (nCount * 3)
                    pColor = ArrayType()
                    depthimg = np.zeros((nFrameHeight, nFrameWidth, 3), dtype=np.uint8)

                    if GetDepthColor(c_uint(nDeviceID), nCount, pDepth, pColor) == SYErrorCode.SYERRORCODE_SUCCESS:
                        memmove(depthimg.ctypes.data, pColor, objFrameData.m_pFrameInfo[i].m_nFrameHeight * objFrameData.m_pFrameInfo[i].m_nFrameWidth * 3)
                        img = cv2.cvtColor(depthimg, cv2.COLOR_BGR2RGB)
                        cv2.imshow("depth_{}".format(nDeviceID), img)

                        # Save frame if required
                        if save_frame:
                            filename = os.path.join(frame_dir, f"depth_{nDeviceID}_{image_counter:04d}.png")
                            cv2.imwrite(filename, img)
                            print(f"Saved: {filename}")

                        itSavePCL = g_mapSavePCL.get(nDeviceID)
                        if itSavePCL is not None:
                            if itSavePCL:
                                if save_pcd:
                                    LP_SYPointCloudData = POINTER(SYPointCloudData)
                                    data_array = (SYPointCloudData * nCount)()
                                    pPCLData = cast(data_array, LP_SYPointCloudData)
                                    if GetDepthPointCloud(nDeviceID, nFrameWidth, nFrameHeight, pDepth,pPCLData) == SYErrorCode.SYERRORCODE_SUCCESS:
                                        filename = os.path.join(pcd_dir, f"depth_{nDeviceID}_{image_counter:04d}.pcd")
                                        # 打开文件并写入.pcd文件头信息
                                        with open(filename, "w") as fp:
                                            fp.write("# .PCD v0.7 - Point Cloud Data file format\n")
                                            fp.write("VERSION 0.7\n")
                                            fp.write("FIELDS x y z rgb\n")
                                            fp.write("SIZE 4 4 4 4\n")
                                            fp.write("TYPE F F F U\n")
                                            fp.write("COUNT 1 1 1 1\n")
                                            fp.write(f"WIDTH  {nCount}\n")
                                            fp.write("HEIGHT 1\n")
                                            fp.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                                            fp.write(f"POINTS {nCount}\n")
                                            fp.write("DATA ascii\n")
                                    
                                            for n in range(nCount):
                                                cTempType = c_ubyte * 4
                                                cTemp = cTempType()
                                                cTemp[1] = c_ubyte(pColor[n * 3])
                                                cTemp[2] = c_ubyte(pColor[n * 3 + 1])
                                                cTemp[3] = c_ubyte(pColor[n * 3 + 2])
                                                nTemp = struct.unpack('I', cTemp)[0]
                                                fp.write(f"{pPCLData[n].m_fltX} {pPCLData[n].m_fltY} {pPCLData[n].m_fltZ} {nTemp}\n")
                                    del pPCLData
                            g_mapSavePCL[nDeviceID] = True
                        image_counter += 1

                    del pColor

                # Move position for the next frame
                nPos += nCount * sizeof(c_short)


if __name__ == "__main__":
    # Print the SDK version
    print("SDKVersion:" + GetSDKVersion())
    
    # Initialize the SDK
    errorCodeInitSDK: SYErrorCode = InitSDK()
    if errorCodeInitSDK != SYErrorCode.SYERRORCODE_SUCCESS:
        PrintErrorCode("InitSDK", errorCodeInitSDK)

    # Search for connected devices
    nDeviceCount = c_int32()  # Store the number of devices found
    errorCodeFindDevice = FindDevice(byref(nDeviceCount), None)
    if errorCodeFindDevice == SYErrorCode.SYERRORCODE_SUCCESS and nDeviceCount.value > 0:
        if nDeviceCount.value <= 0:
            errorCodeRet = SYErrorCode.ERRORCODE_DEVICELISTEMPTY
        else:
            # Create a list to store information about devices
            pDeviceInfo = (SYDeviceInfo * nDeviceCount.value)()
            errorCodeFindDevice = FindDevice(nDeviceCount, pDeviceInfo)
            
            if errorCodeFindDevice == SYErrorCode.SYERRORCODE_SUCCESS:
                pOpen = np.zeros(nDeviceCount.value, dtype=bool)
                # Initialize arrays to store integral time settings
                pIntegralTimeMin = [c_int(0)] * nDeviceCount.value
                pIntegralTimeMax = [c_int(0)] * nDeviceCount.value
                pIntegralTime = [c_int(0)] * nDeviceCount.value
                
                # Loop through each found device
                for i in range(nDeviceCount.value):
                    g_mapSavePCL[pDeviceInfo[i].m_nDeviceID] = True  # Mark device for saving point cloud
                    print("DeviceID ={} Type={}".format(pDeviceInfo[i].m_nDeviceID, pDeviceInfo[i].m_deviceType.value))
                    
                    # Open the device
                    errorCodeOpenDevice: SYErrorCode = OpenDevice(pDeviceInfo[i])
                    if errorCodeOpenDevice == SYErrorCode.SYERRORCODE_SUCCESS:
                        # Retrieve device serial number and hardware version
                        strDeviceSN: str = GetDeviceSN(pDeviceInfo[i].m_nDeviceID)
                        print("DeviceSN:" + strDeviceSN)
                        strDeviceHWVersion: str = GetDeviceHWVersion(pDeviceInfo[i].m_nDeviceID)
                        print("DeviceHWVersion:" + strDeviceHWVersion)
                        
                        # Query supported frame types
                        supportTypeList = list()
                        errorCode = QueryDeviceSupportFrameType(pDeviceInfo[i].m_nDeviceID, supportTypeList)
                        if errorCode == SYErrorCode.SYERRORCODE_SUCCESS:
                            print("Device Support Frame Type:")
                            for j in range(len(supportTypeList)):
                                print("DeviceID ={} FrameType={}".format(pDeviceInfo[i].m_nDeviceID, supportTypeList[j]))
                                
                                # Query supported resolutions for each frame type
                                supportResolutionList = list()
                                errorCodeQueryResolution = QueryDeviceSupportResolution(pDeviceInfo[i].m_nDeviceID, supportTypeList[j], supportResolutionList)
                                if errorCodeQueryResolution == SYErrorCode.SYERRORCODE_SUCCESS:
                                    print("DeviceID ={} FrameType={} Resolution:".format(pDeviceInfo[i].m_nDeviceID, supportTypeList[j]))
                                    for k in range(len(supportResolutionList)):
                                        print("DeviceID ={} FrameType={} Resolution:{}".format(pDeviceInfo[i].m_nDeviceID, supportTypeList[j], supportResolutionList[k]))
                                else:
                                    errorCodeRet = SYErrorCode.SYERRORCODE_FAILED
                                    PrintErrorCode("QueryDeviceSupportResolution", errorCodeRet)

                        # Check if the device is of the CS20_DUAL type
                        deviceType = pDeviceInfo[i].m_deviceType.value
                        if deviceType == SYDeviceTypeEnum.SYDEVICETYPE_CS20_DUAL:
                            print("CS20_DUAL")
                            
                            # Set the frame resolution for the device
                            errorCode = SetFrameResolution(pDeviceInfo[i].m_nDeviceID, SYFrameTypeEnum.SYFRAMETYPE_DEPTH, SYResolutionEnum.SYRESOLUTION_640_480)
                            if errorCode == SYErrorCode.SYERRORCODE_SUCCESS:
                                # Start streaming depth data
                                streamType = SYStreamTypeEnum.SYSTREAMTYPE_DEPTH
                                errorCode = StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType)
                                if errorCode == SYErrorCode.SYERRORCODE_SUCCESS:
                                    g_mapStreamType[pDeviceInfo[i].m_nDeviceID] = streamType
                                    pOpen[i] = True
                                    CreateOpencvWindow(pDeviceInfo[i].m_nDeviceID, streamType)
                                else:
                                    PrintErrorCode("StartStreaming", errorCode)
                            else:
                                PrintErrorCode("SetFrameResolution Depth", errorCode)

                # Loop to manage frame data and user interactions
                while True:
                    for nDeviceIndex in range(0, nDeviceCount.value):
                        if pOpen[nDeviceIndex]:
                            pFrameData = POINTER(SYFrameData)()
                            errorCodeLastFrame: SYErrorCode = GetLastFrameData(pDeviceInfo[nDeviceIndex].m_nDeviceID, byref(pFrameData))
                            if errorCodeLastFrame == SYErrorCode.SYERRORCODE_SUCCESS:
                                ProcessFrameData(pDeviceInfo[nDeviceIndex].m_nDeviceID, pFrameData)
                            else:
                                pass  # Optionally handle error here
                    bBreak = False
                    res: int = cv2.waitKey(30)  # Wait for key press with a 30ms delay
                    if res == 27:  # ESC key to break the loop
                        bBreak = True
                    elif res == 85:  # U key (increase integral time)
                        for nDeviceIndex in range(nDeviceCount.value):
                            nMin = c_int(0)
                            nMax = c_int(0)
                            errorCodeIntegralTime = GetIntegralTimeRange(pDeviceInfo[nDeviceIndex].m_nDeviceID, SYResolutionEnum.SYRESOLUTION_640_480, nMin, nMax)
                            if errorCodeIntegralTime == SYErrorCode.SYERRORCODE_SUCCESS:
                                pIntegralTimeMin[nDeviceIndex] = nMin
                                pIntegralTimeMax[nDeviceIndex] = nMax
                                print("Current device integral time range:{} - {}".format(pIntegralTimeMin[nDeviceIndex].value, pIntegralTimeMax[nDeviceIndex].value))
                            else:
                                PrintErrorCode("GetIntegralTimeRange", errorCodeIntegralTime)
                            nIntegralTime = c_int(0)
                            errorCodeIntegralTime = GetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, nIntegralTime)
                            pIntegralTime[nDeviceIndex] = nIntegralTime
                            if errorCodeIntegralTime == SYErrorCode.SYERRORCODE_SUCCESS:
                                print("Current device integral time:{}".format(pIntegralTime[nDeviceIndex].value))
                            else:
                                PrintErrorCode("GetIntegralTime", errorCodeIntegralTime)
                            if pIntegralTime[nDeviceIndex].value <= pIntegralTimeMin[nDeviceIndex].value:
                                print("IntegralTime already be Min")
                            else:
                                pIntegralTime[nDeviceIndex].value += 100
                                if pIntegralTime[nDeviceIndex].value >= pIntegralTimeMax[nDeviceIndex].value:
                                    pIntegralTime[nDeviceIndex] = pIntegralTimeMax[nDeviceIndex]
                                errorCodeIntegralTime = SetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, pIntegralTime[nDeviceIndex])
                                if errorCodeIntegralTime == SYErrorCode.SYERRORCODE_SUCCESS:
                                    print("SetIntegralTime:{}".format(pIntegralTime[nDeviceIndex].value))
                                else:
                                    PrintErrorCode("SetIntegralTime", errorCodeIntegralTime)
                    elif res == 68:  # D key (decrease integral time)
                        for nDeviceIndex in range(nDeviceCount.value):
                            nMin = c_int(0)
                            nMax = c_int(0)
                            errorCodeIntegralTime = GetIntegralTimeRange(pDeviceInfo[nDeviceIndex].m_nDeviceID, SYResolutionEnum.SYRESOLUTION_640_480, nMin, nMax)
                            if errorCodeIntegralTime == SYErrorCode.SYERRORCODE_SUCCESS:
                                pIntegralTimeMin[nDeviceIndex] = nMin
                                pIntegralTimeMax[nDeviceIndex] = nMax
                                print("Current device integral time range:{} - {}".format(pIntegralTimeMin[nDeviceIndex].value, pIntegralTimeMax[nDeviceIndex].value))
                            else:
                                PrintErrorCode("GetIntegralTimeRange", errorCodeIntegralTime)
                            nIntegralTime = c_int(0)
                            errorCodeIntegralTime = GetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, nIntegralTime)
                            pIntegralTime[nDeviceIndex] = nIntegralTime
                            if errorCodeIntegralTime == SYErrorCode.SYERRORCODE_SUCCESS:
                                print("Current device integral time:{}".format(pIntegralTime[nDeviceIndex].value))
                            else:
                                PrintErrorCode("GetIntegralTime", errorCodeIntegralTime)
                            if pIntegralTime[nDeviceIndex].value <= pIntegralTimeMin[nDeviceIndex].value:
                                print("IntegralTime already be Min")
                            else:
                                pIntegralTime[nDeviceIndex].value -= 100
                                if pIntegralTime[nDeviceIndex].value <= pIntegralTimeMin[nDeviceIndex].value:
                                    pIntegralTime[nDeviceIndex] = pIntegralTimeMin[nDeviceIndex]
                                errorCodeIntegralTime = SetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, pIntegralTime[nDeviceIndex])
                                if errorCodeIntegralTime == SYErrorCode.SYERRORCODE_SUCCESS:
                                    print("SetIntegralTime:{}".format(pIntegralTime[nDeviceIndex].value))
                                else:
                                    PrintErrorCode("SetIntegralTime", errorCodeIntegralTime)
                    elif res == 70:  # F key (toggle filter)
                        for nDeviceIndex in range(nDeviceCount.value):
                            bFilter = c_bool(False)
                            errorCodeFilter = GetFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFilter)
                            if errorCodeFilter == SYErrorCode.SYERRORCODE_SUCCESS:
                                print("GetFilter Success, bFilter = {}".format(bFilter.value))
                            else:
                                PrintErrorCode("GetFilter", errorCodeFilter)
                            bFilter = c_bool(not bFilter.value)
                            errorCodeFilter = SetFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFilter)
                            if errorCodeFilter == SYErrorCode.SYERRORCODE_SUCCESS:
                                print("SetFilter Success, bFilter = {}".format(bFilter.value))
                            else:
                                PrintErrorCode("SetFilter", errorCodeFilter)
                    elif res == 77:  # M key (toggle mirror)
                        for nDeviceIndex in range(nDeviceCount.value):
                            bMirror = c_bool(False)
                            errorCodeMirror = GetMirror(pDeviceInfo[nDeviceIndex].m_nDeviceID, bMirror)
                            if errorCodeMirror == SYErrorCode.SYERRORCODE_SUCCESS:
                                print("GetMirror Success, bMirror = {}".format(bMirror.value))
                            else:
                                PrintErrorCode("GetMirror", errorCodeMirror)
                            bMirror = c_bool(not bMirror.value)
                            errorCodeMirror = SetMirror(pDeviceInfo[nDeviceIndex].m_nDeviceID, bMirror)
                            if errorCodeMirror == SYErrorCode.SYERRORCODE_SUCCESS:
                                print("SetMirror Success, bMirror = {}".format(bMirror.value))
                            else:
                                PrintErrorCode("SetMirror", errorCodeMirror)
                    elif res == 73:  # I key (toggle flip)
                        for nDeviceIndex in range(nDeviceCount.value):
                            bFlip = c_bool(False)
                            errorCodeFlip = GetFlip(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFlip)
                            if errorCodeFlip == SYErrorCode.SYERRORCODE_SUCCESS:
                                print("GetFlip Success, bFlip = {}".format(bFlip.value))
                            else:
                                PrintErrorCode("GetFlip", errorCodeFlip)
                            bFlip = c_bool(not bFlip.value)
                            errorCodeFlip = SetFlip(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFlip)
                            if errorCodeFlip == SYErrorCode.SYERRORCODE_SUCCESS:
                                print("SetFlip Success, bFlip = {}".format(bFlip.value))
                            else:
                                PrintErrorCode("SetFlip", errorCodeFlip)
                    elif res == 80:  # P key (save point cloud data)
                        for nDeviceIndex in range(nDeviceCount.value):
                            itSavePCL = g_mapSavePCL.get(pDeviceInfo[nDeviceIndex].m_nDeviceID)
                            if itSavePCL is not None:
                                g_mapSavePCL[pDeviceInfo[nDeviceIndex].m_nDeviceID] = True
                    elif res == 86:  # V key (save raw data)
                        for nDeviceIndex in range(nDeviceCount.value):
                            itSaveRAW = g_mapSaveDepthOrRaw.get(pDeviceInfo[nDeviceIndex].m_nDeviceID)
                            if itSaveRAW is not None:
                                g_mapSaveDepthOrRaw[pDeviceInfo[nDeviceIndex].m_nDeviceID] = True
                    if bBreak:
                        break

                # Stop streaming for all devices before exit
                for i in range(nDeviceCount.value):
                    if pOpen[i]:
                        errorCode = StopStreaming(pDeviceInfo[i].m_nDeviceID)

                # Clean up the allocated memory
                del pOpen
                del pIntegralTime
                del pIntegralTimeMin
                del pIntegralTimeMax
            else:
                errorCodeRet = SYErrorCode.SYERRORCODE_FAILED
                PrintErrorCode("FindDevice", errorCodeRet)
    else:
        errorCodeRet = SYErrorCode.SYERRORCODE_FAILED
        PrintErrorCode("FindDevice", errorCodeRet)
