# import struct
# import time
# import cv2
# import numpy as np
# from typing import Dict
# from Data_Collection.SynexensPythonSDK import *
# from collections import deque
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# import time
# from utils import load_configs

# configs = load_configs('./config/lidar.yaml')

# streamtype = configs['streamtype']  # SYStreamTypeEnum.SYSTREAMTYPE_DEPTH
# resolution = configs['resolution']  # SYResolutionEnum.SYRESOLUTION_640_480
# model_path = configs['model_path']

# g_mapStreamType: Dict[int, SYStreamTypeEnum] = {}
# g_mapSavePCL: Dict[int, bool] = {}
# g_mapSaveDepthOrRaw: Dict[int, bool] = {}



# # === Load Trained Model ===
# model = load_model(model_path)

# SEQUENCE_LENGTH = 10
# IMG_SIZE = (92, 92)
# CLASS_LABELS = ["lying", "sitting", "standing", "walking"]  # Update based on your training
# frame_buffer = deque(maxlen=SEQUENCE_LENGTH)


# image_counter = 0


# def CreateOpencvWindow(nDeviceID: int, streamType: SYStreamTypeEnum, bDestoryOld: bool = False) -> None:
#     name: str = ""
#     if streamType == SYStreamTypeEnum.SYSTREAMTYPE_NULL:
#         pass
#     elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RAW:
#         if bDestoryOld:
#             name = "RGBD_depth_{}".format(nDeviceID)
#             cv2.destroyWindow(name)
#             name = "RGBD_RGB_{}".format(nDeviceID)
#             cv2.destroyWindow(name)
#         name = "raw_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTH:
#         if bDestoryOld:
#             name = "raw_{}".format(nDeviceID)
#             cv2.destroyWindow(name)
#         name = "depth_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGB:
#         if bDestoryOld:
#             name = "depth_{}".format(nDeviceID)
#             cv2.destroyWindow(name)
#         name = "RGB_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR:
#         if bDestoryOld:
#             name = "RGB_{}".format(nDeviceID)
#             cv2.destroyWindow(name)
#         name = "depth_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#         name = "ir_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHRGB:
#         if bDestoryOld:
#             name = "ir_{}".format(nDeviceID)
#             cv2.destroyWindow(name)
#         name = "depth_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#         name = "RGB_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIRRGB:
#         name = "depth_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#         name = "ir_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#         name = "RGB_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGBD:
#         if bDestoryOld:
#             name = "depth_{}".format(nDeviceID)
#             cv2.destroyWindow(name)
#             name = "ir_{}".format(nDeviceID)
#             cv2.destroyWindow(name)
#             name = "RGB_{}".format(nDeviceID)
#             cv2.destroyWindow(name)
#         name = "RGBD_depth_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#         name = "RGBD_RGB_{}".format(nDeviceID)
#         cv2.namedWindow(name, cv2.WINDOW_NORMAL)


# def ProcessFrameData(nDeviceID, pFrameData: POINTER(SYFrameData)):
#     if g_mapStreamType.get(nDeviceID) is not None:
#         print("**DeviceID ={} StreamType={}".format(nDeviceID, g_mapStreamType[nDeviceID]))
#         if streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGBD:
#             objFrameData = pFrameData.contents
#             mapIndex: Dict[SYFrameTypeEnum, int] = {}
#             mapPos: Dict[SYFrameTypeEnum, int] = {}
#             nPos: int = 0
#             for nFrameIndex in range(objFrameData.m_nFrameCount):
#                 mapIndex[objFrameData.m_pFrameInfo[nFrameIndex].m_frameType] = nFrameIndex
#                 mapPos[objFrameData.m_pFrameInfo[nFrameIndex].m_frameType] = nPos
#                 nPos += objFrameData.m_pFrameInfo[nFrameIndex].m_nFrameHeight * objFrameData.m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(c_short)
#             itDepthIndex = mapIndex.get(SYFrameTypeEnum.SYFRAMETYPE_DEPTH)
#             itRGBIndex = mapIndex.get(SYFrameTypeEnum.SYFRAMETYPE_RGB)
#             nRGBDWidth = objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameWidth
#             nRGBDHeight = objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameHeight
#             pRGBDDepth = (c_ushort * (nRGBDWidth * nRGBDHeight))()
#             pRGBDRGB = (c_ubyte * (nRGBDWidth * nRGBDHeight * 3))()
#             if itDepthIndex is not None and itRGBIndex is not None:
#                 data_pointer_depth = cast(objFrameData.m_pData + mapPos[SYFrameTypeEnum.SYFRAMETYPE_DEPTH], POINTER(c_ushort))
#                 data_pointer_rgb = cast(objFrameData.m_pData + mapPos[SYFrameTypeEnum.SYFRAMETYPE_RGB], POINTER(c_ubyte))
#                 errorCodeGetRGBD = GetRGBD(nDeviceID,
#                                            objFrameData.m_pFrameInfo[itDepthIndex].m_nFrameWidth,
#                                            objFrameData.m_pFrameInfo[itDepthIndex].m_nFrameHeight, data_pointer_depth,
#                                            objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameWidth,
#                                            objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameHeight, data_pointer_rgb,
#                                            nRGBDWidth, nRGBDHeight, pRGBDDepth, pRGBDRGB)
#                 if errorCodeGetRGBD == SYErrorCode.SYERRORCODE_SUCCESS:
#                     # Depth
#                     nCount = nRGBDHeight * nRGBDWidth

#                     ArrayType = c_ubyte * (nCount * 3)
#                     pColor = ArrayType()

#                     rgbd_depth_bgr = np.zeros((nRGBDHeight, nRGBDWidth, 3), dtype=np.uint8)

#                     gray16 = np.zeros((nRGBDHeight, nRGBDWidth), dtype=np.uint16)
#                     memmove(gray16.ctypes.data, pRGBDDepth, nCount * 2)

#                     if GetDepthColor(nDeviceID, nCount, pRGBDDepth, pColor) == SYErrorCode.SYERRORCODE_SUCCESS:
#                         memmove(rgbd_depth_bgr.ctypes.data, pColor, nRGBDHeight * nRGBDWidth * 3)
#                         rgbd_depth_rgb = cv2.cvtColor(rgbd_depth_bgr, cv2.COLOR_BGR2RGB)
#                         cv2.imshow("RGBD_depth_{}".format(nDeviceID), rgbd_depth_rgb)
#                     else:
#                         tmp = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX)
#                         gray8 = cv2.convertScaleAbs(tmp)
#                         rgbimg = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)
#                         cv2.imshow("RGBD_depth_{}".format(nDeviceID), rgbimg)
#                     del pColor

#                     # RGB
#                     rgbd_bgr = np.zeros((nRGBDHeight, nRGBDWidth, 3), dtype=np.uint8)
#                     memmove(rgbd_bgr.ctypes.data, pRGBDRGB, nRGBDHeight * nRGBDWidth * 3)
#                     rgbd_rgb = cv2.cvtColor(rgbd_bgr, cv2.COLOR_BGR2RGB)
#                     cv2.imshow("RGBD_RGB_{}".format(nDeviceID), rgbd_rgb)
#                 else:
#                     PrintErrorCode("GetRGBD", errorCodeGetRGBD)

#                 del pRGBDDepth
#                 del pRGBDRGB
#         else:
#             objFrameData = pFrameData.contents
#             nPos: int = 0
#             for i in range(objFrameData.m_nFrameCount):
#                 nFrameHeight = objFrameData.m_pFrameInfo[i].m_nFrameHeight
#                 nFrameWidth = objFrameData.m_pFrameInfo[i].m_nFrameWidth
#                 # 计算深度图像的像素点个数
#                 nCount = nFrameHeight * nFrameWidth
#                 if objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_RAW:
#                     pRaw = (c_ushort * nCount)()
#                     ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
#                     ptr_new = cast(ptr_void_new, POINTER(c_ushort))
#                     memmove(pRaw, ptr_new, sizeof(c_ushort) * nCount)
#                     gray16 = np.frombuffer(pRaw, dtype=np.uint16).reshape(nFrameHeight, nFrameWidth)
#                     gray8 = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#                     cv2.imshow("raw_{}".format(nDeviceID), gray8)
#                     # 保存Raw图
#                     if g_mapSaveDepthOrRaw.get(nDeviceID) is not None and g_mapSaveDepthOrRaw[nDeviceID]:
#                         pre = f"{nDeviceID}_{objFrameData.m_pFrameInfo[i].m_nFrameWidth}x{objFrameData.m_pFrameInfo[i].m_nFrameHeight}-{int(time.time())}"
#                         depth_name = f"{pre}.raw"

#                         with open(depth_name, "wb") as fp:
#                             fp.write(objFrameData.m_pData[nPos:].tobytes())

#                         g_mapSaveDepthOrRaw[nDeviceID] = False
#                     nPos += nCount * sizeof(c_short)
#                 elif objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_DEPTH:
#                     # print("I am here")
#                     global image_counter
#                     pDepth = (c_ushort * nCount)()
#                     ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
#                     ptr_new = cast(ptr_void_new, POINTER(c_ushort))
#                     memmove(pDepth, ptr_new, sizeof(c_ushort) * nCount)
#                     ArrayType = c_ubyte * (nCount * 3)
#                     pColor = ArrayType()
#                     depthimg = np.zeros((nFrameHeight, nFrameWidth, 3), dtype=np.uint8)
#                     if GetDepthColor(c_uint(nDeviceID), nCount, pDepth, pColor) == SYErrorCode.SYERRORCODE_SUCCESS:
#                         memmove(depthimg.ctypes.data, pColor,objFrameData.m_pFrameInfo[i].m_nFrameHeight * objFrameData.m_pFrameInfo[i].m_nFrameWidth * 3)
#                         gray = cv2.cvtColor(depthimg, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#                         resized = cv2.resize(gray, IMG_SIZE)  # Resize to (92, 92)
#                         normalized = resized / 255.0  # Normalize to [0, 1]
#                         frame_buffer.append(normalized)  # Shape: (92, 92)

#                         display_frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)  # For visualization

#                         # Run prediction if enough frames
#                         if len(frame_buffer) == SEQUENCE_LENGTH:
#                             input_seq = np.stack(frame_buffer, axis=-1)  # Shape: (92, 92, 10)
#                             input_seq = np.expand_dims(input_seq, axis=0)  # Add batch dimension -> (1, 92, 92, 10)
                            
#                             prediction = model.predict(input_seq, verbose=0)
#                             predicted_class = CLASS_LABELS[np.argmax(prediction)]
#                             confidence = np.max(prediction)

#                             # Draw prediction on the frame
#                             text = f"Activity: {predicted_class} ({confidence*100:.1f}%)"
#                             cv2.putText(display_frame, text, (10, 20),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
#                         else:
#                             # Show buffering status
#                             text = f"Buffering: {len(frame_buffer)}/{SEQUENCE_LENGTH}"
#                             cv2.putText(display_frame, text, (10, 20),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

#         # Show final frame
#                         window_name = f"depth_{nDeviceID}"
#                         cv2.imshow(window_name, cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

#                     else:
#                         gray16 = np.zeros((nFrameHeight, nFrameWidth), dtype=np.uint16)
#                         memmove(gray16.ctypes.data, pDepth, nCount * 2)
#                         tmp = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX)
#                         gray8 = cv2.convertScaleAbs(tmp)
#                         rgbimg = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)
#                         cv2.imshow("depth_{}".format(nDeviceID), rgbimg)
#                     del pColor
#                     nPos += nCount * sizeof(c_short)
#                 elif objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_IR:
#                     pIR = (c_ushort * nCount)()
#                     ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
#                     ptr_new = cast(ptr_void_new, POINTER(c_ushort))
#                     memmove(pIR, ptr_new, sizeof(c_ushort) * nCount)
#                     gray16 = np.frombuffer(pIR, dtype=np.uint16).reshape(nFrameHeight, nFrameWidth)

#                     gray8 = np.zeros((nFrameHeight, nFrameWidth), dtype=np.uint8)
#                     cv2.convertScaleAbs(gray16, gray8, 0.5, 0)

#                     cv2.imshow("ir_{}".format(nDeviceID), gray8)
#                     nPos += nCount * sizeof(c_short)

#                 elif objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_RGB:
#                     data_pointer = cast(objFrameData.m_pData + nPos, POINTER(c_ubyte))
#                     data_array = np.ctypeslib.as_array(data_pointer, shape=(nFrameHeight * nFrameWidth * 2,))
#                     img_yuyv_cvt = data_array.reshape(nFrameHeight, nFrameWidth, 2)
#                     yuyv_array = np.frombuffer(img_yuyv_cvt, dtype=np.uint8)
#                     yuyv_image = yuyv_array.reshape((nFrameHeight, nFrameWidth, 2))
#                     rgb_image = cv2.cvtColor(yuyv_image, cv2.COLOR_YUV2BGR_YUYV)
#                     cv2.imshow("RGB_{}".format(nDeviceID), rgb_image)
#                     nPos += nCount * 3 / 2
import struct
import time
import cv2
import numpy as np
from typing import Dict
from Data_Collection.SynexensPythonSDK import *  # Import Synexens SDK for handling sensor data
from collections import deque
from tensorflow.keras.models import load_model  # For loading the trained Keras model
import time
from utils import load_configs  # To load configuration files (like stream settings)

# Load configuration settings from YAML file
configs = load_configs('./config/lidar.yaml')

# Extract stream type, resolution, and model path from the config
streamtype = configs['streamtype']  # SYStreamTypeEnum.SYSTREAMTYPE_DEPTH
resolution = configs['resolution']  # SYResolutionEnum.SYRESOLUTION_640_480
model_path = configs['model_path']

# Dictionaries for managing stream types, save settings, etc.
g_mapStreamType: Dict[int, SYStreamTypeEnum] = {}
g_mapSavePCL: Dict[int, bool] = {}
g_mapSaveDepthOrRaw: Dict[int, bool] = {}

# Load the trained model for activity recognition
model = load_model(model_path)

# Sequence length for activity recognition (e.g., number of frames to consider for prediction)
SEQUENCE_LENGTH = 10
IMG_SIZE = (92, 92)  # Image size for input to the model
CLASS_LABELS = ["lying", "sitting", "standing", "walking"]  # Possible activity classes
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)  # Buffer to store frames for activity prediction

image_counter = 0  # Counter for images

# Function to create OpenCV windows for displaying sensor data
def CreateOpencvWindow(nDeviceID: int, streamType: SYStreamTypeEnum, bDestoryOld: bool = False) -> None:
    name: str = ""
    if streamType == SYStreamTypeEnum.SYSTREAMTYPE_NULL:
        pass  # No action needed for null stream type
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RAW:
        if bDestoryOld:
            cv2.destroyWindow(f"RGBD_depth_{nDeviceID}")
            cv2.destroyWindow(f"RGBD_RGB_{nDeviceID}")
        name = f"raw_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTH:
        if bDestoryOld:
            cv2.destroyWindow(f"raw_{nDeviceID}")
        name = f"depth_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGB:
        if bDestoryOld:
            cv2.destroyWindow(f"depth_{nDeviceID}")
        name = f"RGB_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR:
        if bDestoryOld:
            cv2.destroyWindow(f"RGB_{nDeviceID}")
        name = f"depth_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = f"ir_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHRGB:
        if bDestoryOld:
            cv2.destroyWindow(f"ir_{nDeviceID}")
        name = f"depth_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = f"RGB_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIRRGB:
        name = f"depth_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = f"ir_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = f"RGB_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    elif streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGBD:
        if bDestoryOld:
            cv2.destroyWindow(f"depth_{nDeviceID}")
            cv2.destroyWindow(f"ir_{nDeviceID}")
            cv2.destroyWindow(f"RGB_{nDeviceID}")
        name = f"RGBD_depth_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        name = f"RGBD_RGB_{nDeviceID}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

# Function to process incoming frame data from the Synexens sensor
def ProcessFrameData(nDeviceID, pFrameData: POINTER(SYFrameData)):
    # Check if a valid stream type is assigned for the device
    if g_mapStreamType.get(nDeviceID) is not None:
        print(f"**DeviceID = {nDeviceID} StreamType = {g_mapStreamType[nDeviceID]}")

        # If the stream type is RGBD (Depth + RGB), process both depth and RGB data
        if streamType == SYStreamTypeEnum.SYSTREAMTYPE_RGBD:
            objFrameData = pFrameData.contents  # Access the frame data

            # Mapping frame data types (depth, RGB) to corresponding frame indices
            mapIndex: Dict[SYFrameTypeEnum, int] = {}
            mapPos: Dict[SYFrameTypeEnum, int] = {}
            nPos: int = 0
            for nFrameIndex in range(objFrameData.m_nFrameCount):
                mapIndex[objFrameData.m_pFrameInfo[nFrameIndex].m_frameType] = nFrameIndex
                mapPos[objFrameData.m_pFrameInfo[nFrameIndex].m_frameType] = nPos
                nPos += objFrameData.m_pFrameInfo[nFrameIndex].m_nFrameHeight * objFrameData.m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(c_short)

            # Extract depth and RGB frame data
            itDepthIndex = mapIndex.get(SYFrameTypeEnum.SYFRAMETYPE_DEPTH)
            itRGBIndex = mapIndex.get(SYFrameTypeEnum.SYFRAMETYPE_RGB)
            nRGBDWidth = objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameWidth
            nRGBDHeight = objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameHeight

            # Allocate memory for depth and RGB data
            pRGBDDepth = (c_ushort * (nRGBDWidth * nRGBDHeight))()
            pRGBDRGB = (c_ubyte * (nRGBDWidth * nRGBDHeight * 3))()

            if itDepthIndex is not None and itRGBIndex is not None:
                # Fetch depth and RGB data from the sensor
                data_pointer_depth = cast(objFrameData.m_pData + mapPos[SYFrameTypeEnum.SYFRAMETYPE_DEPTH], POINTER(c_ushort))
                data_pointer_rgb = cast(objFrameData.m_pData + mapPos[SYFrameTypeEnum.SYFRAMETYPE_RGB], POINTER(c_ubyte))

                # Retrieve RGBD data
                errorCodeGetRGBD = GetRGBD(nDeviceID, objFrameData.m_pFrameInfo[itDepthIndex].m_nFrameWidth,
                                           objFrameData.m_pFrameInfo[itDepthIndex].m_nFrameHeight, data_pointer_depth,
                                           objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameWidth,
                                           objFrameData.m_pFrameInfo[itRGBIndex].m_nFrameHeight, data_pointer_rgb,
                                           nRGBDWidth, nRGBDHeight, pRGBDDepth, pRGBDRGB)

                if errorCodeGetRGBD == SYErrorCode.SYERRORCODE_SUCCESS:
                    # If RGBD data is successfully fetched, process depth and RGB frames
                    nCount = nRGBDHeight * nRGBDWidth

                    # Process depth frame
                    ArrayType = c_ubyte * (nCount * 3)
                    pColor = ArrayType()

                    rgbd_depth_bgr = np.zeros((nRGBDHeight, nRGBDWidth, 3), dtype=np.uint8)

                    gray16 = np.zeros((nRGBDHeight, nRGBDWidth), dtype=np.uint16)
                    memmove(gray16.ctypes.data, pRGBDDepth, nCount * 2)

                    # Convert depth data to color for visualization
                    if GetDepthColor(nDeviceID, nCount, pRGBDDepth, pColor) == SYErrorCode.SYERRORCODE_SUCCESS:
                        memmove(rgbd_depth_bgr.ctypes.data, pColor, nRGBDHeight * nRGBDWidth * 3)
                        rgbd_depth_rgb = cv2.cvtColor(rgbd_depth_bgr, cv2.COLOR_BGR2RGB)
                        cv2.imshow(f"RGBD_depth_{nDeviceID}", rgbd_depth_rgb)
                    else:
                        tmp = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX)
                        gray8 = cv2.convertScaleAbs(tmp)
                        rgbimg = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)
                        cv2.imshow(f"RGBD_depth_{nDeviceID}", rgbimg)

                    del pColor

                    # Process RGB frame
                    rgbd_bgr = np.zeros((nRGBDHeight, nRGBDWidth, 3), dtype=np.uint8)
                    memmove(rgbd_bgr.ctypes.data, pRGBDRGB, nRGBDHeight * nRGBDWidth * 3)
                    rgbd_rgb = cv2.cvtColor(rgbd_bgr, cv2.COLOR_BGR2RGB)
                    cv2.imshow(f"RGBD_RGB_{nDeviceID}", rgbd_rgb)

                else:
                    # Handle error if RGBD data could not be retrieved
                    PrintErrorCode("GetRGBD", errorCodeGetRGBD)

                del pRGBDDepth
                del pRGBDRGB
        else:
            # If not RGBD, process other frame types (e.g., raw, depth, IR, etc.)
            objFrameData = pFrameData.contents
            nPos: int = 0
            for i in range(objFrameData.m_nFrameCount):
                nFrameHeight = objFrameData.m_pFrameInfo[i].m_nFrameHeight
                nFrameWidth = objFrameData.m_pFrameInfo[i].m_nFrameWidth
                nCount = nFrameHeight * nFrameWidth

                # Handle raw frame data
                if objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_RAW:
                    pRaw = (c_ushort * nCount)()
                    ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
                    ptr_new = cast(ptr_void_new, POINTER(c_ushort))
                    memmove(pRaw, ptr_new, sizeof(c_ushort) * nCount)
                    gray16 = np.frombuffer(pRaw, dtype=np.uint16).reshape(nFrameHeight, nFrameWidth)
                    gray8 = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    cv2.imshow(f"raw_{nDeviceID}", gray8)

                    # Optionally save raw depth data
                    if g_mapSaveDepthOrRaw.get(nDeviceID) is not None and g_mapSaveDepthOrRaw[nDeviceID]:
                        pre = f"{nDeviceID}_{objFrameData.m_pFrameInfo[i].m_nFrameWidth}x{objFrameData.m_pFrameInfo[i].m_nFrameHeight}-{int(time.time())}"
                        depth_name = f"{pre}.raw"
                        with open(depth_name, "wb") as fp:
                            fp.write(objFrameData.m_pData[nPos:].tobytes())

                        g_mapSaveDepthOrRaw[nDeviceID] = False
                    nPos += nCount * sizeof(c_short)

                # Handle depth frame data
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
                        gray = cv2.cvtColor(depthimg, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, IMG_SIZE)
                        normalized = resized / 255.0
                        frame_buffer.append(normalized)

                        display_frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

                        # Make prediction if enough frames are in the buffer
                        if len(frame_buffer) == SEQUENCE_LENGTH:
                            input_seq = np.stack(frame_buffer, axis=-1)
                            input_seq = np.expand_dims(input_seq, axis=0)  # Add batch dimension

                            prediction = model.predict(input_seq, verbose=0)
                            predicted_class = CLASS_LABELS[np.argmax(prediction)]
                            confidence = np.max(prediction)

                            # Display prediction on the frame
                            text = f"Activity: {predicted_class} ({confidence*100:.1f}%)"
                            cv2.putText(display_frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
                        else:
                            # Show buffering status
                            text = f"Buffering: {len(frame_buffer)}/{SEQUENCE_LENGTH}"
                            cv2.putText(display_frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

                    window_name = f"depth_{nDeviceID}"
                    cv2.imshow(window_name, cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

                    del pColor
                    nPos += nCount * sizeof(c_short)

                # Handle IR frame data
                elif objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_IR:
                    pIR = (c_ushort * nCount)()
                    ptr_void_new = c_void_p(objFrameData.m_pData + nPos)
                    ptr_new = cast(ptr_void_new, POINTER(c_ushort))
                    memmove(pIR, ptr_new, sizeof(c_ushort) * nCount)
                    gray16 = np.frombuffer(pIR, dtype=np.uint16).reshape(nFrameHeight, nFrameWidth)

                    gray8 = np.zeros((nFrameHeight, nFrameWidth), dtype=np.uint8)
                    cv2.convertScaleAbs(gray16, gray8, 0.5, 0)
                    cv2.imshow(f"ir_{nDeviceID}", gray8)
                    nPos += nCount * sizeof(c_short)

                # Handle RGB frame data
                elif objFrameData.m_pFrameInfo[i].m_frameType == SYFrameTypeEnum.SYFRAMETYPE_RGB:
                    data_pointer = cast(objFrameData.m_pData + nPos, POINTER(c_ubyte))
                    data_array = np.ctypeslib.as_array(data_pointer, shape=(nFrameHeight * nFrameWidth * 2,))
                    img_yuyv_cvt = data_array.reshape(nFrameHeight, nFrameWidth, 2)
                    yuyv_array = np.frombuffer(img_yuyv_cvt, dtype=np.uint8).reshape(nFrameHeight, nFrameWidth, 2)
                    frame = cv2.cvtColor(yuyv_array, cv2.COLOR_YUV2BGR_YUYV)
                    cv2.imshow(f"RGB_{nDeviceID}", frame)
                    nPos += nCount * sizeof(c_short)

            # Handle display update and quitting
            cv2.waitKey(1)

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
