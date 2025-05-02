from ctypes import *

class SYErrorCode(c_int):
    SYERRORCODE_SUCCESS = 0  # Success
    SYERRORCODE_FAILED = 1  # Failed
    SYERRORCODE_DEVICENOTEXIST = 2  # Device does not exist
    SYERRORCODE_DEVICENOTOPENED = 3  # Device not opened
    SYERRORCODE_UNKOWNRESOLUTION = 4  # Unsupported resolution
    SYERRORCODE_DEVICEHANDLEEMPTY = 5  # Device handle is null
    SYERRORCODE_SETOUTPUTFORMATFAILED = 6  # Failed to set device output format
    SYERRORCODE_GETSTREAMCTRLFAILED = 7  # Failed to get video stream control pointer
    SYERRORCODE_STARTSTREAMINGFAILED = 8  # Failed to start video stream
    SYERRORCODE_COMMUNICATEOBJECTEMPTY = 9  # Communication pointer is null
    SYERRORCODE_UNKOWNSN = 10  # Invalid SN number
    SYERRORCODE_STRINGLENGTHOUTRANGE = 11  # String length overflow
    SYERRORCODE_UNKOWNFRAMETYPE = 12  # Invalid frame type
    SYERRORCODE_UNKOWNDEVICETYPE = 13  # Invalid device type
    SYERRORCODE_DEVICEOBJECTEMPTY = 14  # Device object pointer is null
    SYERRORCODE_OBSERVEREMPTY = 15  # Observer pointer is null
    SYERRORCODE_OBSERVERNOTFOUND = 16  # Observer not found
    SYERRORCODE_COUNTOUTRANGE = 17  # Count overflow
    SYERRORCODE_UVCINITFAILED = 18  # UVC initialization failed
    SYERRORCODE_UVCFINDDEVICEFAILED = 19  # Failed to find UVC device
    SYERRORCODE_NOFRAME = 20  # No data frame
    SYERRORCODE_GETAPPFOLDERPATHFAILED = 21  # Failed to get application path
    SYERRORCODE_NOSTREAMING = 22  # Video stream not started
    SYERRORCODE_RECONSTRUCTIONEMPTY = 23  # Algorithm pointer is null
    SYERRORCODE_STREAMINGEXIST = 24  # Video stream already started
    SYERRORCODE_UNKOWNSTREAMTYPE = 25  # Unknown stream type
    SYERRORCODE_DATABUFFEREMPTY = 26  # Data pointer is null

class SYDeviceTypeEnum(c_int):
    SYDEVICETYPE_NULL = 0  # Invalid
    SYDEVICETYPE_CS30_DUAL = 1  # CS30 dual-frequency
    SYDEVICETYPE_CS30_SINGLE = 2  # CS30 single-frequency
    SYDEVICETYPE_CS20_DUAL = 3  # CS20 dual-frequency
    SYDEVICETYPE_CS20_SINGLE = 4  # CS20 single-frequency
    SYDEVICETYPE_CS20_P = 5  # CS20_P
    SYDEVICETYPE_CS40 = 6  # CS40
    SYDEVICETYPE_CS40PRO = 7  # CS40 Pro

class SYStreamTypeEnum(c_int):
    SYSTREAMTYPE_NULL = 0  # Invalid
    SYSTREAMTYPE_RAW = 1  # RAW
    SYSTREAMTYPE_DEPTH = 2  # Depth
    SYSTREAMTYPE_RGB = 3  # RGB
    SYSTREAMTYPE_DEPTHIR = 4  # Depth + IR
    SYSTREAMTYPE_DEPTHRGB = 5  # Depth + RGB
    SYSTREAMTYPE_DEPTHIRRGB = 6  # Depth + IR + RGB
    SYSTREAMTYPE_RGBD = 7  # RGBD (depth + RGB after mapping)
    SYSTREAMTYPE_RAWRGB = 8  # RAW + RGB

class SYFrameTypeEnum(c_int):
    SYFRAMETYPE_NULL = 0  # Invalid
    SYFRAMETYPE_RAW = 1  # RAW
    SYFRAMETYPE_DEPTH = 2  # Depth
    SYFRAMETYPE_IR = 3  # IR
    SYFRAMETYPE_RGB = 4  # RGB

class SYSupportTypeEnum(c_int):
    SYSUPPORTTYPE_NULL = 0  # Invalid
    SYSUPPORTTYPE_DEPTH = 1  # Depth
    SYSUPPORTTYPE_RGB = 2  # RGB
    SYSUPPORTTYPE_RGBD = 3  # RGBD

class SYResolutionEnum(c_int):
    SYRESOLUTION_NULL = 0  # Invalid
    SYRESOLUTION_320_240 = 1  # 320x240
    SYRESOLUTION_640_480 = 2  # 640x480
    SYRESOLUTION_960_540 = 3  # 960x540
    SYRESOLUTION_1920_1080 = 4  # 1920x1080
    SYRESOLUTION_1600_1200 = 5  # 1600x1200

class SYFilterTypeEnum(c_int):
    SYFILTERTYPE_NULL = 0  # Invalid
    SYFILTERTYPE_MEDIAN = 1  # Median filter
    SYFILTERTYPE_AMPLITITUD = 2  # Amplitude filter
    SYFILTERTYPE_EDGE = 3  # Edge filter
    SYFILTERTYPE_SPECKLE = 4  # Speckle filter
    SYFILTERTYPE_OKADA = 5  # Okada threshold
    SYFILTERTYPE_EDGE_MAD = 6  # Edge filter (MAD)
    SYFILTERTYPE_GAUSS = 7  # Gaussian filter
    SYFILTERTYPE_EXTRA = 8  # Reserved
    SYFILTERTYPE_EXTRA2 = 9  # Reserved 2

class SYDeviceInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_nDeviceID', c_uint),
        ('m_deviceType', SYDeviceTypeEnum),
        ('m_nUsbBus', c_uint),
        ('m_nUsbPorts', c_uint * 7),
        ('m_nUsbPortsNumber', c_uint),
        ('m_nUsbDeviceAddress', c_uint),
    ]

class SYFrameInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_frameType', c_int),
        ('m_nFrameHeight', c_int),
        ('m_nFrameWidth', c_int)
    ]

class SYFrameData(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_nFrameCount', c_int),
        ('m_pFrameInfo', POINTER(SYFrameInfo)),
        ('m_pData', c_void_p),
        ('m_nBuffferLength', c_int),
    ]

class SYPointCloudData(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_fltX', c_float),
        ('m_fltY', c_float),
        ('m_fltZ', c_float)
    ]

class SYIntrinsics(Structure):
    _pack_ = 1
    _fields_ = [
        ('m_fltFOV', c_float * 2),
        ('m_fltCoeffs', c_float * 5),
        ('m_fltFocalDistanceX', c_float),
        ('m_fltFocalDistanceY', c_float),
        ('m_fltCenterPointX', c_float),
        ('m_fltCenterPointY', c_float),
        ('m_nWidth', c_int),
        ('m_nHeight', c_int),
    ]
