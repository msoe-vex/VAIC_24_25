import struct
import threading
from threading import Lock
import serial
import time
import math
from filter import LiveFilter
import numpy as np 
import copy

class Position:
    # Status flags for different conditions

    STATUS_CONNECTED    = 0x00000001
    STATUS_NODOTS       = 0x00000002
    STATUS_NORAWBITS    = 0x00000004
    STATUS_NOGROUPS     = 0x00000008
    STATUS_NOBITS       = 0x00000010
    STATUS_PIXELERROR   = 0x00000020
    STATUS_SOLVER       = 0x00000040
    STATUS_ANGLEJUMP    = 0x00000080
    STATUS_POSJUMP      = 0x00000100
    STATUS_NOSOLUTION   = 0x00000200
    STATUS_KALMAN_EST   = 0x00100000

    def __init__(self, frameCount: int, status: int, x: float, y: float, z: float, azimuth: float, elevation: float, rotation: float):
        # Initialization of position attributes
        self.frameCount = frameCount
        self.status = status
        self.x = x
        self.y = y
        self.z = z
        self.azimuth = azimuth
        self.elevation = elevation
        self.rotation = rotation

    def to_Serial(self):
        # Converts the position attributes to serial (binary) format
        return struct.pack('<iiffffff', self.frameCount, self.status, self.x, self.y, self.z, self.azimuth, self.elevation, self.rotation)
    
    def to_JSON(self):
        # Converts the position attributes to JSON format
        outData = {}
        outData['frameCount'] = self.frameCount
        outData['status'] = self.status
        outData['x'] = self.x
        outData['y'] = self.y
        outData['z'] = self.z
        outData['azimuth'] = self.azimuth
        outData['elevation'] = self.elevation
        outData['rotation'] = self.rotation
        return outData

class RobotLocation:
    # Position Modes
    POS_MODE_GPS_BEGIN = 0
    POS_MODE_GPS_ONLY = 1
    POS_MODE_GPS_OBJECTS = 2
    POS_MODE_ENCODER_ONLY = 3

    # Field Types
    FIELD_GPS = 0
    FIELD_BRAIN = 1
    FIELD_RL = 2

    # Constants for conversions
    INCHES_PER_METER = 39.3701
    FEET_PER_METER = 3.28084
    FIELD_LENGTH_FEET = 12
    
    def __init__(self, robot_pos_mode):
        self.__robot_pos_mode = robot_pos_mode
        self.__gpsPosition = Position(0, 0, 0, 0, 0, 0, 0, 0)
        self.__encoderPosition = Position(0, 0, 0, 0, 0, 0, 0, 0)
        self.__last_was_pos = True
        self.__gpsLock = Lock()
        self.__encoderLock = Lock()
    
    def get_pos_mode(self):
        return self.__robot_pos_mode
    
    def set_gps_pos(self, pos_obj, field):
        converted_pos = self.convert_from(field, pos_obj)
        self.__gpsLock.acquire()
        self.__gpsPosition = converted_pos
        self.__gpsLock.release()
    
    def get_gps_pos(self, field):
        self.__gpsLock.acquire()
        ret = self.__gpsPosition
        self.__gpsLock.release()
        return self.convert_to(field, ret)
    
    def set_encoder_pos(self, pos_obj, field):
        converted_pos = self.convert_from(field, pos_obj)
        self.__encoderLock.acquire()
        self.__encoderPosition = converted_pos
        self.__encoderLock.release()
    
    def get_encoder_pos(self, field):
        self.__encoderLock.acquire()
        ret = self.__encoderPosition
        self.__encoderLock.release()
        return self.convert_to(field, ret)
    
    def get_pos_for_objects(self, field):
        gps_modes = [
            RobotLocation.POS_MODE_GPS_BEGIN,
            RobotLocation.POS_MODE_GPS_ONLY,
            RobotLocation.POS_MODE_GPS_OBJECTS,
        ]
        if self.__robot_pos_mode in gps_modes:
            ret = self.get_gps_pos(field)
        else:
            ret = self.get_encoder_pos(field)
        return ret
    
    def get_pos_for_rl(self, field):
        gps_modes = [
            RobotLocation.POS_MODE_GPS_ONLY,
        ]
        if self.__robot_pos_mode in gps_modes:
            ret = self.get_gps_pos(field)
        else:
            ret = self.get_encoder_pos(field)
        return ret
    
    def get_pos_to_set_brain(self, begin_auton, field):
        gps_begin_modes = [
            RobotLocation.POS_MODE_GPS_BEGIN,
            RobotLocation.POS_MODE_GPS_ONLY,
        ]
        gps_always_modes = [
            RobotLocation.POS_MODE_GPS_ONLY,
        ]
        if (begin_auton and self.__robot_pos_mode in gps_begin_modes) \
                or (not self.__last_was_pos and self.__robot_pos_mode in gps_always_modes):
            ret = self.get_gps_pos(field)
            self.__last_was_pos = True
        else:
            ret = None
            self.__last_was_pos = False
        return ret

    # Position conversion methods
    # This class uses GPS-type coordinates internally
    
    @staticmethod
    def convert_from(field, pos_obj):
        if field == RobotLocation.FIELD_BRAIN:
            ret = RobotLocation.pos_from_brain(pos_obj)
        elif field == RobotLocation.FIELD_RL:
            ret = RobotLocation.pos_from_rl(pos_obj)
        else:
            # FIELD_GPS defaults here, no conversion needed
            ret = copy.deepcopy(pos_obj)
        return ret
    
    @staticmethod
    def convert_to(field, pos_obj):
        if field == RobotLocation.FIELD_BRAIN:
            ret = RobotLocation.pos_to_brain(pos_obj)
        elif field == RobotLocation.FIELD_RL:
            ret = RobotLocation.pos_to_rl(pos_obj)
        else:
            # FIELD_GPS defaults here, no conversion needed
            ret = copy.deepcopy(pos_obj)
        return ret
    
    @staticmethod
    def pos_to_brain(pos_obj):
        ret = copy.deepcopy(pos_obj)
        ret.x *= RobotLocation.INCHES_PER_METER
        ret.y *= RobotLocation.INCHES_PER_METER
        ret.z *= RobotLocation.INCHES_PER_METER
        return ret
    
    @staticmethod
    def pos_from_brain(pos_obj):
        ret = copy.deepcopy(pos_obj)
        ret.x /= RobotLocation.INCHES_PER_METER
        ret.y /= RobotLocation.INCHES_PER_METER
        ret.z /= RobotLocation.INCHES_PER_METER
        return ret
    
    @staticmethod
    def pos_to_rl(pos_obj):
        ret = copy.deepcopy(pos_obj)
        ret.x *= RobotLocation.FEET_PER_METER
        ret.x += RobotLocation.FIELD_LENGTH_FEET / 2
        ret.y *= RobotLocation.FEET_PER_METER
        ret.y += RobotLocation.FIELD_LENGTH_FEET / 2
        ret.z *= RobotLocation.FEET_PER_METER
        new_azimuth = 90 - ret.azimuth
        ret.azimuth = np.radians(((new_azimuth + 180) % 360) - 180)
        ret.elevation = np.radians(ret.elevation)
        ret.rotation = np.radians(ret.rotation)
        return ret
    
    @staticmethod
    def pos_from_rl(pos_obj):
        ret = copy.deepcopy(pos_obj)
        ret.x -= RobotLocation.FIELD_LENGTH_FEET / 2
        ret.x /= RobotLocation.FEET_PER_METER
        ret.y -= RobotLocation.FIELD_LENGTH_FEET / 2
        ret.y /= RobotLocation.FEET_PER_METER
        ret.z /= RobotLocation.FEET_PER_METER
        ret.azimuth = (90 - np.degrees(ret.azimuth)) % 360
        ret.elevation = np.degrees(ret.elevation)
        ret.rotation = np.degrees(ret.rotation)
        return ret

class V5GPS:
    # Packet type identifier
    __MAP_PACKET_TYPE = 0x0001


    def __init__(self, robot_loc: RobotLocation, port = None):
        # Initialization of GPS attributes including port, position, and offsets
        self.__dev = port
        self.__started = False
        self.__ser = None
        self.__isConnected = False
        self.__position = Position(0, 0, 0, 0, 0, 0, 0, 0)
        self.__positionLock = Lock()
        self.__robot_loc = robot_loc
        self.__HEADINGOFFSET =  0 # Degree offset of gps
        # When x and y offsets are updated, offsets are automatically converted to meters
        self.__GPSXOFFSET = 0  # GPS offset in default units (meters) (X-axis)
        self.__GPSYOFFSET = 0  # GPS offset in default units (meters) (Y-axis)
        self.__OFFSETUNITS = "meters"

        self.__filter = LiveFilter(10)

    def start(self):
        # Starts the GPS thread
        self.__started = True
        self.__thread = threading.Thread(target=self.__run, args=())
        self.__thread.daemon = True
        self.__thread.start()

    def __run(self):
        # Main method to run the GPS data reading and processing
        # Includes connection establishment, data reading, coordinate transformation, and status handling
        # Setting local variables for GPS Offset from global

        while self.__started:
            port = self.__dev

            try:
                if(port == None):
                    from serial.tools.list_ports import comports
                    devices = [dev for dev in comports() if "GPS" in dev.description and "User" in dev.description]
                    if len(devices) == 0:
                        print("No GPS detected.")
                        time.sleep(1)  # Wait for 1 second before retrying
                        continue
                    else:
                        self.__isConnected = True
                        port = devices[0].device

                print("Connecting to ", port)

                self.__ser = serial.Serial(port, 115200, timeout=10)
                self.__ser.flushInput()
                self.__ser.flushOutput()

                self.__frameCount = 0

                while self.__started:
                    # Read data from the serial port
                    data = self.__ser.read_until(b'\xCC\x33')
                    if(len(data) == 16):
                        self.__frameCount = self.__frameCount + 1
                        status = data[1]
                        x, y, z, az, el, rot = struct.unpack('<hhhhhh', data[2:14])
                        # Converts the data into meters and degrees
                        x = x / 10000.0
                        y = y / 10000.0
                        z = z / 10000.0
                        az = ((az/ 32768.0 * 180.0) - self.__HEADINGOFFSET) % 360
                        el = el / 32768.0 * 180.0
                        rot = rot / 32768.0 * 180.0

                        # Adjusts the x and y position to be true to robot position based on positon of GPS sensor relative to robot
                        theta = math.radians(az)
                        new_GPSXOFFSET = self.__GPSXOFFSET * math.cos(theta) + self.__GPSYOFFSET * math.sin(theta)
                        new_GPSYOFFSET = -self.__GPSXOFFSET * math.sin(theta) + self.__GPSYOFFSET * math.cos(theta)

                        x = x - new_GPSXOFFSET
                        y = y - new_GPSYOFFSET

                        localStatus = Position.STATUS_CONNECTED
                        if(status > 0 and status < 32):
                            localStatus = localStatus | (1 << status)

                        # print( x, y, z, az, el, rot, " status: ", hex(status), " Local Status: ", hex(localStatus))
                        
                        # Apply filter to smooth x, y, and azimuth

                        #save data if it is valid
                        if(status == 20):
                            x, y = self.__filter.update(x, y)
                            self.__positionLock.acquire()
                            self.__position.x = x
                            self.__position.y = y
                            self.__position.z = z
                            self.__position.azimuth = az
                            self.__position.elevation = el
                            self.__position.rotation = rot
                            self.__position.status = localStatus
                            self.__position.frameCount = self.__frameCount

                            self.__robot_loc.set_gps_pos(self.__position, RobotLocation.FIELD_GPS)
                            
                            self.__positionLock.release()

            # To close the serial port gracefully, use Ctrl+C to break the loop
            except serial.SerialException as e:
                print("Could not connect to ", port, ". Exception: ", e)
                self.__isConnected = False
                time.sleep(1)
        
            if(self.__ser.isOpen()):
                self.__ser.close()

            self.__frameCount = 0
            self.__positionLock.acquire()
            self.__position.status = 0
            self.__robot_loc.set_gps_pos(self.__position, RobotLocation.FIELD_GPS)
            self.__positionLock.release()

        print("V5SerialComms thread stopped.")

    def getPosition(self):
        # Retrieves the current position
        self.__positionLock.acquire()
        nowPosition = self.__position
        self.__positionLock.release()

        return nowPosition
    
    def isConnected(self):
        # Checks if the GPS is connected
        return self.__isConnected
    
    def updateOffset(self, newOffset):
        # Updates the offset values for GPS data
        unitDivisor = 1
        if newOffset.unit in ("CM", "cm"):
            unitDivisor = 100
        elif newOffset.unit in ("MM", "mm"):
            unitDivisor = 1000
        elif newOffset.unit in ("IN", "in", "inches"):
            unitDivisor = 39.3701
        elif newOffset.unit not in ("m", "meters", "M"):
            raise Exception("Invalid argument: Unit not accepted")
        self.__HEADINGOFFSET = newOffset.heading_offset
        self.__GPSXOFFSET = newOffset.x / unitDivisor
        self.__GPSYOFFSET = newOffset.y / unitDivisor
        self.__OFFSETUNITS = "meters"   # Units will always be saved as meters and converted from input units

    def stop(self):
        # Stops the GPS thread
        self.__started = False
        self.__thread.join()

    def __del__(self):
        # Destructor to call the stop method when the object is deleted
        self.stop