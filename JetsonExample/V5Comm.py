from ctypes import Array
import struct
import threading
from threading import Lock
import json
from json import JSONEncoder
import serial
import time
from V5Position import Position
from typing import Callable
    
class ImageDetection:
    def __init__(self, x: int, y: int, width: int, height: int):
        # Initialize properties of ImageDetection class for x, y coordinates, width, and height
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def to_Serial(self):
        # Convert ImageDetection properties to serialized binary format
        return struct.pack('<iiii', self.x, self.y, self.width, self.height)
    
    def to_JSON(self):
        # Convert ImageDetection properties to JSON format
        return self.__dict__

class MapDetection:
    def __init__(self, x: float, y: float, z: float):
        # Initialize properties of MapDetection class for x, y, z coordinates
        self.x = x
        self.y = y
        self.z = z

    def to_Serial(self):
        # Convert MapDetection properties to serialized binary format
        return struct.pack('<fff', self.x, self.y, self.z)
    
    def to_JSON(self):
        # Convert MapDetection properties to JSON format
        return self.__dict__
    
class Detection:
    def __init__(self, classID: int, probability: float, depth: float, screenLocation: ImageDetection, mapLocation: MapDetection):
        # Initialize properties of Detection class, including class ID, probability, depth, and locations on screen and on the field
        self.classID = classID
        self.probability = probability
        self.depth = depth
        self.screenLocation = screenLocation
        self.mapLocattion = mapLocation

    def to_Serial(self):
        # Convert Detection properties to serialized binary format
        data = struct.pack('<iff', self.classID, self.probability, self.depth)
        data += self.screenLocation.to_Serial()
        data += self.mapLocattion.to_Serial()
        return data
    
    def to_JSON(self):
        # Convert Detection properties to JSON format
        outData = {}
        outData['class'] = self.classID
        outData['prob'] = self.probability
        outData['depth'] = self.depth
        outData['screenLocation'] = self.screenLocation.to_JSON()
        outData['mapLocation'] = self.mapLocattion.to_JSON()
        return outData


class AIRecord:
    # The AIRecord is what is communicated from the Jetson to the V5 Brain as a detection
    def __init__(self, position: Position, detections: "list[Detection]"):
        # Initialize properties of AIRecord class, including position and detections list
        self.position = position
        self.detections = detections

    def to_Serial(self):
        # Convert AIRecord properties to serialized binary format
        data = struct.pack('<i', len(self.detections))
        data += self.position.to_Serial()
        for det in self.detections:
            data += det.to_Serial()
        return data
    
    def to_JSON(self):
        # Convert AIRecord properties to JSON format
        outData = {}
        outData['position'] = self.position.to_JSON()
        outData['detections'] = [det.to_JSON() for det in self.detections]
        return outData

    POLYNOMIAL_CRC32 = 0x04C11DB7

    __crc32_table = [0] * 256
    __table32Generated = 0

    def __Crc32GenerateTable(self):
        for i in range(256):
            crc_accum = i << 24
            for j in range(8):
                if crc_accum & 0x80000000:
                    crc_accum = (crc_accum << 1) ^ AIRecord.POLYNOMIAL_CRC32
                else:
                    crc_accum = crc_accum << 1
            AIRecord.__crc32_table[i] = crc_accum
        AIRecord.__table32Generated = 1

    def __Crc32Generate(self, data, accumulator):
        i, j = 0, 0
        if not AIRecord.__table32Generated:
            self.__Crc32GenerateTable()
        for j in range(len(data)):
            i = ((accumulator >> 24) ^ data[j]) & 0xFF
            accumulator = (accumulator << 8) ^ AIRecord.__crc32_table[i]
        return accumulator

    def getCRC32(self):
        data = self.to_Serial()
        crc = self.__Crc32Generate(data, 0) & 0xFFFFFFFF
        return crc
    

class Observation:  # TODO: This is just a dummy object right now
    def __init__(self):
        pass

    def __str__(self):
        return 'DUMMY PACKET'
    
    def update(self, new_data: str):
        pass

    def from_String(data: str):
        self = Observation()
        return self


class V5SerialPacket:
    def __init__(self, header: str, content: str):
        # Initialize properties of V5SerialPacket class, including type and detections
        self.__header = header
        self.__content = content

    def to_Serial(self):
        # Convert V5SerialPacket properties to serialized binary format
        data = bytearray()
        data += b'#'
        data += self.__header.encode()
        data += b'|'
        data += self.__content.encode()
        data += b'\n'
        return data
    
    def from_Serial(data: str):
        if len(data) < 1 or data[0] != '#':
            return None
        
        stripped_data = data.rstrip()[1:]
        if '|' in stripped_data:
            pipe_idx = stripped_data.index('|')
            header = stripped_data[:pipe_idx]
            content = stripped_data[pipe_idx+1:]
        else:
            header = stripped_data
            content = ''
        
        self = V5SerialPacket(header, content)
        return self

    def get_header(self):
        return self.__header
    
    def get_content(self):
        return self.__content


class V5SerialComms:  # TODO This is unfinished

    __MAP_PACKET_TYPE = 0x0001

    def __init__(self, port = None, debug = False):
        # Initialize properties of V5SerialComms class, including port, started status, and lock
        self.__dev = port
        self.__started = False
        self.__ser = None
        self.__observation = Observation()
        self.__lock = Lock()
        self.__next_action = None
        self.__action_sent = True
        self.__debug = debug

    def start(self):
        # Start serial communication thread
        self.__started = True
        self.__thread = threading.Thread(target=self.__run, args=())
        self.__thread.daemon = True
        self.__thread.start()

    def __run(self):
        #count = 1
        while self.__started:  # Continue running while the thread is started
            port = self.__dev
            try:
                if(port == None):  # If the port is not specified
                    from serial.tools.list_ports import comports
                    # Find devices that match the V5 description
                    devices = [dev for dev in comports() if "V5" in dev.description and "User" in dev.description]
                    # self.devices = [dev for dev in comports()]
                    # print(self.devices)
                    if(len(devices) == 0):  # and count <= 5):
                        print("No V5 Brain detected.")
                        time.sleep(1)  # Wait for 1 second before retrying
                        #count += 1
                        continue
#                    elif(count > 5):
#                        return None  # Return None if no devices found after 5 tries
#                        break
                    else:
                        port = devices[0].device  # Return None if no devices found after 3 tries
                    
                print("Connecting to ", port)

                # Establish serial connection with the port
                self.__ser = serial.Serial(port, 115200, timeout=10)
                self.__ser.flushInput()
                self.__ser.flushOutput()

                while self.__started:  # Continue reading while thread is started
                    # Read data from the serial port
                    data = self.__ser.readline().decode("utf-8").rstrip()
                    # print(data)
                    packet = V5SerialPacket.from_Serial(data)
                    if packet is None:
                        continue
                    if self.__debug:
                        # Send test message back to the V5 Brain
                        if(packet.get_header() == "autoStart"):
                            self.sendPacket("runAction", "Pick up goal")
                            
                        print(f'Packet received, header: "{packet.get_header()}", data: "{packet.get_content()}"')
                    if packet.get_header() == "observation":
                        #get robot observation
                        self.__lock.acquire()
                        self.__observation.update(packet.get_content())
                        self.__lock.release()
                    elif packet.get_header() == "ready":
                        #send action to robot
                        self.__lock.acquire()
                        self.__observation.update(packet.get_content())
                        if self.__next_action is not None and not self.__action_sent:
                            to_write = V5SerialPacket('action', str(self.__next_action))
                            self.__ser.write(to_write.to_Serial())
                            if self.__debug:
                                print(f'Packet sent, header: "{to_write.get_header()}", contents: "{to_write.get_content()}"')
                            self.__action_sent = True
                        self.__lock.release()

            # To close the serial port gracefully, use Ctrl+C to break the loop
            except serial.SerialException as e:
                print("Could not connect to ", port, ". Exception: ", e)
                time.sleep(1)    # Wait for 1 second before retrying
        
            if(self.__ser.isOpen()):
                self.__ser.close()    # Close the serial port if open

        print("V5SerialComms thread stopped.")

    def sendPacket(self, header: str, body: str):
        # Send a packet with the specified header and body over the serial connection
        if self.__ser and self.__ser.isOpen():
            packet = V5SerialPacket(header, body)
            self.__ser.write(packet.to_Serial())
            if self.__debug:
                print(f'Packet sent, header: "{header}", body: "{body}"')
        else:
            print("Serial connection is not open. Cannot send packet.")

    def getObservationData(self, callback: Callable[[Observation], object]):
        # Aquire lock and do desired query on observation object
        self.__lock.acquire()
        ret = callback(self.__observation)
        self.__lock.release()
        return ret
    
    def setNextAction(self, action):
        self.__lock.acquire()
        self.__next_action = action
        self.__action_sent = False
        self.__lock.release()

    def stop(self):
        # Stop the thread by setting started flag to False and join the thread
        self.__started = False
        self.__thread.join()

    def __del__(self):
        # Destructor to call the stop method when the object is deleted
        self.stop
