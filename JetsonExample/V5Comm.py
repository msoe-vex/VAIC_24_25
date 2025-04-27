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
import numpy as np
import os
import cv2
import glob
import shutil
    
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
        if data is None or len(data) < 1 or data[0] != '#':
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

    def __init__(self, port = None, debug = False, log_folder='auton_logs'):
        # Initialize properties of V5SerialComms class, including port, started status, and lock
        self.__dev = port
        self.__started = False
        self.__ser = None
        self.__lock = Lock()
        self.__debug = debug
        self.__rl = None
        self.__pending_actions = []
        self.__gps_position = (0, 0, 0)
        self.__gps_log_interval = 1
        self.__last_gps_log = 0
        self.__auton_running = False
        self.__last_heartbeat = 0
        self.__heartbeat_timeout = 5
        self.__log_folder = log_folder
        self.__auton_begin = 0
        self.__battery = 0
        self.__last_battery_time = 0
        self.__last_battery_img = None
        self.__last_battery_img_time = None
        self.__connected_devices = 0
        self.__last_conn_devs_time = 0
        self.__last_conn_devs_img = None
        self.__last_conn_devs_img_time = None
        self.__last_camera_img = None
        self.__last_camera_time = 0
        self.__camera_update_interval = 1
        self.__save_n_auton_logs = 100
        self.__detection_log_interval = 0.5
        self.__last_detection_log = 0
        self.__rs_video_writer = None
        self.__rs_image_dims = (320, 120)

    def set_rl(self, rl):
        self.__rl = rl

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
                self.__ser = serial.Serial(port, 115200, timeout=self.__heartbeat_timeout)
                self.__ser.flushInput()
                self.__ser.flushOutput()

                while self.__started:  # Continue reading while thread is started
                    # Read data from the serial port
                    try:
                        data = self.__ser.readline().decode("utf-8").rstrip()
                    except UnicodeDecodeError:
                        continue  # Invalid characters in current data
                    # print(data)

                    self.__lock.acquire()
                    if self.__auton_running and time.time() - self.__last_heartbeat > self.__heartbeat_timeout:
                        self.__auton_running = False
                        self.endAuton()
                    self.__lock.release()

                    packet = V5SerialPacket.from_Serial(data)
                    if packet is None:
                        continue

                    log_line = f'[{time.time():.3f}] Packet received, header: "{packet.get_header()}", data: "{packet.get_content()}"'
                    if self.__debug:
                        print(log_line)
                    
                    if self.__auton_running:
                        self.__lock.acquire()
                        self.addLogLine(log_line)
                        self.saveCameraImage(self.__last_camera_img, self.__last_camera_time)
                        self.__lock.release()

                    if packet.get_header() == "autoStart":
                        self.__lock.acquire()

                        # Reset state
                        self.__pending_actions = []
                        if self.__rl is not None:
                            self.__rl.get_observation().begin_auton()
                        self.__auton_running = True
                        self.__last_heartbeat = time.time()
                        self.__auton_begin = time.time()

                        battery_log_line = f'[{self.__last_battery_time:.3f}] Packet received, header: "battery", data: "{self.__battery}"'
                        conn_devs_log_line = f'[{self.__last_conn_devs_time:.3f}] Packet received, header: "connectedDevices", data: "{self.__connected_devices}"'
                        if self.__last_battery_time < self.__last_conn_devs_time:
                            self.addLogLine(battery_log_line)
                            self.saveCameraImage(self.__last_battery_img, self.__last_battery_img_time)
                            self.addLogLine(conn_devs_log_line)
                            self.saveCameraImage(self.__last_conn_devs_img, self.__last_conn_devs_img_time)
                        else:
                            self.addLogLine(conn_devs_log_line)
                            self.saveCameraImage(self.__last_conn_devs_img, self.__last_conn_devs_img_time)
                            self.addLogLine(battery_log_line)
                            self.saveCameraImage(self.__last_battery_img, self.__last_battery_img_time)

                        self.addLogLine(log_line)
                        self.saveCameraImage(self.__last_camera_img, self.__last_camera_time)
                        
                        # Set brain's initial position
                        self.sendPacket('setPosition', ' '.join([f'{n:.2f}' for n in self.__gps_position]))

                        self.__lock.release()

                    elif packet.get_header() == "heartBeat":
                        self.__lock.acquire()
                        self.__last_heartbeat = time.time()
                        self.__lock.release()
                    
                    elif packet.get_header() == "battery":
                        self.__lock.acquire()
                        self.__battery = int(packet.get_content())
                        self.__last_battery_time = time.time()
                        if not self.__auton_running:
                            self.__last_battery_img = self.__last_camera_img
                            self.__last_battery_img_time = self.__last_camera_time
                        self.__lock.release()
                    
                    elif packet.get_header() == "connectedDevices":
                        self.__lock.acquire()
                        self.__connected_devices = int(packet.get_content())
                        self.__last_conn_devs_time = time.time()
                        if not self.__auton_running:
                            self.__last_conn_devs_img = self.__last_camera_img
                            self.__last_conn_devs_img_time = self.__last_camera_time
                        self.__lock.release()

                    elif packet.get_header() == "ready":
                        #send action to robot
                        if self.__rl is not None:
                            self.__lock.acquire()

                            self.__rl.get_observation().update_from_brain(packet.get_content())
                            self.__rl.get_observation().update_time_remaining()

                            while len(self.__pending_actions) == 0:
                                _, action_list = self.__rl.predict()
                                self.__pending_actions += action_list

                            to_execute = self.__pending_actions.pop(0)
                            to_execute_str = self.serializeAction(to_execute)
                            self.sendPacket('runAction', to_execute_str)

                            self.__lock.release()

            # To close the serial port gracefully, use Ctrl+C to break the loop
            except serial.SerialException as e:
                print("Could not connect to ", port, ". Exception: ", e)
                time.sleep(1)    # Wait for 1 second before retrying
        
            if(self.__ser.isOpen()):
                self.__ser.close()    # Close the serial port if open

        print("V5SerialComms thread stopped.")

    def endAuton(self):
        # We assume self.__lock is held by the caller

        # Delete all but the last n autons
        matching_autons = sorted(glob.glob(os.path.join(self.__log_folder, 'auton_*')))
        for to_delete in matching_autons[:-self.__save_n_auton_logs]:
            shutil.rmtree(to_delete)
        
        # Reset video writer
        if self.__rs_video_writer is not None:
            self.__rs_video_writer.release()
            self.__rs_video_writer = None

    def getLogFolder(self):
        # We assume self.__lock is held by the caller

        # Get the location to store auton stats
        subfolder_name = f'auton_{self.__auton_begin:.3f}'
        folder_name = os.path.join(self.__log_folder, subfolder_name)

        # Create that folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        return folder_name

    def addLogLine(self, line):
        # We assume self.__lock is held by the caller

        folder_name = self.getLogFolder()
        auton_file = os.path.join(folder_name, 'auton.log')

        with open(auton_file, 'a') as f:
            f.write(line + '\n')
    
    def saveCameraImage(self, image, taken_time):
        # We assume self.__lock is held by the caller

        if taken_time != 0 and image is not None:
            found = False

            frame_name = f'{taken_time:.3f}'
            log_file_name = f'realsense.log'
            video_file_name = f'realsense.mp4'
            folder_name = self.getLogFolder()
            log_file_path = os.path.join(folder_name, log_file_name)
            video_file_path = os.path.join(folder_name, video_file_name)

            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as f:
                    file_lines = f.readlines()
                file_lines = [l.strip() for l in file_lines]
                file_lines = [l for l in file_lines if l != '']

                found = frame_name in file_lines
            
            if not found:
                if self.__rs_video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'vp09')
                    self.__rs_video_writer = cv2.VideoWriter(video_file_path, fourcc, 1 / self.__camera_update_interval, self.__rs_image_dims)
                
                self.__rs_video_writer.write(image)
                with open(log_file_path, 'a') as f:
                    f.write(f'{frame_name}\n')
    
    def updateCameraImage(self, image):
        # We assume self.__lock is held by the caller
        
        this_camera_time = time.time()

        if image is not None and this_camera_time - self.__last_camera_time > self.__camera_update_interval:
            resized_img = cv2.resize(image, self.__rs_image_dims, interpolation=cv2.INTER_AREA)
            resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            self.__last_camera_img = resized_rgb
            self.__last_camera_time = this_camera_time
    
    def updateGPSPosition(self, gps_pos):
        # We assume self.__lock is held by the caller
        
        this_gps_time = time.time()

        self.__gps_position = gps_pos
        if this_gps_time - self.__last_gps_log > self.__gps_log_interval:
            log_line = f'[{this_gps_time:.3f}] GPS Coordinates: "{",".join([f"{n:.2f}" for n in gps_pos])}"'
            if self.__debug:
                print(log_line)
            if self.__auton_running:
                self.addLogLine(log_line)
                self.saveCameraImage(self.__last_camera_img, self.__last_camera_time)
            self.__last_gps_log = this_gps_time

    def serializeAction(self, action_tuple):
        # We assume self.__lock is held by the caller

        out = action_tuple[0]
        if action_tuple[1] is not None:
            params = action_tuple[1]
            param_num = 0
            for param in params:
                if action_tuple[0] == 'FORWARD' or action_tuple[0] == 'BACKWARD':

                    # Initial angle (so the robot doesn't turn while following and crash)
                    if param_num == 0:
                        if len(params) >= 4:
                            x_old, y_old, x_new, y_new = params[0:4]
                            initial_theta = np.arctan2(y_new - y_old, x_new - x_old)
                            # Transform angle to the system the brain uses
                            initial_theta = ((np.pi / 2 - initial_theta) * (180 / np.pi)) % 360
                        else:
                            initial_theta = 0.0
                        if action_tuple[0] == 'BACKWARD':
                            initial_theta = (initial_theta + 180) % 360
                        
                        out += f' {initial_theta:.2f}'
                        
                    # Transform coordinate to the system the brain uses
                    param = param * 144 / 12 - 72

                elif action_tuple[0] == 'TURN_TO':
                    # Transform angle to the system the brain uses
                    param = ((np.pi / 2 - param) * (180 / np.pi)) % 360
                
                out += ' '
                if isinstance(param, (float, np.floating)):
                    out += f'{param:.2f}'
                else:
                    out += str(param)
                
                param_num += 1
            
        if self.__rl is not None:
            self.__rl.get_observation().update_from_action(action_tuple[0])
        
        return out

    def sendPacket(self, header: str, body: str):
        # We assume self.__lock is held by the caller

        # Send a packet with the specified header and body over the serial connection
        if self.__ser and self.__ser.isOpen():
            packet = V5SerialPacket(header, body)
            self.__ser.write(packet.to_Serial())
            log_line = f'[{time.time():.3f}] Packet sent, header: "{header}", body: "{body}"'
            if self.__debug:
                print(log_line)
            if self.__auton_running:
                self.addLogLine(log_line)
                self.saveCameraImage(self.__last_camera_img, self.__last_camera_time)
        else:
            print("Serial connection is not open. Cannot send packet.")

    def setDetectionData(self, aiRecord, color_image=None):
        if self.__rl is not None:
            object_types = ['goal', 'red_ring', 'blue_ring', 'both_rings']

            objects = []
            for d in aiRecord.detections:
                d_x = d.mapLocattion.x[0]
                d_y = d.mapLocattion.y[0]
                d_z = d.mapLocattion.z[0]
                d_type = d.classID

                if d_type < 0 or d_type >= len(object_types):
                    d_type_str = 'unknown'
                else:
                    d_type_str = object_types[d_type]

                objects.append({
                    'type': d_type_str,
                    'x': d_x,
                    'y': d_y,
                    'z': d_z,
                })

            self.__rl.get_observation().update_from_camera(objects)

            self.__lock.acquire()

            pos = aiRecord.position
            inches_per_meter = 39.3701
            scaled_x = pos.x * inches_per_meter
            scaled_y = pos.y * inches_per_meter
            gps_pos = (scaled_x, scaled_y, pos.rotation)

            self.updateGPSPosition(gps_pos)
            self.updateCameraImage(color_image)

            this_detection = time.time()
            if this_detection - self.__last_detection_log > self.__detection_log_interval:
                self.__last_detection_log = this_detection

                log_line = f'[{this_detection:.3f}] Object detections: "'
                for obj in objects:
                    if np.isnan(obj['x']) or np.isnan(obj['y']):
                        continue
                    if log_line[-1] != '"':
                        log_line += ';'
                    log_line += f"{obj['type']},{obj['x'] * inches_per_meter:.2f},{obj['y'] * inches_per_meter:.2f}"
                log_line += '"'

                if self.__debug:
                    print(log_line)
                if self.__auton_running:
                    self.addLogLine(log_line)

            self.__lock.release()

    def stop(self):
        # Stop the thread by setting started flag to False and join the thread
        self.__started = False
        self.__thread.join()

    def __del__(self):
        # Destructor to call the stop method when the object is deleted
        self.stop
