import numpy as np
from V5Position import Position, RobotLocation
from sklearn.cluster import KMeans
import V5Comm
import copy

class KMeansObjectPositioner:
    FOV_ANGLE = 90  # Field of view angle in degrees
    FOV_MIN_DISTANCE = 0.2  # Minimum distance (meters) to consider an object in the field of view
    FOV_MAX_DISTANCE = 2.0  # Maximum distance (meters) to consider an object in the field of view

    def __init__(self, num_last_positions, fov_angle=FOV_ANGLE, fov_min_distance=FOV_MIN_DISTANCE, fov_max_distance=FOV_MAX_DISTANCE):
        self.__num_last_positions = num_last_positions
        self.__received_positions = []
        self.__fov_angle = fov_angle
        self.__fov_min_distance = fov_min_distance
        self.__fov_max_distance = fov_max_distance
    
    def in_update_area(self, pos, robot_loc):
        robot_pos_rl = robot_loc.get_pos_for_objects(RobotLocation.FIELD_RL)
        robot_pos = RobotLocation.convert_to(RobotLocation.FIELD_GPS, robot_pos_rl)
        obj_theta = np.arctan2(pos.y - robot_pos.y, pos.x - robot_pos.x)

        theta_diff_degrees = ((np.degrees(obj_theta - robot_pos_rl.azimuth) + 180) % 360 ) - 180
        if np.abs(theta_diff_degrees) > self.__fov_angle / 2:
            return False
        
        distance = np.sqrt((pos.x - robot_pos.x) ** 2 + (pos.y - robot_pos.y) ** 2)
        if distance < self.__fov_min_distance or distance > self.__fov_max_distance:
            return False
        
        return True

    def add_detections(self, robot_loc, detections):
        new_received_positions = []

        for pos in self.__received_positions:
            if self.in_update_area(Position.from_xy(pos[0], pos[1]), robot_loc):
                pos[2] += 1
                if pos[2] < self.__num_last_positions:
                    new_received_positions.append(pos)
            else:
                new_received_positions.append(pos)

        for detection in detections:
            if self.in_update_area(detection, robot_loc):
                new_received_positions.append([detection.x, detection.y, 0])
        
        self.__received_positions = new_received_positions
    
    def get_object_positions(self, robot_loc):
        relevant_locs = []

        for pos in self.__received_positions:
            if self.in_update_area(Position.from_xy(pos[0], pos[1]), robot_loc):
                relevant_locs.append(pos[0:2])

        if len(relevant_locs) == 0:
            return []

        k = int(max(np.round(len(relevant_locs) / self.__num_last_positions), 1))
        kmeans = KMeans(n_clusters=k).fit(np.array(relevant_locs))
        cluster_centers = kmeans.cluster_centers_

        return [Position.from_xy(c[0], c[1]) for c in cluster_centers]

class AIRecordParser:
    def __init__(self, num_last_positions, robot_loc):
        self.__kmeans = []
        self.__n_cls = 4
        for _ in range(self.__n_cls):
            self.__kmeans.append(KMeansObjectPositioner(num_last_positions))
        self.__loc = robot_loc
    
    def translate_detections(self, aiRecord):
        relevant_detections = [[] for _ in range(self.__n_cls)]
        for detection in aiRecord.detections:
            if np.isnan(detection.mapLocation.x) or np.isnan(detection.mapLocation.y):
                continue
            if detection.classID >= len(self._kmeans):
                continue
            detection_pos = Position.from_xy(detection.mapLocation.x, detection.mapLocation.y)
            relevant_detections[detection.classID].append(detection_pos)
        
        new_detections = []
        for i in range(self.__n_cls):
            self.__kmeans[i].add_detections(self.__loc, relevant_detections[i])
            positions = self.__kmeans[i].get_object_positions(self.__loc)

            for pos in positions:
                imageDet = V5Comm.ImageDetection(
                    int(0),
                    int(0),
                    int(0),
                    int(0),
                )
                mapDet = V5Comm.MapDetection(pos.x, pos.y, 0)
                detection = V5Comm.Detection(
                    i,
                    0.999,
                    0,
                    imageDet,
                    mapDet,
                )
                new_detections.append(detection)
            
        ret = copy.deepcopy(aiRecord)
        ret.detections = new_detections
        return ret


if __name__ == '__main__':
    # Group 1: Near (0.5, 0.5)
    group1 = [
        Position(0, 1, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
        Position(0, 1, 0.525, 0.51, 0.0, 0.0, 0.0, 0.0),
        Position(0, 1, 0.49, 0.485, 0.0, 0.0, 0.0, 0.0),
    ]

    # Group 2: Near (-1.25, 1.5)
    group2 = [
        Position(0, 1, -1.25, 1.5, 0.0, 0.0, 0.0, 0.0),
        Position(0, 1, -1.24, 1.515, 0.0, 0.0, 0.0, 0.0),
        Position(0, 1, -1.26, 1.485, 0.0, 0.0, 0.0, 0.0),
    ]

    # Group 3: Near (1.75, -1.6)
    group3 = [
        Position(0, 1, 1.75, -1.6, 0.0, 0.0, 0.0, 0.0),
        Position(0, 1, 1.765, -1.59, 0.0, 0.0, 0.0, 0.0),
        Position(0, 1, 1.74, -1.61, 0.0, 0.0, 0.0, 0.0),
    ]

    positioner = KMeansObjectPositioner(num_last_positions=3, 
                                        fov_angle=180,
                                        fov_min_distance=0,
                                        fov_max_distance=4)
    
    loc = RobotLocation(RobotLocation.POS_MODE_GPS_ONLY)
    loc.set_gps_pos(Position(0, 1, 0, 0, 0, np.pi / 4, 0, 0), RobotLocation.FIELD_RL)
    
    for p1, p2, p3 in zip(group1, group2, group3):
        positioner.add_detections(loc, [p1, p2, p3])
    
    objects = positioner.get_object_positions(loc)

    print("Objects:")
    for obj in objects:
        print(f"Object at ({obj.x}, {obj.y})")
