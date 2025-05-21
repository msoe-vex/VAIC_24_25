import numpy as np
from V5Position import Position, RobotLocation

class ObjectDifferentialPositioner:
    def __init__(self, max_distance=0.5, max_distance_multiplier=2):
        self.__max_distance = max_distance
        self.__max_distance_multiplier = max_distance_multiplier
        self.__last_positions = []
    
    def get_offset(self, pos_list):
        last_positions = self.__last_positions
        self.__last_positions = pos_list.copy()

        max_n = min(len(last_positions), len(pos_list))
        if max_n == 0:
            return 0, 0
        
        pos_list_distances = []
        for pos in pos_list:
            pos_distances = []
            for old_pos in last_positions:
                distance = np.sqrt((pos.x - old_pos.x) ** 2 + (pos.y - old_pos.y) ** 2)
                pos_distances.append([distance, old_pos.x, old_pos.y])
            pos_distances.sort()
            pos_distance, old_x, old_y = pos_distances[0]
            pos_list_distances.append([pos_distance, pos.x, pos.y, old_x, old_y])

        pos_list_distances.sort()
        relevant_pos_list = []
        last_distance = 9999
        for distance, x, y, old_x, old_y in pos_list_distances:
            if distance > self.__max_distance or distance > last_distance * self.__max_distance_multiplier:
                break    
            relevant_pos_list.append([x, y, old_x, old_y])
        
        total_displacement_x = 0
        total_displacement_y = 0
        for x, y, old_x, old_y in relevant_pos_list:
            total_displacement_x += (x - old_x)
            total_displacement_y += (y - old_y)
        
        return total_displacement_x / len(relevant_pos_list), total_displacement_y / len(relevant_pos_list)

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

    positioner = ObjectDifferentialPositioner(max_distance=0.5, max_distance_multiplier=2)
    
    loc = RobotLocation(RobotLocation.POS_MODE_GPS_ONLY)
    loc.set_gps_pos(Position(0, 1, 0, 0, 0, np.pi / 4, 0, 0), RobotLocation.FIELD_RL)
    
    offsets = []
    for p1, p2, p3 in zip(group1, group2, group3):
        offsets.append(positioner.get_offset([p1, p2, p3]))

    print("Offsets:")
    for i, offset in enumerate(offsets):
        print(f"Offset {i} is at ({offset[0]}, {offset[1]})")
