import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import sys
import glob
import os
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

class Obstacle:
    def __init__(self, x, y, r, i):
        self.x = x
        self.y = y
        self.radius = r
        self.ignore_collision = i

class ImageRepo:
    def __init__(self, folder, img_src):
        log_path = os.path.join(folder, img_src + '.log')
        vid_path = os.path.join(folder, img_src + '.mp4')

        self.empty = not os.path.exists(log_path) or not os.path.exists(vid_path)

        if not self.empty:
            with open(log_path, 'r') as f:
                file_lines = f.readlines()
            file_lines = [l.strip() for l in file_lines]
            file_lines = [l for l in file_lines if l != '']
            self.times = [float(l) for l in file_lines]
            self.names = file_lines
            
            vid_reader = cv2.VideoCapture(vid_path)
            finished = False
            self.imgs = []
            while not finished:
                ret, img = vid_reader.read()
                if ret:
                    self.imgs.append(img)
                else:
                    finished = True
            vid_reader.release()

            self.last_img_name = -1
            self.last_img = -1
            self.last_width = -1
            self.last_height = -1
            self.last_ranking = None
        
    def get_img_name(self, timestamp):
        if self.empty:
            return None
        
        ret_idx = -1
        for i, curr_time in enumerate(self.times):
            if curr_time > timestamp:
                ret_idx = i - 1
                break
            if i == len(self.times) - 1:
                ret_idx = i
                break
        
        if ret_idx < 0:
            return None  # Before the first picture
        else:
            return self.names[ret_idx]

    @staticmethod
    def _gen_blue_noise_ranking(dims):
        seed_mask = np.random.randint(0, 10, size=dims[:2])
        seed_mask = np.array(1 - np.minimum(seed_mask, 1), dtype=np.uint8)  # 90% zero, 10% one
        
        rank = np.zeros(dims[:2], dtype=np.uint16)
        
        # Lower ranks: take most clustered ones out of the seed mask until it's empty
        n_ones = np.sum(seed_mask)
        mask1 = np.array(seed_mask, dtype=np.uint8)
        while n_ones > 0:
            D = gaussian_filter(np.array(mask1, dtype=np.float32), sigma=1.5, mode='wrap')
            i, j = np.unravel_index(np.argmax(D * mask1), mask1.shape)
            
            mask1[i, j] = 0
            n_ones -= 1
            rank[i, j] = n_ones
        
        # Middle ranks: add ones to the biggest voids until mask is half full
        n_ones = np.sum(seed_mask)
        mask2 = np.array(seed_mask, dtype=np.uint8)
        while n_ones < np.prod(dims[:2]):
            D = gaussian_filter(np.array(mask2, dtype=np.float32), sigma=1.5, mode='wrap')
            i, j = np.unravel_index(np.argmin(D * (-mask2 + 1) + np.array(mask2, dtype=np.float32) * 999), mask2.shape)
            
            rank[i, j] = n_ones
            mask2[i, j] = 1
            n_ones += 1
        
        # Upper ranks: Invert half-full mask, kill clusters until inverted mask is empty
        inv_mask2 = 1 - mask2
        n_zeros = np.sum(inv_mask2)
        while n_zeros > 0:
            D = gaussian_filter(np.array(inv_mask2, dtype=np.float32), sigma=1.5, mode='wrap')
            i, j = np.unravel_index(np.argmax(D * inv_mask2), inv_mask2.shape)
            
            rank[i, j] = n_ones
            inv_mask2[i, j] = 0
            mask2[i, j] = 1
            n_ones += 1
            n_zeros -= 1
        
        return rank

    @staticmethod
    def _rank_to_thresholds(rank, level_range):
        old_range = np.prod(rank.shape[:2])
        return np.array(rank, dtype=np.float32) * level_range / old_range

    @staticmethod
    def _tile_to_image(arr, img):
        height, width = img.shape[:2]
        height_its, width_its = ((np.array(img.shape[:2]) - 1) // np.array(arr.shape[:2])) + 1
        arr = np.tile(arr, (height_its, width_its))
        arr = arr[:height, :width].reshape((height, width, 1))
        return arr

    @staticmethod
    def _apply_to_image(img, thresholds, level_range):
        img = np.array(img)
        
        under_threshold = np.array(img[img % level_range < thresholds], dtype=np.float32)
        under_corrected = np.floor(under_threshold / level_range) * level_range
        img[img % level_range < thresholds] = np.array(under_corrected, dtype=np.uint8)
        
        over_threshold = np.array(img[img % level_range >= thresholds], dtype=np.float32)
        over_corrected = np.ceil(over_threshold / level_range) * level_range
        img[img % level_range >= thresholds] = np.array(over_corrected, dtype=np.uint8)
        
        return img

    @staticmethod
    def _dither_to_web_safe(img, prev_ranking=None):
        channel_step_size = 51
        img = np.array(img)
        if prev_ranking is not None:
            ranking = prev_ranking
        else:
            ranking = ImageRepo._gen_blue_noise_ranking((64, 64))
        thresholds = ImageRepo._rank_to_thresholds(ranking, channel_step_size)
        thresholds = ImageRepo._tile_to_image(thresholds, img)
        img = ImageRepo._apply_to_image(img, thresholds, channel_step_size)
        return img, ranking

    def get_dithered_img(self, img_name, width, height):
        if self.empty:
            return None
        
        # Implement caching so we don't do costly re-dithering
        if img_name == self.last_img_name and width == self.last_width and height == self.last_height:
            return self.last_img
        
        idx = self.names.index(img_name)
        img = self.imgs[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((width, height))
        img, self.last_ranking = self._dither_to_web_safe(img, self.last_ranking)
        
        self.last_img_name = img_name
        self.last_img = img
        self.last_width = width
        self.last_height = height
        
        return img

class GameRenderer:
    def __init__(self):
        self.permanent_obstacles = [
            Obstacle(3/6, 2/6, 3.5/144, False), # Bottom
            Obstacle(3/6, 4/6, 3.5/144, False), # Top
            Obstacle(2/6, 3/6, 3.5/144, False), # Left
            Obstacle(4/6, 3/6, 3.5/144, False) # Right
            ]
        self.robot_position = [-999, -999, -999]
        self.planned_x = []
        self.planned_y = []
        self.target_theta = -999
        self.last_action = ''
        self.battery = -999
        self.objects = []
        self.gps_position = [-999, -999, -999]
    
    def update_packet(self, header, body):
        self.last_action = ''
        
        pos_headers = ['pos', 'ready', 'setPosition']
        if header in pos_headers:
            if header == 'pos':
                delim = ','
            else:
                delim = ' '
            fields = body.split(delim)
            for i in range(min(len(fields), len(self.robot_position))):
                self.robot_position[i] = float(fields[i])
        elif header == 'runAction':
            fields = body.split(' ')
            action = fields[0]
            params = fields[1:]
            
            self.planned_x = []
            self.planned_y = []
            self.target_theta = -999
            
            move_actions = ['FORWARD', 'BACKWARD']
            if action in move_actions:
                self.target_theta = float(params[0])
                for x, y in zip(params[1::2], params[2::2]):
                    self.planned_x.append(float(x))
                    self.planned_y.append(float(y))
            elif action == 'TURN_TO':
                self.target_theta = float(params[0])
            
            self.last_action = action
        elif header == 'battery':
            self.battery = int(body)
    
    def update_detections(self, detection_str):
        self.objects = []

        objects = detection_str.split(';')
        for obj in objects:
            if obj == '':
                continue
            fields = obj.split(',')

            out = {}
            out['type'] = fields[0]
            out['x'] = float(fields[1])
            out['y'] = float(fields[2])

            self.objects.append(out)
    
    def update_gps(self, gps_str):
        fields = gps_str.split(',')
        for i in range(min(len(fields), len(self.gps_position))):
            self.gps_position[i] = float(fields[i])
    
    def render(self, out_folder, log_line, i, n, line_time, img_repo):
        fig, axes = plt.subplots(1, 2, figsize=(2 * 8, 8))
        
        ax = axes[0]
        ax.set_xlim(-72, 72)
        ax.set_ylim(-72, 72)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        img_ax = axes[1]
        img_width = 685
        img_height = 514
        img_ax.set_xlim(0, img_width)
        img_ax.set_ylim(0, img_height)
        img_ax.set_aspect('equal')
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        
        plt.subplots_adjust(wspace=0.1, left=0.07, right=0.97)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
        
        for obstacle in self.permanent_obstacles:
            circle = patches.Circle(
                (obstacle.x * 144 - 72, obstacle.y * 144 - 72), 
                obstacle.radius * 144, 
                edgecolor='black', facecolor='none', 
                linestyle='dotted', alpha=0.5)
            ax.add_patch(circle)
        
        for obj in self.objects:
            obj_color = 'gray'
            if obj['type'] == 'goal':
                obj_color = 'olive'
            elif obj['type'] == 'red_ring':
                obj_color = 'red'
            elif obj['type'] == 'blue_ring':
                obj_color = 'blue'
            elif obj['type'] == 'both_rings':
                obj_color = 'purple'
            obj_pos = (obj['x'], obj['y'])

            if obj['type'] == 'goal':
                obj_patch = patches.RegularPolygon(obj_pos, 6, radius=0.4 * 12, color=obj_color, alpha=0.7)
            else:
                obj_patch = patches.Circle(obj_pos, 0.3 * 12, color=obj_color, alpha=0.7)

            ax.add_patch(obj_patch)
        
        if -999 not in self.robot_position:
            robot = patches.Rectangle(np.array(self.robot_position[:2]) - np.array([15 / 2, 15 / 2]), 
                  15, 15, color='blue', alpha=0.25, ec='black', transform=ax.transData)
            t = patches.transforms.Affine2D().rotate_deg_around(self.robot_position[0], self.robot_position[1], 90 - self.robot_position[2]) + ax.transData
            robot.set_transform(t)
            ax.add_patch(robot)
            center = np.array(self.robot_position[:2])
            arrow_dx = np.cos(np.radians(90 - self.robot_position[2])) * 12
            arrow_dy = np.sin(np.radians(90 - self.robot_position[2])) * 12
            orientation_arrow = patches.FancyArrow(center[0], center[1], arrow_dx, arrow_dy,
                    width=0.1 * 12, color='yellow', length_includes_head=True, alpha=0.25)
            ax.add_patch(orientation_arrow)
            begin_center = center
        
        if -999 not in self.gps_position:
            gps_center = np.array(self.gps_position[:2])
            gps_arrow_dx = np.cos(np.radians(90 - self.gps_position[2])) * 12
            gps_arrow_dy = np.sin(np.radians(90 - self.gps_position[2])) * 12
            gps_orientation_arrow = patches.FancyArrow(gps_center[0], gps_center[1], gps_arrow_dx, gps_arrow_dy,
                    width=0.1 * 12, color='green', length_includes_head=True, alpha=0.25)
            ax.add_patch(gps_orientation_arrow)
        
        if 0 not in [len(self.planned_x), len(self.planned_y)]:
            ax.plot(self.planned_x, self.planned_y, 'k--', alpha=0.5)
            begin_center = np.array([self.planned_x[0], self.planned_y[0]])
        
        if self.target_theta != -999:
            begin_arrow_dx = np.cos(np.radians(90 - self.target_theta)) * 12 / 4
            begin_arrow_dy = np.sin(np.radians(90 - self.target_theta)) * 12 / 4
            begin_orientation_arrow = patches.FancyArrow(begin_center[0], begin_center[1], begin_arrow_dx, begin_arrow_dy,
                    width=0.1 * 12, color='gray', length_includes_head=True, alpha=0.5)
            ax.add_patch(begin_orientation_arrow)
        
        log_line_broken = []
        max_line_length = 120
        for l in range(0, len(log_line), max_line_length):
            log_line_broken.append(log_line[l:l + max_line_length])
        
        ax.set_xlabel('\n'.join(log_line_broken))
        ax.set_title(f'Log Line {i + 1}')
        
        if self.battery != -999:
            ax.set_ylabel(f'Battery: {self.battery}%')
            ax.yaxis.label.set_color((1 - self.battery / 100, self.battery / 100, 0))
        
        img_ax.set_title('RealSense View')
        img_path = img_repo.get_img_name(line_time)
        if img_path is not None:
            img = img_repo.get_dithered_img(img_path, img_width, img_height)
            img_ax.imshow(img)
            img_ax.invert_yaxis()
        
        file_name = os.path.join(out_folder, f'auton_{i + 1:06d}.png')
        plt.savefig(file_name)
        print(f'Rendered frame {i + 1}/{n}')
        plt.close()

def main(in_folder):
    in_file = os.path.join(in_folder, 'auton.log')
    out_file = os.path.join(in_folder, 'auton.gif')
    frame_folder = os.path.join(in_folder, 'frames')
    
    with open(in_file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l != '']
    
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    render = GameRenderer()
    imgs = ImageRepo(in_folder, 'realsense')
    
    line_times = []
    for i, line in enumerate(lines):
        packet_match = re.match(r'^\[([0-9]+\.[0-9]+)\] Packet (?:sent|received), header: "([^"]*)", (?:data|body): "([^"]*)"$', line)
        detection_match = re.match(r'^\[([0-9]+\.[0-9]+)\] Object detections: "([^"]*)"$', line)
        gps_match = re.match(r'^\[([0-9]+\.[0-9]+)\] GPS Coordinates: "([^"]*)"$', line)
        if packet_match is not None:
            line_time, header, body = packet_match.groups()
            line_time = float(line_time)
            line_times.append(line_time)

            render.update_packet(header, body)
            render.render(frame_folder, line, i, len(lines), line_time, imgs)

        elif detection_match is not None:
            line_time, detection_str = detection_match.groups()
            line_time = float(line_time)
            line_times.append(line_time)

            render.update_detections(detection_str)
            render.render(frame_folder, line, i, len(lines), line_time, imgs)
        
        elif gps_match is not None:
            line_time, gps_str = gps_match.groups()
            line_time = float(line_time)
            line_times.append(line_time)

            render.update_gps(gps_str)
            render.render(frame_folder, line, i, len(lines), line_time, imgs)

    out_frame_paths = sorted(glob.glob(os.path.join(frame_folder, 'auton_*.png')))
    out_frames = []
    out_durations = []
    for i, frame_path in enumerate(out_frame_paths):
        img = Image.open(frame_path)
        img = img.copy()
        os.remove(frame_path)
        out_frames.append(img)
        if i != len(out_frame_paths) - 1:
            out_durations.append(1000 * (line_times[i + 1] - line_times[i]))
        else:
            out_durations.append(1000)
    
    out_frames[0].save(out_file, save_all=True, append_images=out_frames[1:], duration=out_durations, loop=1)
    print()
    print(f'Saved GIF animation to {out_file}')
    
    for i, frame in enumerate(out_frames):
        frame_file = os.path.join(frame_folder, f'auton_{i + 1:06d}.gif')
        frame.save(frame_file)
    print(f"Saved GIF frames to {os.path.join(frame_folder, 'auton_*.gif')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Must specify folder name')
        print('Example: python render_log.py auton_logs/auton_1745276439.050')
        exit(1)
    
    main(sys.argv[1])
    exit(0)
