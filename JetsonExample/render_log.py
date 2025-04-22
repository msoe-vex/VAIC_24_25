import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import sys
import glob
import os
from PIL import Image

class Obstacle:
    def __init__(self, x, y, r, i):
        self.x = x
        self.y = y
        self.radius = r
        self.ignore_collision = i

class RSImageRepo:
    def __init__(self, folder):
        rs_imgs = sorted(glob.glob(os.path.join(folder, 'realsense_*.jpg')))
        
        self.rs_times = []
        self.rs_paths = []
        
        for full_path in rs_imgs:
            rs_match = re.match(r'^.*realsense_([0-9]+\.[0-9]+).jpg$', full_path)
            if rs_match is None:
                continue
            rs_time = float(rs_match.groups()[0])
            
            self.rs_times.append(rs_time)
            self.rs_paths.append(full_path)
            
            self.last_img_path = None
            self.last_img = None
            self.last_width = None
            self.last_height = None
        
    def get_img_path(self, timestamp):
        ret_idx = -1
        for i, curr_time in enumerate(self.rs_times):
            if curr_time > timestamp:
                ret_idx = i - 1
                break
            if i == len(self.rs_times) - 1:
                ret_idx = i
                break
        
        if ret_idx < 0:
            return None  # Before the first picture
        else:
            return self.rs_paths[ret_idx]

    def _get_new_val(self, old_val, nc):
        """
        Get the "closest" colour to old_val in the range [0,1] per channel divided
        into nc values.

        """

        return np.round(old_val * (nc - 1)) / (nc - 1)

    def _fs_dither(self, img, nc):
        """
        Floyd-Steinberg dither the image img into a palette with nc colours per
        channel.

        """

        arr = np.array(img, dtype=float) / 255

        new_height = arr.shape[0]
        new_width = arr.shape[1]
        for ir in range(new_height):
            for ic in range(new_width):
                # NB need to copy here for RGB arrays otherwise err will be (0,0,0)!
                old_val = arr[ir, ic].copy()
                new_val = self._get_new_val(old_val, nc)
                arr[ir, ic] = new_val
                err = old_val - new_val
                # In this simple example, we will just ignore the border pixels.
                if ic < new_width - 1:
                    arr[ir, ic+1] += err * 7/16
                if ir < new_height - 1:
                    if ic > 0:
                        arr[ir+1, ic-1] += err * 3/16
                    arr[ir+1, ic] += err * 5/16
                    if ic < new_width - 1:
                        arr[ir+1, ic+1] += err / 16

        carr = np.array(arr/np.max(arr, axis=(0,1)) * 255, dtype=np.uint8)
        return Image.fromarray(carr)

    def get_dithered_img(self, file_path, width, height):
        # Implement caching so we don't do costly re-dithering
        if file_path == self.last_img_path and width == self.last_width and height == self.last_height:
            return self.last_img
        
        img = mpimg.imread(file_path)
        img = Image.fromarray(img).resize((width, height))
        img = self._fs_dither(img, 3)
        
        self.last_img_path = file_path
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
        img_path = img_repo.get_img_path(line_time)
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
    imgs = RSImageRepo(in_folder)
    
    line_times = []
    for i, line in enumerate(lines):
        packet_match = re.match(r'^\[([0-9]+\.[0-9]+)\] Packet (?:sent|received), header: "([^"]*)", (?:data|body): "([^"]*)"$', line)
        detection_match = re.match(r'^\[([0-9]+\.[0-9]+)\] Object detections: "([^"]*)"$', line)
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
