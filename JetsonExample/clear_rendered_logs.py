import glob
import os
import shutil

if __name__ == "__main__":
    auton_log_folders = glob.glob(os.path.join('auton_logs', 'auton_*'))
    
    for log_folder in auton_log_folders:
        frame_folder = os.path.join(log_folder, 'frames')
        gif_path = os.path.join(log_folder, 'auton.gif')
        
        if os.path.exists(frame_folder):
            shutil.rmtree(frame_folder)
        if os.path.exists(gif_path):
            os.remove(gif_path)
    
    exit(0)
