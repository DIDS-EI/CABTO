import pygame
import pygame_gui
import h5py
import numpy as np
from pathlib import Path
import cv2
import os
from btgym.dataclass import cfg
from btgym.dataclass.state import Obs
import shutil

def add_hdf5_sample(hdf5_path,obs):
    # Open the existing file in read/write mode
    if not os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'w') as f:
            data_group = f.create_group('data')
            f.attrs['total_samples'] = 0

    with h5py.File(hdf5_path, 'r+') as f:
        data_group = f['data']
        current_samples = f.attrs['total_samples']


        # 生成组名
        group_name = f'{current_samples:08d}'
        
        # 如果组已经存在，先删除它
        if group_name in data_group:
            del data_group[group_name]
            
        sample_group = data_group.create_group(f'{current_samples:08d}')
        
        sample_group.create_dataset('rgb', 
                                    data=obs['rgb'],
                                    compression='gzip',
                                    compression_opts=9)
        
        sample_group.create_dataset('depth', 
                                    data=obs['depth'],
                                    compression='gzip',
                                    compression_opts=9)
        
        sample_group.create_dataset('seg_semantic', 
                                    data=obs['seg_semantic'],
                                    compression='gzip',
                                    compression_opts=9)
        
        sample_group.create_dataset('proprio', 
                                    data=obs['proprio'])
        
        sample_group.create_dataset('gripper_open', 
                                    data=obs['gripper_open'])
        
        sample_group.create_dataset('eef_pose', 
                                    data=obs['eef_pose'])

        f.attrs['total_samples'] = current_samples + 1


class HDF5Viewer:
    def __init__(self,hdf5_path):
        pygame.init()
        self.hdf5_path = hdf5_path
        self.window_size = (1280, 720)
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('HDF5 Data Viewer')
        
        # Load HDF5 file first to get total samples
        self.load_hdf5()
        
        # Get total number of samples
        self.total_samples = self.h5_file.attrs['total_samples']
        self.current_sample = None
        
        # Create UIManager
        self.manager = pygame_gui.UIManager(self.window_size)
        self.clock = pygame.time.Clock()
        
        # Create UI elements
        self.setup_ui()
        
        # Create surfaces for images
        self.rgb_surface = None
        self.depth_surface = None
        self.mask_surface = None

        self.run()
    
    def setup_ui(self):
        # Create left list panel
        self.list_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((0, 0), (250, 720)),
            manager=self.manager
        )
        
        # Create data list with actual number of samples
        self.data_list = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((10, 10), (230, 650)),
            item_list=[f"{i:08d}" for i in range(self.total_samples)],
            manager=self.manager,
            container=self.list_panel
        )

        # Add delete button
        self.delete_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 670), (230, 40)),
            text='Delete',
            manager=self.manager,
            container=self.list_panel
        )
        
        # Create right display panel
        self.display_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((260, 0), (1020, 720)),
            manager=self.manager
        )
        
        # Create text display area
        self.text_height = 30
        y_offset = 10
        
        # Add total samples counter
        self.total_samples_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (980, self.text_height)),
            text=f"Total Samples: {self.total_samples}",
            manager=self.manager,
            container=self.display_panel
        )
        
        y_offset += self.text_height + 10
        
        # Task description
        self.task_desc_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (980, self.text_height)),
            text="Task Description: ",
            manager=self.manager,
            container=self.display_panel
        )
        
        # Image display area
        self.image_size = (320, 240)
        y_offset += self.text_height + 10
        self.image_y_offset = y_offset
        y_offset += self.image_size[1]
        
        # Image labels
        self.rgb_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (100, self.text_height)),
            text="RGB Image",
            manager=self.manager,
            container=self.display_panel
        )
        
        self.depth_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((340, y_offset), (100, self.text_height)),
            text="Depth Image",
            manager=self.manager,
            container=self.display_panel
        )
        
        self.mask_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((670, y_offset), (100, self.text_height)),
            text="Mask Image",
            manager=self.manager,
            container=self.display_panel
        )
        
        # Robot state
        y_offset += 30
        self.robot_state_label = pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect((10, y_offset), (980, 300)),
            html_text="Robot State: ",
            manager=self.manager,
            container=self.display_panel
        )

    def load_hdf5(self):
        self.h5_file = h5py.File(self.hdf5_path, 'r')
    
    def process_image(self, img_data, is_rgb=False):
        # Resize image
        img = cv2.resize(img_data, self.image_size)
        
        if is_rgb:
            # Convert BGR to RGB
            img = img
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Convert to 3 channel image
            img = np.stack([img] * 3, axis=-1)
        
        # Create pygame surface
        surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        return surface
        
    def update_display(self, sample_idx):
        self.current_sample = sample_idx
        sample_group = self.h5_file['data'][f'{sample_idx:08d}']
        
        # Update task description
        # task_desc = sample_group['task_description'][()].decode('utf-8')
        # self.task_desc_label.set_text(f"Task Description: {task_desc}")
        
        # Update images
        # RGB image
        rgb_img = sample_group['rgb'][()]
        self.rgb_surface = self.process_image(rgb_img, is_rgb=True)
        
        # Depth image
        depth_img = sample_group['depth'][()]
        depth_img = (depth_img * 255).astype(np.uint8)
        self.depth_surface = self.process_image(depth_img)
        
        # seg_semantic
        semantic_img = sample_group['seg_semantic'][()]
        semantic_img = (semantic_img * 255).astype(np.uint8)
        self.mask_surface = self.process_image(semantic_img)
        
        # # Mask image
        # mask_img = sample_group['object_mask'][()]
        # mask_img = (mask_img * 25).astype(np.uint8)
        # self.mask_surface = self.process_image(mask_img)
        
        # Update robot state
        proprio = sample_group['proprio'][()]
        eef_pose = sample_group['eef_pose'][()]
        state_text = (
            f"proprio: {', '.join([f'{j:.3f}' for j in proprio])}<br>"
            f"End Effector Pose: Position({eef_pose[0]:.3f}, {eef_pose[1]:.3f}, {eef_pose[2]:.3f}), "
            f"Euler({eef_pose[3]:.3f}, {eef_pose[4]:.3f}, {eef_pose[5]:.3f})"
        )
        self.robot_state_label.html_text = state_text
        self.robot_state_label.rebuild()
    
    def draw_images(self):
        if self.rgb_surface:
            self.window.blit(self.rgb_surface, (270, self.image_y_offset))
        if self.depth_surface:
            self.window.blit(self.depth_surface, (600, self.image_y_offset))
        if self.mask_surface:
            self.window.blit(self.mask_surface, (930, self.image_y_offset))
        
    def delete_current_sample(self):
        if self.current_sample is None:
            return
        
        # Create a backup of the original file
        backup_path = str(self.hdf5_path) + '.backup'
        shutil.copy(self.hdf5_path, backup_path)
        
        try:
            # Create a new HDF5 file
            with h5py.File(backup_path, 'r') as src, h5py.File(str(self.hdf5_path) + '.temp', 'w') as dst:
                # Copy attributes
                for key, value in src.attrs.items():
                    if key == 'total_samples':
                        dst.attrs[key] = value - 1
                    else:
                        dst.attrs[key] = value
                
                # Copy all groups except the deleted one
                src_data = src['data']
                dst_data = dst.create_group('data')
                
                new_idx = 0
                for i in range(self.total_samples):
                    old_key = f'{i:08d}'
                    if i != self.current_sample:
                        new_key = f'{new_idx:08d}'
                        src_data.copy(old_key, dst_data, name=new_key)
                        new_idx += 1
            
            # Close current file
            self.h5_file.close()
            
            # Replace the original file
            os.replace(str(self.hdf5_path) + '.temp', self.hdf5_path)
            os.remove(backup_path)
            
            # Reload the file and update UI
            self.load_hdf5()
            self.total_samples = self.h5_file.attrs['total_samples']
            self.current_sample = None
            
            # Update the data list
            self.data_list.set_item_list([f"{i:08d}" for i in range(self.total_samples)])
            
        except Exception as e:
            print(f"Error during deletion: {e}")
            # Restore backup if something went wrong
            if os.path.exists(backup_path):
                os.replace(backup_path, self.hdf5_path)
                self.load_hdf5()
                self.total_samples = self.h5_file.attrs['total_samples']

    def run(self):
        running = True
        while running:
            time_delta = self.clock.tick(60)/1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
                    if event.ui_element == self.data_list:
                        selected_idx = int(event.text)
                        self.update_display(selected_idx)
                
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.delete_button:
                        self.delete_current_sample()

                self.manager.process_events(event)

            self.manager.update(time_delta)
            
            self.window.fill((255, 255, 255))
            self.manager.draw_ui(self.window)
            self.draw_images()
            pygame.display.update()

        pygame.quit()
        self.h5_file.close()

if __name__ == '__main__':
    hdf5_path = Path(__file__).parent / 'robot_data.hdf5'
    viewer = HDF5Viewer(hdf5_path)