from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import sys
import cv2
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width ,get_foot_position
class Tracker:
    def __init__(self, model_path):
        self.model =YOLO(model_path)
        self.tracker = sv.ByteTrack()


    
    def add_position_to_track(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    bbox=track_info['bbox']
                    if object=='ball':
                        position=get_center_of_bbox(bbox)
                    else:
                        position=get_foot_position(bbox)

                    tracks[object][frame_num][track_id]['position']=position

                   

    def interpolate_ball_positions(self, ball_positions):
    # Extract frame numbers and bbox data for track ID 1
        frame_numbers = []
        bboxes = []
    
        for i, frame in enumerate(ball_positions):
            if 1 in frame:  # Check if track ID 1 exists in this frame
                frame_numbers.append(i)
                bboxes.append(frame[1]['bbox'])
    
        if not frame_numbers:
            return ball_positions  # Return original if no ball was detected
        
    # Create DataFrame
        df_ball_positions = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'], index=frame_numbers)
    
    # Reindex for all frames
        all_frames = pd.RangeIndex(start=0, stop=len(ball_positions))
        df_ball_positions = df_ball_positions.reindex(all_frames)
    
    # Interpolate
        df_ball_positions = df_ball_positions.interpolate(method='linear')
        df_ball_positions = df_ball_positions.bfill().ffill()
    
    # Update the interpolated positions
        interpolated_positions = [{} for _ in range(len(ball_positions))]
        for frame_num in all_frames:
            if not pd.isna(df_ball_positions.loc[frame_num]['x1']):
                bbox = df_ball_positions.loc[frame_num].tolist()
                interpolated_positions[frame_num][1] = {"bbox": bbox}
    
        return interpolated_positions
 


   
    
    def detect_frames(self, frames):
        batch_size = 20
        detections=[]
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size],conf=0.1)
            detections += detections_batch
            
        return detections

    def get_object_tracks(self,frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections=self.detect_frames(frames)

        tracks={
            'players':[],
            'referees':[],
            'ball':[]
        }

        for frame_num,detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv= {v:k for k,v in cls_names.items()}
            print(cls_names)

            #Convert to supervision detection format

            detection_supervision = sv.Detections.from_ultralytics(detection)

            

            #Convert goalkeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind]=cls_names_inv['player']
            
            #Track objects
            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
    
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f) 
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2=int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, 
                    center=(x_center, y2),
                    axes=(int(width/1.5),int(width/3)), 
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness= 2,
                    lineType=cv2.LINE_4
                    
                    )
        
        rectangle_width=30
        rectangle_height=16
        x1_rect= x_center-rectangle_width//2
        x2_rect= x_center+rectangle_width//2
        y1_rect= (y2-rectangle_height//2) +15
        y2_rect= (y2+rectangle_height//2) +15
        
        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                            color,
                            cv2.FILLED
                            )
            
            x1_text=x1_rect+8
            if track_id>99:
                x1_text-=8

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+13)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        return frame
    

    def draw_triangle(self, frame, bbox, color):
        y=int(bbox[1])
        x,_=get_center_of_bbox(bbox)

        triangle_points = np.array([[x, y], [x - 10, y -20], [x + 10, y - 20]], np.int32)
        
        cv2.drawContours(frame, [triangle_points], 0, color, thickness=cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        #draw a semi_transparent rectangle 
        overlay=frame.copy()
        cv2.rectangle(overlay, (1350,850),(1900,970),(255,255,255),-1)
        alpha=0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame=team_ball_control[:frame_num+1]
        #get the number of time each team has the ball control
        team_1_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1=team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2=team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame,f"Team 1 Ball Possession: {team_1*100:.2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 2 Ball Possession: {team_2*100:.2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame

    def draw_annotations(self, video_frame, tracks,team_ball_control):
        output_video_frames=[]

        for frame_num, frame in enumerate(video_frame):
            frame=frame.copy()

            player_dict=tracks['players'][frame_num]
            referee_dict=tracks['referees'][frame_num]
            ball_dict=tracks['ball'][frame_num] 

            #Draw players
            for track_id, player in player_dict.items():
                color=player.get("team_color", (0,0,255))
                frame=self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame=self.draw_triangle(frame, player['bbox'], color=(0, 0, 255))

            #Draw referees
            for _, referee in referee_dict.items():
                frame=self.draw_ellipse(frame, referee['bbox'], color=(0, 255, 255))

            #Draw ball
            for track_id, ball in ball_dict.items():
                frame=self.draw_triangle(frame, ball['bbox'], color=(0, 255, 0))
            
            
            #Draw team ball control
            frame=self.draw_team_ball_control(frame,frame_num, team_ball_control)
            
            output_video_frames.append(frame)

        return output_video_frames