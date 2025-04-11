from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read video
    video_frames = read_video('input_videos/Football.mp4')
    

    #Initialize tracker
    tracker = Tracker('models/best1.pt')

    tracks=tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stubs/track_stubs.pkl')

    # Create a simplified ball tracking with consistent track ID of 1
    simplified_ball_tracks = []
    for frame in tracks['ball']:
        if frame:  # If any ball was detected in this frame
        # Get the first ball's bbox (ignoring its original track ID)
            first_ball_id = next(iter(frame))
            simplified_ball_tracks.append({1: {"bbox": frame[first_ball_id]["bbox"]}})
        else:
            simplified_ball_tracks.append({})
    #get object position
    tracker.add_position_to_track(tracks)

    #camera_movement_estimator

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement_stub.pkl')


    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)


    #view Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

# Interpolate with consistent track ID
    tracks['ball'] = tracker.interpolate_ball_positions(simplified_ball_tracks)
    

    #Add speed and distance to trackscle
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)



    #Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        
        for player_id,track in player_track.items():
            team=team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    #Assign Ball Aquisition
    player_assigner=PlayerBallAssigner()
    team_ball_control=[]
    # Default team for the first frame
    
    for frame_num, player_track in enumerate(tracks['players']):    
        ball_bbox=tracks['ball'][frame_num][1]['bbox']
        assigned_player=player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player!=-1:
            tracks['players'][frame_num][assigned_player]['has_ball']=True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
              # No player has the ball
            if team_ball_control:
            # If we have previous frame data, use the last team
                team_ball_control.append(team_ball_control[-1])
            else:
            # This handles the first frame when no player has the ball
            # Find the nearest player to the ball and use their team
                nearest_player_id = player_assigner.find_nearest_player_to_ball(player_track, ball_bbox)
                if nearest_player_id != -1:
                    team_ball_control.append(tracks['players'][frame_num][nearest_player_id]['team'])
                else:
                # In the extremely rare case no players are detected at all
                # Use None or a special value that your draw_team_ball_control can handle
                    team_ball_control.append(None)
        

    team_ball_control=np.array(team_ball_control)


    #Draw  output
    ##Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)
    
    ##Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
 
    ##Draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == "__main__":
    main()
