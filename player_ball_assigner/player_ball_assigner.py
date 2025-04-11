import sys
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance=70
    
    def assign_ball_to_player(self, players, ball_bbox):

        ball_position= get_center_of_bbox(ball_bbox)

        minimum_distance=99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox=player['bbox']

            distance_left=measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right=measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance=min(distance_left,distance_right)

            if distance<self.max_player_ball_distance:
                if distance<minimum_distance:
                    minimum_distance=distance
                    assigned_player=player_id

        
        return assigned_player
    

    def find_nearest_player_to_ball(self, player_track, ball_bbox):
        if not player_track:
            return -1
        
        ball_center = ((ball_bbox[0] + ball_bbox[2]) / 2,(ball_bbox[1] + ball_bbox[3]) / 2)
    
        min_distance = float('inf')
        nearest_player_id = -1
    
        for player_id, track in player_track.items():
            player_bbox = track['bbox']
            player_center = ((player_bbox[0] + player_bbox[2]) / 2,(player_bbox[1] + player_bbox[3]) / 2)
        
            distance = ((player_center[0] - ball_center[0]) ** 2 +(player_center[1] - ball_center[1]) ** 2) ** 0.5
        
            if distance < min_distance:
                min_distance = distance
                nearest_player_id = player_id
    
        return nearest_player_id



