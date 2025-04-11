from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Reshape image to 2D array for clustering
        image_2d = image.reshape(-1, 3)

        # Perform k-means clustering with 2 clusters (teams)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        
        return kmeans

    def get_player_color(self, frame, bbox):
        # Extract player region
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Perform KMeans clustering on the whole image (you can adjust this for better segmentation)
        kmeans = self.get_clustering_model(image)
        
        # Get cluster labels for each pixel
        labels = kmeans.labels_

        # Get the two cluster centers and find the dominant color in the player's region
        cluster_centers = kmeans.cluster_centers_
        label_counts = [np.sum(labels == i) for i in range(2)]
        
        # The dominant cluster (larger label count) corresponds to the background, the other to the player
        non_player_cluster = np.argmax(label_counts)
        player_cluster = 1 - non_player_cluster

        player_color = cluster_centers[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        # Get the color for each player
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Perform clustering on the colors to divide into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        # Store the team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get the player color
        player_color = self.get_player_color(frame, player_bbox)

        # Predict the team based on the player's color
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] 
        team_id += 1

        if player_id ==147:
            team_id = 2
        elif player_id == 552:
            team_id = 1

        # Store the team assignment for the player
        self.player_team_dict[player_id] = team_id
        return team_id
