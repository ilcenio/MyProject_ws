import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
import math
import struct
import numpy as np
from sklearn.cluster import DBSCAN
import tf2_ros
from scipy.optimize import linear_sum_assignment
from tf2_ros import TransformException
import tf2_geometry_msgs

class TrackedObstacle:
    """
    Una classe per mantenere lo stato di un singolo ostacolo tracciato nel tempo.
    """
    def __init__(self, obstacle_id, initial_cluster_cartesian, timestamp, kf_params):
        self.id = obstacle_id
        self.current_cartesian_cluster = []
        self.last_timestamp = timestamp

        # --- Inizializzazione del Filtro di Kalman ---
        # Lo stato [x, y, vx, vy]'. Posizione e velocità.
        initial_center = np.mean(np.array(initial_cluster_cartesian), axis=0)[:2]
        self.X = np.array([initial_center[0], initial_center[1], 0.0, 0.0]).reshape(4, 1)

        # Matrice di covarianza dello stato P: la nostra incertezza.
        self.P = np.diag(kf_params['p'])

        # Matrice di transizione di stato F (definita in update).
        self.F = np.eye(4)

        # Matrice di osservazione H: misuriamo solo la posizione (x, y).
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Covarianza del rumore di misurazione R: quanto ci fidiamo del nostro sensore.
        self.R = np.diag(kf_params['r'])

        # Covarianza del rumore di processo Q: incertezza nel modello di movimento (es. accelerazioni non modellate).
        self.Q = np.diag(kf_params['q'])


    def update(self, cluster_cartesian, timestamp):
        """Aggiorna lo stato dell'ostacolo con una nuova rilevazione."""
        self.current_cartesian_cluster = cluster_cartesian
        
        dt = (timestamp.sec - self.last_timestamp.sec) + (timestamp.nanosec - self.last_timestamp.nanosec) / 1e9
        self.last_timestamp = timestamp

        if dt > 0:
            self.predict(dt)

        # Fase di aggiornamento del filtro
        Z = np.mean(np.array(cluster_cartesian), axis=0)[:2].reshape(2, 1) # Misurazione
        y = Z - self.H @ self.X  # Errore di misurazione
        S = self.H @ self.P @ self.H.T + self.R
        # Calcolo del Guadagno di Kalman in modo numericamente più stabile
        # Invece di K = P @ H.T @ inv(S), risolviamo S @ K.T = (P @ H.T).T
        K_T = np.linalg.solve(S, (self.P @ self.H.T).T)
        K = K_T.T
        self.X = self.X + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def predict(self, dt):
        """Predice lo stato futuro basandosi sul modello di movimento."""
        # Modello a velocità costante: x = x0 + v*dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q

    def get_center(self):
        """Restituisce l'ultima posizione centrale conosciuta."""
        return (self.X[0, 0], self.X[1, 0])

    def get_velocity(self):
        """Restituisce la velocità stimata dal filtro."""
        return (self.X[2, 0], self.X[3, 0])

    def get_last_seen_time(self):
        """Restituisce il timestamp dell'ultima rilevazione."""
        return self.last_timestamp

class ObstacleDetectorNode(Node):
    """
    Un nodo per rilevare ostacoli basandosi sui dati di un LaserScan.
    """
    def __init__(self):
        super().__init__('obstacle_detector_node')

        # --- Dichiarazione dei Parametri ROS ---
        self.declare_parameter('max_tracking_distance', 0.5)
        self.declare_parameter('track_lifetime', 1.0)
        self.declare_parameter('prediction_horizon', 1.5)
        self.declare_parameter('prediction_steps', 5)
        # Parametri del Filtro di Kalman
        self.declare_parameter('kf.p', [0.1, 0.1, 10.0, 10.0])
        self.declare_parameter('kf.r', [0.3, 0.3])
        self.declare_parameter('kf.q', [0.001, 0.001, 0.01, 0.01])
        self.declare_parameter('target_frame', 'odom')
        # Nuovi parametri per DBSCAN
        self.declare_parameter('dbscan_eps', 0.2)
        self.declare_parameter('dbscan_min_samples', 4)

        # --- Lettura dei Parametri ---
        self.MAX_TRACKING_DISTANCE = self.get_parameter('max_tracking_distance').get_parameter_value().double_value
        self.TRACK_LIFETIME = self.get_parameter('track_lifetime').get_parameter_value().double_value
        self.PREDICTION_HORIZON = self.get_parameter('prediction_horizon').get_parameter_value().double_value
        self.PREDICTION_STEPS = self.get_parameter('prediction_steps').get_parameter_value().integer_value
        self.TARGET_FRAME = self.get_parameter('target_frame').get_parameter_value().string_value
        # Lettura parametri DBSCAN
        self.DBSCAN_EPS = self.get_parameter('dbscan_eps').get_parameter_value().double_value
        self.DBSCAN_MIN_SAMPLES = self.get_parameter('dbscan_min_samples').get_parameter_value().integer_value
        
        # Memorizziamo i parametri del KF per passarli ai nuovi track
        self.kf_params = {
            'p': self.get_parameter('kf.p').get_parameter_value().double_array_value,
            'r': self.get_parameter('kf.r').get_parameter_value().double_array_value,
            'q': self.get_parameter('kf.q').get_parameter_value().double_array_value
        }

        # Strutture dati per il tracking
        self.tracked_obstacles = {}  # Dizionario {track_id: TrackedObstacle}
        self.next_obstacle_id = 0

        # TF2 Buffer e Listener per le trasformazioni di coordinate
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Creiamo un subscriber per il topic /scan.
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Creiamo un publisher per pubblicare gli ostacoli come PointCloud2
        self.obstacle_pub = self.create_publisher(
            PointCloud2, 
            '/detected_obstacles', 
            10)

        # Creiamo un publisher per visualizzare le frecce della velocità in RViz
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            '/velocity_markers', 
            10)
        self.get_logger().info('Nodo di rilevamento ostacoli avviato.')

    def scan_callback(self, msg):
        """
        Questa funzione viene chiamata ogni volta che arriva un nuovo messaggio LaserScan.
        """
        # --- Trasformazione dei Punti nel Frame Fisso ---
        try:
            # Richiediamo la trasformazione dal frame del laser al frame target ('odom')
            # USANDO IL TIMESTAMP DEL MESSAGGIO SCAN. Questo è fondamentale per la corretta sincronizzazione.
            # L'uso di rclpy.time.Time() (ora attuale) causerebbe errori di percezione durante la rotazione del robot.
            transform = self.tf_buffer.lookup_transform(
                self.TARGET_FRAME,
                msg.header.frame_id,
                msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)) # Aggiungiamo un timeout per robustezza
        except TransformException as ex:
            self.get_logger().warn(f'Non è stato possibile trasformare {msg.header.frame_id} in {self.TARGET_FRAME}: {ex}')
            return

        # 1. Converti tutti i punti validi in coordinate cartesiane nel frame del LASER
        points_in_laser_frame = self.convert_scan_to_cartesian(msg)
        if not points_in_laser_frame:
            # Se non ci sono punti validi, non c'è nulla da fare.
            # Pubblichiamo un messaggio vuoto per cancellare i vecchi ostacoli se necessario.
            self.update_tracks_and_publish([], msg.header)
            return

        # 2. Applica DBSCAN per trovare i cluster (ancora nel frame del LASER)
        clusters_in_laser_frame = self.cluster_points_dbscan(points_in_laser_frame)

        # 3. Trasforma solo i punti dei cluster trovati nel frame TARGET ('odom')
        cartesian_clusters_odom = self.transform_clusters(clusters_in_laser_frame, transform, msg.header)
        
        # 4. Aggiorna i track, predici le traiettorie e pubblica
        self.update_tracks_and_publish(cartesian_clusters_odom, msg.header)

    def update_tracks_and_publish(self, clusters_cartesian_odom, header):
        """
        Gestisce il ciclo di vita dei track e pubblica le predizioni e i marker di velocità.
        I cluster in input sono già nel frame TARGET_FRAME.
        """
        # --- BEST PRACTICE: Creiamo un nuovo header per i messaggi in uscita ---
        output_header = Header()
        output_header.stamp = header.stamp
        output_header.frame_id = self.TARGET_FRAME

        # --- Associazione Dati con Algoritmo Ungherese (più robusto) ---
        # Se non ci sono track o cluster, la logica di associazione non serve.
        if not self.tracked_obstacles or not clusters_cartesian_odom:
            unmatched_cluster_indices = list(range(len(clusters_cartesian_odom)))
            matched_track_ids = set()
        else:
            track_ids = list(self.tracked_obstacles.keys())
            track_centers = np.array([self.tracked_obstacles[tid].get_center() for tid in track_ids])
            cluster_centers = np.array([np.mean(c, axis=0)[:2] for c in clusters_cartesian_odom])

            # 1. Calcola la matrice dei costi (distanze)
            cost_matrix = np.linalg.norm(track_centers[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2)

            # 2. Applica l'algoritmo di assegnazione
            track_indices, cluster_indices = linear_sum_assignment(cost_matrix)

            # 3. Filtra le associazioni valide (sotto la soglia di distanza)
            matched_track_ids = set()
            unmatched_cluster_indices = set(range(len(clusters_cartesian_odom)))
            
            for track_idx, cluster_idx in zip(track_indices, cluster_indices):
                if cost_matrix[track_idx, cluster_idx] < self.MAX_TRACKING_DISTANCE:
                    track_id = track_ids[track_idx]
                    self.tracked_obstacles[track_id].update(clusters_cartesian_odom[cluster_idx], header.stamp)
                    matched_track_ids.add(track_id)
                    if cluster_idx in unmatched_cluster_indices:
                        unmatched_cluster_indices.remove(cluster_idx)

            unmatched_cluster_indices = list(unmatched_cluster_indices)

        # --- Gestione Ciclo di Vita ---
        marker_array = MarkerArray()
        current_time_sec = header.stamp.sec + header.stamp.nanosec / 1e9

        # Rimuovi i track vecchi e crea i marker di CANCELLAZIONE per pulire RViz
        # Nota: la logica qui è corretta, ma ora `matched_track_ids` è calcolato sopra.
        # Il set di track da rimuovere sono quelli non presenti in `matched_track_ids`
        tracks_to_remove = [
            tid for tid, track in self.tracked_obstacles.items() 
            if tid not in matched_track_ids and 
            current_time_sec - (track.get_last_seen_time().sec + track.get_last_seen_time().nanosec / 1e9) > self.TRACK_LIFETIME
        ]
        for track_id in tracks_to_remove:
            delete_marker = Marker()
            delete_marker.header = output_header
            delete_marker.ns = "velocity_vectors"
            delete_marker.id = track_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
            del self.tracked_obstacles[track_id]

        # Crea nuovi track per i cluster non associati
        for i in unmatched_cluster_indices:
            new_id = self.next_obstacle_id
            self.tracked_obstacles[new_id] = TrackedObstacle(new_id, clusters_cartesian_odom[i], header.stamp, self.kf_params)
            self.next_obstacle_id += 1

        # --- Predizione e Pubblicazione ---
        prediction_points = []
        if self.tracked_obstacles:
            self.get_logger().info(f'Tracciando {len(self.tracked_obstacles)} ostacoli.')

        for track_id, track in self.tracked_obstacles.items():
            # A. Prepara i punti predetti per la costmap
            pos = track.get_center()
            vel = track.get_velocity()
            dt = self.PREDICTION_HORIZON / self.PREDICTION_STEPS
            for i in range(1, self.PREDICTION_STEPS + 1):
                t = i * dt
                prediction_points.append((pos[0] + vel[0] * t, pos[1] + vel[1] * t, 0.0))

            # B. Prepara il marker della velocità (freccia) per RViz
            marker = Marker()
            marker.header = output_header
            marker.ns = "velocity_vectors"
            marker.id = track_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            start_point = Point(x=pos[0], y=pos[1], z=0.1)
            end_point = Point(x=pos[0] + vel[0], y=pos[1] + vel[1], z=0.1)
            marker.points = [start_point, end_point]

            marker.scale.x = 0.05  # Diametro del corpo
            marker.scale.y = 0.1   # Diametro della testa
            marker.scale.z = 0.0   # La lunghezza della testa non è usata per le frecce 2D

            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        # Pubblica i messaggi
        self.marker_pub.publish(marker_array)

        # Costruisci e pubblica il PointCloud2 per la costmap
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_data = struct.pack('<%df' % (len(prediction_points) * 3), *[v for p in prediction_points for v in p])
        point_cloud_msg = PointCloud2(
            header=output_header, height=1, width=len(prediction_points), is_dense=False,            is_bigendian=False, fields=fields, point_step=12, row_step=12 * len(prediction_points),
            data=point_data)
        self.obstacle_pub.publish(point_cloud_msg)

    def convert_scan_to_cartesian(self, scan_msg):
        """Converte i range di un LaserScan in una lista di punti cartesiani [x, y]."""
        points = []
        for i, r in enumerate(scan_msg.ranges):
            # Ignora i punti non validi (inf, nan) e quelli fuori dal range del sensore
            if not math.isfinite(r) or r < scan_msg.range_min or r > scan_msg.range_max:
                continue
            
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            points.append([x, y])
        return points
    
    def cluster_points_dbscan(self, points):
        """
        Usa DBSCAN per raggruppare una lista di punti in coordinate cartesiane.
        Restituisce una lista di cluster, dove ogni cluster è una lista di punti.
        """
        if not points:
            return []

        points_np = np.array(points)
        
        # Applica DBSCAN
        db = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DBSCAN_MIN_SAMPLES).fit(points_np)
        labels = db.labels_

        # Raggruppa i punti per etichetta di cluster
        unique_labels = set(labels)
        clusters = []
        for k in unique_labels:
            if k == -1:
                # -1 è l'etichetta per il rumore, lo ignoriamo
                continue
            
            class_member_mask = (labels == k)
            cluster_points = points_np[class_member_mask]
            clusters.append(cluster_points.tolist())  # Converte in lista di liste
            
        return clusters

    def transform_clusters(self, clusters_in_laser_frame, transform_stamped, header):
        """Trasforma una lista di cluster dal frame del laser al frame target."""
        transformed_clusters = []
        # Estrarre la traslazione e la rotazione (come quaternione) dalla trasformazione
        t = transform_stamped.transform.translation
        q = transform_stamped.transform.rotation
        
        # Creare la matrice di rotazione 3x3 dal quaternione
        # Questo è più efficiente che usare tf2_geometry_msgs per ogni punto
        # Nota: questa è una rotazione 3D, ma la applicheremo a punti 2D (z=0)
        # La formula è standard per la conversione da quaternione a matrice di rotazione
        rot_matrix = np.array([
            [1 - 2*q.y**2 - 2*q.z**2, 2*q.x*q.y - 2*q.z*q.w, 2*q.x*q.z + 2*q.y*q.w],
            [2*q.x*q.y + 2*q.z*q.w, 1 - 2*q.x**2 - 2*q.z**2, 2*q.y*q.z - 2*q.x*q.w],
            [2*q.x*q.z - 2*q.y*q.w, 2*q.y*q.z + 2*q.x*q.w, 1 - 2*q.x**2 - 2*q.y**2]
        ])
        translation_vec = np.array([t.x, t.y, t.z])

        for cluster in clusters_in_laser_frame:
            if not cluster:
                continue
            # Converti il cluster in un array numpy [N, 2] e aggiungi una colonna di zeri -> [N, 3]
            points_np = np.hstack([np.array(cluster), np.zeros((len(cluster), 1))])
            # Applica la rotazione e poi la traslazione a tutti i punti in una sola operazione
            transformed_points = points_np @ rot_matrix.T + translation_vec
            transformed_clusters.append(transformed_points[:, :3].tolist()) # Manteniamo (x, y, z)
        return transformed_clusters
    
def main(args=None):
    rclpy.init(args=args)
    obstacle_detector_node = ObstacleDetectorNode()
    rclpy.spin(obstacle_detector_node)
    obstacle_detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
