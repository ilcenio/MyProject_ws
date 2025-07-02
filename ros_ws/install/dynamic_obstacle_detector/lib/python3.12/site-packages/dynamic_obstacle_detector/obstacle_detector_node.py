import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
import math
import struct
import numpy as np
import tf2_ros
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
        K = self.P @ self.H.T @ np.linalg.inv(S) # Guadagno di Kalman
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
        self.declare_parameter('kf.r', [0.1, 0.1])
        self.declare_parameter('kf.q', [0.01, 0.01, 0.1, 0.1])
        self.declare_parameter('target_frame', 'odom')

        # --- Lettura dei Parametri ---
        self.MAX_TRACKING_DISTANCE = self.get_parameter('max_tracking_distance').get_parameter_value().double_value
        self.TRACK_LIFETIME = self.get_parameter('track_lifetime').get_parameter_value().double_value
        self.PREDICTION_HORIZON = self.get_parameter('prediction_horizon').get_parameter_value().double_value
        self.PREDICTION_STEPS = self.get_parameter('prediction_steps').get_parameter_value().integer_value
        self.TARGET_FRAME = self.get_parameter('target_frame').get_parameter_value().string_value
        
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
            # Chiediamo la trasformazione tra il frame del laser (es. 'base_scan') e il nostro frame target ('odom')
            # Usiamo rclpy.time.Time() per ottenere la trasformazione più recente disponibile.
            transform = self.tf_buffer.lookup_transform(
                self.TARGET_FRAME,
                msg.header.frame_id,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f'Non è stato possibile trasformare {msg.header.frame_id} in {self.TARGET_FRAME}: {ex}')
            return

        # 1. Rileva i cluster di punti dal LaserScan (in coordinate polari)
        polar_clusters = self.cluster_scan(msg)

        # 2. Converte i cluster in coordinate cartesiane NEL FRAME TARGET ('odom')
        cartesian_clusters_odom = []
        for cluster in polar_clusters:
            cart_cluster_odom = []
            for r, theta in cluster:
                # Crea un punto nel frame del laser
                point_in_laser_frame = PointStamped()
                point_in_laser_frame.header = msg.header
                # Converte da polare a cartesiano (nel frame del laser)
                point_in_laser_frame.point.x = r * math.cos(theta)
                point_in_laser_frame.point.y = r * math.sin(theta)
                point_in_laser_frame.point.z = 0.0

                # Trasforma il punto dal frame del laser al frame target ('odom')
                point_in_odom_frame = tf2_geometry_msgs.do_transform_point(point_in_laser_frame, transform)
            
                cart_cluster_odom.append((point_in_odom_frame.point.x, point_in_odom_frame.point.y, 0.0))
        
            if cart_cluster_odom:
                cartesian_clusters_odom.append(cart_cluster_odom)
        
        # 3. Aggiorna i track, predici le traiettorie e pubblica
        self.update_tracks_and_publish(cartesian_clusters_odom, msg.header)

    def update_tracks_and_publish(self, clusters_cartesian_odom, header):
        """
        Gestisce il ciclo di vita dei track e pubblica le predizioni e i marker di velocità.
        I cluster in input sono già nel frame TARGET_FRAME.
        """
        # --- MODIFICATO: Creiamo un nuovo header per i messaggi in uscita ---
        output_header = Header()
        output_header.stamp = header.stamp
        output_header.frame_id = self.TARGET_FRAME

        # --- Associazione Dati ---
        matched_track_ids = set()
        unmatched_cluster_indices = list(range(len(clusters_cartesian_odom)))

        for track_id, track in self.tracked_obstacles.items():
            track_center = track.get_center()
            min_dist = self.MAX_TRACKING_DISTANCE
            best_match_idx = -1

            for i in unmatched_cluster_indices:
                cluster_center_x = np.mean([p[0] for p in clusters_cartesian_odom[i]])
                cluster_center_y = np.mean([p[1] for p in clusters_cartesian_odom[i]])
                dist = math.hypot(track_center[0] - cluster_center_x, track_center[1] - cluster_center_y)
                
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                track.update(clusters_cartesian_odom[best_match_idx], header.stamp)
                matched_track_ids.add(track_id)
                unmatched_cluster_indices.remove(best_match_idx)

        # --- Gestione Ciclo di Vita ---
        marker_array = MarkerArray()
        current_time_sec = header.stamp.sec + header.stamp.nanosec / 1e9

        # Rimuovi i track vecchi e crea i marker di CANCELLAZIONE per pulire RViz
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

    def cluster_scan(self, scan_msg, cluster_dist_threshold=0.1):
        """
        Raggruppa i punti dello scan in cluster.
        Due punti appartengono allo stesso cluster se la loro distanza è inferiore a una soglia.
        """
        clusters = []
        current_cluster = []
        
        for i in range(1, len(scan_msg.ranges)):
            if not (math.isfinite(scan_msg.ranges[i]) and math.isfinite(scan_msg.ranges[i-1])):
                if current_cluster:
                    if len(current_cluster) > 3:
                        clusters.append(current_cluster)
                    current_cluster = []
                continue

            dist_between_points = math.sqrt(
                scan_msg.ranges[i]**2 + scan_msg.ranges[i-1]**2 - 
                2 * scan_msg.ranges[i] * scan_msg.ranges[i-1] * 
                math.cos(scan_msg.angle_increment)
            )

            if dist_between_points < cluster_dist_threshold:
                if not current_cluster:
                    angle = scan_msg.angle_min + (i-1) * scan_msg.angle_increment
                    current_cluster.append((scan_msg.ranges[i-1], angle))
                
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                current_cluster.append((scan_msg.ranges[i], angle))
            else:
                if current_cluster:
                    if len(current_cluster) > 3:
                        clusters.append(current_cluster)
                    current_cluster = []
        
        if current_cluster and len(current_cluster) > 3:
            clusters.append(current_cluster)
            
        return clusters


def main(args=None):
    rclpy.init(args=args)
    obstacle_detector_node = ObstacleDetectorNode()
    rclpy.spin(obstacle_detector_node)
    obstacle_detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
