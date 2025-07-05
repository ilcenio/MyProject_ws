import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Percorsi ai pacchetti necessari
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_turtlebot3_navigation2 = get_package_share_directory('turtlebot3_navigation2')
    pkg_dynamic_obstacle_detector = get_package_share_directory('dynamic_obstacle_detector')

    # Argomenti del launch file
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_path = LaunchConfiguration('map', default=os.path.join(
        pkg_turtlebot3_navigation2, 'map', 'map.yaml'))
    
    # Il nostro file di parametri personalizzato per Nav2
    nav2_params_path = LaunchConfiguration('params_file', default=os.path.join(
        pkg_dynamic_obstacle_detector, 'param', 'custom_nav2_params.yaml'))
    
    # Il nostro mondo personalizzato con l'ostacolo dinamico
    world_path = os.path.join(pkg_turtlebot3_gazebo, 'worlds', 'turtlebot3_dynamic_obstacles.world')

    # Percorso al nostro nuovo file di parametri per il detector
    detector_params_path = os.path.join(
        pkg_dynamic_obstacle_detector, 'param', 'detector_params.yaml')
    
    # 1. Avvio della simulazione, dello spawn del robot e del robot_state_publisher
    # Usiamo il launch file dedicato di turtlebot3_gazebo che fa tutto questo.
    start_simulation_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
             os.path.join(pkg_turtlebot3_gazebo, 'launch', 'turtlebot3_dynamic_obstacles.launch.py')),
        launch_arguments={
            'x_pose': '0.5',
            'y_pose': '-0.5'
        }.items())

    # 2. Avvio del nostro nodo di rilevamento ostacoli
    start_obstacle_detector_node = Node(
        package='dynamic_obstacle_detector',
        executable='obstacle_detector_node',
        name='obstacle_detector_node',
        output='screen',
        parameters=[
            detector_params_path,
            # --- SOVRASCRITTURA PER IL TUNING DEI PEDONI ---
            # Aumentiamo 'eps' per unire le due gambe di un pedone in un unico cluster.
            # Un valore tra 0.3 e 0.4 metri è un buon punto di partenza.
            {'dbscan_eps': 0.35},
            # Potremmo voler aumentare i campioni minimi per essere sicuri di tracciare una persona intera
            {'dbscan_min_samples': 5}
        ])

    # 3. Avvio di Nav2 con i nostri parametri personalizzati
    start_ros2_navigation_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            pkg_turtlebot3_navigation2, 'launch', 'navigation2.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'map': map_path,
            'params_file': nav2_params_path,
            'autostart': 'true',
            # Diciamo al launch file di Nav2 di NON avviare la simulazione/spawnare il robot
            'spawn_robot': 'false',
        }.items())

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        DeclareLaunchArgument('map', default_value=map_path, description='Full path to map file'),
        DeclareLaunchArgument('params_file', default_value=nav2_params_path, description='Full path to param file'),
        
        start_simulation_cmd,
        start_obstacle_detector_node,
        start_ros2_navigation_cmd
    ])
