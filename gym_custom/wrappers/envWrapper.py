"""
Gym wrapper classes that interacts with non-standard or private interfaces of Gym-Duckietown to implement additional
functionality or fix a bug. The usability of these heavily depend on. the versions of Gym-Duckietown.
Most of these were implemented for Gym-Duckietown version 5.0.16
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import logging
from random import randint as ri

import gym
import numpy as np
from gym import spaces
from gym_duckietown.simulator import Simulator
from numpy import sign, sin, cos, tan

logger = logging.getLogger(__name__)


class ParamRandWrapper(gym.Wrapper):
    """
       Adds more randomization options to Gym-Duckietown
       Randomized parameters should be specified by a dictionary which has a 'distribution' key (specifying the type)
       Allowed values: 'uniform', 'categorical'
       For uniform distributions the dict should have a 'low' and 'high' key. low < high
       Random samples will be drawn from Unif[low, high).
       Example:
           'robot_speed': {'distribution': 'uniform',
                            'low': 0.1,
                            'high': 1.2
                           }
       For categorical distributions the dict should have a 'values' key specifying valid values
       Example:
           'frame_skip': {'distribution': 'categorical',
                          'values': [1, 2, 3, 4]
                         }
       """
    distribution_types = ['uniform', 'categorical']

    def __init__(self, env, env_config: dict):
        super(ParamRandWrapper, self).__init__(env)
        self.robot_speed_conf = self.check_random_var_descriptor(env_config.get('robot_speed'))
        self.frame_skip_conf = self.check_random_var_descriptor(env_config.get('frame_skip'))

    def _reset(self):
        simulator = self.unwrapped  # type: Simulator
        if self.robot_speed_conf is not None:
            cur_robot_speed = self.sample_random_var(self.robot_speed_conf)
            simulator.robot_speed = cur_robot_speed
        if self.frame_skip_conf is not None:
            cur_frame_skip = self.sample_random_var(self.frame_skip_conf)
            simulator.frame_skip = cur_frame_skip

    def reset(self, **kwargs):
        self._reset()
        return self.env.reset(**kwargs)

    @classmethod
    def sample_random_var(cls, random_var_descriptor):
        if type(random_var_descriptor) is dict:
            if random_var_descriptor.get('distribution') in 'uniform':
                low = random_var_descriptor.get('low')
                high = random_var_descriptor.get('high')
                return (high - low) * np.random.random() + low
            elif random_var_descriptor.get('distribution') in 'categorical':
                return np.random.choice(random_var_descriptor.get('values'))
            else:
                logger.error("Distribution type should be  one of {}".format(cls.distribution_types))
        else:
            return random_var_descriptor

    @staticmethod
    def check_random_var_descriptor(random_var_descriptor):
        if type(random_var_descriptor) == dict:
            if random_var_descriptor.get('distribution') == 'uniform':
                if random_var_descriptor.get('low') < random_var_descriptor.get('high'):
                    # The descriptor is correct
                    return random_var_descriptor

            if random_var_descriptor.get('distribution') == 'categorical':
                if type(random_var_descriptor.get('values')) == list:
                    return random_var_descriptor

            # The descriptor is a dictionary (and likely intended to be a random value descriptor), but it's invalid
            logger.error("""Randomized parameters should be specified by a dictionary which has a 'distribution' key (specifying the type)
                                Allowed values: 'uniform', 'categorical'
                                For uniform distributions the dict should have a 'low' and 'high' key. low < high
                                Random samples will be drawn from Unif[low, high).
                                Example:
                                    {'type': 'uniform',
                                     'low': 0.,
                                     'high': 1.
                                    }
                                For categorical distributions the dict should have a 'values' key specifying valid values
                                Example:
                                    {'type': 'categorical',
                                     'values': [1, 2, 3, 4]
                                    }"""
                         )

        # This descriptor is not a random value descriptor
        return None


class ActionDelayWrapper(gym.Wrapper):
    """
    Computed actions come into effect later in the time period of a step.
    The maximum delay this wrapper can apply is exactly one step time.
    'action_delay_ratio' key of in env_config dictionary controls the amount of delay this wrapper applies.
    Allowed values:
        Floats in the (0., 1.) interval
        'random' to get random values in each instance of the env
        0.0 disables delays applied by this wrapper
    Recommended values: 0.2 ... 0.5
    Example: action_delay_ratio = 0.25 and the simulator runs with the default 30 FPS frame rate, the delay is
    1/30*0.25 = 33.3 ms * 0.25 = 8.33 ms
    """

    def __init__(self, env):
        super(ActionDelayWrapper, self).__init__(env)
        # self.env_config = env_config
        self.simulator = self.unwrapped  # type: Simulator
        # assert isinstance(self.simulator, Simulator), "Env must be gym_duckietown.simulator.Simulator"

        # if env_config.get('action_delay_ratio', 0.) == "random":
        self.random_ratio = True
        self.ratio_distribution_range = (0.9, 0.95)
        self.action_delay_ratio = np.random.uniform(*self.ratio_distribution_range)
        # else:
        # self.random_ratio = False
        # self.action_delay_ratio = env_config.get('action_delay_ratio', 0.)
        # assert self.action_delay_ratio > 0.0 and self.action_delay_ratio < 1.0, "action_delay_ratio must be in the (0, 1) interval"
        self.last_action = np.zeros(self.action_space.shape)

    def step(self, action):
        delta_time = 1.0 / self.simulator.frame_rate
        # Apply the action delay
        for _ in range(self.simulator.frame_skip):
            # Passing delta time to update physics is not enough, _update_pos() uses the member variable delta_time
            self.simulator.delta_time = delta_time * self.action_delay_ratio

            try:
                self.simulator.update_physics(action=self.last_action, delta_time=None)
                self.simulator.step_count -= 1
            except:
                pass

            # Update physics increments step count but that should ony be incremented once for each step
            # This will happen when self.env.step() is called

        self.last_action = action
        self.simulator.delta_time = delta_time * (1. - self.action_delay_ratio)
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = np.zeros(self.action_space.shape)
        if self.random_ratio:
            self.action_delay_ratio = np.random.uniform(*self.ratio_distribution_range)
        return self.env.reset(**kwargs)


class InconvenientSpawnFixingWrapper(gym.Wrapper):
    """
    Fixes the "Exception: Could not find a valid starting pose after 5000 attempts" in duckietown-gym-daffy 5.0.13
    The robot is first placed in a random drivable tile, than a configuration is sampled on this tile. If the later
    step fails, it is repeated (up to 5000 times). If a tile has too many obstacles on it, it might not have any
    convenient (collision risking) configurations, so another tile should be selected. Instead of selecting a new tile,
    Duckietown gym just raises the above exception.
    This wrapper calls reset() again and again if a new tile has to be sampled.
    .. note::
        ``gym_duckietown.simulator.Simulator.reset()`` is called in ``gym_duckietown.simulator.Simulator.__init__(...)``.
        **Simulator instantiation should also be wrapped in a similar while loop!!!**
    """

    def reset(self, **kwargs):
        spawn_successful = False
        spawn_attempts = 1
        while not spawn_successful:
            try:
                ret = self.env.reset(**kwargs)
                spawn_successful = True
            except Exception as e:
                self.unwrapped.seed_value += 1  # Otherwise it selects the same tile in the next attempt
                self.unwrapped.seed(self.unwrapped.seed_value)
                logger.error("{}; Retrying with new seed: {}".format(e, self.unwrapped.seed_value))
                spawn_attempts += 1
        logger.debug("Reset and Spawn successful after {} attempts".format(spawn_attempts))
        return ret


class ObstacleSpawningWrapper(gym.Wrapper):
    """
    Spawns obstacles (duckies, duckiebots) to random locations on a map.
    WARNING: all other obstacles are removed from the map!
    Obstacle specification must be included in env config in the following form:
    obstacles:
       duckie:
          density: 0.5
          static: true
       duckiebot:
          density: 0
          static: 'false'
    """
    default_object_heights = {'duckie': 0.08,
                              'duckiebot': 0.12,
                              'cone': 0.08,
                              'barrier': 0.08}

    def __init__(self, env):
        super(ObstacleSpawningWrapper, self).__init__(env)
        self.simulator = env.unwrapped  # type: Simulator
        self.env_config = {'spawn_obstacles': True, 'obstacles': {'duckie': {'density': 0.2, 'static': True},
                                                                  'duckiebot': {'density': 0, 'static': False}, }, }
        if self.env_config.get('spawn_obstacles', False):
            self.safe_spawn_objects()

    def reset(self, **kwargs):
        # if self.env_config.get('spawn_obstacles', False):
        # self.safe_spawn_objects()
        return self.env.reset(**kwargs)

    def safe_spawn_objects(self):
        spawn_successful = False
        while not spawn_successful:
            try:
                self._spawn_objects()
                spawn_successful = True
            except (IndexError, TypeError) as e:
                logger.error("Obstacle spawning failed with Exception: {}! Retrying.".format(e))

    def _spawn_objects(self):
        obstacles = []
        drivable_tiles = self.simulator.drivable_tiles
        # Get total obstacle count
        obstacle_cnt = 0
        for kind, descriptor in self.env_config.get('obstacles', {}).items():
            obstacle_cnt += int(descriptor.get('density', 0) * len(drivable_tiles))

        # more than 1 object on a single tile is not allowed, because it can easily create unavoidable obstacles.
        if len(drivable_tiles) < obstacle_cnt:
            obstacle_cnt = len(drivable_tiles)
            logger.warning("Can't spawn more obstacles than the amount of drivable tiles on a map! "
                           "Decrease the density of the obstacles!")

        # Get a unique drivable tile for each object
        tiles = self._sample_tiles(drivable_tiles, obstacle_cnt)

        # Generate coordinates and orientations for the obstacles
        obstacle_idx = 0
        for kind, descriptor in self.env_config.get('obstacles', {}).items():
            kind_cnt = int(descriptor.get('density', 0) * len(drivable_tiles))
            for _ in range(kind_cnt):
                tile = tiles[obstacle_idx]
                pos_on_tile = np.random.random(2)
                while kind == 'duckiebot' and np.linalg.norm(np.array([0.5, 0.5]) - pos_on_tile) > 0.25:
                    # If a duckiebot is placed outside the road, it won't be able to follow a lane
                    # This will raise an exception so a duckiebot is never spawned outside a circle on the tile...
                    pos_on_tile = np.random.random(2)
                pos = tile['coords'] + pos_on_tile
                rotate = np.random.random() * 360 - 180
                height_scaler = 0.8 + np.random.random() * 0.4
                height = self.default_object_heights[kind] * height_scaler
                static = descriptor.get('static', True)
                obstacles.append({'kind': kind,
                                  'pos': pos,
                                  'rotate': rotate,
                                  'height': height,
                                  'static': static})
                obstacle_idx += 1
                if obstacle_idx >= len(tiles):
                    break
            if obstacle_idx >= len(tiles):
                break
        self.simulator._load_objects({'objects': obstacles})
        # self.simulator.collidable_corners = np.zeros_like(self.simulator.collidable_corners) #BUGFIX in the simulator...

    def _sample_tiles(self, drivable_tiles, size):
        """ Returns a list of simulator tiles, which are selected randomly without replacement.
        Replacement is not allowed, because more than 1 object on a single tile can easily create unavoidable obstacles.
        """
        tile_idx = np.random.choice(range(len(drivable_tiles)), size, replace=False)
        tiles = [drivable_tiles[idx] for idx in tile_idx]
        return tiles


class ForwardObstacleSpawnnigWrapper(gym.Wrapper):
    default_object_heights = {'duckie': 0.08,
                              'duckiebot': 0.12,
                              'cone': 0.08,
                              'barrier': 0.08}

    def __init__(self, env):

        self.env_config = {'spawn_obstacles': True, 'obstacles': {'duckie': {'density': 0, 'static': True},
                                                                  'duckiebot': {'density': 0.2, 'static': False}, },
                           'spawn_forward_obstacle': True}
        super(ForwardObstacleSpawnnigWrapper, self).__init__(env)
        self.simulator = env.unwrapped
        self.lateral_pos_perturb_half_width = 0.2 * self.simulator.road_tile_size
        self.orientation_perturb_half_width = np.pi / 4
        if self.env_config.get('spawn_forward_obstacle', False):
            self.safe_spawn_objects()

    def safe_spawn_objects(self):
        spawn_successful = False
        while not spawn_successful:
            try:
                self._spawn_objects()
                spawn_successful = True
            except (IndexError, TypeError) as e:
                logger.error("Obstacle spawning failed with Exception: {}! Retrying.".format(e))

    def reset(self, **kwargs):
        reset_good = False
        while not reset_good:
            ret = self.env.reset(**kwargs)
            curve_point, curve_tangent = self.simulator.closest_curve_point(self.simulator.cur_pos,
                                                                            self.simulator.cur_angle)
            if curve_point is not None and curve_tangent is not None:
                reset_good = True
            else:
                logger.error("Reset before forward object spawning failed curve_point or curve_tangent is None")
        if self.env_config.get('spawn_forward_obstacle', False):
            self.safe_spawn_objects()
        return ret

    def _spawn_objects(self):
        drivable_tiles = self.simulator.drivable_tiles
        tile_size = self.simulator.road_tile_size
        # obj_pose_valid = False
        # while not obj_pose_valid:
        # Get a random point in front of the vehicle
        forward_dist = tile_size * (2. + 2 * np.random.random())
        obj_pos, obj_pos_tangent = self.get_point_on_curve_ahead(forward_dist, self.simulator)
        # Randomise the lateral posotion of the obstacle
        right_normal_to_curve = np.array([obj_pos_tangent[2], 0, -obj_pos_tangent[0]])
        # lateral position can be +- half lane width from the lane center
        obj_pos += right_normal_to_curve * (np.random.random() - 0.5) * self.lateral_pos_perturb_half_width
        # Convert coordinates given in meters to coords in tiles
        obj_pos_tiles = [obj_pos[0] / tile_size, obj_pos[2] / tile_size]

        # Check coordinates
        tile_coords = self.simulator.get_grid_coords(obj_pos)
        tile = self.simulator._get_tile(tile_coords[0], tile_coords[1])
        pos_on_tile, _ = np.modf(obj_pos_tiles)
        # if np.linalg.norm(np.array([0.5, 0.5]) - pos_on_tile) < 0.25 and tile in drivable_tiles:
        #     obj_pose_valid = True
        # else:
        #     logger.warning("Forward obstacle position invalid!")
        # obj_pose_valid = True

        # Object points approximately towrads the tangent at it's position
        obj_orientation = self.dir_vec_to_angle(obj_pos_tangent)
        obj_orientation += (np.random.random() - 0.5) * self.orientation_perturb_half_width
        # Convert orientation in radians to degrees
        obj_orientation_deg = obj_orientation / np.pi * 180

        kind = 'duckiebot'
        static = False
        height_scaler = 0.8 + np.random.random() * 0.4
        height = self.default_object_heights[kind] * height_scaler
        obstacles = [{'kind': kind,
                      'pos': obj_pos_tiles,  # [1.7, 4.65],
                      'rotate': obj_orientation_deg,  # 95, 
                      'height': height,
                      'static': static}]
        self.simulator._load_objects({'objects': obstacles})

        # Gym-Duckietown's default safety radius is small, so the learned follow distance is very small
        # This doesn't work for static obstacles, simulator.collidable_safety_radii must be changed for that...
        for obj in self.simulator.objects:
            obj.safety_radius *= 2
            if obj.domain_rand:
                obj.trim = np.random.uniform(-0.1, 0.1)  # This fixes another bug in the simulator.....

        # BUGFIX in the simulator... Dynamic obstacles leave a phantom static version at their initial position
        if not static:
            self.simulator.collidable_corners = np.zeros_like(self.simulator.collidable_corners)
            self.simulator.collidable_centers = np.zeros_like(self.simulator.collidable_centers)

    def get_point_on_curve_ahead(self, forward_dist, simulator):

        # Find the curve point closest to the agent, and the tangent at that point
        curve_point, curve_tangent = simulator.closest_curve_point(simulator.cur_pos, simulator.cur_angle)
        curve_angle = self.dir_vec_to_angle(curve_tangent)
        iterations = 10
        for _ in range(iterations):
            # Project a point ahead along the curve tangent, then find the closest point to to that
            follow_point = curve_point + curve_tangent * forward_dist / iterations
            curve_point, curve_tangent = simulator.closest_curve_point(follow_point, curve_angle)
            curve_angle = self.dir_vec_to_angle(curve_tangent)

        return curve_point, curve_tangent

    @staticmethod
    def dir_vec_to_angle(dir_vec):
        angle = np.arctan2(-dir_vec[2], dir_vec[0])  # Must equal to simulator.cur_angle
        if angle < 0:
            angle += 2 * np.pi
        return angle


class PrepareLearningWrapper(gym.Wrapper):
    def __init__(self, env=None):
        super(PrepareLearningWrapper, self).__init__(env)
        self.observation_space = spaces.Dict({"direction": spaces.Discrete(4), "view": self.observation_space})

    def reset(self):
        return {
            "direction": 0,
            "view": self.env.reset().astype(np.float32)
        }

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return {"direction": info["direction"], "view": obs.astype(np.float32)}, reward, done, info


class TileWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TileWrapper, self).__init__(env)

        self._tile = None
        self._state = None
        self._curve = 0
        self._direction = None

        self._curve_dicts_3way = [{0: 5, 1: 1}, {-1: 2, 1: 0}, {-1: 4, 0: 3}]
        self._curve_dicts_4way = [{0: 5, 1: 1, -1: 0}, {-1: 2, 1: 0, 0: 0}, {-1: 4, 0: 3, 1: 0}]

        self._cr_dirs = ["south", "east", "north", "west"]
        self._cr_vectors = {
            "south": [0, 1],
            "east": [1, 0],
            "north": [0, -1],
            "west": [-1, 0]
        }

        self._dirs = {
            "laneFollowing": 0,
            "left": 1,
            "forward": 2,
            "right": 3
        }

        self._moves = {
            "curve_left": ["laneFollowing"],
            "straight": ["laneFollowing"],
            "4way": ["left", "forward", "right"],
            "3way_left": [["forward", "right"], ["left", "right"], ["left", "forward"]]
        }

        # self.env.unwrapped.closest_curve_point = self._get_chosen_curve

        self.env.unwrapped._get_curve_points = self._get_pts

    def _gettile(self, tile_coords: list):
        return self.env.unwrapped._get_tile(tile_coords[0], tile_coords[1])

    def _next_crossroad(self, tc: list, pos: list, phi: float):
        cons_next = []  # Рассматриваемые тайлы
        crossroads = ['3way_left', '4way']

        dx = sign(cos(phi))
        dy = -sign(sin(phi))
        atphi = abs(tan(phi))

        tcx = tc[0] + dx
        tcy = tc[1] + dy

        if dy == 0 or dx == 0:  # Если взгляд перпендикулярен тайлу
            cons_next.append([tcx, tcy])
        else:  # Сравнение углов, чтобы выбрать нужный тайл
            # Соотношение сторон прямоугольника, в который попадает луч взгляда
            tga = (-dx * (tc[1] + int(dy > 0)) + dx * pos[2]) \
                  / (dx * (tc[0] + int(dx > 0)) - dx * pos[0])

            if atphi < abs(tga):
                cons_next.append([tcx, tc[1]])
            elif atphi > abs(tga):
                cons_next.append([tc[0], tcy])
            else:
                if atphi <= 1:
                    cons_next.append([tcx, tc[1]])
                if atphi >= 1:
                    cons_next.append([tc[0], tcy])
        # Проверка тайлов на принадлежность перекрёсткам и выбор из двух тайлов (в случае неопределённости направления взгляда)
        tile1 = self._gettile(cons_next[0])
        if len(cons_next) == 1 and tile1 and tile1['kind'] in crossroads:
            return self._gettile(cons_next[0])['kind']
        if len(cons_next) > 1:
            tile2 = self._gettile(cons_next[1])
            if tile1 and tile1['kind'] in crossroads and tile2 and tile2['kind'] in crossroads:
                return self._gettile(cons_next[ri(0, 1)])['kind']
            if tile1 and tile1['kind'] in crossroads:
                return tile1['kind']
            if tile2 and tile2['kind'] in crossroads:
                return tile2['kind']
            return None
        return None

    def _directions(self, ppos: list, npos: list, nkind: str, cr_dir: int) -> list:
        if nkind != "3way_left":
            if nkind in self._moves.keys():
                return [(self._dirs[i], i) for i in self._moves[nkind]]
            return []

        v_bot = [npos[0] - ppos[0], npos[1] - ppos[1]]
        if np.linalg.norm(v_bot) != 1:
            return [(0, 'laneFollowing')]
        v_cr = self._cr_vectors[self._cr_dirs[cr_dir]]
        index = int(np.dot(v_bot, v_cr)) + 1
        print("vecs: ", v_bot, v_cr)
        print("dot: ", np.dot(v_bot, v_cr))
        return [(self._dirs[i], i) for i in self._moves[nkind][index]]

    def _curve_index(self, ppos: list, npos: list, nkind: str, cr_dir: int):
        state = self._state
        curve = 0
        if state != None:
            v_bot = [npos[0] - ppos[0], npos[1] - ppos[1]]
            v_cr = self._cr_vectors[self._cr_dirs[cr_dir]]
            dot = int(np.dot(v_bot, v_cr))
            if state == 0 and dot != 1:
                curve = 1
            elif state != 0:
                index = state - 1
                if nkind == "3way_left":
                    print("curveind: ", index)
                    curve = self._curve_dicts_3way[index][dot]
                elif nkind == "4way_left":
                    curve = self._curve_dicts_4way[index][dot]
        return curve

    def _get_chosen_curve(self, pos, angle):
        i, j = self.env.unwrapped.get_grid_coords(pos)
        tile = self.env.unwrapped._get_tile(i, j)

        if tile is None or not tile["drivable"]:
            return None, None

        # Find curve with largest dotproduct with heading
        curves = tile["curves"]
        cps = curves[self._curve]

        from gym_duckietown import simulator

        t = simulator.bezier_closest(cps, pos)
        point = simulator.bezier_point(cps, t)
        tangent = simulator.bezier_tangent(cps, t)

        return point, tangent

    def _get_pts(self, curves, angle):
        return curves[self._curve]

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, done, info = super(TileWrapper, self).step(action)

        cur_info = self.env.unwrapped.get_agent_info()

        next_crossroad = self._next_crossroad(cur_info["Simulator"]["tile_coords"],
                                              cur_info["Simulator"]["cur_pos"],
                                              cur_info["Simulator"]["cur_angle"])

        cur_tile = self._gettile(cur_info["Simulator"]["tile_coords"])
        if not self._tile:
            self._state = 0
            self._tile = cur_tile
            self._direction = 'laneFollowing'
        elif self._tile is not cur_tile:
            ppos, npos, nkind, cr_dir = self._tile["coords"], cur_tile["coords"], cur_tile["kind"], cur_tile["angle"]
            directions = self._directions(ppos, npos, nkind, cr_dir)
            chosen_direction_idx, self._direction = directions[ri(0, len(directions) - 1)]
            self._state = chosen_direction_idx
            self._curve = self._curve_index(ppos, npos, nkind, cr_dir)
            self._tile = cur_tile


        info["Simulator"]["next_crossroad"] = next_crossroad
        info["direction_name"] = self._direction
        info["direction"] = self._state
        info["curve_index"] = self._curve
        return obs, reward, done, info
