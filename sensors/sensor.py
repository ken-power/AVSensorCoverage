import numpy as np
from scipy.spatial.transform import Rotation as R


# this class contains generic sensor properties and functions that are used by every sensortype. it acts as a parent
# class for camera lidar and radar
class Sensor:
    def __init__(self, position, name):
        self.position = np.array(position)
        self.name = name
        self.points = None
        self.mesh = None
        self.calculation_result = None
        self.covered_points = None
        self.number_covered_points = None
        self.covered_indices = None
        self.occluded_points = None
        self.number_occluded_points = None
        self.occluded_indices = None
        self.coordinate_system = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self.covered_volume = None
        self.occluded_volume = None
        self.metrics = np.zeros(shape=(18, 1))
        self.fraction_occluded = None

        # plot position is a parameter used in the automatic screenshot creation for every sensor.
        self.plot_position = np.zeros(3)
        if self.position[0] <= 0:
            self.plot_position[0] = self.position[0] - 5
        else:
            self.plot_position[0] = self.position[0] + 5
        if self.position[1] <= 0:
            self.plot_position[1] = self.position[1] - 5
        else:
            self.plot_position[1] = self.position[1] + 5

        self.plot_position[2] = self.position[2] + 5

    # function to rotate the sensor using pyvista functions. if local = false, rotation about global coordinate axis
    def rotate(self, local=True, pitch=0, yaw=0, roll=0):
        # rotate meshes
        if local:
            self.mesh = self.mesh.rotate_vector(
                self.coordinate_system[:, 1], pitch, point=self.position
            )
            self.mesh = self.mesh.rotate_vector(
                self.coordinate_system[:, 2], yaw, point=self.position
            )
            self.mesh = self.mesh.rotate_vector(
                self.coordinate_system[:, 0], roll, point=self.position
            )
        else:
            self.mesh = self.mesh.rotate_y(pitch, self.position)
            self.mesh = self.mesh.rotate_z(yaw, self.position)
            self.mesh = self.mesh.rotate_x(roll, self.position)

        # rotate local coordinate system
        r = R.from_euler("yzx", [pitch, yaw, roll], degrees=True)
        r = r.as_matrix()
        self.coordinate_system = np.matmul(self.coordinate_system, r)

    # function to translate the sensor using a pyvista function
    def translate(self, x=0, y=0, z=0):
        self.mesh = self.mesh.translate((x, y, z))
        self.position = np.array(
            [self.position[0] + x, self.position[1] + y, self.position[2] + z]
        )


    def __check_occlusion(self, rays, intersection_points, occluded_points):
        # Ensure numpy arrays
        intersection_points = np.asarray(intersection_points, dtype=float)
        occluded_points = np.asarray(occluded_points, dtype=float)
        rays = np.asarray(rays, dtype=int)

        # If there are no intersections at all
        if intersection_points.size == 0:
            return np.array([], dtype=int)

        # ---- Normalize shapes ----
        # intersection_points: want shape (N, 3)
        if intersection_points.ndim == 1:
            # handle single point (3,) or flattened array (N*3,)
            if intersection_points.size % 3 != 0:
                # shape is inconsistent → treat as no valid hits
                return np.array([], dtype=int)
            intersection_points = intersection_points.reshape(-1, 3)

        # occluded_points: want shape (N, 3)
        if occluded_points.ndim == 1:
            if occluded_points.size % 3 != 0:
                return np.array([], dtype=int)
            occluded_points = occluded_points.reshape(-1, 3)

        # rays: want shape (N,)
        if rays.ndim == 0:
            rays = rays.reshape(1)

        # Align lengths in case fallback returned mismatched sizes
        n = min(len(intersection_points), len(occluded_points), len(rays))
        if n == 0:
            return np.array([], dtype=int)

        intersection_points = intersection_points[:n]
        occluded_points = occluded_points[:n]
        rays = rays[:n]

        # ---- Filter out NaN intersections ----
        if intersection_points.ndim != 2 or intersection_points.shape[1] != 3:
            # Something is still weird → bail out safely
            return np.array([], dtype=int)

        valid_mask = ~np.isnan(intersection_points).any(axis=1)
        if not np.any(valid_mask):
            return np.array([], dtype=int)

        intersection_points = intersection_points[valid_mask]
        occluded_points = occluded_points[valid_mask]
        rays = rays[valid_mask]

        # ---- Distance comparison ----
        coll_vectors = intersection_points - np.tile(self.position, (len(intersection_points), 1))
        vectors = occluded_points - np.tile(self.position, (len(occluded_points), 1))

        length_coll = np.sqrt(np.sum(coll_vectors ** 2, axis=1))
        length = np.sqrt(np.sum(vectors ** 2, axis=1))

        valid = length_coll < length
        return rays[valid]
    
    # callable function to determine the points that are occluded by the vehicle
    def is_occluded_matrix(self, occlusion_mesh):
        origins = np.tile(self.position, (len(self.covered_points), 1))
        direction_vectors = self.covered_points - origins

        points, rays = occlusion_mesh.multi_ray_trace(
            origins, direction_vectors, first_point=True
        )[0:2]

        if rays.size > 0:
            occluded_points = self.covered_points[rays]
            rays = self.__check_occlusion(rays, points, occluded_points)
        else:
            rays = np.array([], dtype=int)

        occluded_indices = self.covered_indices[rays] if rays.size > 0 else np.array([], dtype=int)

        result = np.full(len(self.calculation_result), True)
        if occluded_indices.size > 0:
            np.put(result, occluded_indices, False)

        self.occluded_points = self.covered_points[rays] if rays.size > 0 else np.array([])
        self.calculation_result = self.calculation_result & result
        self.occluded_indices = occluded_indices
        self.number_occluded_points = len(occluded_indices)

    # function to set the sensor metrics, that is called after the calculation is done
    def set_metrics(self, grid, indexes=None, all_metrics=True):
        # calculate volume metrics
        cell_volume = grid.mesh.spacing[0] ** 3
        self.occluded_volume = round(self.occluded_indices.size * cell_volume, 2)
        self.covered_volume = round(self.covered_indices.size * cell_volume, 2)
        self.fraction_occluded = round(
            100 * self.occluded_volume / (self.covered_volume + self.occluded_volume), 1
        )

        if all_metrics:
            indexes = np.arange(18)

        # get the area indices to calculate the sensors performance in every area
        area_indices = None
        for index in indexes:
            data = self.__single_sensor_data(grid)
            match index:
                case 0:
                    area_indices = grid.far_front_left_indices
                case 1:
                    area_indices = grid.far_front_center_indices
                case 2:
                    area_indices = grid.far_front_right_indices
                case 3:
                    area_indices = grid.near_front_left_indices
                case 4:
                    area_indices = grid.near_front_center_indices
                case 5:
                    area_indices = grid.near_front_right_indices
                case 6:
                    area_indices = grid.far_left_indices
                case 7:
                    area_indices = grid.near_left_indices
                case 8:
                    area_indices = grid.near_right_indices
                case 9:
                    area_indices = grid.far_right_indices
                case 10:
                    area_indices = grid.near_rear_left_indices
                case 11:
                    area_indices = grid.near_rear_center_indices
                case 12:
                    area_indices = grid.near_rear_right_indices
                case 13:
                    area_indices = grid.far_rear_left_indices
                case 14:
                    area_indices = grid.far_rear_center_indices
                case 15:
                    area_indices = grid.far_rear_right_indices
                case 16:
                    area_indices = grid.car_area_indices
                case 17:
                    area_indices = grid.outside_indices

            # divide the covered points in an area by the number of points in an area
            data = data[area_indices]
            indices = np.nonzero(data == 1)[0]
            self.metrics[index] = round((indices.size / area_indices.size) * 100, 1)

    # helper function that is used to bring the number of rows in the sensor calculation_result to the number of total
    # points of the grid. this way, the area indices can be used correctly on the calculation result    
    def __single_sensor_data(self, grid):
        data = np.zeros(grid.points.shape[0])
        # Put 2 on car-area points – skip if no car points exist
        if grid.car_points_indices is not None:
            np.put(data, grid.car_points_indices, 2)
        np.put(data, grid.outside_indices, self.calculation_result) # type: ignore[arg-type]
        return data
    