import logging
import pickle
import time
import pyvista as pv

from args import args
from environment.grid import Grid
from environment.slice import Slice
from plotting.report import create_report
from plotting.plots import create_plots
from plotting.plot_helpers import metrics, setup_plot_args, output_folder
from sensors.sensor_helpers import load_sensorset
from utils.gui import GUI

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(divide='ignore', invalid='ignore') 
np.seterr(over='ignore', invalid='ignore') 

from embree_bypass import *   # applies the VTK fallback

# PROGRAM OPTIONS
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


# === EMBREE BYPASS: VTK-ONLY MULTI RAY TRACE (ARM64 NATIVE) ===
if not getattr(pv, "embree_available", False):
    def _f(self, origins, direction_vectors, first_point=True):
        import numpy as np

        origins = np.asarray(origins, dtype=float)
        direction_vectors = np.asarray(direction_vectors, dtype=float)

        # Sanity check: require Nx3 arrays
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError(f"origins must be (N,3), got {origins.shape}")
        if direction_vectors.ndim != 2 or direction_vectors.shape[1] != 3:
            raise ValueError(f"direction_vectors must be (N,3), got {direction_vectors.shape}")

        # Normalize directions
        norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_dirs = direction_vectors / norms

        # Ray length = mesh diagonal
        far_points = origins + unit_dirs * self.length

        # Preallocate outputs: one point + one index per ray
        n_rays = origins.shape[0]
        hit_points = np.empty((n_rays, 3), dtype=float)
        hit_indices = np.empty((n_rays,), dtype=int)

        for i, (origin, end) in enumerate(zip(origins, far_points)):
            try:
                point, ind = self.ray_trace(origin, end, first_point=first_point)

                # Convert to arrays for consistent handling
                point = np.asarray(point, dtype=float)
                ind = np.asarray(ind, dtype=int)

                # No hits → mark as NaN / -1
                if point.size < 3 or ind.size < 1:
                    hit_points[i, :] = np.nan
                    hit_indices[i] = -1
                    continue

                # Reshape "point" to (M,3) and take the first hit
                point = point.reshape(-1, 3)
                first_point = point[0]
                first_ind = int(ind.ravel()[0])

                hit_points[i, :] = first_point
                hit_indices[i] = first_ind
            except Exception:
                # If ray_trace itself blows up, treat as no hit
                hit_points[i, :] = np.nan
                hit_indices[i] = -1

        # Return same structure as Embree multi_ray_trace: (points, ray_ids, cells)
        return hit_points, hit_indices, []  # cells not used

    pv.PolyData.multi_ray_trace = _f  # type: ignore
    print("VTK fallback active (arm64 native)")
# === END PATCH ===


def run(args):
    """Main function for the program.

    Args:
        args: input arguments provided by YAML-config file or command line.
    """

    logging.info("Starting Programm")
    sensors = load_sensorset(args.sensor_setup)

    if args.gui_mode:
        gui_instance = GUI()
        gui_instance.run()
        args.update(gui_instance.get_inputs())
    logging.info("Inputs evaluated -> now loading vehicle")

    vehicle = pv.read(args.vehicle_path).triangulate()
    logging.info("Vehicle loaded -> creating grid")

    grid = Grid(
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        dim_z=args.dim_z,
        spacing=args.spacing,
        advanced=args.advanced,
        car=vehicle,
        center=args.origin,
        dist=args.nearfield_dist,
    )
    logging.info("Grid created -> starting single sensor coverage calculation")

    ix = 1
    max_ix = len(sensors)
    for sensor in sensors:
        logging.info(f"Calculating Single Sensor {ix} of {max_ix}")
        sensor.calculate_coverage(grid, vehicle)
        ix += 1
    logging.info("Finished single sensor calculation -> calculating grid coverage")

    grid.combine_data(sensors)
    grid.set_metrics_no_condition()
    grid.set_metrics_condition(
        n1=args.conditions.N1,
        n2=args.conditions.N2,
        n6=args.conditions.N6,
        n7=args.conditions.N7,
        n8=args.conditions.N8,
    )
    logging.info("Grid coverage calculated -> preparing report and plots")

    slices = [
        Slice(grid, 1.5, normal="x"),
        Slice(grid, 0, normal="y"),
        Slice(grid, 0.01),
    ]
    slices.extend(
        [
            Slice(grid, i * args.slice.distance)
            for i in range(1, args.slice.number, 1)
        ]
    )

    plot_args = setup_plot_args(
        metrics["n_sensor_technologies"], car_value=grid.car_value
    )

    if args.create_report:
        logging.info("Creating report")
        create_report(
            sensors,
            slices,
            vehicle,
            grid,
            args.save_path,
            args.folder_name,
            plot_args,
            n1=args.conditions.N1,
            n2=args.conditions.N2,
            n6=args.conditions.N6,
            n7=args.conditions.N7,
            n8=args.conditions.N8,
        )

    if not args.no_plots:
        logging.info("creating plots")
        create_plots(
            grid,
            sensors,
            vehicle,
            args.save_path,
            args.folder_name,
            slices[0],
            slices[1],
            slices[2],
        )
        logging.info("Report created -> finished")

    if args.save_variables:
        save_path = output_folder(args.save_path, args.folder_name) / "save_data.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({"grid": grid, "vehicle": vehicle, "sensors": sensors,
                         "args": args}, f)


if __name__ == "__main__":
    start_time = time.time()
    logging.info("Starting main")
    run(args)
    logging.info("Success")
    logging.info(f"Results saved: {output_folder(args.save_path, args.folder_name)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time is {elapsed_time}")
