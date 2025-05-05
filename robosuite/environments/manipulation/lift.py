from collections import OrderedDict

import numpy as np
import time

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class Lift(ManipulationEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True, #TODO:
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        ##TODO:
        self.cube_positions = [
            np.array([0.0, 0.2, 0.82]),
            np.array([-0.2, 0, 0.82]),
            np.array([0, -0.2, 0.82]),
            np.array([0.2, 0.0, 0.82])
        ]
        self.cube_quat = np.array([1, 0, 0, 0])
        self.current_cube_index = 0
        self.cube_move_interval = 100  # how many steps before auto-move
        self._steps_since_cube_move = 0
        ##TODO:

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
    
    # def reset_in_increments(self):
    #     for i in range(10000):
    #         start = time.time()

    #         if i % 20:
    #             object_placements = self.placement_initializer.sample()

    #             for obj_pos, obj_quat, obj in object_placements.values():
    #                 self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    #         elapsed = time.time() - start
    #         diff = 1 / 25 - elapsed
    #         if diff > 0:
    #             time.sleep(diff)
    
    def reward(self, action=None):
        """
        reward function

        using a distance-based reward for reaching the cube
        (the closer the gripper is to the cube, the higher the reward)
        
        a small positive reward is given for staying close, and a larger
        reward is given upon successful lift

        Args:
            action (np array): [NOT USED]
        Returns:
            float: reward value
        """
        reward = 0.0
        gripper = self.robots[0].gripper
        cube_body = self.cube.root_body
        self._steps_since_cube_move += 1
        # start = time.time()

        #distance between the gripper and the cube
        dist = self._gripper_to_target(gripper=gripper, target=cube_body, target_type="body", return_distance=True)

        #defining a maximum distance for consideration(like further away gives minimal reward)
        max_reach_distance = 1  #TODO: adjust later??

        #calculate a reaching reward based on the distance
        if dist < max_reach_distance:
            # reaching_reward = ((dist + 0.5) ** (-2))
            # reaching_reward = 2 - np.exp(1.0 * dist)
            reaching_reward = (1 - np.tanh(3.0 * dist))
                # reaching_reward = 1 - np.tanh(15.0 * dist)
            # if (dist < 0.13699):
                # if (dist > 0.25):
                #     reaching_reward = 9 * ((0.5 - dist) ** 2)
                # else:
                #     reaching_reward = (-0.01125 * ((0.25 * dist) ** (-2))) + 5.5
            # reaching_reward = 25 - (2 ** ((dist) + 4.4))
            # reaching_reward = (1 - (-0.00125 * ((dist + 0.1) ** (-4)))) - 1
                #reaching_reward = 10 - np.exp(2 * dist)
                # print("Dist: ", dist)
                # print("Reward", reaching_reward)
            #if ()
            #reaching_reward = (max_reach_distance - dist) / max_reach_distance
            reward += reaching_reward  #scale the reaching reward
            #print("Reward prior to success", reward)

        #check if the cube has been lifted
        if self._check_success(): #or (self._steps_since_cube_move % self.cube_move_interval == 0):
            #print("Steps: ", self._steps_since_cube_move)
            # if self._check_success():
            #     reward += 1.0 * (1 - self._steps_since_cube_move / 150)
            #     reward += 2.5  # Larger reward for touching
                #self.cube_move_interval = self._steps_since_cube_move
                #TODO:
                #compare isntead with the previous time it tried to contact the cube at that same position
                    #use a global varibale for an array, index using the same cube index as the positions,
                        #compare self._steps_since_cube_move with that earlier steps, see if imporvement
                            #reward accordingly.

                #print("Reward: ", reward)

            reward += 5
            # self._reset_internal()
            super()._reset_internal()
            
            # print("reward: ", reward)
            # object_placements = self.placement_initializer.sample()
            # for obj_pos, obj_quat, obj in object_placements.values():
            #     self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

            # else:
            #     reward -= 0.1
            
            self._steps_since_cube_move = 0
            self.current_cube_index = (self.current_cube_index + 1) % len(self.cube_positions)
            new_pos = self.cube_positions[self.current_cube_index]
            self.sim.data.set_joint_qpos(
                self.cube.joints[0],
                np.concatenate([new_pos, self.cube_quat])
            )

        # print("Reward: ", reward)
        #scale reward if requested -- idk what this does but it was there before lol
        if self.reward_scale is not None:
            reward *= self.reward_scale

        #for i in range(10000):
            #start = time.time()

            #if i % 20:
                # object_placements = self.placement_initializer.sample()

                # for obj_pos, obj_quat, obj in object_placements.values():
                #     self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

            # elapsed = time.time() - start
            # diff = 1 / 25 - elapsed
            # if elapsed % 100 == 0:
            #     object_placements = self.placement_initializer.sample()
            #     for obj_pos, obj_quat, obj in object_placements.values():
            #         self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
    
            # if diff > 0:
            #     time.sleep(diff)

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.035],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.035],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
            obj_type="all",
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
            # self.placement_initializer.add_objects(self.cube0)
            # self.placement_initializer.add_objects(self.cube1)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.25, 0.25],
                y_range=[-0.25, 0.25],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube], #TODO: edited cube #
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            sensors = [cube_pos, cube_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to cube position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # # Reset all object positions using initializer sampler if we're not directly loading from an xml
        # if not self.deterministic_reset:

        #     # Sample from the placement initializer for all objects
        #     object_placements = self.placement_initializer.sample()

        #     # Loop through all objects and reset their positions
        #     for obj_pos, obj_quat, obj in object_placements.values():
        #         self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        self.current_cube_index = 0
        initial_pos = self.cube_positions[self.current_cube_index]
        self.sim.data.set_joint_qpos(
            self.cube.joints[0],
            np.concatenate([initial_pos, self.cube_quat])
        )

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        # cube_height = self.sim.data.body_xpos[self.cube_body_id][2] #TODO: this is for lifting!
        # table_height = self.model.mujoco_arena.table_offset[2]
        gripper = self.robots[0].gripper['right']
        #print("Gripper contents:", gripper) #TODO:
                # print("Gripper contents:", gripper)
                # print("Gripper attributes:", dir(gripper))
                # gripper_closed = all(gripper["joints"][i] < gripper["joint_limits"][i][1] - 0.01 for i in range(len(gripper["joints"])))

        #print("working??")
        gripper_to_cube_dist = self._gripper_to_target(gripper=gripper, target=self.cube.root_body, target_type="body", return_distance=True)
        in_contact = gripper_to_cube_dist < 0.075  #TODO: can adjust this threshold

        # cube is higher than the table top above a margin
        # return cube_height > table_height + 0.04 #TODO: for lifting!
        #print(in_contact)
        return in_contact #and gripper_closed #TODO: returning true?
    def set_target_position(self, target):
        pass