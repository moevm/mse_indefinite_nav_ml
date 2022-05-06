import itertools
from ctypes import POINTER

import numpy as np
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.graphics import bezier_draw
from gym_duckietown.simulator import get_dir_vec, CAMERA_FORWARD_DIST, get_agent_corners, draw_axes
from pyglet import gl

from gym_custom.env import logger


class Simulator2(DuckietownEnv):
    def _render_img(
        self,
        width: int,
        height: int,
        multi_fbo,
        final_fbo,
        img_array,
        top_down: bool = True,
        segment: bool = False,
    ) -> np.ndarray:
        """
        Render an image of the environment into a frame buffer
        Produce a numpy RGB array image as output
        """

        if not self.graphics:
            return np.zeros((height, width, 3), np.uint8)

        # Switch to the default context
        # This is necessary on Linux nvidia drivers
        # pyglet.gl._shadow_window.switch_to()
        self.shadow_window.switch_to()

        if segment:
            gl.glDisable(gl.GL_LIGHT0)
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_COLOR_MATERIAL)
        else:
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_COLOR_MATERIAL)

        # note by default the ambient light is 0.2,0.2,0.2
        # ambient = [0.03, 0.03, 0.03, 1.0]
        ambient = [0.3, 0.3, 0.3, 1.0]

        gl.glEnable(gl.GL_POLYGON_SMOOTH)

        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, (gl.GLfloat * 4)(*ambient))
        # Bind the multisampled frame buffer
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)
        gl.glViewport(0, 0, width, height)

        # Clear the color and depth buffers

        c0, c1, c2 = self.horizon_color if not segment else [255, 0, 255]
        gl.glClearColor(c0, c1, c2, 1.0)
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(self.cam_fov_y, width / float(height), 0.04, 100.0)

        # Set modelview matrix
        # Note: we add a bit of noise to the camera position for data augmentation
        pos = self.cur_pos
        angle = self.cur_angle
        # logger.info('Pos: %s angle %s' % (self.cur_pos, self.cur_angle))
        if self.domain_rand:
            pos = pos + self.randomization_settings["camera_noise"]

        x, y, z = pos + self.cam_offset
        dx, dy, dz = get_dir_vec(angle)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        if self.draw_bbox:
            y += 0.8
            gl.glRotatef(90, 1, 0, 0)
        elif not top_down:
            y += self.cam_height
            gl.glRotatef(self.cam_angle[0], 1, 0, 0)
            gl.glRotatef(self.cam_angle[1], 0, 1, 0)
            gl.glRotatef(self.cam_angle[2], 0, 0, 1)
            gl.glTranslatef(0, 0, CAMERA_FORWARD_DIST)

        if top_down:
            a = (self.grid_width * self.road_tile_size) / 2
            b = (self.grid_height * self.road_tile_size) / 2
            fov_y_deg = self.cam_fov_y
            fov_y_rad = np.deg2rad(fov_y_deg)
            H_to_fit = max(a, b) + 0.1  # borders

            H_FROM_FLOOR = H_to_fit / (np.tan(fov_y_rad / 2))

            look_from = a, H_FROM_FLOOR, b
            look_at = a, 0.0, b - 0.01
            up_vector = 0.0, 1.0, 0
            gl.gluLookAt(*look_from, *look_at, *up_vector)
        else:
            look_from = x, y, z
            look_at = x + dx, y + dy, z + dz
            up_vector = 0.0, 1.0, 0.0
            gl.gluLookAt(*look_from, *look_at, *up_vector)

        # Draw the ground quad
        gl.glDisable(gl.GL_TEXTURE_2D)
        # background is magenta when segmenting for easy isolation of main map image
        gl.glColor3f(*self.ground_color if not segment else [255, 0, 255])  # XXX
        gl.glPushMatrix()
        gl.glScalef(50, 0.01, 50)
        self.ground_vlist.draw(gl.GL_QUADS)
        gl.glPopMatrix()

        # Draw the ground/noise triangles
        if not segment:
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.1, 0.0)
            self.tri_vlist.draw(gl.GL_TRIANGLES)
            gl.glPopMatrix()

        # Draw the road quads
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        add_lights = False
        if add_lights:
            for i in range(1):
                li = gl.GL_LIGHT0 + 1 + i
                # li_pos = [i + 1, 1, i + 1, 1]

                li_pos = [0.0, 0.2, 3.0, 1.0]
                diffuse = [0.0, 0.0, 1.0, 1.0] if i % 2 == 0 else [1.0, 0.0, 0.0, 1.0]
                ambient = [0.0, 0.0, 0.0, 1.0]
                specular = [0.0, 0.0, 0.0, 1.0]
                spot_direction = [0.0, -1.0, 0.0]
                logger.debug(
                    li=li, li_pos=li_pos, ambient=ambient, diffuse=diffuse, spot_direction=spot_direction
                )
                gl.glLightfv(li, gl.GL_POSITION, (gl.GLfloat * 4)(*li_pos))
                gl.glLightfv(li, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
                gl.glLightfv(li, gl.GL_DIFFUSE, (gl.GLfloat * 4)(*diffuse))
                gl.glLightfv(li, gl.GL_SPECULAR, (gl.GLfloat * 4)(*specular))
                gl.glLightfv(li, gl.GL_SPOT_DIRECTION, (gl.GLfloat * 3)(*spot_direction))
                # gl.glLightfv(li, gl.GL_SPOT_EXPONENT, (gl.GLfloat * 1)(64.0))
                gl.glLightf(li, gl.GL_SPOT_CUTOFF, 60)

                gl.glLightfv(li, gl.GL_CONSTANT_ATTENUATION, (gl.GLfloat * 1)(1.0))
                # gl.glLightfv(li, gl.GL_LINEAR_ATTENUATION, (gl.GLfloat * 1)(0.1))
                gl.glLightfv(li, gl.GL_QUADRATIC_ATTENUATION, (gl.GLfloat * 1)(0.2))
                gl.glEnable(li)

        cx, cy = self.get_grid_coords(pos)
        # For each grid tile
        for i, j in itertools.product(range(self.grid_width), range(self.grid_height)):

            # Get the tile type and angle
            tile = self._get_tile(i, j)

            if tile is None:
                continue

            # kind = tile['kind']
            angle = tile["angle"]
            color = tile["color"]
            texture = tile["texture"]

            # logger.info('drawing', tile_color=color)

            gl.glColor4f(*color)

            gl.glPushMatrix()
            TS = self.road_tile_size
            gl.glTranslatef((i + 0.5) * TS, 0, (j + 0.5) * TS)
            gl.glRotatef(angle * 90 + 180, 0, 1, 0)

            # gl.glEnable(gl.GL_BLEND)
            # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            # Bind the appropriate texture
            texture.bind(segment)

            self.road_vlist.draw(gl.GL_QUADS)
            # gl.glDisable(gl.GL_BLEND)

            gl.glPopMatrix()

            if self.draw_curve and tile["drivable"] and i == cx and j == cy:
                # Find curve with largest dotproduct with heading
                curves = tile["curves"]
                pts = self._get_curve_points(curves, angle)
                bezier_draw(pts, n=20, red=True)

        # For each object
        for obj in self.objects:
            obj.render(draw_bbox=self.draw_bbox, segment=segment, enable_leds=self.enable_leds)

        # Draw the agent's own bounding box
        if self.draw_bbox:
            corners = get_agent_corners(pos, angle)
            gl.glColor3f(1, 0, 0)
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f(corners[0, 0], 0.01, corners[0, 1])
            gl.glVertex3f(corners[1, 0], 0.01, corners[1, 1])
            gl.glVertex3f(corners[2, 0], 0.01, corners[2, 1])
            gl.glVertex3f(corners[3, 0], 0.01, corners[3, 1])
            gl.glEnd()

        if top_down:
            gl.glPushMatrix()
            gl.glTranslatef(*self.cur_pos)
            gl.glScalef(1, 1, 1)
            gl.glRotatef(self.cur_angle * 180 / np.pi, 0, 1, 0)
            # glColor3f(*self.color)
            self.mesh.render()
            gl.glPopMatrix()
        draw_xyz_axes = False
        if draw_xyz_axes:
            draw_axes()
        # Resolve the multisampled frame buffer into the final frame buffer
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, multi_fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, final_fbo)
        gl.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)
        gl.glReadPixels(
            0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_array.ctypes.data_as(POINTER(gl.GLubyte))
        )

        # Unbind the frame buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Flip the image because OpenGL maps (0,0) to the lower-left corner
        # Note: this is necessary for gym.wrappers.Monitor to record videos
        # properly, otherwise they are vertically inverted.
        img_array = np.ascontiguousarray(np.flip(img_array, axis=0))

        return img_array

    def _get_curve_points(self, curves, angle):
        curve_headings = curves[:, -1, :] - curves[:, 0, :]
        curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
        dirVec = get_dir_vec(angle)
        dot_prods = np.dot(curve_headings, dirVec)

        # Current ("closest") curve drawn in Red
        return curves[np.argmax(dot_prods)]