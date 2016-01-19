# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import matplotlib.pyplot as plt

from hyperspy.drawing.widgets import Widget2DBase, ResizersMixin


class Draggable2DCircle(Widget2DBase, ResizersMixin):

    """Draggable2DCircle is a symmetric, Cicle-patch based widget, which can
    be dragged, and resized by keystrokes/code.
    """

    def __init__(self, axes_manager, **kwargs):
        super(Draggable2DCircle, self).__init__(axes_manager, **kwargs)
        self.size_step = 0.5

    def _do_snap_size(self, value=None):
        value = np.array(value) if value is not None else self._size
        snap_spacing = self.axes[0].scale * self.size_step
        for i in xrange(2):
            value[i] = round(value[i] / snap_spacing) * snap_spacing
        return value

    def _set_size(self, value):
        """Setter for the 'size' property. Calls _size_changed to handle size
        change, if the value has changed.
        """
        # Override so that r_inner can be 0
        value = np.minimum(value, [ax.size for ax in self.axes])
        # Changed from base:
        min_sizes = np.array((self.axes[0].scale, 0))
        value = np.maximum(value, (self.size_step * min_sizes))
        if self.snap_size:
            value = self._do_snap_size(value)
        if np.any(self._size != value):
            self._size = value
            self._size_changed()

    def increase_size(self):
        """Increment all sizes by one step. Applied via 'size' property.
        """
        s = np.array(self.size)
        if self.size[1] > 0:
            s += self.size_step * self.axes[0].scale
        else:
            s[0] += self.size_step * self.axes[0].scale
        self.size = s

    def decrease_size(self):
        """Decrement all sizes by one step. Applied via 'size' property.
        """
        s = np.array(self.size)
        if self.size[1] > 0:
            s -= self.size_step * self.axes[0].scale
        else:
            s[0] -= self.size_step * self.axes[0].scale
        self.size = s

    def get_centre(self):
        return self.position

    def _get_patch_xy(self):
        """Returns the xy coordinates of the patch. In this implementation, the
        patch is centered on the position.
        """
        return self.position

    def _set_patch(self):
        """Sets the patch to a matplotlib Circle with the correct geometry.
        The geometry is defined by _get_patch_xy, and size.
        """
        super(Draggable2DCircle, self)._set_patch()
        xy = self._get_patch_xy()
        ro, ri = self.size
        self.patch = [plt.Circle(
            xy, radius=ro,
            animated=self.blit,
            fill=False,
            lw=self.border_thickness,
            ec=self.color,
            picker=True,)]

    def _validate_pos(self, value):
        """Constrict the position within bounds.
        """
        value = (min(value[0], self.axes[0].high_value - self._size[0] +
                     0.5 * self.axes[0].scale),
                 min(value[1], self.axes[1].high_value - self._size[0] +
                     0.5 * self.axes[1].scale))
        value = (max(value[0], self.axes[0].low_value + self._size[0] -
                     0.5 * self.axes[0].scale),
                 max(value[1], self.axes[1].low_value + self._size[0] -
                     0.5 * self.axes[1].scale))
        return super(Draggable2DCircle, self)._validate_pos(value)

    def get_size_in_indices(self):
        return np.array(self._size / self.axes[0].scale)

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch[0].center = self._get_patch_xy()
            self._update_resizers()
            self.draw_patch()

    def _update_patch_size(self):
        if self.is_on() and self.patch is not None:
            ro, ri = self.size
            self.patch[0].radius = ro
            self._update_resizers()
            self.draw_patch()

    def _update_patch_geometry(self):
        if self.is_on() and self.patch is not None:
            ro, ri = self.size
            self.patch[0].center = self._get_patch_xy()
            self.patch[0].radius = ro
            self._update_resizers()
            self.draw_patch()

    def _onmousemove(self, event):
        'on mouse motion move the patch if picked'
        if self.picked is True and event.inaxes:
            x = event.xdata
            y = event.ydata
            if self.resizer_picked is False:
                x -= self.pick_offset[0]
                y -= self.pick_offset[1]
                self.position = (x, y)
            else:
                rad_vect = np.array((x, y)) - self.position
                radius = np.sqrt(np.sum(rad_vect**2))
                s = list(self.size)
                s[0] = radius
                self.size = s

    def _get_resizer_pos(self):
        r = self._size[0]
        rsize = self._get_resizer_size() / 2

        positions = []
        rp = np.array(self._get_patch_xy())
        p = rp - (r, 0) - rsize             # Left
        positions.append(p)
        p = rp - (0, r) - rsize             # Top
        positions.append(p)
        p = rp + (r, 0) - rsize             # Right
        positions.append(p)
        p = rp + (0, r) - rsize             # Bottom
        positions.append(p)
        return positions