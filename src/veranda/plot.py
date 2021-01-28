# import os
# import numpy as np
# from matplotlib.widgets import Button, AxesWidget
# from PIL import Image
#
# def split_axis_bounds(bounds, proportions):
#     if sum(proportions) != 1:
#         raise Exception("")
#
#     min_x, min_y, width, height = bounds
#
#     widths = [width*proportion for proportion in proportions]
#     bounds_split = [(min_x, min_y, widths[0], height)]
#     for i in range(1, len(widths)):
#         prev_bound = bounds_split[i-1]
#         bounds_split.append((prev_bound[0] + prev_bound[2], min_y, widths[i], height))
#
#     return bounds_split
#
#
# # TODO add label format
# class RasterStackSlider(AxesWidget):
#     """
#     A slider representing a floating point range.
#
#     Create a slider from *valmin* to *valmax* in axes *ax*. For the slider to
#     remain responsive you must maintain a reference to it. Call
#     :meth:`on_changed` to connect to the slider event.
#
#     Attributes
#     ----------
#     val : float
#         Slider value.
#     """
#
#     def __init__(self, raster_stack, ax, valmin=None, valmax=None, valstep=1, valinit=None, label_fmt=None, **kwargs):
#         """
#         Parameters
#         ----------
#         ax : Axes
#             The Axes to put the slider in.
#
#         label : str
#             Slider label.
#
#         valmin : float
#             The minimum value of the slider.
#
#         valmax : float
#             The maximum value of the slider.
#
#         valinit : float, optional, default: 0.5
#             The slider initial position.
#
#         valfmt : str, optional, default: "%1.2f"
#             Used to format the slider value, fprint format string.
#
#         closedmin : bool, optional, default: True
#             Indicate whether the slider interval is closed on the bottom.
#
#         closedmax : bool, optional, default: True
#             Indicate whether the slider interval is closed on the top.
#
#         slidermin : Slider, optional, default: None
#             Do not allow the current slider to have a value less than
#             the value of the Slider `slidermin`.
#
#         slidermax : Slider, optional, default: None
#             Do not allow the current slider to have a value greater than
#             the value of the Slider `slidermax`.
#
#         dragging : bool, optional, default: True
#             If True the slider can be dragged by the mouse.
#
#         valstep : float, optional, default: None
#             If given, the slider will snap to multiples of `valstep`.
#
#         orientation : str, 'horizontal' or 'vertical', default: 'horizontal'
#             The orientation of the slider.
#
#         Notes
#         -----
#         Additional kwargs are passed on to ``self.poly`` which is the
#         :class:`~matplotlib.patches.Rectangle` that draws the slider
#         knob.  See the :class:`~matplotlib.patches.Rectangle` documentation for
#         valid property names (e.g., `facecolor`, `edgecolor`, `alpha`).
#         """
#         AxesWidget.__init__(self, ax)
#
#         self.raster_stack = raster_stack
#         self.valmin = valmin if valmin is not None else 0
#         self.valmax = valmax if valmax is not None else len(raster_stack)-1
#         self.valstep = valstep
#         self.drag_active = False
#
#         valinit = self._value_in_bounds(valinit)
#         if valinit is None:
#             valinit = self.valmin
#         self.val = valinit
#         self.valinit = valinit
#
#         self.connect_event('button_press_event', self._update)
#         self.connect_event('button_release_event', self._update)
#         self.connect_event('motion_notify_event', self._update)
#         valmin_label = str(raster_stack.inventory.index[self.valmin])
#         valmax_label = str(raster_stack.inventory.index[self.valmax])
#
#         bounds = split_axis_bounds(ax.get_position().bounds, [0.1, 0.8, 0.1])
#         self.slider_ax = ax.get_figure().add_axes(bounds[1])
#         ax.set_yticks([])
#         ax.set_xticks([])
#         ax.set_navigate(False)
#         ax.axis('off')
#         self.slider_ax.set_yticks([])
#         self.slider_ax.set_xlim((self.valmin, self.valmax))
#         self.slider_ax.set_xticks([])
#         self.slider_ax.set_navigate(False)
#
#         self.ax.text(-0.02, 0.5, valmin_label, transform=ax.transAxes,
#                      verticalalignment='center', horizontalalignment='right')
#         self.ax.text(1.02, 0.5, valmax_label, transform=ax.transAxes,
#                      verticalalignment='center', horizontalalignment='left')
#
#         for val in np.arange(self.valmin + self.valstep, self.valmax, self.valstep):
#             self.slider_ax.axvline(val, 0, 1, color='r', lw=1)
#
#         self.poly = self.slider_ax.axvspan(self.valmin, valinit, 0, 1, **kwargs)
#
#         left_ax = ax.get_figure().add_axes(bounds[0])
#         right_ax = ax.get_figure().add_axes(bounds[2])
#         left_ax.axis('off')
#         right_ax.axis('off')
#         go_left_img = Image.open(os.path.join(os.path.dirname(__file__), "data", "go_left.png"))
#         go_right_img = Image.open(os.path.join(os.path.dirname(__file__), "data", "go_right.png"))
#         self.left_button = Button(left_ax, "", go_left_img)
#         self.right_button = Button(right_ax, "", go_right_img)
#         def left_button_update(event):
#             val = self.val - self.valstep
#             val = self._value_in_bounds(val)
#             self.set_val(val)
#         def right_button_update(event):
#             val = self.val + self.valstep
#             val = self._value_in_bounds(val)
#             self.set_val(val)
#         self.left_button.on_clicked(left_button_update)
#         self.right_button.on_clicked(right_button_update)
#
#
#
#         self.cnt = 0
#         self.observers = {}
#
#         self.set_val(valinit)
#
#     def _value_in_bounds(self, val):
#         """Makes sure *val* is with given bounds."""
#         if self.valstep:
#             val = np.round((val - self.valmin) / self.valstep) * self.valstep
#             val += self.valmin
#
#         if val <= self.valmin:
#             val = self.valmin
#         elif val >= self.valmax:
#             val = self.valmax
#
#         return val
#
#     def _update(self, event):
#         """Update the slider position."""
#
#         if self.ignore(event) or event.button != 1:
#             return
#
#         if event.name == 'button_press_event' and event.inaxes == self.slider_ax:
#             self.drag_active = True
#             event.canvas.grab_mouse(self.slider_ax)
#
#         if not self.drag_active:
#             return
#         elif ((event.name == 'button_release_event') or
#               (event.name == 'button_press_event' and
#                event.inaxes != self.slider_ax)):
#             self.drag_active = False
#             event.canvas.release_mouse(self.slider_ax)
#             return
#
#         val = self._value_in_bounds(event.xdata)
#         if val not in [None, self.val]:
#             self.set_val(val)
#
#     def set_val(self, val):
#         """
#         Set slider value to *val*
#
#         Parameters
#         ----------
#         val : float
#         """
#
#         xy = self.poly.xy
#         xy[2] = val, 1
#         xy[3] = val, 0
#         self.poly.xy = xy
#
#         self.slider_ax.set_xlabel(self.raster_stack.inventory.index[int(val)])
#
#         if self.drawon:
#             self.slider_ax.figure.canvas.draw_idle()
#
#         self.val = val
#         if not self.eventson:
#             return
#         for cid, func in self.observers.items():
#             func(val)
#
#     def on_changed(self, func):
#         """
#         When the slider value is changed call *func* with the new
#         slider value
#
#         Parameters
#         ----------
#         func : callable
#             Function to call when slider is changed.
#             The function must accept a single float as its arguments.
#
#         Returns
#         -------
#         cid : int
#             Connection id (which can be used to disconnect *func*)
#         """
#         cid = self.cnt
#         self.observers[cid] = func
#         self.cnt += 1
#         return cid
#
#     def disconnect(self, cid):
#         """
#         Remove the observer with connection id *cid*
#
#         Parameters
#         ----------
#         cid : int
#             Connection id of the observer to be removed
#         """
#         try:
#             del self.observers[cid]
#         except KeyError:
#             pass
#
#     def reset(self):
#         """Reset the slider to the initial value"""
#         if self.val != self.valinit:
#             self.set_val(self.valinit)