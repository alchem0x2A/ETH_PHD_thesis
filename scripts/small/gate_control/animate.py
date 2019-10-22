import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import cv2
from scipy.optimize import fsolve


file_video = "./IMG_5583.MOV"
file_IV = "./IV_cycles.csv"
file_MCA = "./contact.bmp"
file_bare_IV = "./bare_IV_control.csv"

IV_data = numpy.genfromtxt(file_IV, delimiter=",",
                           skip_header=13)
I_d = IV_data[:, 1]
I_g = IV_data[:, 3]  / 5 / 10 ** -9
cycle = IV_data[:, -2]
V_g = IV_data[:, 0]
t_IV = IV_data[:, 4]

IV_bare = numpy.genfromtxt(file_bare_IV,
                           delimiter=",",
                           skip_header=13,
                           skip_footer=34)
V_G_i = IV_bare[:, 0]
I_t_i = IV_bare[:, 1]
func_IV = interp1d(V_G_i, I_t_i, kind="linear")

VCC = 5
Rs = 12e6
R2 = 2e6

def func_R(R, It):
    V_G = VCC / (R + Rs + R2) * R2
    I_L = VCC / (R + Rs + R2)
    I_R = func_IV(V_G)
    return It - I_L - I_R

def solve_R(It):
    return fsolve(func_R, x0=1e6, args=(It))
solve_R_vec = numpy.vectorize(solve_R)

video_capt = cv2.VideoCapture(file_video)
fps = video_capt.get(cv2.CAP_PROP_FPS)
max_frames = video_capt.get(cv2.CAP_PROP_FRAME_COUNT)
print(max_frames)

crop_box = ((198, 258),
            (925, 1495))

# the box for the LED
focus_box = ((567, 1108),
             (592, 1135))


fig = plt.figure(figsize=(9, 2.5))
ax1 = fig.add_subplot(131)
ax1.set_axis_off()
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.text(x=0, y=0.93, s="(a)",
         transform=fig.transFigure)
ax2.text(x=0.33, y=0.93, s="(b)",
         transform=fig.transFigure)
ax3.text(x=0.68, y=0.93, s="(c)",
         transform=fig.transFigure)

# fig = plt.figure(figsize=(5, 4))
plt.style.use("science")
# ax1 = fig.add_subplot(111)

# t = []
# avg = []
# count = 0

# ret = True
# while ret is True:
#     ret, frame = video_capt.read()
#     if ret is True:
#         t.append(count / fps)
#         count += 1
#         color_mean = numpy.mean(numpy.mean(frame[focus_box[0][0] : focus_box[1][0],
#                                                  focus_box[0][1]: focus_box[1][1]], axis=0),
#                                 axis=0)
#         avg.append(color_mean[1])

# video_capt.release()

# t = numpy.array(t)
# avg = numpy.array(avg)

# ax1.plot(t, avg / 255)
# ax1.plot(t_IV, IV_data[:, 1] / max(IV_data[:, 1]))
# plt.show()


offset = 56.5                  # seconds ahead in IV measurement

def cut_frame(frame):
    return frame[crop_box[0][0]: crop_box[1][0],
                 crop_box[0][1]: crop_box[1][1]]

current_frame = 0
video_capt.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # start on zero frame
ret, frame = video_capt.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
im_video = ax1.imshow(cut_frame(frame_rgb), interpolation="bicubic")
im_video.set_zorder(1)
t_IV = t_IV - offset

func_Id_t = interp1d(t_IV, I_d, "linear")
func_Ig_t = interp1d(t_IV, I_g, "linear")
func_Vg_t = interp1d(t_IV, V_g, "linear")
func_cycle = interp1d(t_IV, cycle, "linear")

t_list = numpy.array([0])
Id_list = func_Id_t(t_list)
Ig_list = func_Ig_t(t_list)
Vg_list = func_Vg_t(t_list)
current_cycle = func_cycle(0)
t_text = ax1.set_title("{:.2f} s, Cycle {:d}".format(0, int(current_cycle)))

line_1_old, = ax2.plot([], [], color="#b2b2b2", alpha=0.6)
line_1,  = ax2.plot(Vg_list, Id_list)
line_1_dot, = ax2.plot(Vg_list[-1], Id_list[-1], "o")
ax2.set_xlim(-50, 100)
ax2.set_xlabel("$V_{\\mathrm{G}}$ (V)")
ax2.set_ylabel("$I_{\\mathrm{tot}}$ (A)")
ax2.set_ylim(10e-9, 50e-6)
ax2.set_yscale("log")

line_2_old, = ax3.plot([], [], color="#b2b2b2", alpha=0.6)
line_2, = ax3.plot(Vg_list, Ig_list, color="#1ab215")
line_2_dot, = ax3.plot(Vg_list[-1], Id_list[-1], "o")

ax3.set_xlim(-50, 100)
# ax3.set_ylim(-50, 50)
ax3.set_ylim(1e6, 5e7)
ax3.set_xlabel("$V_{\\mathrm{G}}$ (V)")
ax3.set_ylabel("$I_{\\mathrm{SG}}$ (nA)")

# inset
# from mpl_toolkits.axes_grid.inset_locator import inset_axes
# inset_ax1 = inset_axes(ax1,
#                        height="40%",
#                        width="40%",
#                        loc=3)   # lower left
# inset_ax1.set_axis_off()
# bmp_data = cv2.imread(file_MCA)
# im_inset = inset_ax1.imshow(bmp_data, cmap="binary_r")
# im_inset.set_zorder(2)
plt.tight_layout()

artists = [im_video, line_1, line_2,  # 0-2
           t_text,                    # 3
           line_1_dot, line_1_old,    # 4-5
           line_2_dot, line_2_old,]   # 6-7


skip_frame = 5



def init():
    global current_frame
    global t_list
    current_frame = 0
    current_time = current_frame / fps
    current_cycle = int(func_cycle(current_time))
    t_list = numpy.array([current_time])
    Id_list = func_Id_t(t_list)
    Ig_list = func_Ig_t(t_list)
    Vg_list = func_Vg_t(t_list)
    cycles = func_cycle(t_list)
    condition_old = numpy.where(cycles < current_cycle)
    condition_current = numpy.where(cycles >= current_cycle)
    # videos
    video_capt.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # start on zero frame
    ret_, frame_ = video_capt.read()
    frame_rgb_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
    artists[0].set_array(cut_frame(frame_rgb_))
    # I-V
    artists[1].set_data(Vg_list[condition_current], Id_list[condition_current])
    artists[4].set_data(Vg_list[-1], Id_list[-1])
    artists[5].set_data(Vg_list[condition_old], Id_list[condition_old])

    # Ig-V
    artists[2].set_data(Vg_list[condition_current], Ig_list[condition_current])
    artists[6].set_data(Vg_list[-1], Ig_list[-1])
    artists[7].set_data(Vg_list[condition_old], Ig_list[condition_old])
    
    artists[3].set_text("{:.2f} s, Cycle {:d}".format(current_time, current_cycle))
    return artists

def run(frame_num):
    # variables
    global current_frame
    global t_list
    current_frame = skip_frame * frame_num
    current_time = current_frame / fps
    current_cycle = int(func_cycle(current_time))
    t_list = numpy.hstack([t_list, current_time])
    Id_list = func_Id_t(t_list)
    Ig_list = func_Ig_t(t_list)
    R_list = solve_R_vec(Id_list)
    # print(Id_list[-1], R_list[-1y])
    Vg_list = func_Vg_t(t_list)
    cycles = func_cycle(t_list)
    condition_old = numpy.where(cycles < current_cycle)
    condition_current = numpy.where(cycles >= current_cycle)
    # videos
    video_capt.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # start on zero frame
    ret_, frame_ = video_capt.read()
    frame_rgb_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
    artists[0].set_array(cut_frame(frame_rgb_))
    # I-V
    artists[1].set_data(Vg_list[condition_current], Id_list[condition_current])
    artists[4].set_data(Vg_list[-1], Id_list[-1])
    artists[5].set_data(Vg_list[condition_old], Id_list[condition_old])

    # Ig-V
    # artists[2].set_data(Vg_list[condition_current], Ig_list[condition_current])
    # artists[6].set_data(Vg_list[-1], Ig_list[-1])
    # artists[7].set_data(Vg_list[condition_old], Ig_list[condition_old])
    artists[2].set_data(Vg_list[condition_current], R_list[condition_current])
    artists[6].set_data(Vg_list[-1], R_list[-1])
    artists[7].set_data(Vg_list[condition_old], R_list[condition_old])
    
    artists[3].set_text("{:.2f} s, Cycle {:d}".format(current_time, current_cycle))
    return artists
    
num_frames = min(int(max_frames) // skip_frame,
                 int(t_IV[-1] * fps) // skip_frame)
# num_frames = 10
print(num_frames)
ani = animation.FuncAnimation(fig, func=run,
                              frames=range(0, num_frames - 1),
                              init_func=init,
                              interval=50, repeat=False, blit=True)
# plt.show()
ani.save("gate_control.mp4", writer="ffmpeg",
         dpi=300,
         bitrate=2000,
         codec="libx264",
         extra_args=["-pix_fmt", "yuv420p",
                     "-crf", "20"])







