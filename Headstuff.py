import pandas as pd
import numpy as np
data = list()
i = 0
name = input("Enter the name")
# with open(f'D:/dropbox/Dropbox/Samples/{name}/example_recording_{name}.txt') as f:
with open(f'./{name}/example_recording_{name}.txt') as f:
    for line in f.readlines():
        if i ==0:
            temp = line.split()
            print(temp)
            print(temp.insert(4,'hi'))
            data.append(temp)
            i+=1
        if not line.startswith('MSG'):
            data.append(line.split())
df = pd.DataFrame(data[2:],columns = data[0])
df['avgx'] = pd.to_numeric(df['avgx'])
df['avgy'] = pd.to_numeric(df['avgy'])
df = df.reset_index()
df_usable = df[df['index']%3 ==0]
df_usable.drop(['index'],axis = 1,inplace = True)
df_usable = df_usable.reset_index()
import cv2
import av
import numpy as np
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
import pyautogui
cap = cv2.VideoCapture(f'{name}/2022-10-24 14-14-34_{name}.mp4')
# print(cap)
# stream = container.add_stream('mpeg4',rate = 10)
# stream.width = 1920
# stream.height = 1080
# stream.codec_context.time_base = Fraction(1,10)
my_pts = 0
last_pts = 0

# def gaussian(x,sx):
#     y = x
#     sy = sx
#     x0 = x/2
#     y0 = y/2
#     M = np.zeros([y,x],dtype = float)
#     for i in range(x):
#         for j in range(y):
#             M[j,i] = np.exp(-1.0*(((float(i)-x0)**2/(2*sx*sx))+ ((float(y)-y0)**2/(2*sy*sy)) ) )
#     return M
gwh = 200
displaysize = (1080,1920)
gsdwh = gwh/6
def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(-1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M
gauss = gaussian(gwh,gsdwh)
strt = int(gwh/2)
heatmapsize = int(displaysize[1] + 2*strt), int(displaysize[0]+2*strt)
heatmap = np.zeros(heatmapsize, dtype=float)





def heatgenerator(i,heatmap):
    heatmapsize = int(displaysize[1] + 2 * strt), int(displaysize[0] + 2 * strt)
#     heatmap = np.zeros(heatmapsize, dtype=float)
    print(i)
    # get x and y coordinates
    # x and y - indexes of heatmap array. must be integers
#     j = i-1
    x = strt + int(df_usable['avgx'][i-1]) - int(gwh / 2)
    y = strt + int(df_usable['avgy'][i-1]) - int(gwh / 2)
    # correct Gaussian size if either coordinate falls outside of
    # display boundaries
    if (not 0 < x < displaysize[1]) or (not 0 < y < displaysize[0]):
        hadj = [0, gwh];
        vadj = [0, gwh]
        if 0 > x:
            hadj[0] = abs(x)
            x = 0
        elif displaysize[0] < x:
            hadj[1] = gwh - int(x - displaysize[0])
        if 0 > y:
            vadj[0] = abs(y)
            y = 0
        elif dispsize[1] < y:
            vadj[1] = gwh - int(y - displaysize[1])
        # add adjusted Gaussian to the current heatmap
        try:
            heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * 0.1
        except:
            # fixation was probably outside of display
            pass
    else:
        # add Gaussian to the current heatmap
        if i ==467:
            print('hi')
        print(heatmap[y:y + gwh, x:x + gwh].shape,gauss.shape)
        heatmap[y:y + gwh, x:x + gwh] += gauss * 0.1
    temp = heatmap.copy()
    hm = heatmap[strt:displaysize[1] + strt, strt:displaysize[0] + strt]
    lowbound = np.mean(heatmap[heatmap>0])
    heatmap[heatmap<lowbound] = np.NaN
#     if j%100 ==1:
#         for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                          key= lambda x: -x[1])[:10]:
#             print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    return hm,temp
# lowbound = np.mean(heatmap[heatmap>0])
# heatmap[heatmap<lowbound] = np.NaN


displaysize = (1920, 1080)
dispsize = displaysize
container = av.open('./Samples/Ram2/temp.avi',mode = 'w')
stream = container.add_stream('mpeg4',rate = 10)
stream.width = 1800
stream.height = 1200
stream.codec_context.time_base = Fraction(1,10)
my_pts = 0
last_pts = 0
SCREEN_SIZE = tuple(pyautogui.size())
# define the codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")
# frames per second
fps = 20.0

out = cv2.VideoWriter(f'./{name}/temp{i}.avi', fourcc, fps, (SCREEN_SIZE))
for i in range(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
# for i in range(1,int(100)):
    if i%2==0:
        continue
    if i ==1:
        temp = np.zeros(heatmapsize, dtype=float)
    # create the video write object


    # the time you want to record in seconds
    heatmapp,temp = heatgenerator(int(i/2)+1,temp)
    ret,frame = cap.read()
#     print(frame.shape)
    screen = np.zeros((dispsize[1],dispsize[0],3), dtype='uint8')

    w, h = len(frame[0]), len(frame)
#     print(w,h)
    # x and y position of the image on the display
    x = int(dispsize[0]/2 - w/2)
    y = int(dispsize[1]/2 - h/2)
    # draw the image on the screen
#     print(y+h,x+w)
    screen[y:y+h,x:x+w,:] += frame
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
    # create a figure
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0,dispsize[0],0,dispsize[1]])

    ax.imshow(screen)#, origin='upper')
#     fig.savefig('tst.png')
    ax.imshow(heatmapp, cmap='jet', alpha=0.5)
    ax.invert_yaxis()
    fig.savefig('test.png')
    plt.close(fig)
    img = cv2.imread('test.png')
    out.write(img)
out.release()

#     hmax = sns.kdeplot(df.iloc[:i].x, df.iloc[:i].y, cmap="viridis", shade=True, bw=0.1)
#     hmax.imshow(frame, zorder=1,alpha=.5)
#     plt.savefig('save_as_a_png.png',dpi = 300)
#     img = cv2.imread('save_as_a_png.png')
#     frame = av.VideoFrame.from_ndarray(np.array(img,dtype = np.uint8),format = 'rgb24')
#     frame.pts = my_pts
#     last_pts = my_pts
#     if last_pts == my_pts:
#         my_pts +=1
#     i +=1
#     for packet in stream.encode(frame):
#         container.mux(packet)