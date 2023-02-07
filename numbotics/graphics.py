from vpython import *
from vpython.no_notebook import stop_server
import numpy as np


GFX_ON = False

col_used = 0
col_dict = {0: vec(0.85,0.0,0.0), #color.red,
            1: vec(0.3,0.3,0.3)}
'''
            1: color.blue,
            2: color.green,
            3: color.purple,
            4: color.orange,
            5: color.magenta,
            6: color.cyan,
            7: color.yellow}
'''

def inst_gfx(height=600,
             width=800,
             rnge=2.0,
             grid=True,
             xlim=[-1.0,1.0],
             ylim=[-1.0,1.0],
             zlim=[-1.0,1.0],
             grid_res=10,
             line_res=0.0025,
             title='Numbotics'):

    global scene
    global GFX_ON

    if not GFX_ON:
        GFX_ON = True

        scene = canvas()

        scene.height = height
        scene.width = width
        scene.range = rnge
        scene.title = title
        scene.background = vec(0.82,0.82,0.82)

        if grid:
            # making graphics coordinates intuitive
            ytmp = ylim[:]
            ztmp = zlim[:]
            zlim = xlim[:]
            xlim = ytmp[:]
            ylim = ztmp[:]

            scene.lights = []
            local_light(pos=vec(xlim[1],ylim[1],zlim[1]),
                        color=color.white)

            box(pos=vec(xlim[0]+xlim[1],ylim[0]+ylim[1],zlim[0]),
                size=vec(xlim[1]-xlim[0],ylim[1]-ylim[0],line_res/2.0),
                axis=vec(0,0,0),
                color=vec(1.0,1.0,1.0))

            box(pos=vec(xlim[0],ylim[0]+ylim[1],zlim[0]+zlim[1]),
                size=vec(line_res/2.0,ylim[1]-ylim[0],zlim[1]-zlim[0]),
                axis=vec(0,0,0),
                color=vec(1.0,1.0,1.0))

            box(pos=vec(xlim[0]+xlim[1],ylim[0],zlim[0]+zlim[1]),
                size=vec(xlim[1]-xlim[0],line_res/2.0,zlim[1]-zlim[0]),
                axis=vec(0,0,0),
                color=vec(1.0,1.0,1.0))

            for i in range(grid_res+1):

                a = float(i)/float(grid_res)
                xval = ((1-a)*xlim[0])+(a*xlim[1])
                yval = ((1-a)*ylim[0])+(a*ylim[1])
                zval = ((1-a)*zlim[0])+(a*zlim[1])

                size_fac = float(i==grid_res)*3.0 + float(i!=grid_res)

                cylinder(pos=vec(xlim[0],ylim[0],zval),
                        size=vec(xlim[1]-xlim[0],line_res*size_fac,line_res*size_fac),
                        axis=vec(1,0,0),
                        color=vec(0.15,0.15,0.15))
                cylinder(pos=vec(xval,ylim[0],zlim[0]),
                        size=vec(zlim[1]-zlim[0],line_res*size_fac,line_res*size_fac),
                        axis=vec(0,0,1),
                        color=vec(0.15,0.15,0.15))

                cylinder(pos=vec(xlim[0],ylim[0],zval),
                        size=vec(ylim[1]-ylim[0],line_res*size_fac,line_res*size_fac),
                        axis=vec(0,1,0),
                        color=vec(0.15,0.15,0.15))
                cylinder(pos=vec(xlim[0],yval,zlim[0]),
                        size=vec(zlim[1]-zlim[0],line_res,line_res),
                        axis=vec(0,0,1),
                        color=vec(0.15,0.15,0.15))

                cylinder(pos=vec(xval,ylim[0],zlim[0]),
                        size=vec(ylim[1]-ylim[0],line_res,line_res),
                        axis=vec(0,1,0),
                        color=vec(0.15,0.15,0.15))
                cylinder(pos=vec(xlim[0],yval,zlim[0]),
                        size=vec(xlim[1]-xlim[0],line_res,line_res),
                        axis=vec(1,0,0),
                        color=vec(0.15,0.15,0.15))

def kill_gfx():
    stop_server()

def get_keys():
    return keysdown()

def gfx_rate(hz):
    rate(hz)

def spat_to_graph(x):
    # type cast to float to support both (3,) and (3,1) shapes
    return vec(float(x[1]),float(x[2]),float(x[0]))


def pris_link(prev_mat, this_mat):
    l_box = box()

def trans_pris_link(link, prev_mat, this_mat):
    pass


def rev_link(prev_mat, this_mat):
    global col_used
    pos = spat_to_graph(prev_mat[0:3,3])
    pos_diff = this_mat[0:3,3]-prev_mat[0:3,3]
    ax = spat_to_graph(pos_diff)
    length = np.linalg.norm(pos_diff)
    l_cyl = cylinder(pos=pos,
                     axis=ax,
                     size=vec(length,0.05,0.05),
                     color=col_dict[col_used%len(col_dict)])
    col_used += 1
    return l_cyl

def rot_rev_link(link, prev_mat, this_mat):
    pos_diff = this_mat[0:3,3]-prev_mat[0:3,3]
    ax = spat_to_graph(pos_diff)
    link.axis=ax
    link.pos=spat_to_graph(prev_mat[0:3,3])


