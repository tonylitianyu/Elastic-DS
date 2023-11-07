from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

class MouseTrajectory:
    lock = None 
    def __init__(self, points, indexing=0, store_mat = 0, data_dir='./data/', snaps_dir='./snaps/'):
        self.press      = None
        self.background = None
        self.points     = points
        self.x_data     = []
        self.y_data     = []
        self.t_data     = []
        self.l_data     = []
        self.label      = 0
        self.t0         = 0
        self.indexing   = indexing
        self.store_mat  = store_mat
        self.data_dir   = data_dir
        self.snaps_dir  = snaps_dir

    def connect(self):
        ''' connect to all the events we need 
        '''
        self.cidpress = self.points.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.points.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.points.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        '''
        on button press we will see if the mouse is over us and store some data
        '''
        if event.inaxes != self.points.axes: return

        x, y     = event.xdata, event.ydata
        self.t0  = time.time()
        t = 0
        self.x_data.append(x)
        self.y_data.append(y)
        self.t_data.append(t)
        self.l_data.append(self.label)

        # print('l=%d, t=%1.5f, x=%1.2f, y=%1.2f' % (self.label, t, x, y))
        self.press = x, y
        MouseTrajectory.lock = self

        # update plot's data  
        self.points.set_data(x,y)
        canvas = self.points.figure.canvas
        axes   = self.points.axes
        self.points.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.points.axes.bbox)

        # restore background
        canvas.restore_region(self.background)
        # redraw just the points
        axes.draw_artist(self.points)
        # fill in the axes pointsangle
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        '''
        on motion we will move the points if the mouse is over us
        '''
        if MouseTrajectory.lock is not self:
            return
        if event.inaxes != self.points.axes: return

        x = event.xdata
        y = event.ydata
        t = time.time() - self.t0

        # For debugging
        # print('l=%d, t=%1.5f, x=%1.2f, y=%1.2f' % (self.label, t, x, y))

        # Stores trajectories
        self.x_data.append(x)
        self.y_data.append(y)
        self.t_data.append(t)
        self.l_data.append(self.label)

        # update plot's data  
        self.points.set_data(self.x_data,self.y_data)
        self.update_figure()        


    def on_release(self, event):
        '''
        on release we reset the pressed data
        '''
        if MouseTrajectory.lock is not self:
            return

        self.press = None
        MouseTrajectory.lock = None
        
        # turn off the points animation property and reset the background
        self.points.set_animated(False)
        self.background = None

        # redraw the full figure
        self.points.figure.canvas.draw()

        # Increase label for demonstrated trajectories
        self.label = self.label + 1


    def update_figure(self):
        '''
        updates the points on figure
        '''
        canvas = self.points.figure.canvas
        axes   = self.points.axes            
        # restore background
        canvas.restore_region(self.background)
        # redraw just the points
        axes.draw_artist(self.points)
        # fill in the axes rectangle
        canvas.blit(axes.bbox)

    def disconnect(self):
        '''
        disconnect all the stored connection ids
        '''
        self.points.figure.canvas.mpl_disconnect(self.cidpress)
        self.points.figure.canvas.mpl_disconnect(self.cidrelease)
        self.points.figure.canvas.mpl_disconnect(self.cidmotion)

    def snap_callback(self, event):
        '''
        take snapshot of current state
        ''' 
        print('Created snapshot of figure')
        
        # If indexing set to 1, time-index will be added to figure name
        if self.indexing == 1:
            figure_name = '%shuman_demonstrated_trajectories_%s.png'% (self.snaps_dir, time.strftime("%b%d_%H:%M:%S", time.gmtime()))
        else:
            figure_name = '%shuman_demonstrated_trajectories.png'% (self.snaps_dir)

        plt.savefig(figure_name, bbox_inches='tight')


    def store_callback(self, event):
        '''
        store the recorded trajectories in pickle file
        '''
        print('Saving data to file')

        data_name = 'human_demonstrated_trajectories'
        time_str  = time.strftime("%b%d_%H:%M:%S", time.gmtime())

        # If indexing set to 1, time-index will be added to figure name
        if self.indexing == 1:
            file_name = self.data_dir + data_name + '_' + time_str + '.dat'
        elif self.indexing == 2:
            file_name = self.data_dir + data_name + '_1' + '.dat'
        else:
            file_name = self.data_dir + data_name + '.dat'
        
        header = "# l, t, x, y\n"        
        with open(file_name, 'w') as f:
            f.write(header)
            for i in range(len(self.l_data)):
                f.write('{:d} {:.4f} {:.4f} {:.4f}\n'.format(self.l_data[i], self.t_data[i], self.x_data[i], self.y_data[i]))

        if self.store_mat == 1:        
            # Create a dictionary
            adict = {}
            adict['labels']      = self.l_data
            adict['time-stamps'] = self.t_data
            adict['x-coords']    = self.x_data
            adict['y-coords']    = self.y_data
            if self.indexing == 1:
                file_name = self.data_dir + data_name + '_' + time_str + '.mat'
            else:
                file_name = self.data_dir + data_name + '.mat'
            sio.savemat(file_name, adict)

        # Fast numpy-way, can't modify format though        
        # demos = np.column_stack((self.t_data, self.x_data, self.y_data))
        # np.savetxt(file_name, demos, header=header)

    def clear_callback(self, event):    
        '''
        clear drawn data and restart trajectory labels
        '''
        print('Inside clear data callback')
        self.x_data     = []
        self.y_data     = []
        self.t_data     = []
        self.l_data     = []
        self.label      = 0
        self.t0         = 0

        # update plot's data  
        self.points.set_data(self.x_data,self.y_data)
        self.points.set_animated(True)

        canvas = self.points.figure.canvas                
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.points.axes.bbox)
        self.update_figure()


def load_trajectories(file_name):
    '''reads trajectory data from a text file'''
    with open(file_name, 'r') as f:
        line = f.readline()
        l = []
        t = []
        header = []
        x = []
        y = []        
        while line:
            if line.startswith('#'):
                header.append(line)
            else:
                data = line.split(' ')
                l.append(int(data[0]))
                t.append(float(data[1]))
                x.append(float(data[2]))
                y.append(float(data[3]))                            
            line = f.readline()
    return l,t,x,y