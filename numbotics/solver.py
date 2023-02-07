import numpy as np
import numbotics.logger as nlog


# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods
class ARK():
    def __init__(self, c, a, b, b_star, dt=1e-3, err_low=1e-12, err_high=1e-8, auto_dec=True, auto_inc=False):

        assert err_low < err_high

        if not auto_dec and auto_inc:
            nlog.warning('numerical instability likely when auto increment on and auto decrement off')

        self.dt = dt
        self.err_low = err_low
        self.err_high = err_high

        self.auto_dec = auto_dec
        self.auto_inc = auto_inc

        self.c = c

        self.a = a

        self.b = b
        self.b_star = b_star


    def step(self, fun, init_t, init_x, T):

        t = init_t
        x = init_x

        while t < T:
            k = fun(t,x)
            for i in range(len(self.a)):
                k = np.concatenate((k,fun(
                    t+self.c[i]*self.dt,
                    x+np.expand_dims(
                        self.dt*np.sum(
                            np.multiply(k,self.a[i]),
                            axis=1),
                        1))),
                    axis=1)

            err = np.linalg.norm(self.dt*np.sum(np.multiply(k,(self.b-self.b_star)),axis=1))

            if err < self.err_high:
                x += self.dt*np.expand_dims(np.sum(np.multiply(k,self.b),axis=1),1)
                t += self.dt
                if (err < self.err_low) and self.auto_inc and (self.dt < (T-init_t)/2.0):
                    self.dt *= 2.0

            elif self.auto_dec:
                self.dt /= 2.0

        return t-init_t, x

    
class ARK45(ARK):
    def __init__(self, dt=1e-3, err_low=1e-12, err_high=1e-8, auto_dec=True, auto_inc=False):
        c = np.array([1.0/4.0,
                      3.0/8.0,
                      12.0/13.0,
                      1.0,
                      1.0/2.0])

        a = [np.array([1.0/4.0]),
             np.array([3.0/32.0, 9.0/32.0]),
             np.array([1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0]),
             np.array([439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0]),
             np.array([-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0])]

        b = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0])
        b_star = np.array([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0])

        super().__init__(c, a, b, b_star, dt, err_low, err_high, auto_dec, auto_inc)


class ARK23(ARK):
    def __init__(self, dt=1e-3, err_low=1e-10, err_high=1e-6, auto_dec=True, auto_inc=False):
        c = np.array([1.0/2.0,
                      3.0/4.0,
                      1.0])

        a = [np.array([1.0/2.0]),
             np.array([0.0, 3.0/4.0]),
             np.array([2.0/9.0, 1.0/3.0, 4.0/9.0])]

        b = np.array([2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0])
        b_star = np.array([7.0/24.0, 1.0/4.0, 1.0/3.0, 1.0/8.0])

        super().__init__(c, a, b, b_star, dt, err_low, err_high, auto_dec, auto_inc)


class ARK12(ARK):
    def __init__(self, dt=1e-3, err_low=1e-8, err_high=1e-4, auto_dec=True, auto_inc=False):
        c = np.array([1.0])

        a = [np.array([1.0])]

        b = np.array([0.5, 0.5])
        b_star = np.array([1.0, 0.0])

        super().__init__(c, a, b, b_star, dt, err_low, err_high, auto_dec, auto_inc)



class Eul():
    def __init__(self, dt=1e-3, nudge=1.0):
        self.dt = dt
        self.nudge = nudge

    def step(self, fun, init_t, init_x, T):
        t = init_t
        x = init_x

        while t < T:
            k = fun(t,x)
            x += k*(self.dt*self.nudge)
            t += self.dt

        return t-init_t, x
