'''
Copyright (C) 2015 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from .arm2base import Arm2Base
import numpy as np

class Arm(Arm2Base):
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, **kwargs): 
        
        Arm2Base.__init__(self, **kwargs)

        # arm model parameters
        self.m1   = 1.4    # segment mass
        self.m2   = 1.1
        s1   = 0.11   # segment center of mass
        s2   = 0.16
        i1   = 0.025  # segment moment of inertia
        i2   = 0.045
        #b11 = b22 = b12 = b21 = 0.0
        b11_  = 0.7    # joint friction
        b22_  = 0.8  
        b12_  = 0.08 
        b21_  = 0.08 

        tau = 0.04 # actuator time constant (sec)
        
        self.reset() # set to init_q and init_dq

    def apply_torque(self, u, dt=None):
        if dt is None: 
            dt = self.dt

        tau = 0.04     # actuator time constant (sec)

        # arm model parameters
        m1_   = 1.4    # segment mass
        m2_   = 1.1
        l1_   = self.l1     # segment length
        l2_   = self.l2
        s1_   = 0.11   # segment center of mass
        s2_   = 0.16
        i1_   = 0.025  # segment moment of inertia
        i2_   = 0.045
        #b11_ = b22_ = b12_ = b21_ = 0.0
        b11_  = 0.7    # joint friction
        b22_  = 0.8  
        b12_  = 0.08 
        b21_  = 0.08
        g = 9.81        # earth's acceleration due to gravity 

        #------------------------ compute inertia I and extra torque H --------
        # temp vars
        mls = m2_* l1_*s2_
        iml = i1_ + i2_ + m2_*l1_**2
        dd = i2_ * iml - i2_**2
        sy = np.sin(self.q[1])
        cy = np.cos(self.q[1])

        # inertia
        #I_11 = iml + 2 * mls * cy
        #I_12 = i2_ + mls * cy
        #I_22 = i2_ * np.ones_like(cy)
        # aditya modified:
        I_11 = iml + 2 * mls * cy + m1_*s1_**2 + m2_*s2_**2
        I_12 = i2_ + mls * cy + m2_*s2_**2
        I_22 = i2_ * np.ones_like(cy) + m2_*s2_**2

        # determinant
        #det = dd - mls**2 * cy**2
        det = I_11*I_22 - I_12**2 

        # inverse inertia I1
        #I1_11 = i2_ / det
        #I1_12 = (-i2_ - mls * cy) / det
        #I1_22 = (iml + 2 * mls * cy) / det
        I1_11 = I_22 / det
        I1_12 = -I_12 / det
        I1_22 = I_11 / det

        # temp vars
        sw = np.sin(self.q[1])
        cw = np.cos(self.q[1])
        y = self.dq[0]
        z = self.dq[1]

        # extra torque H (Coriolis, centripetal, friction, gravity-aditya)
        H = np.array([-mls * (2 * y + z) * z * sw + b11_ * y + b12_ * z + (m1_*s1_+m2_*l1_)*g*np.sin(self.q[0]) + m2_*s2_*g*np.sin(self.q[0]+self.q[1]) ,
            mls * y**2 * sw + b22_ * z + b12_ * y + m2_*s2_*g*np.sin(self.q[0]+self.q[1]) ])
        #H = np.array([-mls * (2 * y + z) * z * sw + b11_ * y + b12_ * z + m1_*s1_*g*np.sin(self.q[0]) ,
        #    mls * y**2 * sw + b22_ * z + b12_ * y + m2_*s2_*g*np.sin(self.q[0]+self.q[1]) ])

        #------------- compute xdot = inv(I) * (torque - H) ------------ 
        torque = u.T - H

        self.q += dt * self.dq
        self.dq += dt * np.array([(I1_11 * torque[0] + I1_12 * torque[1]),
                                   (I1_12 * torque[0] + I1_22 * torque[1])])

        # transfer to next time step 
        self.t += dt

    def evolveFns(self, q, dq, u):
        ### CAUTION: when copying-pasting from apply_torque() above,
        ###  REMEMBER to change the self.q[] etc to q[], else you'll use the wrong values!!!
        tau = 0.04     # actuator time constant (sec)

        # arm model parameters
        m1_   = 1.4    # segment mass
        m2_   = 1.1
        l1_   = self.l1     # segment length
        l2_   = self.l2
        s1_   = 0.11   # segment center of mass
        s2_   = 0.16
        i1_   = 0.025  # segment moment of inertia
        i2_   = 0.045
        #b11_ = b22_ = b12_ = b21_ = 0.0
        b11_  = 0.7    # joint friction
        b22_  = 0.8  
        b12_  = 0.08 
        b21_  = 0.08 
        g = 9.81        # earth's acceleration due to gravity 

        #------------------------ compute inertia I and extra torque H --------
        # temp vars
        mls = m2_* l1_*s2_
        iml = i1_ + i2_ + m2_*l1_**2
        dd = i2_ * iml - i2_**2
        sy = np.sin(q[1])
        cy = np.cos(q[1])

        # inertia
        #I_11 = iml + 2 * mls * cy
        #I_12 = i2_ + mls * cy
        #I_22 = i2_ * np.ones_like(cy)
        # aditya modified:
        I_11 = iml + 2 * mls * cy + m1_*s1_**2 + m2_*s2_**2
        I_12 = i2_ + mls * cy + m2_*s2_**2
        I_22 = i2_ * np.ones_like(cy) + m2_*s2_**2

        # determinant
        #det = dd - mls**2 * cy**2
        det = I_11*I_22 - I_12**2 

        # inverse inertia I1
        #I1_11 = i2_ / det
        #I1_12 = (-i2_ - mls * cy) / det
        #I1_22 = (iml + 2 * mls * cy) / det
        I1_11 = I_22 / det
        I1_12 = -I_12 / det
        I1_22 = I_11 / det

        # temp vars
        sw = np.sin(q[1])
        cw = np.cos(q[1])
        y = dq[0]
        z = dq[1]

        # extra torque H (Coriolis, centripetal, friction, gravity-aditya)
        H = np.array([-mls * (2 * y + z) * z * sw + b11_ * y + b12_ * z + (m1_*s1_+m2_*l1_)*g*np.sin(q[0]) + m2_*s2_*g*np.sin(q[0]+q[1]) ,
            mls * y**2 * sw + b22_ * z + b12_ * y + m2_*s2_*g*np.sin(q[0]+q[1]) ])
        #H = np.array([-mls * (2 * y + z) * z * sw + b11_ * y + b12_ * z + m1_*s1_*g*np.sin(q[0]),
        #    mls * y**2 * sw + b22_ * z + b12_ * y + m2_*s2_*g*np.sin(q[0]+q[1]) ])

        #------------- compute xdot = inv(I) * (torque - H) ------------ 
        torque = u.T - H

        # return qdot and dqdot
        return  dq, \
                np.array([(I1_11 * torque[0] + I1_12 * torque[1]),
                    (I1_12 * torque[0] + I1_22 * torque[1])])
        #return  np.zeros(2),np.zeros(2)
        #return  -q,-dq
