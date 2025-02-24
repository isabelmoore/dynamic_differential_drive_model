#!/usr/bin/env python3

class PID:
    '''
    kp, ki, kd are gains
    satLower and satUpper are control saturation thresholds
    '''
    def __init__(self, kp=1., ki=0., kd=0., satLower=-100., satUpper=100.):
        self.errorNow = 0.
        self.errorTotal = 0.
        self.errorPrev = 0.
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.satLower = satLower
        self.satUpper = satUpper

    def setGains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def setSaturation(self, satLower, satUpper):
        self.satLower = satLower
        self.satUpper = satUpper

    def computeControl(self, error):
        self.errorPrev = self.errorNow
        self.errorNow = error
        self.errorTotal += error

        # Compute control
        u = \
                self.kp * self.errorNow + \
                self.ki * self.errorTotal + \
                self.kd * (self.errorNow - self.errorPrev)

        # Integrator anti-windup via clamping and control saturation
        if u > self.satUpper:
            u = self.satUpper
            self.errorTotal -= self.errorNow
        elif u < self.satLower:
            u = self.satLower
            self.errorTotal -= self.errorNow

        return u
