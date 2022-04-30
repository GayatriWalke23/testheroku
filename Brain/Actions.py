class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3

        
        '''self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()'''
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        self.state = None
    '''angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32),
                               gear=np.array(raw_obs['gear'], dtype=np.float32),
                               accel=np.array(obs2['accel'], dtype=np.float32),
                               getBrake=np.array(obs2['brake'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               segment_radius=0,#np.array(raw_obs['segment_radius'], dtype=np.float32),
                               next_segment_radius=0,#np.array(raw_obs['next_segment_radius'], dtype=np.float32),
                               car_yaw=np.array(raw_obs['focus'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32))
'''
    def drive(self):

        angle = self.state["angle"]
        dist = self.state["trackPos"]
        steer = ((angle - dist*0.5)/self.steer_lock)


        rpm = self.state["rpm"]
        gear = self.state.getGear()
        
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        
        if up and rpm > 7000:
            gear += 1
        
        if not up and rpm < 3000:
            gear -= 1
        
        self.control.setGear(gear)
    

        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0

        return steer,gear,accel
