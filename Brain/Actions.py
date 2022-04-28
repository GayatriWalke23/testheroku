class Actions():
    def __init__(self):
        self.steer=0
        self.accl=0
        self.brake=0
        self.straightTargetSpeed  = 100
        self.turnTargetSpeed  = 50


PI_HALF = Math.PI / 2.0# // 90 deg
PI_FOURTHS = Math.PI / 4.0# // 45 deg
RAD_PER_DEG = Math.PI / 180.0#
DEG_PER_RAD = 1.0 / RAD_PER_DEG#

DEFAULT_MIN_SPEED = 50#
DEFAULT_MAX_SPEED = 250#

GEAR_MAX = 6#      //
RPM_MAX = 9500#    //
ACCEL_MAX = 1.0#   //
ACCEL_DELTA = 0.5# // maximum rate of change in acceleration signal, avoid spinning out

BRAKING_MAX = -0.5#   // braking signal <= BRAKING_MAX
BRAKING_DELTA = 0.05# // dampen braking to avoid lockup, max rate of chage in braking
WHEELSPIN_ACCEL_DELTA = 0.025#
WHEELSPIN_MAX = 5.0# // greater than this value --> loss of control

PURE_PURSUIT_K = 0.35# // bias - increase to reduces steering sensitivity
PURE_PURSUIT_L = 2.4#  // approx vehicle wheelbase
PURE_PURSUIT_2L = 2 * PURE_PURSUIT_L#
MAX_STEERING_ANGLE_DEG = 21# // steering lock
USE_STEERING_FILTER = false# //
STEERING_FILTER_SIZE = 5#    //

EDGE_AVOIDANCE_ENABLED = true#
EDGE_MAX_TRACK_POS = 0.85# // track edge limit
EDGE_STEERING_INPUT = 0.0075# // slightly steer away from edge

STALLED_TIMEOUT = 5# // # seconds of no significant movement

# TODO: implement mode for when vehicle is off track, e.g., a spin or missed turn

LEFT_SIDE = 1#
RIGHT_SIDE = -1#
MIDDLE = 0#
Q1 = 1#
Q2 = 2#
Q3 = 3#
Q4 = 4#
OFF_TRACK_TARGET_SPEED = 20#

k_p= 0.2
k_i= 0
k_d= 0




'''control(sensors: SensorData): SimAction {
    let action = new SimAction();
    if (this.isStalled(sensors)) {
      action.restartRace = true;
    } else {
      this.computeSteering(sensors, action);
      this.computeSpeed(sensors, action);
    }
    return action;
  }
'''

# * steer towards longest distance measure
# * greater tgt angle -> increased turn angle
# * increased turn angle -> slower tgt speed
#
# todo: account for vehicle angle to track, currently assume parallel with track centerline


def computeSteering(sensors, action):

    targetAngle = this.computeTargetAngle(sensors)

    # alpha (a) = angle of longest sensor (... -20, -10, 0, 10, 20, ...)
    rawSteeringAngleRad = -math.atan(  PURE_PURSUIT_2L * math.sin(targetAngle) /  (PURE_PURSUIT_K * sensors.speed))
    rawSteeringAngleDeg = rawSteeringAngleRad * DEG_PER_RAD

    # normalize between[-1,1]
    normalizedSteeringAngle = Utils.clamp(rawSteeringAngleDeg / MAX_STEERING_ANGLE_DEG, -1.0, 1.0)

    if (USE_STEERING_FILTER):
      this.steeringCmdFilter.push(normalizedSteeringAngle)
      steer = this.steeringCmdFilter.get()
    else:
      steer = normalizedSteeringAngle


    # On straight segments, correct for vehicle drift near edge of track
    if (EDGE_AVOIDANCE_ENABLED and sensors.isOnTrack()):
      edgeSteeringCorrection = 0
      if (sensors.trackPos > EDGE_MAX_TRACK_POS and sensors.angle < 0.005) :# too far left
        edgeSteeringCorrection = -EDGE_STEERING_INPUT
      elif (sensors.trackPos < -EDGE_MAX_TRACK_POS and sensors.angle > -0.005) :# too far right
        edgeSteeringCorrection = EDGE_STEERING_INPUT

    steer += edgeSteeringCorrection
    return  steer



def computeSpeed(SensorData, action):
    accel = 0
    gear = sensors.gear
    brakingZone = sensors.maxDistance < sensors.speedX / 1.5
    targetSpeed = 0
    hasWheelSpin = false

    if (sensors.isOnTrack()):
      if(brakingZone):
        targetSpeed = Math.max(DEFAULT_MIN_SPEED, sensors.maxDistance)
      else:
        targetSpeed = DEFAULT_MAX_SPEED

      # detect wheel spin
      frontWheelAvgSpeed = (sensors.wheelSpinVelocity[0] + sensors.wheelSpinVelocity[1]) / 2.0;
      rearWheelAvgSpeed = (sensors.wheelSpinVelocity[2] + sensors.wheelSpinVelocity[3]) / 2.0;
      slippagePercent = frontWheelAvgSpeed / rearWheelAvgSpeed * 100.0;

      wheelSpinDelta = math.abs(
          ((sensors.wheelSpinVelocity[0] + sensors.wheelSpinVelocity[1]) / 2) -
          ((sensors.wheelSpinVelocity[2] + sensors.wheelSpinVelocity[3]) / 2))

      hasWheelSpin = sensors.speedX > 5.0 and slippagePercent < 80.0
      if (hasWheelSpin) :#excessive wheelspin preempts normal accel/decel calc
        accel = this.curAccel - WHEELSPIN_ACCEL_DELTA

      else: #{ // off track
        targetSpeed = OFF_TRACK_TARGET_SPEED


    if (not hasWheelSpin):
      this.speedController.setTarget(targetSpeed)
      accel = this.speedController.update(sensors.speed)


  # returns targetAngle in radians
def computeTargetAngle(sensors):

    targetAngle = sensors.maxDistanceAngle * RAD_PER_DEG

    if (sensors.isOffTrack()) :
      targetAngle = this.computeRecoveryTargetAngle(sensors)

    return targetAngle


#returns targetAngle in radians
def computeRecoveryTargetAngle(sensors):

    targetAngle = 0
    trackPos = sensors.trackPos

    # clockwise Q1=[0,90), Q2=[90,+], Q3=[-90,-], Q4=[0,-90)
    quadrant = 0
    if (sensors.angle >= 0.0 and sensors.angle < PI_HALF) :
      quadrant = Q1
    elif (sensors.angle >= PI_HALF):
      quadrant = Q2
    elif (sensors.angle <= -PI_HALF):
      quadrant = Q3
    else:
      quadrant = Q4


    trackSide = MIDDLE
    if (trackPos > 1.0):
      trackSide = LEFT_SIDE
    elif (trackPos < -1.0) :
      trackSide = RIGHT_SIDE


    if quadrant==Q1:
        if (trackSide == LEFT_SIDE):
          targetAngle = PI_FOURTHS - sensors.angle
        elif (trackSide == RIGHT_SIDE) :
          targetAngle = -PI_FOURTHS

    elif quadrant==Q2:
        if (trackSide == RIGHT_SIDE):
          targetAngle = PI_FOURTHS
        else :# // left or middle
          targetAngle = -PI_FOURTHS


    elif quadrant==Q3:
        if (trackSide == LEFT_SIDE):
          targetAngle = -PI_FOURTHS
        else: # // right or middle
          targetAngle = PI_FOURTHS


    elif quadrant==Q4:
        if (trackSide == LEFT_SIDE):
            targetAngle = PI_FOURTHS
        elif (trackSide == RIGHT_SIDE):
            targetAngle = -PI_FOURTHS - sensors.angle

        # Note: when trackSide == MIDDLE -> normal steering logic applied

    return targetAngle

def isStalled(SensorData):
    this.speedMonitor.push(sensors.speed)
    return this.speedMonitor.get() < 2.0




