from direct.gui.DirectGui import DirectSlider


declarations = """
// Beginning of `boids.declarations`
uniform float radius;  // The detection radius.

// Value accumulators for the boid.
uint otherVecs = 0;  // This counts how many boids we actually work on.
vec3 cohere = vec3(0);
vec3 align = vec3(0);
vec3 separate = vec3(0);
// End of `boids.declarations`
"""[1:-1]
processing = """
  // Beginning of `boids.processing`
  // `a` is the current boid, `b` the nearby boid.
  vec3 toBoid = b.pos - a.pos;
  float dist = length(toBoid);
  if (dist <= radius) {  // Only consider boids that we can detect
    otherVecs++;
    cohere += b.pos;
    align += b.dir;
    separate += -normalize(toBoid) * (1.0 - (length(toBoid) / radius));
  // End of `boids.processing`
}
"""[1:-1]
combining = """
  // Beginning of `boids.combining`
  // This happens after looping over all nearby boids.
  // Relevant variables first...
  float dt = 1.0/60.0;
  vec3 pos = boids[boidIdx].pos;
  vec3 dir = boids[boidIdx].dir;

  // Boid rules
  if (otherVecs > 0) {
    // So far, `cohere` and `align` are the sum of the positions / 
    // directions of the boids around us. Dividing by their number
    // yields the average position / direction, and subtracting our own
    // value yields how much we'd have to steer to fully move ourselves
    // to the center of mass / fully align our direction.
    cohere = cohere / otherVecs - pos;
    align = align / otherVecs - dir;
    separate = separate / otherVecs;
  }

  // Wall repulsion
  // Notes from spherical boundary experiment
  // vec3 toCenter = vec3(0.5) - pos;
  // if (length(toCenter) > (0.5 - wallRepDist)) {
  //   wallRep =  normalize(toCenter) * dt;
  // }
  //
  // Box wall repulsion
  // * We're repulsed by all six walls
  // * We're in the 0-1 cube
  float wallRepDist = 0.2;
  vec3 wallRep = vec3(0);
  float wallDistToNeg = min(min(pos.x, pos.y), pos.z);
  vec3 posV = vec3(1) - pos;
  float wallDistToPos = min(min(posV.x, posV.y), posV.z);
  float wallDist = min(wallDistToNeg, wallDistToPos);
  if (wallDist < wallRepDist) {  // Too close to wall, go to center.
    float ratio = 1.0 - wallDist / wallRepDist;
    wallRep = normalize(vec3(0.5) - pos) * ratio;
  }
  
  vec3 nextPos;
  vec3 nextDir;
  vec3 steer = vec3(0);
  float steerForce = 30.0;

  steer += align * 0.5;
  steer += cohere * 0.3;
  steer += separate * 0.2;
  steer += wallRep * 0.01;
  steer *= steerForce;
  dir += steer * dt ;
  dir = normalize(dir) * clamp(length(dir), 0.1, 0.25);
  nextPos = pos + dir * dt;
  nextDir = dir;

  // Boundary condition: Hard walls
  nextPos = min(nextPos, 1.0);
  nextPos = max(nextPos, 0.0);

  // Limit minimum and maximum speed
  //nextDir = clampVec(nextDir);

  // Write values into the boid
  boids[boidIdx].nextPos = nextPos;
  boids[boidIdx].nextDir = nextDir;
  // End of `boids.combining`
"""[1:-1]


def make_ui(mover):
    def set_radius(*args, **kwargs):
        r = slider_radius['value']
        mover.set_shader_arg('radius', r)
        print(f"Detection radius: {r}")
    slider_radius = DirectSlider(
        parent=base.a2dTopLeft,
        frameSize=(0, 1, -0.03, 0.03),
        pos=(0.02+0.45, 0, -0.05),
        text="Detection radius",
        text_scale=0.05,
        text_pos=(-0.25, -0.015),
        range=(0.0, 1.0),
        value=mover.shader_args['radius'],
        pageSize=0.01,
        command=set_radius,
    )
