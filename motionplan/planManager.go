package motionplan

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/edaniels/golog"
	"go.viam.com/utils"

	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
)

const (
	defaultOptimalityMultiple = 2.0
	defaultFallbackTimeout    = 1.5

	// set this to true to get collision penetration depth, which is useful for debugging.
	getCollisionDepth = false
)

// planManager is intended to be the single entry point to motion planners, wrapping all others, dealing with fallbacks, etc.
// Intended information flow should be:
// motionplan.PlanMotion() -> SolvableFrameSystem.SolveWaypointsWithOptions() -> planManager.planSingleWaypoint().
type planManager struct {
	*planner
	frame *solverFrame
	fs    referenceframe.FrameSystem
}

func newPlanManager(frame *solverFrame, fs referenceframe.FrameSystem, logger golog.Logger, seed int) (*planManager, error) {
	//nolint: gosec
	p, err := newPlanner(frame, rand.New(rand.NewSource(int64(seed))), logger, newBasicPlannerOptions())
	if err != nil {
		return nil, err
	}
	return &planManager{p, frame, fs}, nil
}

// PlanSingleWaypoint will solve the solver frame to one individual pose. If you have multiple waypoints to hit, call this multiple times.
// Any constraints, etc, will be held for the entire motion.
func (pm *planManager) PlanSingleWaypoint(ctx context.Context,
	seedMap map[string][]referenceframe.Input,
	goalPos spatialmath.Pose,
	worldState *referenceframe.WorldState,
	motionConfig map[string]interface{},
) ([][]referenceframe.Input, error) {
	seed, err := pm.frame.mapToSlice(seedMap)
	if err != nil {
		return nil, err
	}
	seedPos, err := pm.frame.Transform(seed)
	if err != nil {
		return nil, err
	}

	var cancel func()

	// set timeout for entire planning process if specified
	if timeout, ok := motionConfig["timeout"].(float64); ok {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	}
	if cancel != nil {
		defer cancel()
	}

	// If we are world rooted, translate the goal pose into the world frame
	if pm.frame.worldRooted {
		tf, err := pm.frame.fss.Transform(seedMap, referenceframe.NewPoseInFrame(pm.frame.goalFrame.Name(), goalPos), referenceframe.World)
		if err != nil {
			return nil, err
		}
		goalPos = tf.(*referenceframe.PoseInFrame).Pose()
	}

	var goals []spatialmath.Pose
	var opts []*plannerOptions

	// linear motion profile has known intermediate points, so solving can be broken up and sped up
	if profile, ok := motionConfig["motion_profile"]; ok && profile == LinearMotionProfile {
		pathStepSize, ok := motionConfig["path_step_size"].(float64)
		if !ok {
			pathStepSize = defaultPathStepSize
		}
		numSteps := PathStepCount(seedPos, goalPos, pathStepSize)

		from := seedPos
		for i := 1; i < numSteps; i++ {
			by := float64(i) / float64(numSteps)
			to := spatialmath.Interpolate(seedPos, goalPos, by)
			goals = append(goals, to)
			opt, err := pm.plannerSetupFromMoveRequest(from, to, seedMap, worldState, motionConfig)
			if err != nil {
				return nil, err
			}
			opts = append(opts, opt)

			from = to
		}
		seedPos = from
	}
	goals = append(goals, goalPos)
	opt, err := pm.plannerSetupFromMoveRequest(seedPos, goalPos, seedMap, worldState, motionConfig)
	if err != nil {
		return nil, err
	}
	opts = append(opts, opt)

	planners := make([]motionPlanner, 0, len(opts))
	// Set up planners for later execution
	for _, opt := range opts {
		// Build planner
		var randseed *rand.Rand
		if seed, ok := opt.extra["rseed"].(int); ok {
			//nolint: gosec
			randseed = rand.New(rand.NewSource(int64(seed)))
		} else {
			//nolint: gosec
			randseed = rand.New(rand.NewSource(int64(pm.randseed.Int())))
		}

		pathPlanner, err := opt.PlannerConstructor(
			pm.frame,
			randseed,
			pm.logger,
			opt,
		)
		if err != nil {
			return nil, err
		}
		planners = append(planners, pathPlanner)
	}

	// If we have multiple sub-waypoints, make sure the final goal is not unreachable.
	if len(goals) > 1 {
		// Viability check; ensure that the waypoint is not impossible to reach
		_, err = pm.getSolutions(ctx, goalPos, seed)
		if err != nil {
			return nil, err
		}
	}

	resultSlices, err := pm.planAtomicWaypoints(ctx, goals, seed, planners)
	if err != nil {
		if len(goals) > 1 {
			err = fmt.Errorf("failed to plan path for valid goal: %w", err)
		}
		return nil, err
	}
	return resultSlices, nil
}

// planAtomicWaypoints will plan a single motion, which may be composed of one or more waypoints. Waypoints are here used to begin planning
// the next motion as soon as its starting point is known. This is responsible for repeatedly calling planSingleAtomicWaypoint for each
// intermediate waypoint. Waypoints here refer to points that the software has generated to.
func (pm *planManager) planAtomicWaypoints(
	ctx context.Context,
	goals []spatialmath.Pose,
	seed []referenceframe.Input,
	planners []motionPlanner,
) ([][]referenceframe.Input, error) {
	// A resultPromise can be queried in the future and will eventually yield either a set of planner waypoints, or an error.
	// Each atomic waypoint produces one result promise, all of which are resolved at the end, allowing multiple to be solved in parallel.
	resultPromises := []*resultPromise{}

	// try to solve each goal, one at a time
	for i, goal := range goals {
		// Check if ctx is done between each waypoint
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		pathPlanner := planners[i]
		// Plan the single waypoint, and accumulate objects which will be used to constrauct the plan after all planning has finished
		newseed, future, err := pm.planSingleAtomicWaypoint(ctx, goal, seed, pathPlanner, nil)
		if err != nil {
			return nil, err
		}
		seed = newseed
		resultPromises = append(resultPromises, future)
	}

	resultSlices := [][]referenceframe.Input{}

	// All goals have been submitted for solving. Reconstruct in order
	for _, future := range resultPromises {
		steps, err := future.result(ctx)
		if err != nil {
			return nil, err
		}
		resultSlices = append(resultSlices, steps...)
	}

	return resultSlices, nil
}

// planSingleAtomicWaypoint attempts to plan a single waypoint. It may optionally be pre-seeded with rrt maps; these will be passed to the
// planner if supported, or ignored if not.
func (pm *planManager) planSingleAtomicWaypoint(
	ctx context.Context,
	goal spatialmath.Pose,
	seed []referenceframe.Input,
	pathPlanner motionPlanner,
	maps *rrtMaps,
) ([]referenceframe.Input, *resultPromise, error) {
	if parPlan, ok := pathPlanner.(rrtParallelPlanner); ok {
		// rrtParallelPlanner supports solution look-ahead for parallel waypoint solving
		// This will set that up, and if we get a result on `endpointPreview`, then the next iteration will be started, and the steps
		// for this solve will be rectified at the end.
		endpointPreview := make(chan node, 1)
		solutionChan := make(chan *rrtPlanReturn, 1)
		utils.PanicCapturingGo(func() {
			pm.planParallelRRTMotion(ctx, goal, seed, parPlan, endpointPreview, solutionChan, maps)
		})
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		default:
		}

		select {
		case nextSeed := <-endpointPreview:
			return nextSeed.Q(), &resultPromise{future: solutionChan}, nil
		case planReturn := <-solutionChan:
			if planReturn.planerr != nil {
				return nil, nil, planReturn.planerr
			}
			steps := planReturn.toInputs()
			return steps[len(steps)-1], &resultPromise{steps: steps}, nil
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		}
	} else {
		// This ctx is used exclusively for the running of the new planner and timing it out.
		plannerctx, cancel := context.WithTimeout(ctx, time.Duration(pathPlanner.opt().Timeout*float64(time.Second)))
		defer cancel()
		steps, err := pathPlanner.plan(plannerctx, goal, seed)
		if err != nil {
			return nil, nil, err
		}
		// Update seed for the next waypoint to be the final configuration of this waypoint
		seed = steps[len(steps)-1]
		return seed, &resultPromise{steps: steps}, nil
	}
}

// planParallelRRTMotion will handle planning a single atomic waypoint using a parallel-enabled RRT solver. It will handle fallbacks
// as necessary.
func (pm *planManager) planParallelRRTMotion(
	ctx context.Context,
	goal spatialmath.Pose,
	seed []referenceframe.Input,
	pathPlanner rrtParallelPlanner,
	endpointPreview chan node,
	solutionChan chan *rrtPlanReturn,
	maps *rrtMaps,
) {
	var err error
	// If we don't pass in pre-made maps, initialize and seed with IK solutions here
	if maps == nil {
		planSeed := initRRTSolutions(ctx, pathPlanner, goal, seed)
		if planSeed.planerr != nil || planSeed.steps != nil {
			solutionChan <- planSeed
			return
		}
		maps = planSeed.maps
	}

	// publish endpoint of plan if it is known
	var nextSeed node
	if len(maps.goalMap) == 1 {
		pm.logger.Debug("only one IK solution, returning endpoint preview")
		for key := range maps.goalMap {
			nextSeed = key
		}
		if endpointPreview != nil {
			endpointPreview <- nextSeed
			endpointPreview = nil
		}
	}

	// This ctx is used exclusively for the running of the new planner and timing it out.
	plannerctx, cancel := context.WithTimeout(ctx, time.Duration(pathPlanner.opt().Timeout*float64(time.Second)))
	defer cancel()

	plannerChan := make(chan *rrtPlanReturn, 1)

	// start the planner
	utils.PanicCapturingGo(func() {
		pathPlanner.rrtBackgroundRunner(plannerctx, goal, seed, &rrtParallelPlannerShared{maps, endpointPreview, plannerChan})
	})

	// Wait for results from the planner. This will also handle calling the fallback if needed, and will ultimately return the best path
	select {
	case <-ctx.Done():
		// Error will be caught by monitoring loop
		return
	default:
	}

	select {
	case finalSteps := <-plannerChan:
		// We didn't get a solution preview (possible error), so we get and process the full step set and error.

		mapSeed := finalSteps.maps

		// Create fallback planner
		var fallbackPlanner motionPlanner
		if pathPlanner.opt().Fallback != nil {
			var randseed *rand.Rand
			if seed, ok := pathPlanner.opt().extra["rseed"].(int); ok {
				//nolint: gosec
				randseed = rand.New(rand.NewSource(int64(seed)))
			} else {
				//nolint: gosec
				randseed = rand.New(rand.NewSource(int64(pm.randseed.Int())))
			}

			fallbackPlanner, err = pathPlanner.opt().Fallback.PlannerConstructor(
				pm.frame,
				randseed,
				pm.logger,
				pathPlanner.opt().Fallback,
			)
			if err != nil {
				fallbackPlanner = nil
			}
		}

		// If there was no error, check path quality. If sufficiently good, move on.
		// If there *was* an error, then either the fallback will not error and will replace it, or the error will be returned
		if finalSteps.err() == nil {
			if fallbackPlanner != nil {
				if ok, score := goodPlan(finalSteps, pathPlanner.opt()); ok {
					pm.logger.Debugf("got path with score %f, close enough to optimal %f", score, maps.optNode.cost)
					fallbackPlanner = nil
				} else {
					pm.logger.Debugf("path with score %f not close enough to optimal %f, falling back", score, maps.optNode.cost)

					// If we have a connected but bad path, we recreate new IK solutions and start from scratch
					// rather than seeding with a completed, known-bad tree
					mapSeed = nil
				}
			}
		}

		// Start smoothing before initializing the fallback plan. This allows both to run simultaneously.
		smoothChan := make(chan []node, 1)
		utils.PanicCapturingGo(func() {
			smoothChan <- pathPlanner.smoothPath(ctx, finalSteps.steps)
		})
		var alternateFuture *resultPromise

		// Run fallback only if we don't have a very good path
		if fallbackPlanner != nil {
			_, alternateFuture, err = pm.planSingleAtomicWaypoint(
				ctx,
				goal,
				seed,
				fallbackPlanner,
				mapSeed,
			)
			if err != nil {
				alternateFuture = nil
			}
		}

		// Receive the newly smoothed path from our original solve, and score it
		finalSteps.steps = <-smoothChan
		_, score := goodPlan(finalSteps, pathPlanner.opt())

		// If we ran a fallback, retrieve the result and compare to the smoothed path
		if alternateFuture != nil {
			alternate, err := alternateFuture.result(ctx)
			if err == nil {
				// If the fallback successfully found a path, check if it is better than our smoothed previous path.
				// The fallback should emerge pre-smoothed, so that should be a non-issue
				altCost := EvaluatePlan(alternate, pathPlanner.opt().DistanceFunc)
				if altCost < score {
					pm.logger.Debugf("replacing path with score %f with better score %f", score, altCost)
					finalSteps = &rrtPlanReturn{steps: stepsToNodes(alternate)}
				} else {
					pm.logger.Debugf("fallback path with score %f worse than original score %f; using original", altCost, score)
				}
			}
		}

		solutionChan <- finalSteps
		return

	case <-ctx.Done():
		return
	}
}

// This is where the map[string]interface{} passed in via `extra` is used to decide how planning happens.
func (pm *planManager) plannerSetupFromMoveRequest(
	from, to spatialmath.Pose,
	seedMap map[string][]referenceframe.Input,
	worldState *referenceframe.WorldState,
	planningOpts map[string]interface{},
) (*plannerOptions, error) {
	// Start with normal options
	opt := newBasicPlannerOptions()

	opt.extra = planningOpts

	// add collision constraints
	selfCollisionConstraint, err := newSelfCollisionConstraint(pm.frame, seedMap, []*Collision{}, getCollisionDepth)
	if err != nil {
		return nil, err
	}
	obstacleConstraint, err := newObstacleConstraint(pm.frame, pm.fs, worldState, seedMap, []*Collision{}, getCollisionDepth)
	if err != nil {
		return nil, err
	}
	opt.AddConstraint(defaultObstacleConstraintName, obstacleConstraint)
	opt.AddConstraint(defaultSelfCollisionConstraintName, selfCollisionConstraint)

	// error handling around extracting motion_profile information from map[string]interface{}
	var motionProfile string
	profile, ok := planningOpts["motion_profile"]
	if ok {
		motionProfile, ok = profile.(string)
		if !ok {
			return nil, errors.New("could not interpret motion_profile field as string")
		}
	}

	// convert map to json, then to a struct, overwriting present defaults
	jsonString, err := json.Marshal(planningOpts)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(jsonString, opt)
	if err != nil {
		return nil, err
	}

	var planAlg string
	alg, ok := planningOpts["planning_alg"]
	if ok {
		planAlg, ok = alg.(string)
		if !ok {
			return nil, errors.New("could not interpret planning_alg field as string")
		}
		switch planAlg {
		// TODO(pl): make these consts
		case "cbirrt":
			opt.PlannerConstructor = newCBiRRTMotionPlanner
		case "rrtstar":
			// no motion profiles for RRT*
			opt.PlannerConstructor = newRRTStarConnectMotionPlanner
			// TODO(pl): more logic for RRT*?
			return opt, nil
		default:
			// use default, already set
		}
	}

	switch motionProfile {
	case LinearMotionProfile:
		// Linear constraints
		linTol, ok := planningOpts["line_tolerance"].(float64)
		if !ok {
			// Default
			linTol = defaultLinearDeviation
		}
		orientTol, ok := planningOpts["orient_tolerance"].(float64)
		if !ok {
			// Default
			orientTol = defaultLinearDeviation
		}
		constraint, pathDist := NewAbsoluteLinearInterpolatingConstraint(from, to, linTol, orientTol)
		opt.AddConstraint(defaultLinearConstraintName, constraint)
		opt.pathDist = pathDist
	case PseudolinearMotionProfile:
		tolerance, ok := planningOpts["tolerance"].(float64)
		if !ok {
			// Default
			tolerance = defaultPseudolinearTolerance
		}
		constraint, pathDist := NewProportionalLinearInterpolatingConstraint(from, to, tolerance)
		opt.AddConstraint(defaultPseudolinearConstraintName, constraint)
		opt.pathDist = pathDist
	case OrientationMotionProfile:
		tolerance, ok := planningOpts["tolerance"].(float64)
		if !ok {
			// Default
			tolerance = defaultOrientationDeviation
		}
		constraint, pathDist := NewSlerpOrientationConstraint(from, to, tolerance)
		opt.AddConstraint(defaultOrientationConstraintName, constraint)
		opt.pathDist = pathDist
	case PositionOnlyMotionProfile:
		opt.SetMetric(NewPositionOnlyMetric())
	case FreeMotionProfile:
		// No restrictions on motion
		fallthrough
	default:
		if planAlg == "" {
			// set up deep copy for fallback
			try1 := deepAtomicCopyMap(planningOpts)
			// No need to generate tons more IK solutions when the first alg will do it

			// time to run the first planning attempt before falling back
			try1["timeout"] = defaultFallbackTimeout
			try1["planning_alg"] = "rrtstar"
			try1Opt, err := pm.plannerSetupFromMoveRequest(from, to, seedMap, worldState, try1)
			if err != nil {
				return nil, err
			}

			try1Opt.Fallback = opt
			opt = try1Opt
		}
	}
	return opt, nil
}

// check whether the solution is within some amount of the optimal.
func goodPlan(pr *rrtPlanReturn, opt *plannerOptions) (bool, float64) {
	solutionCost := math.Inf(1)
	if pr.steps != nil {
		if pr.maps.optNode.cost <= 0 {
			return true, solutionCost
		}
		solutionCost = EvaluatePlan(pr.toInputs(), opt.DistanceFunc)
		if solutionCost < pr.maps.optNode.cost*defaultOptimalityMultiple {
			return true, solutionCost
		}
	}

	return false, solutionCost
}

// Copy any atomic values.
func deepAtomicCopyMap(opt map[string]interface{}) map[string]interface{} {
	optCopy := map[string]interface{}{}
	for k, v := range opt {
		optCopy[k] = v
	}
	return optCopy
}
