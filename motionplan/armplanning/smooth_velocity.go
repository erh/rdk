package armplanning

import (
	"context"
	"fmt"
	"math"
	"time"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan"
	"go.viam.com/rdk/motionplan/ik"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
)

// findCartesianBreakpoints identifies trajectory indices where the Cartesian direction
// changes significantly OR where segments exceed a maximum length.
// Returns a sorted list of indices that always includes 0 and len(traj)-1.
func findCartesianBreakpoints(
	fs *referenceframe.FrameSystem,
	traj []*referenceframe.LinearInputs,
	angleThresholdDeg float64,
	logger logging.Logger,
) ([]int, error) {
	if len(traj) < 3 {
		return []int{0, len(traj) - 1}, nil
	}

	// Compute Cartesian positions for each trajectory step, per frame.
	type vec3 struct{ x, y, z float64 }
	positionsByFrame := map[string][]vec3{}

	for i, li := range traj {
		poses, err := li.ComputePoses(fs)
		if err != nil {
			return nil, fmt.Errorf("FK error at step %d: %w", i, err)
		}
		for frameName, pif := range poses {
			pt := pif.Pose().Point()
			positionsByFrame[frameName] = append(positionsByFrame[frameName], vec3{pt.X, pt.Y, pt.Z})
		}
	}

	cosThreshold := math.Cos(angleThresholdDeg * math.Pi / 180.0)

	breakpoints := map[int]bool{0: true, len(traj) - 1: true}

	// Add distance-based breakpoints to prevent segments from being too long
	// Long segments struggle to achieve good velocity consistency
	// 125mm provides good balance between segment consistency and boundary discontinuities
	maxSegmentDistanceMM := 125.0

	for frameName, positions := range positionsByFrame {
		if len(positions) < 2 {
			continue
		}

		accumulatedDistance := 0.0

		for i := 1; i < len(positions); i++ {
			// Calculate distance from previous point
			dx := positions[i].x - positions[i-1].x
			dy := positions[i].y - positions[i-1].y
			dz := positions[i].z - positions[i-1].z
			dist := math.Sqrt(dx*dx + dy*dy + dz*dz)
			accumulatedDistance += dist

			// If we've accumulated >150mm since last breakpoint, add one
			if accumulatedDistance > maxSegmentDistanceMM {
				breakpoints[i] = true
				logger.Debugf("smoothVelocities distance breakpoint at step %d frame=%s (%.1fmm from last breakpoint)",
					i, frameName, accumulatedDistance)
				accumulatedDistance = 0.0
			}
		}
	}

	for frameName, positions := range positionsByFrame {
		if len(positions) < 3 {
			continue
		}

		for i := 1; i < len(positions)-1; i++ {
			// Direction from i-1 to i
			dx1 := positions[i].x - positions[i-1].x
			dy1 := positions[i].y - positions[i-1].y
			dz1 := positions[i].z - positions[i-1].z
			norm1 := math.Sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)

			// Direction from i to i+1
			dx2 := positions[i+1].x - positions[i].x
			dy2 := positions[i+1].y - positions[i].y
			dz2 := positions[i+1].z - positions[i].z
			norm2 := math.Sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)

			if norm1 < 1e-9 || norm2 < 1e-9 {
				continue
			}

			dot := (dx1*dx2 + dy1*dy2 + dz1*dz2) / (norm1 * norm2)
			// Clamp for numerical stability
			if dot > 1 {
				dot = 1
			}
			if dot < -1 {
				dot = -1
			}

			if dot < cosThreshold {
				angleDeg := math.Acos(dot) * 180.0 / math.Pi
				logger.Debugf("smoothVelocities breakpoint at step %d frame=%s angle=%0.1f°", i, frameName, angleDeg)
				breakpoints[i] = true
			}
		}
	}

	// Sort breakpoints
	sorted := make([]int, 0, len(breakpoints))
	for idx := range breakpoints {
		sorted = append(sorted, idx)
	}
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	return sorted, nil
}

var optimizeCallCount = 0

// optimizeForVelocityConsistency uses nlopt to find a joint configuration that:
// 1. Maintains the target Cartesian pose (critical for linear constraints)
// 2. Improves velocity consistency compared to the original
// 3. Satisfies all constraints
func optimizeForVelocityConsistency(
	ctx context.Context,
	psc *planSegmentContext,
	targetPoses referenceframe.FrameSystemPoses,
	originalConfig, prevConfig, nextConfig *referenceframe.LinearInputs,
	logger logging.Logger,
) (*referenceframe.LinearInputs, error) {
	optimizeCallCount++
	if optimizeCallCount%50 == 1 {
		logger.Infof("optimizeForVelocityConsistency: call #%d", optimizeCallCount)
	}

	// Create nlopt solver - maximize iterations for best velocity consistency
	// Allow inexact solutions since we care more about velocity than perfect convergence
	solver, err := ik.CreateNloptSolver(logger, 1000, false, false, 500*time.Millisecond)
	if err != nil {
		return nil, err
	}

	// Get limits for all DOFs from the schema
	limits := psc.pc.lis.GetLimits()

	// Compute target joint velocities from neighbors
	// We want each individual joint to have constant velocity
	prevJoints := prevConfig.GetLinearizedInputs()
	origJoints := originalConfig.GetLinearizedInputs()
	nextJoints := nextConfig.GetLinearizedInputs()

	targetJoints := make([]float64, len(origJoints))
	for i := range origJoints {
		// Target: linear interpolation between prev and next
		targetJoints[i] = (prevJoints[i] + nextJoints[i]) / 2.0
	}

	// Generate multiple seeds to explore different IK solutions
	seeds := [][]float64{}
	seedLimits := [][]referenceframe.Limit{}

	// Seed 1: Original configuration
	seeds = append(seeds, originalConfig.GetLinearizedInputs())
	seedLimits = append(seedLimits, limits)

	// Seed 2: Target joints (ideal for velocity)
	seeds = append(seeds, targetJoints)
	seedLimits = append(seedLimits, limits)

	// Seed 3: Blend of original and target
	blendSeed := make([]float64, len(targetJoints))
	for i := range targetJoints {
		if i < len(origJoints) {
			blendSeed[i] = 0.7*targetJoints[i] + 0.3*origJoints[i]
		} else {
			blendSeed[i] = targetJoints[i]
		}
	}
	seeds = append(seeds, blendSeed)
	seedLimits = append(seedLimits, limits)

	// Build cost function that prioritizes joint-level velocity consistency
	costFunc := func(ctx context.Context, joints []float64) float64 {
		config, err := psc.pc.lis.FloatsToInputs(joints)
		if err != nil {
			return math.Inf(1)
		}

		// 1. Constraint checking FIRST (hard constraint)
		// The constraint checker will enforce Cartesian accuracy via linear constraints
		_, constraintErr := psc.checker.CheckStateFSConstraints(ctx, &motionplan.StateFS{
			Configuration: config,
			FS:            psc.pc.fs,
		})
		if constraintErr != nil {
			return 10000000.0 // Very hard constraint - invalid configuration
		}

		// 2. ONLY priority: Individual joint velocity consistency
		// We want to hit the target joint values that give constant velocity
		// Since constraints are satisfied, focus purely on velocity
		jointVelocityError := 0.0
		for i, targetVal := range targetJoints {
			if i < len(joints) {
				deviation := math.Abs(joints[i] - targetVal)
				jointVelocityError += deviation * deviation // Square to heavily penalize
			}
		}

		return jointVelocityError
	}

	// Solve with multiple seeds to explore solution space
	solutionChan := make(chan *ik.Solution, 30)

	go func() {
		defer close(solutionChan)
		_, _, _ = solver.Solve(ctx, solutionChan, seeds, seedLimits, costFunc, 0)
	}()

	// Get best solution
	var bestSolution *ik.Solution
	for sol := range solutionChan {
		if bestSolution == nil || sol.Score < bestSolution.Score {
			bestSolution = sol
		}
	}

	// Compute original joint velocity error for comparison
	origVelocityError := 0.0
	for i, targetVal := range targetJoints {
		if i < len(origJoints) {
			deviation := math.Abs(origJoints[i] - targetVal)
			origVelocityError += deviation * deviation
		}
	}

	if bestSolution == nil {
		logger.Infof("optimizeForVelocityConsistency: no solution found, using original (orig velocity error=%0.6f)",
			origVelocityError)
		return originalConfig, nil
	}

	// Check if optimization actually improved things
	result, err := psc.pc.lis.FloatsToInputs(bestSolution.Configuration)
	if err != nil {
		logger.Infof("optimizeForVelocityConsistency: FloatsToInputs failed, using original")
		return originalConfig, nil
	}

	// Compute new joint velocity error
	resultJoints := result.GetLinearizedInputs()
	newVelocityError := 0.0
	for i, targetVal := range targetJoints {
		if i < len(resultJoints) {
			deviation := math.Abs(resultJoints[i] - targetVal)
			newVelocityError += deviation * deviation
		}
	}

	improvement := (1.0 - newVelocityError/origVelocityError) * 100.0

	// Log every 50th call to avoid spam
	if optimizeCallCount%50 == 1 {
		logger.Infof("optimizeForVelocityConsistency #%d: score=%0.2f origVelErr=%0.6f newVelErr=%0.6f improvement=%.1f%%",
			optimizeCallCount, bestSolution.Score, origVelocityError, newVelocityError, improvement)
	}

	// Only use optimized result if it actually improved or score is very good
	if bestSolution.Score > 100.0 && newVelocityError >= origVelocityError {
		logger.Infof("optimizeForVelocityConsistency: optimization didn't improve, using original")
		return originalConfig, nil
	}

	return result, nil
}

// optimizeBreakpointConfiguration uses nlopt to adjust a breakpoint configuration to:
// 1. Achieve the target Cartesian pose
// 2. Enable smooth velocity transitions across the breakpoint
// 3. Satisfy all constraints
func optimizeBreakpointConfiguration(
	ctx context.Context,
	psc *planSegmentContext,
	targetPoses referenceframe.FrameSystemPoses,
	initialConfig *referenceframe.LinearInputs,
	logger logging.Logger,
) (*referenceframe.LinearInputs, error) {
	// Create nlopt solver for breakpoint optimization
	solver, err := ik.CreateNloptSolver(logger, 300, false, false, 300*time.Millisecond)
	if err != nil {
		return nil, err
	}

	// Get limits for all DOFs from the schema
	limits := psc.pc.lis.GetLimits()

	// Build cost function that prioritizes Cartesian accuracy at breakpoints
	costFunc := func(ctx context.Context, joints []float64) float64 {
		config, err := psc.pc.lis.FloatsToInputs(joints)
		if err != nil {
			return math.Inf(1)
		}

		cost := 0.0

		// 1. Cartesian position error (high priority at breakpoints)
		poses, err := config.ComputePoses(psc.pc.fs)
		if err != nil {
			return math.Inf(1)
		}

		for frameName, targetPIF := range targetPoses {
			actualPIF, ok := poses[frameName]
			if !ok {
				continue
			}
			dist := spatialmath.PoseDelta(actualPIF.Pose(), targetPIF.Pose())
			cost += dist.Point().Norm() // mm
			// Orientation error
			orientDist := dist.Orientation().AxisAngles()
			cost += math.Abs(orientDist.Theta) * 100.0 // radians * mm
		}

		// 2. Constraint checking
		_, constraintErr := psc.checker.CheckStateFSConstraints(ctx, &motionplan.StateFS{
			Configuration: config,
			FS:            psc.pc.fs,
		})
		if constraintErr != nil {
			cost += 10000.0
		}

		// 3. Small penalty for large deviations from initial config (smoothness)
		delta := psc.pc.configurationDistanceFunc(&motionplan.SegmentFS{
			StartConfiguration: initialConfig,
			EndConfiguration:   config,
			FS:                 psc.pc.fs,
		})
		cost += delta * 0.01

		return cost
	}

	// Solve with nlopt
	seedFloats := initialConfig.GetLinearizedInputs()
	solutionChan := make(chan *ik.Solution, 10)

	go func() {
		defer close(solutionChan)
		_, _, _ = solver.Solve(ctx, solutionChan, [][]float64{seedFloats}, [][]referenceframe.Limit{limits}, costFunc, 0)
	}()

	// Get best solution
	var bestSolution *ik.Solution
	for sol := range solutionChan {
		if bestSolution == nil || sol.Score < bestSolution.Score {
			bestSolution = sol
		}
	}

	if bestSolution == nil || bestSolution.Score > 1.0 {
		// If optimization didn't find a good solution, return initial
		logger.Debugf("optimizeBreakpointConfiguration: no good solution found, using initial")
		return initialConfig, nil
	}

	result, err := psc.pc.lis.FloatsToInputs(bestSolution.Configuration)
	if err != nil {
		return initialConfig, nil
	}

	logger.Debugf("optimizeBreakpointConfiguration: score=%0.2f", bestSolution.Score)
	return result, nil
}

func smoothVelocities(ctx context.Context,
	segmentContexts []*planSegmentContext,
	traj []*referenceframe.LinearInputs,
	logger logging.Logger,
) ([]*referenceframe.LinearInputs, []int, error) {
	if len(segmentContexts)+1 != len(traj) {
		// because we put the start position
		return nil, nil, fmt.Errorf("lengths wrong segmentContexts: %d linearinputs: %d", len(segmentContexts), len(traj))
	}

	fs := segmentContexts[0].pc.fs

	// We'll determine per-segment whether to use optimization or interpolation
	// based on whether interpolation succeeds

	// Find breakpoints where Cartesian direction changes significantly
	breakpoints, err := findCartesianBreakpoints(fs, traj, 5, logger)
	if err != nil {
		return traj, nil, err
	}

	logger.Infof("smoothVelocities found %d breakpoints: %v", len(breakpoints), breakpoints)

	// Build a new trajectory by linearly interpolating within each segment
	newTraj := make([]*referenceframe.LinearInputs, len(traj))

	// Copy breakpoint positions directly
	for _, bp := range breakpoints {
		newTraj[bp] = traj[bp]
	}

	// First pass: Try to optimize breakpoints for better Cartesian accuracy
	// This uses nlopt to solve IK for breakpoint poses
	for _, bp := range breakpoints {
		if bp == 0 || bp == len(traj)-1 {
			// Keep start and end fixed
			continue
		}

		// Get the target Cartesian pose for this breakpoint
		targetPoses, err := traj[bp].ComputePoses(fs)
		if err != nil {
			logger.Debugf("smoothVelocities: FK error at breakpoint %d: %v", bp, err)
			continue
		}

		// Pick the segment context
		pscIdx := bp - 1
		if pscIdx >= len(segmentContexts) {
			pscIdx = len(segmentContexts) - 1
		}
		if pscIdx < 0 {
			continue
		}
		psc := segmentContexts[pscIdx]

		// Try to optimize this breakpoint configuration
		optimized, err := optimizeBreakpointConfiguration(
			ctx, psc, targetPoses, traj[bp], logger.Sublogger("bp"),
		)
		if err == nil {
			// Verify path to/from this breakpoint using original trajectory
			prevOK := true
			nextOK := true
			if bp > 0 {
				prevNode := newTraj[bp-1]
				if prevNode == nil {
					prevNode = traj[bp-1]
				}
				prevOK = psc.checkPath(ctx, prevNode, optimized, false) == nil
			}
			if bp < len(traj)-1 {
				nextNode := traj[bp+1]
				if newTraj[bp+1] != nil {
					nextNode = newTraj[bp+1]
				}
				nextOK = psc.checkPath(ctx, optimized, nextNode, false) == nil
			}

			if prevOK && nextOK {
				newTraj[bp] = optimized
				logger.Debugf("smoothVelocities: optimized breakpoint %d", bp)
			}
		}
	}

	replaced := 0

	// Second pass: Process points between breakpoints
	// Try interpolation first; use optimization only if many points fail
	for seg := 0; seg < len(breakpoints)-1; seg++ {
		segStart := breakpoints[seg]
		segEnd := breakpoints[seg+1]
		segLen := segEnd - segStart

		if segLen <= 1 {
			continue
		}

		startJoints := newTraj[segStart]
		endJoints := newTraj[segEnd]

		// First attempt: Try simple joint-space interpolation
		interpSuccesses := 0
		interpFailures := 0
		tempResults := make([]*referenceframe.LinearInputs, segLen+1)
		tempResults[0] = startJoints
		tempResults[segLen] = endJoints

		for i := segStart + 1; i < segEnd; i++ {
			by := float64(i-segStart) / float64(segLen)

			interpJoints, err := referenceframe.InterpolateFS(fs, startJoints, endJoints, by)
			if err != nil {
				interpFailures++
				continue
			}

			pscIdx := i - 1
			if pscIdx >= len(segmentContexts) {
				pscIdx = len(segmentContexts) - 1
			}
			if pscIdx < 0 {
				pscIdx = 0
			}
			psc := segmentContexts[pscIdx]

			_, constraintErr := psc.checker.CheckStateFSConstraints(ctx, &motionplan.StateFS{
				Configuration: interpJoints,
				FS:            fs,
			})

			if constraintErr == nil {
				tempResults[i-segStart] = interpJoints
				interpSuccesses++
			} else {
				interpFailures++
			}
		}

		totalPoints := segEnd - segStart - 1
		successRate := float64(interpSuccesses) / float64(totalPoints)

		// If interpolation works well (>80% success), use it
		if successRate > 0.8 {
			logger.Debugf("smoothVelocities seg %d: using joint-space interpolation (%d/%d succeeded)",
				seg, interpSuccesses, totalPoints)
			for i := segStart + 1; i < segEnd; i++ {
				if tempResults[i-segStart] != nil {
					newTraj[i] = tempResults[i-segStart]
					replaced++
				} else {
					newTraj[i] = traj[i]
				}
			}
		} else {
			// Interpolation failed for many points: Use optimization
			logger.Debugf("smoothVelocities seg %d: using optimization (interp only %d/%d succeeded)",
				seg, interpSuccesses, totalPoints)

			// Pre-compute target Cartesian poses from original trajectory
			targetPosesBySeg := make([]referenceframe.FrameSystemPoses, segLen+1)
			for i := segStart; i <= segEnd; i++ {
				poses, err := traj[i].ComputePoses(fs)
				if err != nil {
					logger.Infof("smoothVelocities seg %d FK error at %d: %v", seg, i, err)
					// Keep original on error
					for j := segStart + 1; j < segEnd; j++ {
						newTraj[j] = traj[j]
					}
					continue
				}
				targetPosesBySeg[i-segStart] = poses
			}

			// Optimize each interior point
			for i := segStart + 1; i < segEnd; i++ {
				pscIdx := i - 1
				if pscIdx >= len(segmentContexts) {
					pscIdx = len(segmentContexts) - 1
				}
				if pscIdx < 0 {
					pscIdx = 0
				}
				psc := segmentContexts[pscIdx]

				prevJoints := newTraj[i-1]
				if prevJoints == nil {
					prevJoints = traj[i-1]
				}
				nextJoints := traj[i+1]
				if i+1 < len(newTraj) && newTraj[i+1] != nil {
					nextJoints = newTraj[i+1]
				}

				targetPoses := targetPosesBySeg[i-segStart]
				optimized, err := optimizeForVelocityConsistency(
					ctx, psc, targetPoses, traj[i], prevJoints, nextJoints, logger.Sublogger("opt"),
				)
				if err == nil {
					newTraj[i] = optimized
					replaced++
				} else {
					newTraj[i] = traj[i]
				}
			}

			// Iterative refinement: now that we have first-pass optimized trajectory,
			// measure actual joint deltas and do a second optimization pass
			// focusing on the worst joints
			logger.Debugf("smoothVelocities seg %d: starting iterative refinement pass", seg)

			for pass := 0; pass < 2; pass++ {
				// Compute joint delta stats for this segment
				segTraj := make(motionplan.Trajectory, segEnd-segStart+1)
				for i := segStart; i <= segEnd; i++ {
					if newTraj[i] == nil {
						segTraj[i-segStart] = map[string][]referenceframe.Input{}
					} else {
						// Convert LinearInputs to Trajectory format
						step := make(map[string][]referenceframe.Input)
						for k, v := range newTraj[i].Items() {
							step[k] = v
						}
						segTraj[i-segStart] = step
					}
				}

				// Analyze which joints need improvement
				stats := TrajectoryDeltaStats(segTraj)
				if stats == nil || len(stats) == 0 {
					break
				}

				// Find joints with StdDev > threshold
				needsWork := []int{}
				for _, s := range stats {
					if s.StdDev > 0.0002 { // Target improvement for joints above 0.0002
						needsWork = append(needsWork, s.JointIdx)
					}
				}

				if len(needsWork) == 0 {
					logger.Debugf("smoothVelocities seg %d pass %d: all joints below threshold, stopping", seg, pass)
					break
				}

				logger.Infof("smoothVelocities seg %d pass %d: refining %d joints with high StdDev", seg, pass, len(needsWork))

				// Re-optimize points focusing on problematic joints
				// This time use ACTUAL neighbors from the current trajectory
				improvedCount := 0
				for i := segStart + 1; i < segEnd; i++ {
					pscIdx := i - 1
					if pscIdx >= len(segmentContexts) {
						pscIdx = len(segmentContexts) - 1
					}
					if pscIdx < 0 {
						pscIdx = 0
					}
					psc := segmentContexts[pscIdx]

					// Use actual optimized neighbors
					prevJoints := newTraj[i-1]
					nextJoints := newTraj[i+1]
					if prevJoints == nil || nextJoints == nil {
						continue
					}

					targetPoses := targetPosesBySeg[i-segStart]
					optimized, err := optimizeForVelocityConsistency(
						ctx, psc, targetPoses, newTraj[i], prevJoints, nextJoints, logger.Sublogger("refine"),
					)
					if err == nil {
						// Only accept if it improves things
						oldJoints := newTraj[i].GetLinearizedInputs()
						newJoints := optimized.GetLinearizedInputs()
						prevJ := prevJoints.GetLinearizedInputs()
						nextJ := nextJoints.GetLinearizedInputs()

						// Compute old and new velocity errors
						oldErr := 0.0
						newErr := 0.0
						for _, jointIdx := range needsWork {
							if jointIdx >= len(oldJoints) {
								continue
							}
							oldTarget := (prevJ[jointIdx] + nextJ[jointIdx]) / 2.0
							oldDev := oldJoints[jointIdx] - oldTarget
							newDev := newJoints[jointIdx] - oldTarget
							oldErr += oldDev * oldDev
							newErr += newDev * newDev
						}

						if newErr < oldErr*0.95 { // Accept if 5%+ improvement
							newTraj[i] = optimized
							improvedCount++
						}
					}
				}
				logger.Infof("smoothVelocities seg %d pass %d: improved %d points", seg, pass, improvedCount)

				if improvedCount == 0 {
					break // No more improvements possible
				}
			}
		}
	}

	total := len(traj) - len(breakpoints)
	logger.Infof("smoothVelocities replaced %d/%d intermediate trajectory points (%d segments)",
		replaced, total, len(breakpoints)-1)
	return newTraj, breakpoints, nil
}
