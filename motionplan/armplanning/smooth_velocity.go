package armplanning

import (
	"context"
	"fmt"
	"math"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan"
	"go.viam.com/rdk/referenceframe"
)

// findCartesianBreakpoints identifies trajectory indices where the Cartesian direction
// changes significantly. Returns a sorted list of indices that always includes 0 and len(traj)-1.
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

	replaced := 0

	// Interpolate within each segment between consecutive breakpoints
	for seg := 0; seg < len(breakpoints)-1; seg++ {
		segStart := breakpoints[seg]
		segEnd := breakpoints[seg+1]
		segLen := segEnd - segStart

		if segLen <= 1 {
			continue
		}

		startJoints := traj[segStart]
		endJoints := traj[segEnd]

		for i := segStart + 1; i < segEnd; i++ {
			by := float64(i-segStart) / float64(segLen)

			interpJoints, err := referenceframe.InterpolateFS(fs, startJoints, endJoints, by)
			if err != nil {
				logger.Infof("smoothVelocities seg %d step %d interpolation error: %v", seg, i, err)
				newTraj[i] = traj[i]
				continue
			}

			// Pick the segment context for constraint checking
			pscIdx := i
			if pscIdx >= len(segmentContexts) {
				pscIdx = len(segmentContexts) - 1
			}
			psc := segmentContexts[pscIdx]

			_, constraintErr := psc.checker.CheckStateFSConstraints(ctx, &motionplan.StateFS{
				Configuration: interpJoints,
				FS:            fs,
			})

			if constraintErr == nil {
				newTraj[i] = interpJoints
				replaced++
			} else {
				logger.Infof("smoothVelocities seg %d step %d/%d interp fails constraints: %v",
					seg, i, len(traj)-1, constraintErr)
				newTraj[i] = traj[i]
			}
		}
	}

	total := len(traj) - len(breakpoints)
	logger.Infof("smoothVelocities replaced %d/%d intermediate trajectory points (%d segments)",
		replaced, total, len(breakpoints)-1)
	return newTraj, breakpoints, nil
}
